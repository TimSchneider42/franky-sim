import logging
import socket
import threading
import time
from unittest.mock import Mock

import numpy as np
import pytest

from franka_sim.franka_protocol import COMMAND_PORT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock robot state for testing
MOCK_ROBOT_STATE = {"q": np.zeros(7), "dq": np.zeros(7), "tau_J": np.zeros(7)}


def wait_for_server(port, max_retries=20, retry_delay=0.2):
    """Helper function to wait for server to start"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} to connect to server...")
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.settimeout(1.0)
            test_socket.connect(("localhost", port))
            test_socket.close()
            logger.info("Successfully connected to server")
            return True
        except (ConnectionRefusedError, socket.timeout) as e:
            logger.warning(f"Connection attempt failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            continue
        except Exception as e:
            logger.error(f"Unexpected error while connecting: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            continue
    return False


@pytest.fixture
def mock_genesis_sim():
    """Fixture that provides a mocked Genesis simulator"""
    mock_robot = Mock()
    # MOCK_ROBOT_STATE is a dict in old code, we might need a mock RobotState
    from pathlib import Path

    import pinocchio as pin

    from franka_sim.franka_robot_state import FrankaRobotState

    mock_state = FrankaRobotState()
    mock_robot.state = mock_state

    # Load real pinocchio model to avoid C++ argument matching errors
    urdf_path = Path(__file__).parent.parent / "franka_sim" / "assets" / "fr3.urdf"
    if urdf_path.exists():
        mock_robot.model = pin.buildModelFromUrdf(str(urdf_path))
        mock_robot.data = mock_robot.model.createData()
        mock_robot.ee_frame_name = "fr3_hand_tcp"
    else:
        # Fallback to a dummy model if URDF doesn't exist
        mock_robot.model = pin.Model()
        mock_robot.data = pin.Data(mock_robot.model)
        mock_robot.ee_frame_name = "dummy_frame"

    mock_sim = Mock()
    mock_sim.robots = [mock_robot]
    mock_sim.get_robots.return_value = [mock_robot]
    return mock_sim


@pytest.fixture
def sim_server(mock_genesis_sim):
    """Fixture that provides a server with mocked Genesis simulator"""
    # First ensure no existing server is running
    try:
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.settimeout(1.0)
        test_socket.connect(("localhost", COMMAND_PORT))
        test_socket.close()
        raise RuntimeError("Port already in use. Make sure no other server is running.")
    except (ConnectionRefusedError, socket.timeout):
        pass

    from franka_sim.franka_sim_server import FrankaSimServer

    server = FrankaSimServer(sim=mock_genesis_sim)
    server_thread = threading.Thread(target=server.run_forever)
    server_thread.daemon = True

    try:
        logger.info("Starting server thread...")
        server.start()
        server_thread.start()
        logger.info("Waiting for server to start...")

        if not wait_for_server(COMMAND_PORT):
            logger.error("Server failed to start after maximum retries")
            server.stop()
            server_thread.join(timeout=1.0)
            raise RuntimeError("Server failed to start")

        logger.info("Server started successfully")
        yield server

    finally:
        logger.info("Cleaning up server...")
        try:
            # Make sure the server is stopped properly
            server.running = False

            # Stop the server (which calls cleanup)
            server.stop()

            # Join the server thread with a longer timeout
            if server_thread.is_alive():
                server_thread.join(timeout=3.0)

            # Additional socket cleanup - with better error handling
            for rs in server.robot_servers:
                if hasattr(rs, "client_socket") and rs.tcp_socket is not None:
                    try:
                        rs.tcp_socket.shutdown(socket.SHUT_RDWR)
                    except (socket.error, AttributeError) as e:
                        logger.debug(f"Error during client socket shutdown: {e}")
                    try:
                        rs.tcp_socket.close()
                    except (socket.error, AttributeError) as e:
                        logger.debug(f"Error during client socket close: {e}")
                    rs.tcp_socket = None

                if hasattr(rs, "udp_socket") and rs.udp_socket is not None:
                    try:
                        rs.udp_socket.close()
                    except (socket.error, AttributeError) as e:
                        logger.debug(f"Error during UDP socket close: {e}")
                    rs.udp_socket = None

                if hasattr(rs, "server_socket") and rs.server_socket is not None:
                    try:
                        rs.server_socket.shutdown(socket.SHUT_RDWR)
                    except (socket.error, AttributeError) as e:
                        logger.debug(f"Error during server socket shutdown: {e}")
                    try:
                        rs.server_socket.close()
                    except (socket.error, AttributeError) as e:
                        logger.debug(f"Error during server socket close: {e}")
                    rs.server_socket = None

            # Wait longer for sockets to fully close
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Error during server cleanup: {e}")
            # Continue with test teardown even if cleanup fails


@pytest.fixture
def tcp_client():
    """Fixture that provides a TCP client socket"""
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.settimeout(5.0)  # Increased timeout for reliability
    yield client
    try:
        client.shutdown(socket.SHUT_RDWR)
    except socket.error:
        pass
    try:
        client.close()
    except socket.error:
        pass


@pytest.fixture
def udp_client():
    """Fixture that provides a UDP client socket"""
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.settimeout(5.0)  # Increased timeout for reliability
    yield client
    try:
        client.close()
    except socket.error:
        pass


@pytest.fixture(autouse=True)
def cleanup_sockets():
    """Fixture to ensure sockets are cleaned up after each test"""
    yield
    # Clean up any lingering sockets in TIME_WAIT state
    time.sleep(0.2)  # Increased delay to allow socket cleanup
