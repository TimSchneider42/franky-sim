from __future__ import annotations

import logging
import threading
import time
from typing import Iterable, Optional

from .base_simulator import BaseSimulator
from .franka_robot_server import FrankaRobotServer

logger = logging.getLogger(__name__)


class LocalHostnames:
    """Iterable over all usable 127.x.y.z loopback addresses (excluding .0.0.0 and .255.255.255)."""

    def __iter__(self):
        for x in range(256):
            for y in range(256):
                for z in range(256):
                    # Skip the Network Address (127.0.0.0)
                    if x == 0 and y == 0 and z == 0:
                        continue

                    # Skip the Broadcast Address (127.255.255.255)
                    if x == 255 and y == 255 and z == 255:
                        continue

                    yield f"127.{x}.{y}.{z}"


class SimulationServer:
    """Ties a BaseSimulator to one FrankaRobotServer per robot and runs the control loop."""

    def __init__(
        self,
        sim: BaseSimulator,
        hostname_candidates: Iterable[str] = LocalHostnames(),
    ):
        self._hostname_candidates = hostname_candidates
        self.sim: BaseSimulator = sim
        self.robot_servers: list[FrankaRobotServer] = []
        self.running: bool = False
        self.async_thread: Optional[threading.Thread] = None

    @property
    def _remaining_hostname_candidates(self) -> Iterable[str]:
        for h in self._hostname_candidates:
            if h not in {r.hostname for r in self.robot_servers}:
                yield h

    def init(self) -> None:
        """Create and bind a FrankaRobotServer for each robot, then start the simulation."""
        for robot in self.sim.robots:
            rs = FrankaRobotServer(robot, self._remaining_hostname_candidates)
            rs.init()
            self.robot_servers.append(rs)

        self.sim.start()

    def run_once(self, realtime: bool | float = True):
        """Process commands, step the simulation, and broadcast state for one 1 ms tick.

        Args:
            realtime: If True or a positive float, sleep to maintain 1 kHz wall-clock rate
                      (float acts as a speed multiplier; False disables sleeping).
        """
        start_time = time.time()

        for rs in self.robot_servers:
            rs.process_commands()

        self.sim.step()

        for rs in self.robot_servers:
            rs.send_state()

        time.sleep(max(0.0, 0.001 * float(realtime) - (time.time() - start_time)))

    def run_forever(self, realtime: bool | float = True):
        """Block and run the control loop until stop() is called or KeyboardInterrupt."""
        self.running = True
        while self.running:
            self.run_once(realtime)

    def run_async(self, realtime: bool | float = True):
        """Start the control loop in a background daemon thread."""
        self.async_thread = threading.Thread(target=self.run_forever, args=(realtime,), daemon=True)
        self.async_thread.start()

    def cleanup(self) -> None:
        """Stop the control loop, join the async thread if running, and clean up all servers."""
        self.running = False
        if self.async_thread and self.async_thread.is_alive():
            if self.async_thread is not threading.current_thread():
                self.async_thread.join()

        for rs in self.robot_servers:
            rs.cleanup()

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
