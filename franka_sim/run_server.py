#!/usr/bin/env python3
import argparse
import logging

from franka_sim.franka_genesis_sim import FrankaGenesisSim
from franka_sim.franka_sim_server import FrankaSimServer


def main() -> None:
    """Run the Franka simulation server."""
    parser = argparse.ArgumentParser(description="Run a Franka simulation server")
    parser.add_argument(
        "-v",
        "--vis",
        action="store_true",
        default=False,
        help="Enable visualization of the Genesis simulator",
    )
    parser.add_argument(
        "-V", "--verbose", action="store_true", default=False, help="Enable verbose logging"
    )
    args = parser.parse_args()

    # Configure detailed logging for debugging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Configure logging to silence Numba debug output
    logging.getLogger("numba").setLevel(logging.WARNING)

    print(f"Starting Franka Simulation Server {'with' if args.vis else 'without'} visualization")
    print("Connect to the server using 'localhost' or '127.0.0.1' as the robot IP address")
    print("Press Ctrl+C to stop the server")

    # Create the simulation
    sim = FrankaGenesisSim(enable_vis=args.vis)
    sim.start()

    # Start the server with the simulation
    server = FrankaSimServer(sim=sim)
    server.start()

    try:
        server.run_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        sim.stop()


if __name__ == "__main__":
    main()
