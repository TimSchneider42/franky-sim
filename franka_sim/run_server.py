#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging

from franka_sim import SimulationServer
from franka_sim.genesis_sim import GenesisSimulation


def main() -> None:
    """Run the Franka simulation server."""
    parser = argparse.ArgumentParser(
        description="Run a standard Franka simulation server with one robot."
    )
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

    # Create the simulation
    with GenesisSimulation(enable_visualization=args.vis) as sim:
        robot = sim.add_robot()
        with SimulationServer(sim) as server:
            print(f"Connect to the server using '{robot.hostname}' as the robot IP address")
            print("Press Ctrl+C to stop the server")
            try:
                server.run_forever()
            except KeyboardInterrupt:
                print("\nShutting down server...")


if __name__ == "__main__":
    main()
