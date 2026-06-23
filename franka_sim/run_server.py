#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging

from franka_sim import SimulationServer

AVAILABLE_SIMULATORS = {}
for s in ["genesis_sim"]:
    try:
        module = getattr(__import__(f"franka_sim.{s}"), s)
        name = s[:-4]
        simulator = getattr(module, [e for e in dir(module) if e.lower() == name + "simulator"][0])
        display_name = simulator.__name__[: -len("simulator")]
        AVAILABLE_SIMULATORS[name] = (simulator, display_name)
        print(f"Successfully loaded {display_name} simulator.")
    except ImportError:
        pass


def main() -> None:
    """Run the Franka simulation server."""
    parser = argparse.ArgumentParser(
        description="Run a standard Franka simulation server with one robot."
    )
    parser.add_argument(
        "-r",
        "--render",
        action="store_true",
        default=False,
        help="Render a visualization of the simulator",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )
    parser.add_argument(
        "-s",
        "--simulator",
        choices=AVAILABLE_SIMULATORS.keys(),
        default=list(AVAILABLE_SIMULATORS.keys())[0],
        help="Simulator to use.",
    )
    args = parser.parse_args()

    # Configure detailed logging for debugging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Configure logging to silence Numba debug output
    logging.getLogger("numba").setLevel(logging.WARNING)

    Simulator, display_name = AVAILABLE_SIMULATORS[args.simulator]

    print(
        f"Starting {display_name} simulation server {'with' if args.render else 'without'} "
        "visualization"
    )

    # Create the simulation
    with Simulator(enable_visualization=args.render) as sim:
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
