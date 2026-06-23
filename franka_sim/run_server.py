#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Type

import franka_sim
from franka_sim import BaseSimulator

SIMULATOR_NAMES = [
    e.stem[: -len("simulator") - 1]
    for e in Path(franka_sim.__file__).parent.iterdir()
    if e.stem.endswith("_simulator") and e.stem != "base_simulator"
]


def load_simulator(
    simulator_name: str, force_success: bool = True
) -> tuple[Type[franka_sim.BaseSimulator], str] | None:
    if simulator_name == "auto":
        for name in SIMULATOR_NAMES:
            simulator = load_simulator(name)
            if simulator is not None:
                return simulator
        return None
    try:
        module_name = f"{simulator_name}_simulator"
        module = getattr(__import__(f"franka_sim.{module_name}"), module_name)
        simulator = getattr(
            module,
            [e for e in dir(module) if e.lower() == simulator_name + "simulator"][0],
        )
        display_name = simulator.__name__[: -len("simulator")]
        print(f"Successfully loaded {display_name} simulator.")
        return simulator, display_name
    except ImportError:
        if force_success:
            raise ValueError(f"Simulator {simulator_name} could not be loaded.")
        return None


def main() -> None:
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
        choices=["auto"] + SIMULATOR_NAMES,
        default="auto",
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

    Simulator, display_name = load_simulator(args.simulator)

    print(
        f"Starting {display_name} simulation server {'with' if args.render else 'without'} "
        "visualization"
    )

    # Create the simulation
    with Simulator(enable_visualization=args.render) as sim:
        robot = sim.add_robot()
        with franka_sim.SimulationServer(sim) as server:
            print(f"Connect to the server using '{robot.hostname}' as the robot IP address")
            print("Press Ctrl+C to stop the server")
            try:
                server.run_forever()
            except KeyboardInterrupt:
                print("\nShutting down server...")


if __name__ == "__main__":
    main()
