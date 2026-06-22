from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from .base_simulator import BaseSimulator
from .robot_server import RobotServer

logger = logging.getLogger(__name__)


class SimulationServer:
    def __init__(
        self, sim: BaseSimulator, hostnames: Callable[[int], str] = lambda i: f"127.0.0.{i + 1}"
    ):
        self.robot_hostnames: Callable[[int], str] = hostnames
        self.sim: BaseSimulator = sim
        self.robot_servers: list[RobotServer] = []
        self.running: bool = False
        self.async_thread: Optional[threading.Thread] = None

    def init(self) -> None:
        for i, robot in enumerate(self.sim.robots):
            rs = RobotServer(robot, self.robot_hostnames(i))
            rs.init()
            self.robot_servers.append(rs)

    def run_once(self, realtime: bool | float = True):
        start_time = time.time()

        for rs in self.robot_servers:
            rs.process_commands()

        self.sim.step()

        for rs in self.robot_servers:
            rs.send_state()

        time.sleep(max(0.0, 0.001 * float(realtime) - (time.time() - start_time)))

    def run_forever(self, realtime: bool | float = True):
        self.running = True
        while self.running:
            self.run_once(realtime)

    def run_async(self, realtime: bool | float = True):
        self.async_thread = threading.Thread(target=self.run_forever, args=(realtime,), daemon=True)
        self.async_thread.start()

    def cleanup(self) -> None:
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
