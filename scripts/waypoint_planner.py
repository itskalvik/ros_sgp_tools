#! /usr/bin/env python3
import rclpy
from rclpy.executors import MultiThreadedExecutor

from base_planner import BasePathPlanner


class WaypointPathPlanner(BasePathPlanner):
    MISSION_TYPE = "Waypoint"

    def _init_mission_models_and_waypoints(self) -> None:
        self.get_logger().info("Waypoint mission: no models to initialize.")
        if self.waypoints is None or len(self.waypoints) == 0:
            raise RuntimeError("Waypoint mission requires waypoints in mission.plan")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WaypointPathPlanner()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        while rclpy.ok() and not node.shutdown_requested:
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
