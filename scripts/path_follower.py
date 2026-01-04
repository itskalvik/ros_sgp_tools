#! /usr/bin/env python3
import sys
import argparse
import importlib

import rclpy
from ros_sgp_tools.srv import Waypoint


class WaypointServiceClient(Node):
    """Get a new waypoint from the Waypoint Service"""
    def __init__(self):
        super().__init__('waypoint_service_client')
        self.client = self.create_client(Waypoint, 'waypoint')
        while not self.client.wait_for_service(timeout_sec=1.0):
            rclpy.spin_once(self, timeout_sec=1.0)
        self.request = Waypoint.Request()

    def get_waypoint(self, ok=True):
        self.request.ok = ok
        future = self.client.call_async(self.request)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result.new_waypoint:
            return [result.waypoint.x, result.waypoint.y]
        else: 
            self.get_logger().info('Mission complete')
            return


def _load_controller_class(controller_name: str):
    """
    Dynamically import the requested controller base class.
    """
    if controller_name == "aqua2":
        mod = importlib.import_module("aqua2_control.controller")
        return getattr(mod, "Controller")
    elif controller_name == "mavros":
        mod = importlib.import_module("mavros_control.controller")
        return getattr(mod, "Controller")
    else:
        raise ValueError(f"Unknown controller '{controller_name}'")


def _parse_args(argv):
    """
    Parse only our custom args; leave the rest for rclpy/ROS args.
    """
    parser = argparse.ArgumentParser(
        description="Waypoint path follower (select controller backend)."
    )
    parser.add_argument(
        "--controller",
        choices=["mavros", "aqua2"],
        default="mavros",
        help="Controller backend to use.",
    )
    # Keep unknown args (e.g. --ros-args ...) to pass to rclpy.init
    return parser.parse_known_args(argv)


def build_path_follower_node(controller_name: str):
    """
    Create a PathFollower class that inherits from the selected Controller base.
    """
    ControllerBase = _load_controller_class(controller_name)

    class PathFollower(ControllerBase):
        def __init__(self):
            # Match each original controller's constructor arguments
            if controller_name == "aqua2":
                super().__init__(
                    default_depth=0.5,
                    default_speed=0.5,
                    acceptance_radius=1.1,
                )
            else:  # mavros
                super().__init__(
                    navigation_type=0,
                    start_mission=False,
                )

            self._shutdown_requested: bool = False
            self.waypoint_service = WaypointServiceClient()
            self.mission()

        def mission(self):
            """
            IPP mission:
              - MAVROS: GUIDED -> set home -> arm -> waypoints -> disarm
              - Aqua2: waypoints -> disarm
            """
            if controller_name == "mavros":
                self.get_logger().info("Engaging GUIDED mode")
                if self.set_mode("GUIDED"):
                    self.get_logger().info("GUIDED mode engaged")

                self.get_logger().info("Setting current position as home")
                if self.set_home(self.vehicle_position[0], self.vehicle_position[1]):
                    self.get_logger().info("Home position set")

                self.get_logger().info("Arming")
                if self.arm(True):
                    self.get_logger().info("Armed")

            while rclpy.ok():
                waypoint = self.waypoint_service.get_waypoint()
                if waypoint is None:
                    break

                self.get_logger().info(f"Visiting waypoint: {waypoint[0]} {waypoint[1]}")
                if self.go2waypoint([waypoint[0], waypoint[1]]):
                    self.get_logger().info("Reached waypoint")

            self.get_logger().info("Disarming")
            if self.arm(False):
                self.get_logger().info("Disarmed")

            self.request_shutdown("Mission complete")

        @property
        def shutdown_requested(self) -> bool:
            return self._shutdown_requested

        def request_shutdown(self, reason: str = "") -> None:
            if reason:
                self.get_logger().info(f"Shutdown requested: {reason}")
            self._shutdown_requested = True

    return PathFollower


def main(args=None):
    argv = sys.argv[1:] if args is None else args
    parsed, ros_args = _parse_args(argv)

    # Initialize ROS with the remaining args (including --ros-args ...)
    rclpy.init(args=ros_args)

    try:
        PathFollower = build_path_follower_node(parsed.controller)
        node = PathFollower()

        while rclpy.ok() and not node.shutdown_requested:
            rclpy.spin_once(node, timeout_sec=0.1)

    except KeyboardInterrupt:
        # If node exists, log; otherwise just exit
        try:
            node.get_logger().info("Keyboard interrupt, shutting down")
        except Exception:
            pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()
