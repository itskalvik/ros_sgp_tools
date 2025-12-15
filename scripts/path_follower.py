#! /usr/bin/env python3

from mavros_control.controller import Controller
from ros_sgp_tools.srv import Waypoint
from rclpy.node import Node
import rclpy


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

class PathFollower(Controller):

    def __init__(self):
        super().__init__(navigation_type=0,
                         start_mission=False)

        self._shutdown_requested: bool = False

        # Create the waypoint service client
        self.waypoint_service = WaypointServiceClient()
        self.mission()

    def mission(self):
        """IPP mission"""
        self.get_logger().info('Engaging GUIDED mode')
        if self.set_mode('GUIDED'):
            self.get_logger().info('GUIDED mode Engaged')

        self.get_logger().info('Setting current positon as home')
        if self.set_home(self.vehicle_position[0], self.vehicle_position[1]):
            self.get_logger().info('Home position set')

        self.get_logger().info('Arming')
        if self.arm(True):
            self.get_logger().info('Armed')

        while rclpy.ok():
            waypoint = self.waypoint_service.get_waypoint()
            if waypoint is None:
                break
            self.get_logger().info(f'Visiting waypoint: {waypoint[0]} {waypoint[1]}')
            if self.go2waypoint([waypoint[0], waypoint[1], 0.0]):
                self.get_logger().info(f'Reached waypoint')

        self.get_logger().info('Disarming')
        if self.arm(False):
            self.get_logger().info('Disarmed')

        self.request_shutdown("Mission complete")

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_requested

    def request_shutdown(self, reason: str = "") -> None:
        if reason:
            self.get_logger().info(f"Shutdown requested: {reason}")
        self._shutdown_requested = True

def main(args=None):
    rclpy.init(args=args)
    path_follower = PathFollower()

    try:
        while rclpy.ok() and not path_follower.shutdown_requested:
            rclpy.spin_once(path_follower, timeout_sec=0.1)
    except KeyboardInterrupt:
        path_follower.get_logger().info('Keyboard interrupt, shutting down')
    finally:
        path_follower.destroy_node()
        rclpy.shutdown()    

if __name__ == '__main__':
    main()