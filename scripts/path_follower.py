#! /usr/bin/env python3

from mavros_control.controller import Controller
from ros_sgp_tools.srv import Waypoint
from rclpy.node import Node
import rclpy
from shapely.geometry import Polygon, LineString
from utils import get_mission_plan, calculate_bounded_path
from ament_index_python.packages import get_package_share_directory
import os
import numpy as np


class WaypointServiceClient(Node):
    """Get a new waypoint from the Waypoint Service"""
    def __init__(self):
        super().__init__('waypoint_service_client')
        self.client = self.create_client(Waypoint, 'waypoint')
        while not self.client.wait_for_service(timeout_sec=1.0):
            rclpy.spin_once(self, timeout_sec=1.0)
        self.request = Waypoint.Request()

    def get_waypoint(self, ok=True, timeout_sec=30):
        self.request.ok = ok
        future = self.client.call_async(self.request)
        rclpy.spin_until_future_complete(self, future, 
                                         timeout_sec=timeout_sec)
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
        
        plan_fname = os.path.join(get_package_share_directory('ros_sgp_tools'), 
                                                              'launch', 'data',
                                                              'mission.plan')
        self.declare_parameter('geofence_plan', plan_fname)
        plan_fname = self.get_parameter('geofence_plan').get_parameter_value().string_value
        self.get_logger().info(f'GeoFence Plan File: {plan_fname}')
        self.fence_vertices, *_ = get_mission_plan(plan_fname,
                                                                  get_waypoints=False)
        if not np.array_equal(self.fence_vertices[0], self.fence_vertices[-1]):
            fence_vertices = np.vstack([self.fence_vertices, self.fence_vertices[0]])
        self.fence_polygon = Polygon(fence_vertices)

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

        self.prev_waypoint = None
        while rclpy.ok():
            waypoint = self.waypoint_service.get_waypoint()
            if waypoint is None:
                break
            if self.prev_waypoint is None:
                self.get_logger().info(f'Visiting waypoint: {waypoint[0]} {waypoint[1]}')
                if self.go2waypoint([waypoint[0], waypoint[1], 0.0]):
                    self.get_logger().info(f'Reached waypoint')
            else:
                point_a = (waypoint[0], waypoint[1])
                point_b = (self.prev_waypoint[0], self.prev_waypoint[1])
                line = LineString([point_a, point_b])

                if not self.fence_polygon.contains(line):
                    self.get_logger().info(f'Planned path to waypoint leaves the boundary. Replanning.')
                    bounded_path = calculate_bounded_path(point_a, point_b,self.fence_polygon)
                    self.get_logger().info(f'Bounded path: {bounded_path}')

                    for j, point in enumerate(bounded_path):
                        self.get_logger().info(f'Visiting waypoint .{j+1}: {point}')
                        if self.go2waypoint([point[0], point[1], 0.0]):
                            self.get_logger().info(f'Reached waypoint .{j+1}')

            self.prev_waypoint = waypoint

        self.get_logger().info('Disarming')
        if self.arm(False):
            self.get_logger().info('Disarmed')

        self.get_logger().info('Mission complete')


def main(args=None):
    rclpy.init(args=args)
    path_follower = PathFollower()
    rclpy.spin_once(path_follower)

if __name__ == '__main__':
    main()
