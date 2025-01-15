#! /usr/bin/env python3
import utm
import json
import time
import numpy as np
from collections import deque
from nav_msgs.msg import Odometry

import rclpy
from aqua2_navigation.swimmer import SwimmerAPI


# Extract geofence and home location from QGC plan file
def get_mission_plan(fname):
    with open(fname, "r") as infile:
        data = json.load(infile)
        waypoints = data['mission']['items'][1]['TransectStyleComplexItem']['VisualTransectPoints']
    return np.array(waypoints)[::2]

class AquaController(SwimmerAPI):

    def __init__(self):
        SwimmerAPI.__init__(self)
        self.start_spin()

        # Sets the (0,0,0) point in the local coordinates to the current pose
        self.zero_local_pose()

        # Calibrates the Robot. Important to do as part of the pre-operation process
        self.calibrate()
        # time.sleep(2)

        # Sets the Robot to Swimmode. Must be calibrated first.
        self.set_mode("swimmode")

        # Sets the Robot's Autopilot to depth mode. In depth mode, the Robot will ignore pitch and roll targets.
        self.set_autopilot_mode("depth")

        self.set_acceptance_radius(1.5)

        self.velocity = 0.01
        self.velocity_buffer = deque([0.01])
        self.create_subscription(Odometry, 
                                 '/aqua/dvl/velocity',
                                 self.vehicle_odom_callback, 1)
        
        # Declare parameters
        self.declare_parameter('geofence_plan', '')
        self.geofence_plan = self.get_parameter('geofence_plan').get_parameter_value().string_value
        self.get_logger().info(f'Geofence Plan: {self.geofence_plan}')
        
    def vehicle_odom_callback(self, msg):
        """Callback function for vehicle odom topic subscriber.
        Computes nominal linear velocity"""
        velocity = np.hypot(msg.twist.twist.linear.x,
                            msg.twist.twist.linear.y)
        if len(self.velocity_buffer) > 50:
            self.velocity_buffer.popleft()
        if velocity > 0.2:
            self.velocity_buffer.append(velocity)
        self.velocity = np.mean(self.velocity_buffer)
        self.waypoint_distance = self.distance_to_target

    def go2waypoint(self, goal):
        self.swim_to_wp(speed=1.0, depth=5.0, 
                        x=goal[0], y=goal[1], 
                        blocking=False)
        while not self.is_at_waypoint():
            rclpy.spin_once(self, timeout_sec=1.0)

    def mission(self):
        if len(self.geofence_plan) > 0:
            waypoints = get_mission_plan(self.geofence_plan)
            waypoints = utm.from_latlon(waypoints[:, 0], waypoints[:, 1])
            waypoints = np.vstack([waypoints[0], waypoints[1]]).T
            waypoints -= waypoints[0]
            waypoints = np.round(waypoints)
        else:
            waypoints = [[0.0, 0.0],
                         [5.0, 0.0],
                         [5.0, 5.0],
                         [0.0, 5.0],
                         [0.0, 0.0]]

        for i in range(len(waypoints)):
            self.get_logger().info(f'Visiting waypoint {i}')
            if self.go2waypoint(waypoints[i]):
                self.get_logger().info(f'Reached waypoint {i}')
        self.get_logger().info(f'Mission Complete!')
            

def main():
    node = AquaController()
    node.mission()

if __name__ == '__main__':
    main()