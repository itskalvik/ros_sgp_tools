#! /usr/bin/env python3

from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode, CommandBool, CommandTOL
from geographic_msgs.msg import GeoPoseStamped
from sensor_msgs.msg import NavSatFix

import rclpy
from rclpy.node import Node
from  time import sleep

import numpy as np


class MissionPlanner(Node):

    def __init__(self):
        super().__init__('MissionPlanner')
        self.get_logger().info('Initializing')

        # Create QoS profiles
        # STATE_QOS used for state topics, like ~/state, ~/mission/waypoints etc.
        STATE_QOS = rclpy.qos.QoSProfile(
            depth=10, 
            durability=rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        # SENSOR_QOS used for most of sensor streams
        SENSOR_QOS = rclpy.qos.qos_profile_sensor_data

        # Create subscribers
        self.vehicle_state_subscriber = self.create_subscription(
            State, 'mavros/state', self.vehicle_state_callback, STATE_QOS)
        self.vehicle_pose_subscriber = self.create_subscription(
            NavSatFix, 'mavros/global_position/global', 
            self.vehicle_position_callback, SENSOR_QOS)
        
        # Create publishers
        self.setpoint_position_publisher = self.create_publisher(
            GeoPoseStamped, 'mavros/setpoint_position/global', SENSOR_QOS)

        # Create service clients
        self.set_mode_client = self.create_client(SetMode, 'mavros/set_mode')
        while not self.set_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Set mode service not available, waiting again...')
        self.get_logger().info('Set mode service available')

        self.arm_client = self.create_client(CommandBool, 'mavros/cmd/arming')
        while not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Arming service not available, waiting again...')
        self.get_logger().info('Arming service available')

        # Initialize variables
        self.vehicle_state = State()
        self.vehicle_position = np.array([0., 0.])
        self.arm_request = CommandBool.Request()
        self.set_mode_request = SetMode.Request()
        self.setpoint_position = GeoPoseStamped()

        # Wait to get the state of the vehicle
        rclpy.spin_once(self, timeout_sec=5.0)

    def at_waypoint(self, waypoint, tolerance=0.0000007):
        """Check if the vehicle is at the waypoint."""
        dist = np.linalg.norm(self.vehicle_position[:2] - np.array(waypoint)[:2])
        if dist < tolerance:
            return True
        else:
            return False

    def vehicle_state_callback(self, vehicle_state):
        """Callback function for vehicle_state topic subscriber."""
        self.vehicle_state = vehicle_state

    def vehicle_position_callback(self, position):
        """Callback function for vehicle position topic subscriber."""
        self.vehicle_position = np.array([position.latitude, 
                                          position.longitude,
                                          position.altitude])

    def arm(self, state=True, timeout=30):
        """Arm/Disarm the vehicle"""
        self.arm_request.value = state
        if state:
            str_state = 'Arm'
        else:
            str_state = 'Disarm'
        
        start_time = self.get_clock().now().to_msg().sec
        last_request = start_time-6.0
        while self.vehicle_state.armed != state:
            # Send the command only once every 5 seconds
            if self.get_clock().now().to_msg().sec - last_request < 5.0:
                rclpy.spin_once(self, timeout_sec=1.0)
                continue

            future = self.arm_client.call_async(self.arm_request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)

            # Timeout for retry
            last_request = self.get_clock().now().to_msg().sec
            if self.vehicle_state.armed != state and \
                last_request - start_time > timeout:
               
                self.get_logger().info(f'Timeout: Failed to {str_state}')
                return False

        return True

    def takeoff(self, altitude=20.0, timeout=30):
        """Takeoff the vehicle to the given height"""

        takeoff_client = self.create_client(CommandTOL, 'mavros/cmd/takeoff')
        while not takeoff_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Takeoff service not available, waiting again...')
        self.get_logger().info('Takeoff service available')

        takeoff_request = CommandTOL.Request()
        takeoff_request.min_pitch = 0.0
        takeoff_request.yaw = 0.0
        takeoff_request.latitude = self.vehicle_position[0]
        takeoff_request.longitude = self.vehicle_position[1]
        takeoff_request.altitude = altitude

        # Send request
        future = takeoff_client.call_async(takeoff_request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        return future.result()

    def land(self, altitude=0.0, timeout=30):
        """Land the vehicle to the given height"""

        self.land_client = self.create_client(CommandTOL, 'mavros/cmd/land')
        while not self.land_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Land service not available, waiting again...')
        self.get_logger().info('Land service available')

        land_request = CommandTOL.Request()
        land_request.min_pitch = 0.0
        land_request.yaw = 0.0
        land_request.latitude = self.vehicle_position[0]
        land_request.longitude = self.vehicle_position[1]
        land_request.altitude = altitude

        # Send request
        future = self.land_client.call_async(land_request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        return future.result()

    def engage_mode(self, mode="GUIDED", timeout=30):
        """Set the vehicle mode"""
        self.set_mode_request.custom_mode = mode
        
        start_time = self.get_clock().now().to_msg().sec
        last_request = start_time-6.0
        while self.vehicle_state.mode != mode:
            # Send the command only once every 5 seconds
            if self.get_clock().now().to_msg().sec - last_request < 5.0:
                rclpy.spin_once(self, timeout_sec=1.0)
                continue

            future = self.set_mode_client.call_async(self.set_mode_request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)

            # Timeout for retry
            last_request = self.get_clock().now().to_msg().sec
            if self.vehicle_state.mode != mode and \
                last_request - start_time > timeout:
               
                self.get_logger().info(f'Timeout: Failed to engage {mode} mode')
                return False

        return True
    
    def go2waypoint(self, waypoint, timeout=900):
        """Go to waypoint (latitude, longitude) when in GUIDED mode and armed"""
        self.setpoint_position.pose.position.latitude = waypoint[0]
        self.setpoint_position.pose.position.longitude = waypoint[1]
        self.setpoint_position.pose.position.altitude = 20.0

        start_time = self.get_clock().now().to_msg().sec
        last_request = start_time-6.0
        while not self.at_waypoint(waypoint):
            # Send the command only once every 5 seconds
            if self.get_clock().now().to_msg().sec - last_request < 5.0:
                rclpy.spin_once(self, timeout_sec=1.0)
                continue

            self.setpoint_position.header.stamp = self.get_clock().now().to_msg()
            self.setpoint_position_publisher.publish(self.setpoint_position)
            last_request = self.get_clock().now().to_msg().sec

            # Timeout for retry
            if not self.at_waypoint(waypoint) and \
                last_request - start_time > timeout:
               
                self.get_logger().info(f'Timeout: Failed to go to waypoint: {waypoint[0]}, {waypoint[1]}')
                return False

        return True

    def mission(self):
        """GUIDED mission"""

        self.mission_altitude = 20.0

        self.get_logger().info('Engaging GUIDED mode')
        if self.engage_mode('GUIDED'):
            self.get_logger().info('GUIDED mode Engaged')

        self.get_logger().info('Arming')
        if self.arm(True):
            self.get_logger().info('Armed')            

        self.get_logger().info('Taking off')
        if self.takeoff(self.mission_altitude):
            sleep(60.0) # Wait for landing
        self.get_logger().info('Takeoff complete')

        self.get_logger().info('Visiting waypoint 1')
        if self.go2waypoint([35.30704387683425, 
                             -80.7360063599907, 
                             self.mission_altitude]):
            self.get_logger().info('Reached waypoint 1')

        self.get_logger().info('Visiting waypoint 2')
        if self.go2waypoint([35.30604267884529, 
                             -80.73600329951549, 
                             self.mission_altitude]):
            self.get_logger().info('Reached waypoint 2')

        self.get_logger().info('Landing')
        if self.land():
            sleep(60.0) # Wait for landing
        self.get_logger().info('Landing complete')

        self.get_logger().info('Disarming')
        if self.arm(False):
            self.get_logger().info('Disarmed')

def main(args=None):
    rclpy.init(args=args)

    mission_planner = MissionPlanner()
    rclpy.spin_once(mission_planner)
    mission_planner.mission()


if __name__ == '__main__':
    main()