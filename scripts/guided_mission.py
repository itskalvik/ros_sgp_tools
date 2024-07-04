#! /usr/bin/env python3

from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode, CommandBool
from geographic_msgs.msg import GeoPoseStamped
from sensor_msgs.msg import NavSatFix

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.executors import MultiThreadedExecutor

from time import sleep
import numpy as np


class MissionPlanner(Node):

    def __init__(self):
        super().__init__('MissionPlanner')

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
            State, '/mavros/state', self.vehicle_state_callback, STATE_QOS)
        self.vehicle_pose_subscriber = self.create_subscription(
            NavSatFix, '/mavros/global_position/global', 
            self.vehicle_position_callback, SENSOR_QOS)
        
        # Create publishers
        self.setpoint_position_publisher = self.create_publisher(
            GeoPoseStamped, '/mavros/setpoint_position/global', SENSOR_QOS)

        # Create service clients
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        while not self.set_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Set mode service not available, waiting again...')
        self.get_logger().info('Set mode service available')

        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        while not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Arming service not available, waiting again...')
        self.get_logger().info('Arming service available')

        # Initialize variables
        self.vehicle_state = State()
        self.vehicle_position = np.array([0., 0.])
        self.arm_request = CommandBool.Request()
        self.set_mode_request = SetMode.Request()
        self.setpoint_position = GeoPoseStamped()

    def at_waypoint(self, waypoint, tolerance=0.00005):
        """Check if the vehicle is at the waypoint."""
        dist = np.linalg.norm(self.vehicle_position - np.array(waypoint))
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
                                          position.longitude])

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
                sleep(1)
                continue

            self.arm_client.call_async(self.arm_request)
            last_request = self.get_clock().now().to_msg().sec

            # Timeout
            if self.vehicle_state.armed != state and \
                last_request - start_time > timeout:
               
                self.get_logger().info(f'Timeout: Failed to {str_state}')
                return False

        return True

    def engage_mode(self, mode="GUIDED", timeout=30):
        """Set the vehicle mode"""
        self.set_mode_request.custom_mode = mode
        
        start_time = self.get_clock().now().to_msg().sec
        last_request = start_time-6.0
        while self.vehicle_state.mode != mode:
            # Send the command only once every 5 seconds
            if self.get_clock().now().to_msg().sec - last_request < 5.0:
                sleep(1)
                continue

            self.set_mode_client.call_async(self.set_mode_request)
            last_request = self.get_clock().now().to_msg().sec

            # Timeout
            if self.vehicle_state.mode != mode and \
                last_request - start_time > timeout:
               
                self.get_logger().info(f'Timeout: Failed to engage {mode} mode')
                return False

        return True
    
    def go2waypoint(self, waypoint, timeout=900):
        """Go to waypoint (latitude, longitude) when in GUIDED mode and armed"""
        self.setpoint_position.pose.position.latitude = waypoint[0]
        self.setpoint_position.pose.position.longitude = waypoint[1]

        start_time = self.get_clock().now().to_msg().sec
        last_request = start_time-6.0
        while not self.at_waypoint(waypoint):
            # Send the command only once every 5 seconds
            if self.get_clock().now().to_msg().sec - last_request < 5.0:
                sleep(1)
                continue

            self.setpoint_position.header.stamp = self.get_clock().now().to_msg()
            self.setpoint_position_publisher.publish(self.setpoint_position)
            last_request = self.get_clock().now().to_msg().sec

            # Timeout
            if not self.at_waypoint(waypoint) and \
                last_request - start_time > timeout:
               
                self.get_logger().info(f'Timeout: Failed to go to waypoint: {waypoint[0]}, {waypoint[1]}')
                return False

        return True

    def mission(self):
        """GUIDED mission"""

        sleep(5) # Wait to get the state of the vehicle

        self.get_logger().info('Engaging MANUAL mode')
        if self.engage_mode('MANUAL'):
            self.get_logger().info('MANUAL mode Engaged')

        self.get_logger().info('Engaging GUIDED mode')
        if self.engage_mode('GUIDED'):
            self.get_logger().info('GUIDED mode Engaged')

        self.get_logger().info('Arming')
        if self.arm(True):
            self.get_logger().info('Armed')

        self.get_logger().info('Visiting waypoint 1')
        if self.go2waypoint([35.30684387683425, -80.7360063599907]):
            self.get_logger().info('Reached waypoint 1')

        self.get_logger().info('Visiting waypoint 2')
        if self.go2waypoint([35.30674267884529, -80.73600329951549]):
            self.get_logger().info('Reached waypoint 2')

        self.get_logger().info('Visiting waypoint 3')
        if self.go2waypoint([35.30684275566786, -80.73612370299257]):
            self.get_logger().info('Reached waypoint 3')

        self.get_logger().info('Visiting waypoint 4')
        if self.go2waypoint([35.30679876645213, -80.73623439122146]):
            self.get_logger().info('Reached waypoint 4')

        self.get_logger().info('Disarming')
        if self.arm(False):
            self.get_logger().info('Disarmed')

        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    try:
        mission_planner = MissionPlanner()

        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(mission_planner)
        executor.create_task(mission_planner.mission)
        executor.spin()

    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        rclpy.shutdown()
    

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
