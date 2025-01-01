#! /usr/bin/env python3

from mavros_msgs.srv import SetMode, CommandBool, CommandHome, CommandTOL
from geographic_msgs.msg import GeoPoseStamped
from pygeodesy.geoids import GeoidPGM
from sensor_msgs.msg import NavSatFix
from mavros_msgs.msg import State
from nav_msgs.msg import Odometry
from rclpy.node import Node
import rclpy

import numpy as np
from time import sleep
from collections import deque

_egm96 = GeoidPGM('/usr/share/GeographicLib/geoids/egm96-5.pgm', kind=-3)


class MissionPlanner(Node):

    def __init__(self, use_altitude=False):
        super().__init__('MissionPlanner')
        self.get_logger().info('Initializing')

        self.use_altitude = use_altitude

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
        self.vehicle_odom_subscriber = self.create_subscription(
            Odometry, 'mavros/local_position/odom',
            self.vehicle_odom_callback, SENSOR_QOS)
            
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
        self.arm_request = CommandBool.Request()
        self.set_mode_request = SetMode.Request()
        self.setpoint_position = GeoPoseStamped()
        self.vehicle_position = np.array([0., 0., 0.])
        self.velocity = 0.01
        self.velocity_buffer = deque([0.01])
        self.waypoint_distance = -1

        # Wait to get the state of the vehicle
        rclpy.spin_once(self, timeout_sec=5.0)

    def haversine(self, pt1, pt2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        https://stackoverflow.com/a/29546836
        """
        lon1, lat1 = pt1[:, 0], pt1[:, 1]
        lon2, lat2 = pt2[:, 0], pt2[:, 1]
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        m = 6378.137 * c * 1000
        return m

    def at_waypoint(self, waypoint, xy_tolerance=0.7, z_tolerance=0.3):
        """Check if the vehicle is at the waypoint."""
        self.waypoint_distance = self.haversine(self.vehicle_position[:2].reshape(1, -1), 
                                                np.array(waypoint[:2]).reshape(1, -1))[0]
        if self.use_altitude:
            z_dist = np.abs(self.vehicle_position[2] - waypoint[2])
        else: 
            z_dist = 0.0

        if self.waypoint_distance < xy_tolerance and z_dist < z_tolerance:
            return True
        else:
            return False
        
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

    def vehicle_state_callback(self, vehicle_state):
        """Callback function for vehicle_state topic subscriber."""
        self.vehicle_state = vehicle_state

    def vehicle_position_callback(self, position):
        """Callback function for vehicle position topic subscriber."""
        self.vehicle_position[0] = position.latitude
        self.vehicle_position[1] = position.longitude

        # Offset to convert ellipsoid to AMSL height
        offset = _egm96.height(self.vehicle_position[0], self.vehicle_position[1])
        self.vehicle_position[2] = position.altitude-offset

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
    
    def takeoff(self, altitude=20.0, timeout=30, time_lim=900):
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
        self.get_logger().info('Taking off')
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if future.result():
            waypoint = [self.vehicle_position[0], 
                        self.vehicle_position[1],
                        altitude]
            start_time = self.get_clock().now().to_msg().sec
            while not self.at_waypoint(waypoint):
                if self.get_clock().now().to_msg().sec - start_time > time_lim:
                    self.get_logger().info(f'Timeout: Failed to takeoff')
                    return False
                rclpy.spin_once(self, timeout_sec=1.0)
            return True
        else: 
            self.get_logger().info(f'COMMAND_ACK: Failed to takeoff')
            return False  

    def land(self, altitude=0.0, timeout=30, time_lim=900):
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
        self.get_logger().info('Landing')
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if future.result():
            waypoint = [self.vehicle_position[0], 
                        self.vehicle_position[1],
                        altitude]
            start_time = self.get_clock().now().to_msg().sec
            while not self.at_waypoint(waypoint):
                if self.get_clock().now().to_msg().sec - start_time > time_lim:
                    self.get_logger().info(f'Timeout: Failed to land')
                    return False
                rclpy.spin_once(self, timeout_sec=1.0)
            return True
        else: 
            self.get_logger().info(f'COMMAND_ACK: Failed to land')
            return False  
    
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
    
    def set_home(self, latitude, longitude, altitude=None, timeout=900):
        """Set the home position"""

        home_client = self.create_client(CommandHome, 'mavros/cmd/set_home')
        while not home_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Set home service not available, waiting again...')
        self.get_logger().info('Set home service available')

        set_home_request = CommandHome.Request()
        set_home_request.latitude = latitude
        set_home_request.longitude = longitude
        if altitude is not None:
            set_home_request.altitude = altitude

        future = home_client.call_async(set_home_request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.result().success:
            self.get_logger().info(f'Timeout: Failed to set home')
            return False

        return True
    
    def go2waypoint(self, waypoint, timeout=900):
        """Go to waypoint (latitude, longitude) when in GUIDED mode and armed"""
        self.setpoint_position.pose.position.latitude = waypoint[0]
        self.setpoint_position.pose.position.longitude = waypoint[1]
        if self.use_altitude:
            self.setpoint_position.pose.position.altitude = waypoint[2]
        else:
            waypoint = waypoint[:2]

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

            # Timeout
            if not self.at_waypoint(waypoint) and \
                last_request - start_time > timeout:
               
                self.get_logger().info(f'Timeout: Failed to go to waypoint: {waypoint[0]}, {waypoint[1]}')
                return False

        return True

    def mission(self):
        """GUIDED mission ArduRover/ArduCopter"""
        sleep(4.0) # Wait to get position
        mission_altitude = self.vehicle_position[2]

        self.get_logger().info('Engaging GUIDED mode')
        if self.engage_mode('GUIDED'):
            self.get_logger().info('GUIDED mode Engaged')

        self.get_logger().info('Arming')
        if self.arm(True):
            self.get_logger().info('Armed')

        self.get_logger().info('Setting current positon as home')
        if self.set_home(self.vehicle_position[0], 
                         self.vehicle_position[1]):
            self.get_logger().info('Home position set')

        if self.use_altitude:
            if self.takeoff(mission_altitude+20.0):
                self.get_logger().info('Takeoff complete')

        self.get_logger().info('Visiting waypoint 1')
        if self.go2waypoint([35.30684387683425, 
                             -80.7360063599907, 
                             mission_altitude+20.0]):
            self.get_logger().info('Reached waypoint 1')

        self.get_logger().info('Visiting waypoint 2')
        if self.go2waypoint([35.30674267884529, 
                             -80.73600329951549, 
                             mission_altitude+25.0]):
            self.get_logger().info('Reached waypoint 2')

        if self.use_altitude:
            if self.land(mission_altitude):
                self.get_logger().info('Landing complete')

        self.get_logger().info('Disarming')
        if self.arm(False):
            self.get_logger().info('Disarmed')


def main(args=None):
    rclpy.init(args=args)

    mission_planner = MissionPlanner(use_altitude=False)
    rclpy.spin_once(mission_planner)
    mission_planner.mission()


if __name__ == '__main__':
    main()
