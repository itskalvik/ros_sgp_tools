import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from launch.actions import GroupAction, IncludeLaunchDescription
from launch_ros.actions import PushRosNamespace
from ament_index_python import get_package_share_directory


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ros_sgp_tools',
            executable='offline_ipp.py',
            name='offline_ipp',
            parameters=[
                {'num_waypoints': 10},
                {'num_robots': 1}
            ]
        ),
        
        Node(
            package='ros_sgp_tools',
            executable='online_ipp.py',
            namespace='robot_0',
            name='online_ipp'
        ),

        Node(
            package='ros_sgp_tools',
            executable='depth_publisher.py',
            namespace='robot_0',
            name='depth_publisher'
        ),

        Node(
            package='ros_sgp_tools',
            executable='ipp_mission.py',
            namespace='robot_0',
            name='ipp_mission'
        ),

        GroupAction(
            actions=[
                # push_ros_namespace to set namespace of included nodes
                PushRosNamespace('robot_0'),
                IncludeLaunchDescription(
                    XMLLaunchDescriptionSource(
                        os.path.join(
                            get_package_share_directory('ros_sgp_tools'),
                            'launch/mavros.launch')),
                    launch_arguments={
                        'fcu_url': 'udp://0.0.0.0:14551@',
                        'pluginlists_yaml': '$(find-pkg-share ros_sgp_tools)/launch/apm_pluginlists.yaml',
                    }.items()
                ),
            ]
        )
    ])
