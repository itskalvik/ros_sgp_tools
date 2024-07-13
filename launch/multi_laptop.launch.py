from launch_ros.actions import Node
from launch import LaunchDescription


def generate_launch_description():    
    return LaunchDescription([
        Node(
            package='ros_sgp_tools',
            executable='offline_ipp.py',
            name='offline_ipp',
            parameters=[
                {'num_waypoints': 10},
                {'num_robots': 2}
            ]
        ),
    ])
