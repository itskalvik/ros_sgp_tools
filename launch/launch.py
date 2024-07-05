from launch import LaunchDescription
from launch_ros.actions import Node

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
            name='online_ipp'
        ),

        Node(
            package='ros_sgp_tools',
            executable='ipp_mission.py',
            name='ipp_mission'
        )
    ])
