from launch_ros.actions import Node
from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():    
    num_robots = 2
    num_waypoints = 15 # For each robot
    geofence_plan = PathJoinSubstitution([FindPackageShare('ros_sgp_tools'),
                                          'launch',
                                          'lake.plan'])
    
    return LaunchDescription([
        Node(
            package='ros_sgp_tools',
            executable='offline_ipp.py',
            name='OfflineIPP',
            parameters=[
                {'num_waypoints': num_waypoints,
                 'num_robots': num_robots,
                 'geofence_plan': geofence_plan
                }
            ]
        ),
    ])
