from launch_ros.actions import Node
from launch import LaunchDescription
from launch_ros.actions import PushRosNamespace
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import GroupAction, IncludeLaunchDescription
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource


def generate_launch_description():
    namespace = 'robot_0'
    num_robots = 1
    num_waypoints = 20
    geofence_plan = PathJoinSubstitution([FindPackageShare('ros_sgp_tools'),
                                          'launch',
                                          'lake.plan'])
    
    return LaunchDescription([
        Node(
            package='ros_sgp_tools',
            executable='offline_ipp.py',
            name='offline_ipp',
            parameters=[
                {'num_waypoints': num_waypoints,
                 'num_robots': num_robots,
                 'geofence_plan': geofence_plan
                }
            ]
        ),
        
        Node(
            package='ros_sgp_tools',
            executable='online_ipp.py',
            namespace=namespace,
            name='online_ipp',
            parameters=[
                {'num_waypoints': num_waypoints,
                 'num_robots': num_robots,
                 'geofence_plan': geofence_plan
                }
            ]
        ),

        Node(
            package='ros_sgp_tools',
            executable='lake_depth_publisher.py',
            namespace=namespace,
            name='lake_depth_publisher'
        ),

        Node(
            package='ros_sgp_tools',
            executable='ipp_mission.py',
            namespace=namespace,
            name='ipp_mission'
        ),

        GroupAction(
            actions=[
                # push_ros_namespace to set namespace of included nodes
                PushRosNamespace(namespace),
                IncludeLaunchDescription(
                    XMLLaunchDescriptionSource([
                        PathJoinSubstitution([
                            FindPackageShare('ros_sgp_tools'),
                            'launch',
                            'mavros.launch'
                        ])
                    ]),
                    launch_arguments={
                    }.items()
                )
            ]
        )
    ])
