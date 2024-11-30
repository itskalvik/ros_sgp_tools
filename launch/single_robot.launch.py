from launch_ros.actions import Node
from launch import LaunchDescription
from launch_ros.actions import PushRosNamespace
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import GroupAction, IncludeLaunchDescription
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource

import sys
import psutil

# Sanity check to avoid crashing the system because of low memory
total_memory = psutil.virtual_memory().total + psutil.swap_memory().total
if total_memory < 4e9:
    sys.exit("Low memory (less than 4GB), increase swap size!")

def generate_launch_description():
    namespace = 'robot_0'
    data_type = 'SerialPing2'
    num_robots = 1
    num_waypoints = 20
    sampling_rate = 2
    adaptive_ipp = True
    fake_data = False
    start_foxglove = False
    geofence_plan = PathJoinSubstitution([FindPackageShare('ros_sgp_tools'),
                                          'launch', 'data',
                                          'mission.plan'])
    nodes = []

    offline_planner = Node(package='ros_sgp_tools',
                           executable='offline_ipp.py',
                           name='OfflineIPP',
                           parameters=[
                                {'num_waypoints': num_waypoints,
                                 'num_robots': num_robots,
                                 'sampling_rate': sampling_rate,
                                 'geofence_plan': geofence_plan
                                }
                           ])
    nodes.append(offline_planner)

    online_planner = Node(package='ros_sgp_tools',
                          executable='online_ipp.py',
                          namespace=namespace,
                          name='OnlineIPP',
                          parameters=[
                              {'data_type': data_type,
                               'adaptive_ipp': adaptive_ipp
                              }
                          ])
    nodes.append(online_planner)

    mission_planner = Node(package='ros_sgp_tools',
                           executable='ipp_mission.py',
                           namespace=namespace,
                           name='IPPMission')
    nodes.append(mission_planner)

    mavros = GroupAction(
                    actions=[
                        # push_ros_namespace to set namespace of included nodes
                        PushRosNamespace(namespace),
                        # MAVROS
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
                        ),
                    ]
                )
    nodes.append(mavros)

    if start_foxglove:
        foxglove = GroupAction(
                        actions=[
                            # push_ros_namespace to set namespace of included nodes
                            PushRosNamespace(namespace),
                            # Foxglove (web-based rviz)
                            IncludeLaunchDescription(
                                XMLLaunchDescriptionSource([
                                    PathJoinSubstitution([
                                        FindPackageShare('foxglove_bridge'),
                                        'launch',
                                        'foxglove_bridge_launch.xml'
                                    ])
                                ]),
                            )
                        ]
                    )
        nodes.append(foxglove)
   
    if fake_data and data_type=='SerialPing2':
        print("Publishing Fake Sonar Data")
        sensor = Node(package='ros_sgp_tools',
                      executable='lake_depth_publisher.py',
                      name='FakeSonarData',
                      namespace=namespace)
        nodes.append(sensor)

    if data_type=='Ping2':
        sensor = Node(package='ping_sonar_ros',
                      executable='ping1d_node',
                      name='Ping2',
                      namespace=namespace,
                      parameters=[
                        {'port': '/dev/ttyUSB0'}
                      ])
        nodes.append(sensor)       

    return LaunchDescription(nodes)