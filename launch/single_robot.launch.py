import os
import sys
import psutil

from launch_ros.actions import Node
from launch import LaunchDescription
from launch_ros.actions import PushRosNamespace
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import GroupAction, IncludeLaunchDescription
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource

def get_var(var, default):
    try:
        return os.environ[var]
    except:
        return default

def generate_launch_description():
    # Sanity check to avoid crashing the system because of low memory
    total_memory = psutil.virtual_memory().total + psutil.swap_memory().total
    if total_memory < 4e9:
        sys.exit("Low memory (less than 4GB), increase swap size!")

    # Get parameter values
    namespace = get_var('NAMESPACE', 'robot_0')
    data_type = get_var('DATA_TYPE' ,'SerialPing2')
    num_waypoints = int(get_var('NUM_WAYPOINTS', 20))
    sampling_rate = int(get_var('SAMPLING_RATE', 2))
    adaptive_ipp = True if get_var('ADAPTIVE_IPP', 'True')=='True' else False
    start_foxglove =  True if get_var('START_FOXGLOVE', 'False')=='True' else False
    fake_data =  True if get_var('FAKE_DATA', 'True')=='True' else False
    
    num_robots = 1
    geofence_plan = PathJoinSubstitution([FindPackageShare('ros_sgp_tools'),
                                          'launch', 'data',
                                          'mission.plan'])
    
    print("\nParameters:")
    print("===========")
    print(f"NAMESPACE: {namespace}")
    print(f"DATA_TYPE: {data_type}")
    print(f"NUM_WAYPOINTS: {num_waypoints}")
    print(f"SAMPLING_RATE: {sampling_rate}")
    print(f"ADAPTIVE_IPP: {adaptive_ipp}")
    print(f"START_FOXGLOVE: {start_foxglove}\n")
    print(f"FAKE_DATA: {fake_data}\n")

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
        ping2_port = get_var('PING2_PORT', '/dev/ttyUSB0')
        sensor = Node(package='ping_sonar_ros',
                      executable='ping1d_node',
                      name='Ping2',
                      namespace=namespace,
                      parameters=[
                        {'port': ping2_port}
                      ])
        nodes.append(sensor)       

    return LaunchDescription(nodes)