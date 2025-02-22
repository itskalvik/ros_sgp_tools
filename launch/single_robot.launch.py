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
    data_buffer_size = int(get_var('DATA_BUFFER_SIZE', 100))
    train_param_inducing = True if get_var('TRAIN_PARAM_INDUCING', 'False')=='True' else False
    num_param_inducing = int(get_var('NUM_PARAM_INDUCING', 40))
    adaptive_ipp = True if get_var('ADAPTIVE_IPP', 'True')=='True' else False
    fake_data =  True if get_var('FAKE_DATA', 'True')=='True' else False
    data_folder = get_var('DATA_FOLDER', '')
    fcu_url = get_var('FCU_URL', 'udp://0.0.0.0:14550@')
    ping2_port = get_var('PING2_PORT', '/dev/ttyUSB0')

    num_robots = 1
    geofence_plan = PathJoinSubstitution([FindPackageShare('ros_sgp_tools'),
                                          'launch', 'data',
                                          'mission.plan'])
    
    print("\nParameters:")
    print("===========")
    print(f"DATA_TYPE: {data_type}")
    print(f"NUM_WAYPOINTS: {num_waypoints}")
    print(f"SAMPLING_RATE: {sampling_rate}")
    print(f"DATA_BUFFER_SIZE: {data_buffer_size}")
    print(f"TRAIN_PARAM_INDUCING: {train_param_inducing}")
    print(f"NUM_PARAM_INDUCING': {num_param_inducing}")
    print(f"ADAPTIVE_IPP: {adaptive_ipp}")
    print(f"FAKE_DATA: {fake_data}")
    print(f"FCU_URL: {fcu_url}")
    if data_type=='Ping2':
        print(f"PING2_PORT: {ping2_port}")
    print('')

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
                               'adaptive_ipp': adaptive_ipp,
                               'data_folder': data_folder,
                               'data_buffer_size': data_buffer_size,
                               'train_param_inducing': train_param_inducing,
                               'num_param_inducing': num_param_inducing
                              }
                          ])
    nodes.append(online_planner)

    path_follower = Node(package='ros_sgp_tools',
                         executable='path_follower.py',
                         namespace=namespace,
                         name='PathFollower')
    nodes.append(path_follower)

    mavros = GroupAction(
                    actions=[
                        # push_ros_namespace to set namespace of included nodes
                        PushRosNamespace(namespace),
                        # MAVROS
                        IncludeLaunchDescription(
                            XMLLaunchDescriptionSource([
                                PathJoinSubstitution([
                                    FindPackageShare('mavros_control'),
                                    'launch',
                                    'mavros.launch'
                                ])
                            ]),
                            launch_arguments={
                                "fcu_url": fcu_url
                            }.items()
                        ),
                    ]
                )
    nodes.append(mavros)

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
                        {'port': ping2_port}
                      ])
        nodes.append(sensor)       

    return LaunchDescription(nodes)