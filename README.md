- Install [ROS 2 Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html#install-ros-2-packages)
  ```
  sudo apt install ros-humble-desktop
  echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
  source ~/.bashrc
  ```
- Create a ROS 2 workspace
  ```
  mkdir -p ~/ros2_ws/src
  cd ~/ros2_ws/
  colcon build
  echo "source $HOME/ros2_ws/install/setup.bash" >> ~/.bashrc
  source ~/.bashrc
  ```
- Install [Gazebo Garden](https://gazebosim.org/docs/garden/install_ubuntu)
- Install [ArduPilot SITL](https://ardupilot.org/dev/docs/building-setup-linux.html#building-setup-linux)
- Install [ardupilot_gazebo](https://github.com/ArduPilot/ardupilot_gazebo?tab=readme-ov-file#installation)
- Get [SITL_Models repo](https://github.com/ArduPilot/SITL_Models)
- Setup environment variables

  ```
  echo "export GZ_VERSION=garden" >> ~/.bashrc
  echo "export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/ardupilot_gazebo/build:${GZ_SIM_SYSTEM_PLUGIN_PATH}" >> ~/.bashrc
  echo "export GZ_SIM_RESOURCE_PATH=$HOME/ardupilot_gazebo/models:$HOME/ardupilot_gazebo/worlds:$HOME/SITL_Models/Gazebo/models:$HOME/SITL_Models/Gazebo/worlds:$GZ_SIM_RESOURCE_PATH" >> ~/.bashrc
  ```
- Install mavros
  ```
  sudo apt install ros-humble-mavros*
  ```