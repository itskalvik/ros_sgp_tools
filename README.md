<p align="center">
  <img src=".assets/SGP-Tools.png#gh-light-mode-only" alt="SGP-Tools Logo" width="600"/>
  <img src=".assets/logo_dark.png#gh-dark-mode-only" alt="SGP-Tools Logo" width="600"/>
</p>

<p align="center">
  <em>A ROS 2 companion package for <a href="http://sgp-tools.com">SGP-Tools</a>, a Python library for efficient sensor placement and informative path planning
</em>
</p>

**SGP-Tools** is a powerful and flexible Python library for solving **Sensor Placement and Informative Path Planning (IPP)** problems. It enables scalable and efficient solutions for environmental monitoring tasks such as tracking air and water quality, soil moisture, or temperature.

This ROS 2 package integrates SGP-Tools with the ROS ecosystem, enabling deployment of its advanced algorithms on real-world robotic platforms.

## Features ✨
* **Online & Adaptive Path Planning** 

  Execute IPP missions in real time. Specifically designed for deployment on [ArduPilot-based vehicles](https://ardupilot.org/copter/docs/common-use-cases-and-applications.html).

* **Realistic SITL Simulation**

  Fully compatible with Gazebo and ArduPilot SITL, enabling robust software-in-the-loop testing in safe, controlled environments.

* **Data Visualization**

  Visualize collected data and reconstructed maps using [Foxglove](https://foxglove.dev/), or perform detailed analysis via the provided Jupyter notebook.

* **Flexible Sensor Integration**

  Easily configurable to support a variety of sensors, such as the [Ping1D sonar](https://bluerobotics.com/store/sonars/echosounders/ping-sonar-r2-rp/), making it adaptable to different mission requirements.

* **Pre-configured Docker Environment**

  Get started quickly using **[docker-sgp-tools](https://github.com/itskalvik/docker-sgp-tools)** — a ready-to-use Docker container with all dependencies and simulation tools pre-installed.

## Installation 🛠️

1.  **Create a ROS 2 workspace and clone the repository:**

    ```bash
    mkdir -p ~/ros2_ws/src
    cd ~/ros2_ws/src
    git clone https://github.com/itskalvik/ros_sgp_tools.git
    ```

2.  **Install Python dependencies:**

    ```bash
    cd ros_sgp_tools
    python3 -m pip install -r requirements.txt
    ```

3.  **Build and source the package:**

    ```bash
    cd ~/ros2_ws
    rosdep install --from-paths src --ignore-src -y
    colcon build --symlink-install
    source ~/ros2_ws/install/setup.bash
    ```

## Usage 🚀

### 🏄‍♂️ Running an ASV Mission

To launch an Informative Path Planning (IPP) mission using an ArduPilot-based Autonomous Surface Vehicle (ASV), use the provided launch file:

```bash
ros2 launch ros_sgp_tools asv.launch.py
```

This command launches all the necessary components:
* `path_planner.py`: Generates the informative path.
* `path_follower.py`: Guides the ASV along the planned path.
* `mavros.launch`: Enables communication with the ArduPilot flight controller.
* **Sensor Node**: Launches the relevant sensor driver (e.g., Ping1D), as defined in your configuration.

To start the mission, run the following command in your terminal:

### 🗂️ Mission Configuration

Configuration is managed through a YAML file, a plan file, and environment variables.

**config.yaml**

Located in `ros_sgp_tools/launch/data/`. Accepts parameters from the [SGP-Tools API](https://www.sgp-tools.com/api/index.html).
  * **`robot`**: Sensor type, data buffer size, mission type (`AdaptiveIPP`, `IPP`, or `Waypoint`), random seed.
  * **`sensor`**: Sensor-specific settings.
  * **`ipp_model`**: Number of waypoints, optimization strategy, etc.
  * **`param_model`**: Online parameter learning model configuration.
  * **`hyperparameters`**: Initial GP model (`ipp_model` and `param_model`) hyperparameters.
  * **`tsp`**: Configuration for the Traveling Salesperson Problem solver.

**mission.plan**

Also in `ros_sgp_tools/launch/data/`. Defines the survey area and start location via a polygon geofence created in [QGroundControl](https://qgroundcontrol.com/).


**Environment Variables**

  * `DATA_FOLDER`: Path to save mission logs (default: launch directory).
  * `FCU_URL`: ArduPilot connection string (default:`udp://0.0.0.0:14550@`).
  
Set variables with:

```
export <parameter_name>=<parameter_value>
```

**Note:** This package requires **at least 4GB of available memory**, including swap. Ensure your system is appropriately configured.

### 🛰️ Visualizing Mission Data

After completing a mission, visualize the collected data using:
```bash
ros2 launch ros_sgp_tools visualize_data.launch.py mission_log:=<log-folder-name>
```

This will launch:
* `data_visualizer.py`: Publishes the point cloud from mission logs.
* `foxglove_bridge`: Connects to Foxglove for real-time visualization.

#### Launch Arguments
* `mission_log`: The folder containing the mission log to visualize (default: most recent).
* `num_samples`: Number of points to sample in the point cloud (default: `5000`). 

#### Custom Visualizer Configuration
To fine-tune the visualization, place a `viz_config.yaml` file in the target log folder before launching the visualizer. You can specify:
  * **`hyperparameters`**: GP model hyperparameters for visualization.
  * **`optimizer`**: Optimizer parameters passed to `sgptools`.