<p align="center">
<a href="http://sgp-tools.com">
  <img src=".assets/SGP-Tools.png#gh-light-mode-only" alt="SGP-Tools Logo" width="600"/>
  <img src=".assets/logo_dark.png#gh-dark-mode-only" alt="SGP-Tools Logo" width="600"/>
</a>
</p>

<p align="center">
  <em>A ROS 2 companion package for <a href="http://sgp-tools.com">SGP-Tools</a>, a Python library for efficient sensor placement and informative path planning
</em>
</p>

SGP-Tools is a powerful and flexible Python library for solving Sensor Placement and Informative Path Planning (IPP) problems. It enables efficient and scalable solutions for environmental monitoring tasks, such as tracking air and water quality, soil moisture, or temperature. This ROS 2 package integrates SGP-Tools with the ROS ecosystem, allowing for the deployment of these advanced algorithms on robotic hardware.

## Features ✨
**Online & Adaptive Path Planning:** Execute IPP missions on real-world hardware. The package is specifically designed for deployment on [ArduPilot-based vehicles](https://ardupilot.org/copter/docs/common-use-cases-and-applications.html).

**Realistic SITL Simulation:** Test missions in a safe, controlled environment before deployment. The package is fully compatible with Gazebo and ArduPilot SITL, allowing for robust software-in-the-loop testing.

**Data Visualization:** Visualize collected data and the reconstructed map with [Foxglove](https://foxglove.dev/), or use the provided Jupyter notebook for post-mission analysis.

**Flexible Sensor Integration:** The package can be easily configured to work with a variety of sensors, such as the [Ping1D sonar](https://bluerobotics.com/store/sonars/echosounders/ping-sonar-r2-rp/), making it adaptable to different data collection needs.

**Pre-configured Docker Environment:** Get up and running in minutes with **[docker-sgp-tools](https://github.com/itskalvik/docker-sgp-tools)**! A Docker container with a pre-configured simulation environment, including all necessary dependencies and tools to start development immediately.

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

### Running an ASV Mission

To start an Informative Path Planning (IPP) mission with an Ardupilot-based Autonomous Surface Vehicle (ASV), use the `asv.launch.py` launch file.

This will launch all the necessary components:
* `path_planner.py`: The core node for generating the informative path.
* `path_follower.py`: The node that controls the ASV to follow the generated path.
* `mavros.launch`: The bridge for communication with the Ardupilot flight control unit.
* `Sensor Node`: The appropriate sensor driver as defined in your configuration (e.g., Ping1D).

To start the mission, run the following command in your terminal:

```bash
ros2 launch ros_sgp_tools asv.launch.py
```

### Mission Configuration

The IPP mission is configured through two files and environment variables:

* **config.yaml:** Located in `ros_sgp_tools/launch/data/`, this file configures the IPP method using arguments from the `sgptools` library.
  * **`robot`**: Robot settings, including sensor type, data buffer size, mission type (`AdaptiveIPP`, `IPP`, or `Waypoint`), and seed (for reproducibility).
  * **`sensor`**: Configuration for the sensors.
  * **`ipp_model`**: IPP model settings, such as the number of waypoints and optimization method.
  * **`param_model`**: Configuration for the online learning model.
  * **`hyperparameters`**: Initial values for the GP models' (`ipp_model` and `param_model`) hyperparameters.
  * **`tsp`**: Configuration for the Traveling Salesperson Problem (TSP) solver.
* **mission.plan:** Located in `ros_sgp_tools/launch/data/`, this file defines the survey area and launch position using a **polygon-shaped** geofence created in [QGC](https://qgroundcontrol.com/).
* **Environment Variables:**
  * `DATA_FOLDER`: The folder where mission logs are saved. Defaults to the launch directory.
  * `FCU_URL`: The Ardupilot device connection URL. Defaults to `udp://0.0.0.0:14550@`.
  
  Set a variable using the export command:
  ```
  export <parameter_name>=<parameter_value>
  ```

**Note:** This package requires over 4GB of available memory (including swap). Ensure your system has sufficient swap space allocated.

### Visualizing Mission Data

After a mission, use the `visualize_data.launch.py` launch file to inspect the results. It processes a mission log and publishes the reconstructed map as a point cloud for viewing in Foxglove.

This launch file starts:
* `data_visualizer.py`: Publishes the point cloud data.
* `foxglove_bridge`: Connects to Foxglove for visualization.

#### Launch Arguments
* `mission_log`: The specific mission log folder to visualize. Defaults to the most recent log in the current directory.
* `num_samples`: The number of points for visualizing the point cloud. Defaults to `5000`.

To visualize a specific log, run:
```bash
ros2 launch ros_sgp_tools visualize_data.launch.py mission_log:=<log-folder-name>
```

#### Visualizer Configuration
* **viz_config.yaml:** To customize the visualizer, place this file inside the mission log folder you are viewing.

  * **`hyperparameters`**: Specify the GP model's hyperparameters.
  * **`optimizer`**: Provide optimizer arguments from the `sgptools` library.