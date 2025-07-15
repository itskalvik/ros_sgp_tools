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

SGP-Tools is a powerful and flexible Python library designed for solving Sensor Placement and Informative Path Planning (IPP) problems, enabling efficient and scalable solutions for environment monitoring, e.g., monitoring air/water quality, soil moisture, or temperature. This ROS 2 package integrates SGP-Tools with the ROS ecosystem, allowing for easy deployment of advanced IPP algorithms on robotic hardware.

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

## Launch Files & Usage

### Running an ASV Mission

The `asv.launch.py` launch file is used to start an Informative Path Planning (IPP) mission with an Ardupilot-based Autonomous Surface Vehicle (ASV).

This will launch all the necessary components, including:
* `path_planner.py`: The core node for generating the informative path.
* `path_follower.py`: The node that controls the ASV to follow the generated path.
* `mavros.launch`: The bridge for communication with the Ardupilot flight control unit.
* `Sensor Node`: The appropriate sensor driver as defined in your configuration (e.g., Ping1D).

To start the mission, run the following command in your terminal:

```bash
ros2 launch ros_sgp_tools asv.launch.py
```

### Visualizing Mission Data

Once a mission is complete, you can use the `visualize_data.launch.py` launch file to inspect the results. This launch file starts the `data_visualizer.py` node to process a mission log and publishes the reconstructed map as a point cloud for viewing in Foxglove.


* `data_visualizer.py`: publishes the point cloud data.
* `foxglove_bridge`: to visualize the data in Foxglove.

The launch file accepts the following arguments:

* `mission_log`: The specific mission log folder to visualize. Defaults to the most recent mission log in the current directory.
* `num_samples`: The number of points to use for visualizing the point cloud. Defaults to `5000`.
* `kernel`: The kernel function used for the terrain estimation model. Defaults to `RBF`.

To visualize the data from a specific log folder, run:

```bash
ros2 launch ros_sgp_tools visualize_data.launch.py mission_log:=<log-folder-name>
```

## Configuration

The behavior of the `ros_sgp_tools` package can be configured using the `config.yaml` file located in `ros_sgp_tools/launch/data/`. This file allows you to configure the IPP method with all the arguments available in the `sgptools` library.

* **`robot`**: Configuration for the robot, including the sensor type, data buffer size, and mission type (`AdaptiveIPP`, `IPP`, or `Waypoint`).
* **`sensor`**: Configuration for the sensors.
* **`ipp_model`**: Configuration for the informative path planning model, including the number of waypoints, optimization method, and constraints. The `method` and `optimizer` parameters correspond to the available options in the `sgptools` library.
* **`param_model`**: Configuration for the parameter model used for online learning.
* **`hyperparameters`**: Initial values for the hyperparameters of the GP models.
* **`tsp`**: Configuration for the Traveling Salesperson Problem (TSP) solver.

---

## Nodes

### `path_planner.py`

This is the main node for informative path planning. It subscribes to sensor data, updates a Gaussian Process model of the environment, and plans a path to collect more informative data. It provides a service to get the next waypoint and subscribes to the estimated time of arrival to the next waypoint.

### `path_follower.py`

This node controls the vehicle to follow the waypoints provided by the `path_planner`. It's a service client that requests the next waypoint from the `path_planner` and navigates the vehicle to it.

### `data_visualizer.py`

This node is used to visualize the collected data and the reconstructed map. It reads the mission log, processes the data, and publishes it as a point cloud, which can be visualized in tools like Foxglove.
