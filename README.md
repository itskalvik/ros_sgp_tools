<p align="center">
  <img src=".assets/SGP-Tools.png#gh-light-mode-only" alt="SGP-Tools Logo" width="600"/>
  <img src=".assets/logo_dark.png#gh-dark-mode-only" alt="SGP-Tools Logo" width="600"/>
</p>

<p align="center">
  <em>A ROS 2 companion package for <a href="http://sgp-tools.com">SGP-Tools</a>, a Python library for efficient sensor placement and informative path planning
</em>
</p>

**SGP-Tools** is a Python library for **Sensor Placement and Informative Path Planning (IPP)**. This ROS 2 package bridges SGP-Tools to real-world robots, providing out-of-the-box adaptive and online path planning for field robotics and simulation.

## ✨ Features
* **Online & Adaptive Path Planning** 

  Run IPP missions in real time on [ArduPilot-based vehicles](https://ardupilot.org/copter/docs/common-use-cases-and-applications.html). Supports adaptive re-planning using streaming sensor data.

* **Flexible Mission Types**
  
  Choose between `AdaptiveIPP`, `IPP`, or `Waypoint` missions via config.

* **Sensor Integration**

  Supports a variety of sensors:
  * **Ping1D Sonar** (real & Gazebo simulation)
  * Pressure sensors
  * GPS
  * ZED stereo cameras (beta version) 
  
  Add new sensors by extending `sensors.py`.

* **Realistic Simulation**

  Works seamlessly with Gazebo and ArduPilot SITL.

* **Data Logging & Visualization**

  Mission data is saved as HDF5 logs and can be visualized in **[Foxglove](https://foxglove.dev/)** or analyzed in the provided **Jupyter notebook**.

  Easily configurable to support a variety of sensors, such as the [Ping1D sonar](https://bluerobotics.com/store/sonars/echosounders/ping-sonar-r2-rp/), making it adaptable to different mission requirements.

* **Docker Support**

  **[docker-sgp-tools](https://github.com/itskalvik/docker-sgp-tools)** provides a one-command, ready-to-go container for simulation and development.

## Installation 🛠️

1.  **Create and initialize a ROS 2 workspace:**

    ```bash
    mkdir -p ~/ros2_ws/src
    cd ~/ros2_ws/src
    git clone https://github.com/itskalvik/ros_sgp_tools.git
    ```

2.  **Install Python & ROS dependencies:**

    ```bash
    cd ~/ros2_ws/src/ros_sgp_tools
    python3 -m pip install -r requirements.txt
    cd ~/ros2_ws
    rosdep install --from-paths src --ignore-src -y
    ```

3.  **Build and source the package:**

    ```bash
    cd ~/ros2_ws
    colcon build --symlink-install
    source ~/ros2_ws/install/setup.bash
    ```

## 🚀 Usage

### 🏄‍♂️ Running an ASV Mission

To start an **Informative Path Planning (IPP) mission** with an ArduPilot-based ASV:

```bash
ros2 launch ros_sgp_tools asv.launch.py
```

This command launches all the necessary components:
* `path_planner.py`: Online informative path planner node.
* `path_follower.py`: Vehicle guidance and waypoint following.
* **Sensor Node**: As selected in `config.yaml` (e.g., Ping1D or Gazebo).
* MAVROS and flight control interface.

**Environment Variables**

Set these before launching for custom runs:

  * `DATA_FOLDER`: Path for saving mission logs (default: launch directory).
  * `FCU_URL`: MAVLink/ArduPilot connection string (default:`udp://0.0.0.0:14550@`).

Example:
```
export DATA_FOLDER=~/asv_logs
export FCU_URL=udp://0.0.0.0:14550@
```

### 🗂️ Mission Configuration

Configuration is via YAML files and plan files in `ros_sgp_tools/launch/data/`:

**config.yaml**
Defines the robot, mission, models, sensors, and solver parameters.

**Key sections:**
  * `robot`: sensor type, mission type, buffer sizes, random seed.
  * `sensor`: sensor-specific settings (e.g., serial port for Ping1D).
  * `ipp_model`: number of waypoints, solver type, transform/budget.
  * `param_model`: online GP/SSGP model for parameter learning.
  * `hyperparameters`: initial GP model params (kernel, noise, etc.).
  * `tsp`: TSP solver configuration.

**Tip:** See [SGP-Tools API](https://www.sgp-tools.com/api/index.html) documentation for parameter references.

**mission.plan**

Geofence and home location in [QGroundControl](https://qgroundcontrol.com/) plan file format.

* Used to define the survey polygon and start point.
* Optionally includes explicit waypoint lists for `Waypoint` missions.

### 🛰️ Visualization

After a mission, visualize results with:

```bash
ros2 launch ros_sgp_tools visualize_data.launch.py mission_log:=<log-folder-name>
```

This starts:

* `data_visualizer.py`: Publishes point cloud reconstructions from log.
* `foxglove_bridge`: Connects to [Foxglove](https://foxglove.dev/) for real-time visualization.

#### Launch Arguments
* `mission_log`: Name of log folder (default: most recent).
* `num_samples`: Number of points to sample in the point cloud (default: `5000`). 

#### Custom Visualizer Configuration
Place a `viz_config.yaml` in the log directory to customize:
  * **`hyperparameters`**: Override GP model for reconstruction.
  * **`optimizer`**: Pass optimizer args to SGP-Tools for visualization.

### 📄 Node & Script Overview
| File                 | Description                                                                                                                             |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `asv.launch.py`      | Main launch file: brings up all mission nodes and interfaces, configures sensor node based on YAML, checks minimum memory requirements. |
| `path_planner.py`    | Online IPP planner; supports Adaptive/Static/Waypoint, manages mission logs, sensor fusion, GP/SSGP learning, waypoint service.         |
| `path_follower.py`   | Drives the ASV using waypoints, interacts with the planner through a service, arms/disarms vehicle, manages mission progress.           |
| `sensors.py`         | Sensor abstraction layer. Supports GPS, Ping1D, simulated sonar, pressure, ZED. Easy to extend for new devices.                         |
| `data_visualizer.py` | Reads mission logs, reconstructs spatial field, publishes 3D point cloud to ROS2 for visualization.                                     |
| `utils.py`           | Utilities for coordinate transforms, mission file parsing, normalization, and data scaling.                                             |

### 🤖 Sensor Support
#### Supported sensors out of the box:

* **Ping1D:** Real and Gazebo-simulated.
* **GPS:** ArduPilot via MAVROS.
* **Pressure:** For relative altitude/depth.
* **ZED Camera:** 3x3 grid extraction of depth (beta version).
* **GazeboPing1D:** For simulation.

#### To add new sensors:
Extend the **SensorCallback** class in **sensors.py** with appropriate ROS topic and data extraction logic.

### 🪵🪓 Logging & Data Format
* **Mission logs:**
  Saved as HDF5 in `DATA_FOLDER/IPP-mission-<timestamp>/`.
  * Includes: waypoints, geofence, all sensor measurements, mission params.

* **Configuration & plan files** are copied into the log folder for reproducibility.

* **Jupyter notebook** (`postprocess.ipynb`) provided for custom, offline analysis.

### 😵 Troubleshooting & Tips

* **Memory:** 

  At least 4GB RAM (including swap) is required; the launch file checks this.

* **Simulation:**

  For Gazebo/SITL, ensure ArduPilot and simulation environment are properly installed.

* **Parameter Tuning:**

  Use `viz_config.yaml` to test different GP hyperparameters on collected data.

* **Extending Missions:**

  To run custom paths, edit `mission.plan` or provide your own via QGroundControl.