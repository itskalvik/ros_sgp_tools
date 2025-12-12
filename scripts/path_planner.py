#! /usr/bin/env python3

import os
import shutil
import importlib
import traceback
from threading import Lock
from typing import List, Tuple, Optional

import h5py
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from time import gmtime, strftime

import tensorflow as tf
import gpflow

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import (
    MutuallyExclusiveCallbackGroup,
    ReentrantCallbackGroup,
)
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Point
from message_filters import ApproximateTimeSynchronizer
from ament_index_python.packages import get_package_share_directory

from ros_sgp_tools.srv import Waypoint

from gpflow.config import default_float
from sgptools.methods import get_method
from sgptools.kernels import get_kernel
from sgptools.utils.tsp import run_tsp, resample_path
from sgptools.utils.misc import polygon2candidates, project_waypoints, get_inducing_pts
from sgptools.utils.gpflow import get_model_params, optimize_model
from sgptools.core.transformations import IPPTransform
from sgptools.core.osgpr import init_osgpr

from utils import *  # LatLonStandardScaler, get_mission_plan, haversine, RunningStats, etc.


class PathPlanner(Node):
    """
    Informative path / coverage planner with multiple mission types:
      - Waypoint: use static mission plan waypoints.
      - IPP: offline informative path planning with GP model.
      - AdaptiveIPP: online adaptive IPP with GP parameter updates.
      - Coverage: two-stage coverage mission:
          1) initial exploratory path for data collection,
          2) kernel fit from data and offline coverage planning,
             then execute coverage path.
    """

    def __init__(self) -> None:
        super().__init__('path_planner')
        self.get_logger().info('Initializing')

        # Internal flags/state
        self._shutdown_requested: bool = False
        self.mission_type: str = 'Waypoint'  # default, overwritten by config

        # Core initialization steps
        self._load_config_and_plan()
        self._init_random_seeds()
        if self.mission_type in ('IPP', 'AdaptiveIPP', 'Coverage'):
            # Only non-Waypoint missions require GP hyperparameters
            self._init_gp_settings()
        self._init_objective_and_scaler()
        self._init_mission_models_and_waypoints()
        self._init_data_store()
        self._init_distances_and_state()
        self._init_sensors_and_sync()
        self._init_timers_and_services()

        self.get_logger().info('PathPlanner initialization complete')

    # -------------------------------------------------------------------------
    # Initialization helpers
    # -------------------------------------------------------------------------

    def _load_config_and_plan(self) -> None:
        """Load mission plan, configuration, and basic parameters."""
        # Mission/plan file
        default_plan_fname = os.path.join(
            get_package_share_directory('ros_sgp_tools'),
            'launch',
            'data',
            'mission.plan',
        )
        self.declare_parameter('geofence_plan', default_plan_fname)
        self.plan_fname: str = self.get_parameter('geofence_plan').value
        self.get_logger().info(f'GeoFence Plan File: {self.plan_fname}')

        # Config file
        default_config_fname = os.path.join(
            get_package_share_directory('ros_sgp_tools'),
            'launch',
            'data',
            'config.yaml',
        )
        self.declare_parameter('config_file', default_config_fname)
        self.config_fname: str = self.get_parameter('config_file').value
        self.get_logger().info(f'Config File: {self.config_fname}')

        if not os.path.exists(self.plan_fname):
            raise FileNotFoundError(f"Geofence plan file not found: {self.plan_fname}")
        if not os.path.exists(self.config_fname):
            raise FileNotFoundError(f"Config file not found: {self.config_fname}")

        with open(self.config_fname, 'r') as file:
            self.config = yaml.safe_load(file)

        robot_cfg = self.config.get('robot')
        if robot_cfg is None:
            raise RuntimeError("Missing 'robot' section in config.yaml")

        # Mission type
        self.mission_type = robot_cfg.get('mission_type', 'Waypoint')
        self.get_logger().info(f"Mission type: {self.mission_type}")

        if self.mission_type in ('IPP', 'AdaptiveIPP', 'Coverage'):
            # For IPP/AdaptiveIPP/Coverage we require hyperparameters
            if 'hyperparameters' not in self.config:
                raise RuntimeError(
                    "Missing 'hyperparameters' section in config.yaml "
                    "for IPP/AdaptiveIPP/Coverage mission."
                )

        # Base data folder
        self.declare_parameter('data_folder', '')
        base_data_folder = self.get_parameter('data_folder').value
        self.get_logger().info(f'Base Data Folder: {base_data_folder}')
        self.base_data_folder = base_data_folder

    def _init_random_seeds(self) -> None:
        """Initialize random seeds for reproducibility."""
        robot_cfg = self.config.get('robot', {})
        self.seed: Optional[int] = robot_cfg.get('seed')

        if self.seed is not None:
            tf.random.set_seed(self.seed)
            np.random.seed(self.seed)
            self.get_logger().info(f'Using seed: {self.seed}')
        else:
            self.get_logger().info('No seed specified; using non-deterministic behavior')

    def _init_gp_settings(self) -> None:
        """Configure gpflow global settings based on kernel type (non-Waypoint only)."""
        hyper_cfg = self.config['hyperparameters']
        self.kernel_name: str = hyper_cfg['kernel_function']

        if self.kernel_name in ['Attentive', 'NeuralSpectral']:
            gpflow.config.set_default_float(np.float32)
            gpflow.config.set_default_jitter(1e-1)
            self.get_logger().info('Using float32 and high jitter for deep kernel')
        else:
            gpflow.config.set_default_float(np.float64)
            gpflow.config.set_default_jitter(1e-6)
            self.get_logger().info('Using float64 and standard jitter')

    def _init_objective_and_scaler(self) -> None:
        """Set up objective point cloud, mission type, and coordinate scaler."""
        # Get mission plan (fence and start location) and optional waypoints
        if self.mission_type == 'Waypoint':
            self.fence_vertices, self.start_location, waypoints = get_mission_plan(
                self.plan_fname, get_waypoints=True
            )
            self.waypoints = waypoints[:, :2]
        else:
            self.fence_vertices, self.start_location = get_mission_plan(
                self.plan_fname, get_waypoints=False
            )
            self.waypoints = None  # will be set by IPP/Coverage model if needed

        ipp_cfg = self.config.get('ipp_model', {})
        num_samples = ipp_cfg.get('num_samples', 1000)

        # Candidate set for objective
        self.X_objective = polygon2candidates(
            self.fence_vertices, num_samples=num_samples, seed=self.seed
        )
        self.X_objective = np.array(self.X_objective).reshape(-1, 2)

        self.X_scaler = LatLonStandardScaler()
        self.X_scaler.fit(self.X_objective)
        self.X_objective = self.X_scaler.transform(self.X_objective)
        self.X_objective = self.X_objective.astype(default_float())

        # Normalize start location
        self.start_location = self.X_scaler.transform(
            np.array([self.start_location[:2]])
        )

        # Normalize waypoints for Waypoint mission
        if self.mission_type == 'Waypoint' and self.waypoints is not None:
            self.waypoints = self.X_scaler.transform(self.waypoints)

    def _init_mission_models_and_waypoints(self) -> None:
        """Initialize IPP, Coverage, and/or parameter models, and ensure waypoints are set."""
        # Optional IPP model & param model handle
        self.ipp_model = None
        self.ipp_model_config = None
        self.ipp_model_kwargs = None
        self.distance_budget: Optional[float] = None
        self.num_waypoints: Optional[int] = None
        self.param_model = None
        self.param_model_config = None
        self.param_model_kwargs = None
        self.param_model_method = None
        self.train_param_inducing = True
        self.num_param_inducing = 0
        self.kernel_name = getattr(self, "kernel_name", None)

        # Coverage-specific state
        self.coverage_phase: str = "none"   # "initial" or "coverage"
        self.coverage_planned: bool = False
        self.coverage_waypoints: Optional[np.ndarray] = None
        self.coverage_model = None
        self.coverage_fovs = None

        mission_type = self.mission_type

        if mission_type in ('IPP', 'AdaptiveIPP', 'Coverage'):
            hyper_cfg = self.config['hyperparameters']
            self.kernel_name = hyper_cfg['kernel_function']
            kernel_kwargs = hyper_cfg.get('kernel', {})
            kernel = get_kernel(self.kernel_name)(**kernel_kwargs)
            noise_variance = float(hyper_cfg['noise_variance'])

            if mission_type == 'IPP':
                # Static IPP path: IPP model only, no param model
                self.init_models(
                    init_ipp_model=True,
                    init_param_model=False,
                    kernel=kernel,
                    noise_variance=noise_variance,
                )
            elif mission_type == 'AdaptiveIPP':
                # Adaptive IPP: both IPP and param model
                self.init_models(
                    init_ipp_model=True,
                    init_param_model=True,
                    kernel=kernel,
                    noise_variance=noise_variance,
                )
            elif mission_type == 'Coverage':
                # Two-stage coverage mission:
                #   1) Initial exploratory path (TSP over candidates),
                #   2) Later: kernel fit + coverage planning (from collected data).
                coverage_cfg = self.config.get('ipp_model', {})
                self.num_waypoints = coverage_cfg.get('num_waypoints', 20)

                self.get_logger().info(
                    f"Coverage mission: generating initial exploratory path with "
                    f"{self.num_waypoints} points."
                )
                X_init = get_inducing_pts(
                    self.X_objective,
                    num_inducing=self.num_waypoints,
                    seed=self.seed,
                )
                self.get_logger().info("Running TSP solver to get initial coverage path...")
                X_init, _ = run_tsp(
                    X_init,
                    start_nodes=self.start_location,
                    **self.config.get('tsp', {}),
                )
                X_init = np.array(X_init)[0]
                X_init = project_waypoints(X_init, self.X_objective)

                self.waypoints = X_init
                self.coverage_phase = "initial"
                self.coverage_planned = False
                self.coverage_model = None
                self.coverage_waypoints = None
                self.coverage_fovs = None
        elif mission_type == 'Waypoint':
            # Pure waypoint mission: NO IPP / Coverage / param model; no hyperparameters required
            self.get_logger().info(
                'Waypoint mission: skipping IPP, Coverage, and parameter model initialization.'
            )
        else:
            raise ValueError(f'Invalid mission type: {mission_type}')

        # After models or direct plan parsing, waypoints must be defined
        if self.waypoints is None:
            raise RuntimeError('Waypoints not initialized after mission setup')

    def _init_data_store(self) -> None:
        """Create HDF5 data store and save initial mission info."""
        time_stamp = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        # Create mission-specific folder
        self.data_folder = os.path.join(
            self.base_data_folder, f'IPP-mission-{time_stamp}'
        )
        os.makedirs(self.data_folder, exist_ok=True)

        # Preserve plan and config for reproducibility
        shutil.copy(self.plan_fname, self.data_folder)
        shutil.copy(self.config_fname, self.data_folder)

        data_fname = os.path.join(self.data_folder, 'mission-log.hdf5')
        self.data_file = h5py.File(data_fname, "a")

        # Datasets for collected data
        self.dset_X = self.data_file.create_dataset(
            "X",
            (0, 2),
            maxshape=(None, 2),
            dtype=np.float64,
            chunks=True,
        )
        self.dset_y = self.data_file.create_dataset(
            "y",
            (0, 1),
            maxshape=(None, 1),
            dtype=np.float64,
            chunks=True,
        )

        # Save fence and initial path
        self.data_file.create_dataset(
            "fence_vertices",
            self.fence_vertices.shape,
            dtype=np.float64,
            data=self.fence_vertices,
        )

        fname = f"waypoints_{-1}-{strftime('%H-%M-%S', gmtime())}"
        self.data_file.create_dataset(
            fname,
            self.waypoints.shape,
            dtype=np.float64,
            data=self.X_scaler.inverse_transform(self.waypoints),
        )
        self.plot_paths(fname, self.waypoints, update_waypoint=0)

    def _init_distances_and_state(self) -> None:
        """Compute waypoint distances and initialize runtime / buffer state."""
        # Compute distances between waypoints for waypoint arrival estimation
        lat_lon_waypoints = self.X_scaler.inverse_transform(self.waypoints)
        self.distances = haversine(lat_lon_waypoints[1:], lat_lon_waypoints[:-1])

        # Setup runtime/data-related state
        self.data_X: List[np.ndarray] = []
        self.data_y: List[np.ndarray] = []
        self.all_X: List[np.ndarray] = []   # all samples for coverage kernel fit
        self.all_y: List[np.ndarray] = []
        self.current_waypoint: int = -1
        self.data_lock = Lock()
        self.waypoints_lock = Lock()
        self.runtime_est: float = 0.0
        self.heading_velocity: float = 1.0

        robot_cfg = self.config.get('robot', {})
        self.data_buffer_size: int = robot_cfg.get('data_buffer_size', 1000)

        # Running statistics for sensor normalization
        self.stats = RunningStats()

    def _init_sensors_and_sync(self) -> None:
        """Set up sensor objects, subscribers, and time synchronization."""
        sensors_module = importlib.import_module('sensors')
        self.sensors = []
        sensor_subscribers = []
        sensor_group = ReentrantCallbackGroup()

        # GPS is always used
        gps_obj = getattr(sensors_module, 'GPS')()
        self.sensors.append(gps_obj)
        sensor_subscribers.append(
            gps_obj.get_subscriber(self, callback_group=sensor_group)
        )

        # Optional secondary sensor
        robot_cfg = self.config.get('robot', {})
        sensor_name = robot_cfg.get('sensor', 'Altitude')
        if sensor_name != 'Altitude':
            sensor_obj = getattr(sensors_module, sensor_name)()
            self.sensors.append(sensor_obj)
            sensor_subscribers.append(
                sensor_obj.get_subscriber(self, callback_group=sensor_group)
            )

        self.time_sync = ApproximateTimeSynchronizer(
            sensor_subscribers, queue_size=10, slop=0.1, sync_arrival_time=True
        )
        self.time_sync.registerCallback(self.data_callback)

    def _init_timers_and_services(self) -> None:
        """Set up timers, services, and other ROS interfaces."""
        # Timer to update parameters and waypoints; ensure single instance at a time
        timer_group = MutuallyExclusiveCallbackGroup()
        self.timer = self.create_timer(
            5.0, self.update_with_data, callback_group=timer_group
        )

        # Waypoint service
        self.create_service(Waypoint, 'waypoint', self.waypoint_service_callback)

        # ETA / current waypoint info
        self.create_subscription(
            Float32MultiArray,
            'mavros/waypoint_eta',
            self.eta_callback,
            qos_profile_sensor_data,
        )

    # -------------------------------------------------------------------------
    # Properties / lifecycle
    # -------------------------------------------------------------------------

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_requested

    def request_shutdown(self, reason: str = "") -> None:
        if reason:
            self.get_logger().info(f"Shutdown requested: {reason}")
        self._shutdown_requested = True

    def destroy_node(self) -> None:
        """Ensure clean shutdown of resources."""
        try:
            if hasattr(self, "data_file") and self.data_file is not None:
                self.data_file.flush()
                self.data_file.close()
        except Exception as e:
            self.get_logger().error(f"Failed to close data file cleanly: {e}")
        finally:
            super().destroy_node()

    # -------------------------------------------------------------------------
    # Coverage mission: kernel fitting + coverage planning
    # -------------------------------------------------------------------------

    def plan_coverage_from_data(self) -> None:
        """
        Fit the kernel from collected data (initial path) and generate a
        coverage path, similar in spirit to the benchmark script:
          1) Use all_X/all_y as training data.
          2) Fit kernel with get_model_params using config['hyperparameters'].
          3) Compute max prior variance over X_objective.
          4) Use ipp_model settings to build coverage model and optimize.
        """
        if self.coverage_planned:
            return

        if len(self.all_X) == 0:
            self.get_logger().warning(
                "No data collected for coverage kernel fitting; "
                "using X_objective with dummy zero targets."
            )
            X_train = self.X_objective.copy()
            y_train = np.zeros((X_train.shape[0], 1), dtype=float)
        else:
            X_train = np.array(self.all_X, dtype=float).reshape(-1, 2)
            y_train = np.array(self.all_y, dtype=float).reshape(-1, 1)

        # Scale X_train into GP space
        X_train_scaled = self.X_scaler.transform(X_train).astype(default_float())
        y_train = y_train.astype(default_float())

        # Build base kernel from config hyperparameters
        hyper_cfg = self.config['hyperparameters']
        kernel_kwargs = hyper_cfg.get('kernel', {})
        base_kernel = get_kernel(self.kernel_name)(**kernel_kwargs)

        self.get_logger().info(
            f"Fitting GP kernel '{self.kernel_name}' on "
            f"{X_train_scaled.shape[0]} samples for coverage..."
        )

        # Fit kernel + noise variance (like benchmark get_model_params)
        _, noise_variance, kernel, init_model = get_model_params(
            X_train=X_train_scaled,
            y_train=y_train,
            kernel=base_kernel,
            return_model=True,
            verbose=False,
        )

        # Compute max prior variance over candidate set
        prior_mean, prior_var = init_model.predict_f(self.X_objective)
        max_prior_var = float(prior_var.numpy().max())
        self.get_logger().info(f"Max prior variance on objective set: {max_prior_var:.4f}")

        # Coverage configuration
        coverage_cfg = self.config.get('ipp_model', {})
        method_name = coverage_cfg.get('method', 'HexCoverage')
        num_sensing = len(self.X_objective)

        # var_threshold: either explicit, or variance_ratio * max_prior_var
        var_threshold = coverage_cfg.get('var_threshold')
        if var_threshold is None:
            var_ratio = coverage_cfg.get('variance_ratio', 0.5)
            var_threshold = max_prior_var * float(var_ratio)
            self.get_logger().info(
                f"Coverage var_threshold not provided; using "
                f"max_prior_var * variance_ratio = {max_prior_var:.4f} * {var_ratio:.2f} "
                f"= {var_threshold:.4f}"
            )
        else:
            self.get_logger().info(f"Using explicit coverage var_threshold={var_threshold}")

        optimizer_kwargs = coverage_cfg.get('optimizer', {})

        # Instantiate coverage planner (similar to benchmark's cmodel)
        coverage_model_cls = get_method(method_name)
        self.coverage_model = coverage_model_cls(
            num_sensing=num_sensing,
            X_objective=self.X_objective,
            kernel=kernel,
            noise_variance=noise_variance,
        )

        self.get_logger().info(
            f"Running coverage planner optimize() with method={method_name}..."
        )
        X_sol, fovs = self.coverage_model.optimize(
            var_threshold=var_threshold,
            return_fovs=True,
            start_nodes=self.start_location,
            **optimizer_kwargs,
        )
        X_sol = np.array(X_sol)[0]
        X_sol = project_waypoints(X_sol, self.X_objective)

        self.coverage_fovs = fovs
        self.coverage_waypoints = X_sol
        self.coverage_planned = True

        self.plot_paths(
            f"coverage_solution-{strftime('%H-%M-%S', gmtime())}",
            self.coverage_waypoints,
            update_waypoint=-1,
        )

        self.get_logger().info(
            f"Coverage planner produced {len(self.coverage_waypoints)} waypoints."
        )

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def eta_callback(self, msg: Float32MultiArray) -> None:
        """
        Callback to get the current waypoint ETA and heading velocity.
        Also used to update the remaining distance to the current waypoint.
        """
        self.heading_velocity = msg.data[2]

        # Ensure index is valid
        if 0 <= self.current_waypoint < len(self.distances):
            with self.waypoints_lock:
                self.distances[self.current_waypoint] = msg.data[1]

    def waypoint_service_callback(
        self, request: Waypoint.Request, response: Waypoint.Response
    ) -> Waypoint.Response:
        """
        Handle requests for the next waypoint.
        For Coverage:
          - First, execute initial path.
          - After initial path is done, fit kernel and generate coverage path.
        """
        if not request.ok:
            self.get_logger().error(
                'Path follower failed to reach a waypoint; requesting shutdown of online planner.'
            )
            self.request_shutdown("Path follower error")
            response.new_waypoint = False
            return response

        self.current_waypoint += 1

        # Coverage mission: handle phase transitions
        if self.mission_type == 'Coverage':
            # Finished current phase's path?
            if self.coverage_phase == "initial" and self.current_waypoint >= len(self.waypoints):
                self.get_logger().info(
                    "Initial coverage path complete; fitting kernel and planning coverage path..."
                )
                try:
                    self.plan_coverage_from_data()
                except Exception as e:
                    self.get_logger().error(
                        f"Coverage planning failed with error: {e}\n{traceback.format_exc()}"
                    )
                    self.request_shutdown("Coverage planning failed")
                    response.new_waypoint = False
                    return response

                # Switch to coverage phase and start at first coverage waypoint
                self.coverage_phase = "coverage"
                self.current_waypoint = 0

                with self.waypoints_lock:
                    self.waypoints = self.coverage_waypoints
                    lat_lon_waypoints = self.X_scaler.inverse_transform(self.waypoints)
                    self.distances = haversine(
                        lat_lon_waypoints[1:], lat_lon_waypoints[:-1]
                    )
                    waypoint = self.waypoints[self.current_waypoint].reshape(1, -1)
                    waypoint = self.X_scaler.inverse_transform(waypoint)[0]

                self.get_logger().info(
                    f"Starting coverage path, waypoint index {self.current_waypoint}"
                )
                response.new_waypoint = True
                response.waypoint = Point(
                    x=float(waypoint[0]),
                    y=float(waypoint[1]),
                )
                return response

            elif self.coverage_phase == "coverage" and self.current_waypoint >= len(
                self.waypoints
            ):
                # Final mission completion
                self.get_logger().info("Coverage mission complete.")
                response.new_waypoint = False
                return response

        # Non-coverage (or ongoing coverage) behavior:
        if self.current_waypoint >= len(self.waypoints):
            response.new_waypoint = False
            return response

        self.get_logger().info(f'Current waypoint: {self.current_waypoint}')

        with self.waypoints_lock:
            waypoint = self.waypoints[self.current_waypoint].reshape(1, -1)
            waypoint = self.X_scaler.inverse_transform(waypoint)[0]
            response.new_waypoint = True
            response.waypoint = Point(x=float(waypoint[0]), y=float(waypoint[1]))

        return response

    def data_callback(self, *args) -> None:
        """Collect synchronized sensor data."""
        # Use data only when the vehicle is moving (avoids failed Cholesky in OSGPR)
        if self.current_waypoint > 0 and self.current_waypoint < len(self.waypoints):
            position = self.sensors[0].process_msg(args[0])

            if len(args) == 1:
                data_X = [position[:2]]
                data_y = [position[2]]
            else:
                # position data is used by only a few sensors
                data_X, data_y = self.sensors[1].process_msg(
                    args[1], position=position
                )

            # Update running stats
            self.stats.push(data_y, per_dim=True)

            with self.data_lock:
                self.data_X.extend(data_X)
                self.data_y.extend(data_y)
                # Accumulate all data for coverage kernel fit as well
                if self.mission_type == "Coverage":
                    self.all_X.extend(data_X)
                    self.all_y.extend(data_y)

    # -------------------------------------------------------------------------
    # Model initialization and updates (used only for IPP/AdaptiveIPP)
    # -------------------------------------------------------------------------

    def init_models(
        self,
        init_ipp_model: bool = True,
        init_param_model: bool = True,
        kernel=None,
        noise_variance: float = 1e-3,
    ) -> None:
        """
        Initialize IPP model and/or parameter model.
        `kernel` and `noise_variance` are provided by _init_mission_models_and_waypoints.
        """
        if init_ipp_model:
            self.ipp_model_config = self.config['ipp_model']
            self.num_waypoints = self.ipp_model_config['num_waypoints']

            # Sample uniform random initial waypoints and compute initial paths
            X_init = get_inducing_pts(
                self.X_objective,
                (self.num_waypoints - 1),
                seed=self.seed,
            )
            self.get_logger().info("Running TSP solver to get the initial IPP path...")
            X_init, _ = run_tsp(
                X_init,
                start_nodes=self.start_location,
                **self.config.get('tsp', {}),
            )
            X_init = np.array(X_init)

            transform_kwargs = self.ipp_model_config.get('transform', {})
            self.distance_budget = None

            # Map distance budget in meters to normalized units
            if transform_kwargs.get('distance_budget') is not None:
                self.distance_budget = transform_kwargs['distance_budget']
                transform_kwargs['distance_budget'] = self.X_scaler.meters2units(
                    self.distance_budget
                )

            transform = IPPTransform(
                Xu_fixed=X_init[:, :1, :],
                **transform_kwargs,
            )

            ipp_model_cls = get_method(self.ipp_model_config['method'])
            self.ipp_model = ipp_model_cls(
                self.num_waypoints,
                X_objective=self.X_objective,
                kernel=kernel,
                noise_variance=noise_variance,
                transform=transform,
                X_init=X_init[0],
            )

            # Project the waypoints to be within the bounds of the environment
            self.get_logger().info("Running IPP solver to update the initial path...")
            self.ipp_model_kwargs = self.ipp_model_config.get('optimizer', {})
            self.waypoints = self.ipp_model.optimize(**self.ipp_model_kwargs)[0]
            self.waypoints = project_waypoints(self.waypoints, self.X_objective)

            if self.distance_budget is not None:
                distance = self.ipp_model.transform.distance(
                    self.waypoints.reshape(-1, 2)
                ).numpy()
                distance = self.X_scaler.units2meters(distance)
                if distance > self.distance_budget:
                    self.get_logger().warn(
                        "Distance budget constraint violated! Consider increasing the "
                        "transform's constraint_weight!"
                    )
                self.get_logger().info(f"Distance Budget: {self.distance_budget:.2f} m")
                self.get_logger().info(f"Path Length: {distance[0]:.2f} m")

            self.get_logger().info(
                f'Initialized {self.ipp_model_config["method"]} IPP model'
            )

        if init_param_model:
            # Initialize the param model
            self.param_model_config = self.config['param_model']
            self.param_model_kwargs = self.param_model_config.get('optimizer', {})
            self.param_model_method = self.param_model_config['method']

            if self.param_model_method == 'SSGP':
                self.train_param_inducing = self.param_model_config.get(
                    'train_inducing', True
                )
                self.num_param_inducing = self.param_model_config['num_inducing']
                self.param_model = init_osgpr(
                    self.X_objective,
                    num_inducing=self.num_param_inducing,
                    kernel=kernel,
                    noise_variance=noise_variance,
                )
            else:
                raise NotImplementedError(
                    f"Unsupported param model method: {self.param_model_method}"
                )

            self.get_logger().info(
                f'Initialized {self.param_model_method} Parameter model'
            )

    def update_with_data(self, force_update: bool = False) -> None:
        """
        Update the hyperparameters and waypoints if the buffer is full
        or if force_update is True and enough data points are available.
        For Waypoint and Coverage missions, this will only log data and plot paths.
        """
        if not self.data_X and not force_update and self.current_waypoint < len(
            self.waypoints
        ):
            # Nothing to do
            return

        enough_buffer = len(self.data_X) > self.data_buffer_size
        enough_forced = force_update and (
            hasattr(self, "num_param_inducing")
            and len(self.data_X) > getattr(self, "num_param_inducing", 0)
        )
        mission_complete = self.current_waypoint >= len(self.waypoints)

        if not (enough_buffer or enough_forced or mission_complete):
            return

        # Make local copies of the data and clear the data buffers
        with self.data_lock:
            data_X = np.array(self.data_X, dtype=default_float()).reshape(-1, 2)
            data_y = np.array(self.data_y, dtype=default_float()).reshape(-1, 1)
            self.data_X = []
            self.data_y = []

        # Update the parameters and waypoints for AdaptiveIPP only
        update_waypoint = -1
        new_waypoints = None

        if (
            self.mission_type == 'AdaptiveIPP'
            and self.current_waypoint < len(self.waypoints)
        ):
            start_time = self.get_clock().now().nanoseconds
            self.update_param(data_X, data_y)
            end_time = self.get_clock().now().nanoseconds
            runtime = (end_time - start_time) / 1e9
            self.get_logger().info(f'Param update time: {runtime:.3f} secs')
            self.runtime_est = runtime

            # Update the waypoints
            start_time = self.get_clock().now().nanoseconds
            new_waypoints, update_waypoint = self.update_waypoints()
            end_time = self.get_clock().now().nanoseconds
            runtime = (end_time - start_time) / 1e9
            self.get_logger().info(f'IPP update time: {runtime:.3f} secs')
            self.runtime_est += runtime
        else:
            if self.mission_type != 'AdaptiveIPP':
                self.get_logger().debug(
                    'Skipping IPP update because mission_type is not AdaptiveIPP'
                )

        # If waypoints were updated, accept waypoints if update waypoint was not already passed
        if update_waypoint != -1 and new_waypoints is not None:
            with self.waypoints_lock:
                if self.current_waypoint < update_waypoint:
                    self.waypoints = new_waypoints
                    lat_lon_waypoints = self.X_scaler.inverse_transform(new_waypoints)
                    self.distances = haversine(
                        lat_lon_waypoints[1:], lat_lon_waypoints[:-1]
                    )
                    num_changed = len(self.waypoints) - (update_waypoint + 1)
                    self.get_logger().info(
                        f'Updated IPP path from waypoint {update_waypoint + 1} onward '
                        f'({num_changed} waypoints modified)'
                    )

        # Dump data to data store
        if data_X.size > 0:
            self.dset_X.resize(self.dset_X.shape[0] + len(data_X), axis=0)
            self.dset_X[-len(data_X):] = data_X

            self.dset_y.resize(self.dset_y.shape[0] + len(data_y), axis=0)
            self.dset_y[-len(data_y):] = data_y

        current_waypoint_idx = (
            self.current_waypoint if self.current_waypoint > -1 else 0
        )
        if self.mission_type == "Coverage" and self.coverage_phase == "coverage":
            current_waypoint_idx += self.num_waypoints
        fname = f"waypoints_{current_waypoint_idx}-{strftime('%H-%M-%S', gmtime())}"

        # Always plot path; include data if available
        X_data_plot = self.X_scaler.transform(data_X) if data_X.size > 0 else None

        if update_waypoint != -1 and new_waypoints is not None:
            lat_lon_waypoints = self.X_scaler.inverse_transform(new_waypoints)
            dset = self.data_file.create_dataset(
                fname,
                self.waypoints.shape,
                dtype=np.float64,
                data=lat_lon_waypoints,
            )
            dset.attrs['update_waypoint'] = update_waypoint

            self.plot_paths(
                fname,
                self.waypoints,
                X_data=X_data_plot,
                update_waypoint=update_waypoint,
            )
        else:
            self.plot_paths(
                fname,
                self.waypoints,
                X_data=X_data_plot,
                update_waypoint=-1,
            )

        # Handle mission completion
        if self.current_waypoint >= len(self.waypoints):
            # For Coverage initial phase, don't shutdown here; shutdown only after
            # the coverage phase is done (handled in waypoint_service).
            if self.mission_type == "Coverage" and self.coverage_phase == "initial":
                self.get_logger().info(
                    "Reached end of initial coverage path; waiting for phase transition."
                )
                return

            # Re-run method to get last batch of data, if any
            if not force_update and self.data_X:
                self.update_with_data(force_update=True)
            self.get_logger().info(
                'Finished mission, requesting shutdown of online planner'
            )
            self.request_shutdown("Mission complete")

    def update_waypoints(self) -> Tuple[np.ndarray, int]:
        """Update the IPP solution and return (new_waypoints, update_waypoint_index)."""
        self.get_logger().info('Updating IPP solution...')

        update_waypoint = self.get_update_waypoint()
        if update_waypoint == -1:
            self.get_logger().info('No waypoint can be safely updated at this time')
            return self.waypoints, update_waypoint

        # Freeze the visited inducing points
        Xu_visited = self.waypoints[: update_waypoint + 1]
        Xu_visited = Xu_visited.reshape(1, -1, 2)
        self.ipp_model.transform.update_Xu_fixed(Xu_visited)

        # Get the new inducing points for the path
        self.ipp_model.update(
            self.param_model.kernel,
            self.param_model.likelihood.variance,
        )
        waypoints = self.ipp_model.optimize(**self.ipp_model_kwargs)[0]

        # Might move waypoints before the current waypoint (reset to avoid update rejection)
        waypoints = project_waypoints(waypoints, self.X_objective)
        waypoints[: update_waypoint + 1] = self.waypoints[: update_waypoint + 1]

        if self.distance_budget is not None:
            distance = self.ipp_model.transform.distance(
                waypoints.reshape(-1, 2)
            ).numpy()
            distance = self.X_scaler.units2meters(distance)
            if distance > self.distance_budget:
                self.get_logger().warn(
                    "Distance budget constraint violated! Consider increasing the "
                    "transform's constraint_weight!"
                )
            self.get_logger().info(f"Distance Budget: {self.distance_budget:.2f} m")
            self.get_logger().info(f"Path Length: {distance[0]:.2f} m")

        return waypoints, update_waypoint

    def update_param(self, X_new: np.ndarray, y_new: np.ndarray) -> None:
        """Update the SSGP parameters (AdaptiveIPP only)."""
        self.get_logger().info('Updating SSGP parameters...')

        # Normalize the data, use running mean and std for sensor data
        X_new = self.X_scaler.transform(X_new)
        X_new = X_new.astype(default_float())

        eps = 1e-6
        y_new = (y_new - self.stats.mean) / (self.stats.std + eps)
        y_new = y_new.astype(default_float())
        self.get_logger().info(f'Data Mean: {self.stats.mean}')
        self.get_logger().info(f'Data Std: {self.stats.std}')

        # Don't update the parameters if the current target is the last waypoint
        if self.num_waypoints is not None and self.current_waypoint >= (
            self.num_waypoints - 1
        ):
            self.get_logger().info(
                'Current waypoint is the last target; skipping parameter update.'
            )
            return

        # Set the inducing points to be along the traversed portion of the planned path
        inducing_variable = np.copy(self.waypoints[: self.current_waypoint + 1])

        # Ensure inducing points do not extend beyond the collected data
        inducing_variable[-1] = X_new[-1]

        # Resample the path to the number of inducing points
        inducing_variable = resample_path(inducing_variable, self.num_param_inducing)

        # Update SSGP with new batch of data
        self.param_model.update((X_new, y_new), inducing_variable=inducing_variable)

        if self.train_param_inducing:
            trainable_variables = None
        else:
            trainable_variables = self.param_model.trainable_variables[1:]

        try:
            optimize_model(
                self.param_model,
                trainable_variables=trainable_variables,
                **self.param_model_kwargs,
            )
        except Exception:
            # Failsafe for Cholesky decomposition failure
            self.get_logger().error(traceback.format_exc())
            self.get_logger().warning(
                "Failed to update parameter model! Resetting parameter model..."
            )
            # Reset param model; keep IPP model as-is
            hyper_cfg = self.config['hyperparameters']
            kernel_kwargs = hyper_cfg.get('kernel', {})
            kernel = get_kernel(self.kernel_name)(**kernel_kwargs)
            noise_variance = float(hyper_cfg['noise_variance'])
            self.init_models(
                init_ipp_model=False,
                init_param_model=True,
                kernel=kernel,
                noise_variance=noise_variance,
            )

        if self.kernel_name == 'RBF':
            self.get_logger().info(
                f'SSGP kernel lengthscales: {self.param_model.kernel.lengthscales.numpy():.4f}'
            )
            self.get_logger().info(
                f'SSGP kernel variance: {self.param_model.kernel.variance.numpy():.4f}'
            )
            self.get_logger().info(
                f'SSGP likelihood variance: {self.param_model.likelihood.variance.numpy():.4f}'
            )

    def get_update_waypoint(self) -> int:
        """Returns the waypoint index that is safe to update."""
        # Do not update the current target waypoint
        with self.waypoints_lock:
            for i in range(self.current_waypoint, len(self.distances)):
                if self.runtime_est <= 0.0 or self.heading_velocity <= 0.0:
                    # If runtime estimate or velocity is invalid, do not update
                    return -1
                if self.distances[i] / self.heading_velocity > self.runtime_est:
                    # Map path edge idx to waypoint index
                    return i + 1

        # Do not update the path if none of waypoints can be
        # updated before the vehicle reaches them
        return -1

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------

    def plot_paths(
        self,
        fname: str,
        waypoints: np.ndarray,
        X_data: Optional[np.ndarray] = None,
        inducing_pts: Optional[np.ndarray] = None,
        update_waypoint: Optional[int] = None,
    ) -> None:
        """Plot current candidate set, path, and optionally data and inducing points."""
        plt.figure()
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.xlabel('X')
        plt.ylabel('Y')

        plt.scatter(
            self.X_objective[:, 0],
            self.X_objective[:, 1],
            marker='.',
            s=1,
            label='Candidates',
        )
        plt.plot(
            waypoints[:, 0],
            waypoints[:, 1],
            label='Path',
            marker='o',
            c='r',
        )

        if update_waypoint is not None and update_waypoint >= 0:
            plt.scatter(
                waypoints[update_waypoint, 0],
                waypoints[update_waypoint, 1],
                label='Update Waypoint',
                zorder=2,
                c='g',
            )

        if X_data is not None:
            plt.scatter(
                X_data[:, 0],
                X_data[:, 1],
                label='Data',
                c='b',
                marker='x',
                zorder=3,
                s=1,
            )

        if inducing_pts is not None:
            plt.scatter(
                inducing_pts[:, 0],
                inducing_pts[:, 1],
                label='Inducing Pts',
                marker='.',
                c='g',
                zorder=4,
                s=2,
            )

        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.savefig(
            os.path.join(self.data_folder, f'{fname}.png'),
            bbox_inches='tight',
        )
        plt.close()


def main(args=None) -> None:
    rclpy.init(args=args)
    online_ipp = PathPlanner()
    executor = MultiThreadedExecutor()
    executor.add_node(online_ipp)

    try:
        while rclpy.ok() and not online_ipp.shutdown_requested:
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        online_ipp.get_logger().info('Keyboard interrupt, shutting down')
    finally:
        executor.shutdown()
        online_ipp.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
