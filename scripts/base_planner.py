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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from time import gmtime, strftime

import tensorflow as tf
import gpflow

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
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
from sgptools.utils.gpflow import optimize_model
from sgptools.core.transformations import IPPTransform
from sgptools.core.osgpr import init_osgpr

from utils import *  # LatLonStandardScaler, get_mission_plan, haversine, RunningStats, etc.


class BasePathPlanner(Node):
    """
    Base node for all mission types. Subclasses implement:
      - _init_mission_models_and_waypoints()
      - (optionally) waypoint_service_callback(), _should_update_models(), etc.
    """

    # Subclasses set this to "Waypoint" / "IPP" / "AdaptiveIPP" / "Coverage"
    MISSION_TYPE: Optional[str] = None

    def __init__(self) -> None:
        super().__init__("path_planner")
        self.get_logger().info("Initializing")

        self._shutdown_requested: bool = False
        self.mission_type: str = self.MISSION_TYPE or "Waypoint"

        self._load_config_and_plan()
        self._init_random_seeds()

        # Only GP-based missions require gpflow settings/hyperparameters
        if self.mission_type in ("IPP", "AdaptiveIPP", "Coverage"):
            self._init_gp_settings()

        self._init_objective_and_scaler()
        self._init_mission_models_and_waypoints()
        self._init_data_store()
        self._init_distances_and_state()
        self._init_sensors_and_sync()
        self._init_timers_and_services()

        self.get_logger().info("PathPlanner initialization complete")

    # -------------------------------------------------------------------------
    # Initialization helpers
    # -------------------------------------------------------------------------

    def _load_config_and_plan(self) -> None:
        default_plan_fname = os.path.join(
            get_package_share_directory("ros_sgp_tools"),
            "launch",
            "data",
            "mission.plan",
        )
        self.declare_parameter("geofence_plan", default_plan_fname)
        self.plan_fname: str = self.get_parameter("geofence_plan").value
        self.get_logger().info(f"GeoFence Plan File: {self.plan_fname}")

        default_config_fname = os.path.join(
            get_package_share_directory("ros_sgp_tools"),
            "launch",
            "data",
            "config.yaml",
        )
        self.declare_parameter("config_file", default_config_fname)
        self.config_fname: str = self.get_parameter("config_file").value
        self.get_logger().info(f"Config File: {self.config_fname}")

        if not os.path.exists(self.plan_fname):
            raise FileNotFoundError(f"Geofence plan file not found: {self.plan_fname}")
        if not os.path.exists(self.config_fname):
            raise FileNotFoundError(f"Config file not found: {self.config_fname}")

        with open(self.config_fname, "r") as file:
            self.config = yaml.safe_load(file)

        robot_cfg = self.config.get("robot")
        if robot_cfg is None:
            raise RuntimeError("Missing 'robot' section in config.yaml")

        cfg_mission = robot_cfg.get("mission_type", "Waypoint")
        if self.MISSION_TYPE is not None and cfg_mission != self.MISSION_TYPE:
            self.get_logger().warning(
                f"Config mission_type='{cfg_mission}' but this node is fixed to "
                f"MISSION_TYPE='{self.MISSION_TYPE}'. Using '{self.MISSION_TYPE}'."
            )
        self.mission_type = self.MISSION_TYPE or cfg_mission
        self.get_logger().info(f"Mission type: {self.mission_type}")

        if self.mission_type in ("IPP", "AdaptiveIPP", "Coverage"):
            if "hyperparameters" not in self.config:
                raise RuntimeError(
                    "Missing 'hyperparameters' section in config.yaml for "
                    "IPP/AdaptiveIPP/Coverage mission."
                )

        self.declare_parameter("data_folder", "")
        self.base_data_folder = self.get_parameter("data_folder").value
        self.get_logger().info(f"Base Data Folder: {self.base_data_folder}")

    def _init_random_seeds(self) -> None:
        robot_cfg = self.config.get("robot", {})
        self.seed: Optional[int] = robot_cfg.get("seed")

        if self.seed is not None:
            tf.random.set_seed(self.seed)
            np.random.seed(self.seed)
            self.get_logger().info(f"Using seed: {self.seed}")
        else:
            self.get_logger().info("No seed specified; using non-deterministic behavior")

    def _init_gp_settings(self) -> None:
        hyper_cfg = self.config["hyperparameters"]
        self.kernel_name: str = hyper_cfg["kernel_function"]

        if self.kernel_name in ["Attentive", "NeuralSpectral"]:
            gpflow.config.set_default_float(np.float32)
            gpflow.config.set_default_jitter(1e-1)
            self.get_logger().info("Using float32 and high jitter for deep kernel")
        else:
            gpflow.config.set_default_float(np.float64)
            gpflow.config.set_default_jitter(1e-6)
            self.get_logger().info("Using float64 and standard jitter")

    def _init_objective_and_scaler(self) -> None:
        self.fence_vertices, self.start_location, waypoints = get_mission_plan(
            self.plan_fname
        )
        self.waypoints = waypoints

        ipp_cfg = self.config.get("ipp_model", {})
        num_samples = ipp_cfg.get("num_samples", 1000)

        self.X_objective = polygon2candidates(
            self.fence_vertices, num_samples=num_samples, seed=self.seed
        )
        self.X_objective = np.array(self.X_objective).reshape(-1, 2)

        self.X_scaler = LatLonStandardScaler()

        # Set origin when using DVL to use local/relative coords
        robot_cfg = self.config.get("robot", {})
        sensor_name = robot_cfg.get("navigation", "GPS")
        if sensor_name == "DVL":
            origin = self.start_location
        else:
            origin = None
        self.X_scaler.fit(self.X_objective, origin=origin)

        self.X_objective = self.X_scaler.transform(self.X_objective).astype(default_float())
        self.fence_vertices_local = self.X_scaler.transform(self.fence_vertices)

        self.start_location = self.X_scaler.transform(np.array([self.start_location[:2]])).astype(
            default_float()
        )

        if self.mission_type == "Waypoint" and self.waypoints is not None:
            self.waypoints = self.X_scaler.transform(self.waypoints).astype(default_float())

    def _init_mission_models_and_waypoints(self) -> None:
        raise NotImplementedError("Subclasses must implement _init_mission_models_and_waypoints()")

    def _init_data_store(self) -> None:
        time_stamp = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        self.data_folder = os.path.join(self.base_data_folder, f"{self.MISSION_TYPE}-mission-{time_stamp}")
        os.makedirs(self.data_folder, exist_ok=True)

        shutil.copy(self.plan_fname, self.data_folder)
        shutil.copy(self.config_fname, self.data_folder)

        data_fname = os.path.join(self.data_folder, "mission-log.hdf5")
        self.data_file = h5py.File(data_fname, "a")

        self.dset_X = self.data_file.create_dataset(
            "X", (0, 2), maxshape=(None, 2), dtype=np.float64, chunks=True
        )
        self.dset_y = self.data_file.create_dataset(
            "y", (0, 1), maxshape=(None, 1), dtype=np.float64, chunks=True
        )

        self.data_file.create_dataset(
            "fence_vertices",
            self.fence_vertices.shape,
            dtype=np.float64,
            data=self.fence_vertices,
        )

        fname = f"initial_path-{strftime('%H-%M-%S', gmtime())}"
        self.data_file.create_dataset(
            fname,
            self.waypoints.shape,
            dtype=np.float64,
            data=self.X_scaler.inverse_transform(self.waypoints),
        )
        self.plot_paths(fname, 
                        self.waypoints, 
                        update_waypoint=0)

    def _init_distances_and_state(self) -> None:
        lat_lon_waypoints = self.X_scaler.inverse_transform(self.waypoints)
        self.distances = haversine(lat_lon_waypoints[1:], lat_lon_waypoints[:-1])

        self.data_X: List[np.ndarray] = []
        self.data_y: List[np.ndarray] = []

        self.current_waypoint: int = -1
        self.data_lock = Lock()
        self.waypoints_lock = Lock()
        self.runtime_est: float = 0.0
        self.heading_velocity: float = 1.0

        robot_cfg = self.config.get("robot", {})
        self.data_buffer_size: int = robot_cfg.get("data_buffer_size", 250)

        self.stats = RunningStats()

    def _init_sensors_and_sync(self) -> None:
        sensors_module = importlib.import_module("sensors")
        self.sensors = []
        sensor_subscribers = []
        sensor_group = ReentrantCallbackGroup()

        robot_cfg = self.config.get("robot", {})

        sensor_name = robot_cfg.get("navigation", "GPS")
        pos_obj = getattr(sensors_module, sensor_name)()
        self.sensors.append(pos_obj)
        sensor_subscribers.append(pos_obj.get_subscriber(self, callback_group=sensor_group))

        sensor_name = robot_cfg.get("sensor", "Altitude")
        if sensor_name != "Altitude":
            sensor_obj = getattr(sensors_module, sensor_name)()
            self.sensors.append(sensor_obj)
            sensor_subscribers.append(sensor_obj.get_subscriber(self, callback_group=sensor_group))

        self.time_sync = ApproximateTimeSynchronizer(
            sensor_subscribers, queue_size=10, slop=0.1, sync_arrival_time=True
        )
        self.time_sync.registerCallback(self.data_callback)

    def _init_timers_and_services(self) -> None:
        timer_group = MutuallyExclusiveCallbackGroup()
        self.timer = self.create_timer(5.0, self.update_with_data, callback_group=timer_group)

        self.create_service(Waypoint, "waypoint", self.waypoint_service_callback)

        self.create_subscription(
            Float32MultiArray,
            "waypoint_eta",
            self.eta_callback,
            qos_profile_sensor_data,
        )

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_requested

    def request_shutdown(self, reason: str = "") -> None:
        if reason:
            self.get_logger().info(f"Shutdown requested: {reason}")
        self._shutdown_requested = True

    def destroy_node(self) -> None:
        try:
            if hasattr(self, "data_file") and self.data_file is not None:
                self.data_file.flush()
                self.data_file.close()
        except Exception as e:
            self.get_logger().error(f"Failed to close data file cleanly: {e}")
        finally:
            super().destroy_node()

    # -------------------------------------------------------------------------
    # Callbacks (shared)
    # -------------------------------------------------------------------------

    def eta_callback(self, msg: Float32MultiArray) -> None:
        self.heading_velocity = msg.data[2]
        if 0 <= self.current_waypoint < len(self.distances):
            with self.waypoints_lock:
                self.distances[self.current_waypoint] = msg.data[1]

    def waypoint_service_callback(self, request: Waypoint.Request, response: Waypoint.Response):
        if not request.ok:
            self.get_logger().error(
                "Path follower failed to reach a waypoint; requesting shutdown."
            )
            self.request_shutdown("Path follower error")
            response.new_waypoint = False
            return response

        self.current_waypoint += 1

        if self.current_waypoint >= len(self.waypoints):
            response.new_waypoint = False
            return response

        self.get_logger().info(f"Current waypoint: {self.current_waypoint}")

        with self.waypoints_lock:
            waypoint = self.waypoints[self.current_waypoint].reshape(1, -1)
            waypoint = self.X_scaler.inverse_transform(waypoint)[0]
            response.new_waypoint = True
            response.waypoint = Point(x=float(waypoint[0]), y=float(waypoint[1]))
        return response

    def data_callback(self, *args) -> None:
        # Only use data while moving (avoids OSGPR failures)
        if self.current_waypoint > 0 and self.current_waypoint < len(self.waypoints):
            position = self.sensors[0].process_msg(args[0])

            if len(args) == 1:
                data_X = [position[:2]]
                data_y = [position[2]]
            else:
                data_X, data_y = self.sensors[1].process_msg(args[1], position=position)

            self.stats.push(data_y, per_dim=True)

            with self.data_lock:
                self.data_X.extend(data_X)
                self.data_y.extend(data_y)

    # -------------------------------------------------------------------------
    # IPP/AdaptiveIPP reusable bits
    # -------------------------------------------------------------------------

    def init_models(
        self,
        init_ipp_model: bool = True,
        init_param_model: bool = True,
        kernel=None,
        noise_variance: float = 1e-3,
    ) -> None:
        if init_ipp_model:
            self.ipp_model_config = self.config["ipp_model"]
            self.num_waypoints = self.ipp_model_config["num_waypoints"]

            X_init = get_inducing_pts(
                self.X_objective, (self.num_waypoints - 1), seed=self.seed
            )
            self.get_logger().info("Running TSP solver to get the initial IPP path...")
            X_init, _ = run_tsp(X_init, start_nodes=self.start_location, **self.config.get("tsp", {}))
            X_init = np.array(X_init)

            transform_kwargs = self.ipp_model_config.get("transform", {})
            self.distance_budget = None
            if transform_kwargs.get("distance_budget") is not None:
                self.distance_budget = transform_kwargs["distance_budget"]
                transform_kwargs["distance_budget"] = self.X_scaler.meters2units(self.distance_budget)

            transform = IPPTransform(Xu_fixed=X_init[:, :1, :], **transform_kwargs)

            ipp_model_cls = get_method(self.ipp_model_config["method"])
            self.ipp_model = ipp_model_cls(
                self.num_waypoints,
                X_objective=self.X_objective,
                kernel=kernel,
                noise_variance=noise_variance,
                transform=transform,
                X_init=X_init[0],
            )

            self.get_logger().info("Running IPP solver to update the initial path...")
            self.ipp_model_kwargs = self.ipp_model_config.get("optimizer", {})
            self.waypoints = self.ipp_model.optimize(**self.ipp_model_kwargs)[0]
            self.waypoints = project_waypoints(self.waypoints, self.X_objective)

            self.get_logger().info(f'Initialized {self.ipp_model_config["method"]} IPP model')

        if init_param_model:
            self.param_model_config = self.config["param_model"]
            self.param_model_kwargs = self.param_model_config.get("optimizer", {})
            self.param_model_method = self.param_model_config["method"]

            if self.param_model_method == "SSGP":
                self.train_param_inducing = self.param_model_config.get("train_inducing", True)
                self.num_param_inducing = self.param_model_config["num_inducing"]
                self.param_model = init_osgpr(
                    self.X_objective,
                    num_inducing=self.num_param_inducing,
                    kernel=kernel,
                    noise_variance=noise_variance,
                )
            else:
                raise NotImplementedError(f"Unsupported param model method: {self.param_model_method}")

            self.get_logger().info(f"Initialized {self.param_model_method} Parameter model")

    def _should_update_models(self) -> bool:
        return False

    def _update_models_and_waypoints(
        self, data_X: np.ndarray, data_y: np.ndarray
    ) -> Tuple[Optional[np.ndarray], int]:
        return None, -1

    def update_param(self, X_new: np.ndarray, y_new: np.ndarray) -> None:

        if not (hasattr(self, "num_param_inducing") and \
                len(X_new) > getattr(self, "num_param_inducing", 0)):
            self.get_logger().info("Skipping parameter update, num data samples less than num inducing points...")
            return False
        
        self.get_logger().info("Updating parameter model...")
        X_new = self.X_scaler.transform(X_new).astype(default_float())

        eps = 1e-6
        y_new = ((y_new - self.stats.mean) / (self.stats.std + eps)).astype(default_float())
        self.get_logger().info(f"Data Mean: {self.stats.mean}")
        self.get_logger().info(f"Data Std: {self.stats.std}")

        if self.current_waypoint >= (self.num_waypoints - 1):
            self.get_logger().info("Current waypoint is last target; skipping parameter update.")
            return False

        inducing_variable = np.copy(self.waypoints[: self.current_waypoint + 1])
        inducing_variable[-1] = X_new[-1]
        inducing_variable = resample_path(inducing_variable, self.num_param_inducing)

        self.param_model.update((X_new, y_new), inducing_variable=inducing_variable)

        trainable_variables = None if self.train_param_inducing else self.param_model.trainable_variables[1:]

        try:
            optimize_model(self.param_model, trainable_variables=trainable_variables, **self.param_model_kwargs)
        except Exception:
            self.get_logger().error(traceback.format_exc())
            self.get_logger().warning("Failed to update parameter model! Resetting parameter model...")

            hyper_cfg = self.config["hyperparameters"]
            kernel_kwargs = hyper_cfg.get("kernel", {})
            kernel = get_kernel(self.kernel_name)(**kernel_kwargs)
            noise_variance = float(hyper_cfg["noise_variance"])
            self.init_models(
                init_ipp_model=False,
                init_param_model=True,
                kernel=kernel,
                noise_variance=noise_variance,
            )

        return True

    def update_waypoints(self) -> Tuple[np.ndarray, int]:
        self.get_logger().info("Updating IPP solution...")

        update_waypoint = self.get_update_waypoint()
        if update_waypoint == -1:
            self.get_logger().info("No waypoint can be safely updated at this time")
            return self.waypoints, update_waypoint

        Xu_visited = self.waypoints[: update_waypoint + 1].reshape(1, -1, 2)
        self.ipp_model.transform.update_Xu_fixed(Xu_visited)

        self.ipp_model.update(self.param_model.kernel, self.param_model.likelihood.variance)
        waypoints = self.ipp_model.optimize(**self.ipp_model_kwargs)[0]

        waypoints = project_waypoints(waypoints, self.X_objective)
        waypoints[: update_waypoint + 1] = self.waypoints[: update_waypoint + 1]
        return waypoints, update_waypoint

    def get_update_waypoint(self) -> int:
        with self.waypoints_lock:
            for i in range(self.current_waypoint, len(self.distances)):
                if self.runtime_est <= 0.0 or self.heading_velocity <= 0.0:
                    return -1
                if self.distances[i] / self.heading_velocity > self.runtime_est:
                    return i + 1
        return -1

    # -------------------------------------------------------------------------
    # Shared periodic update
    # -------------------------------------------------------------------------

    def _should_request_shutdown_on_completion(self) -> bool:
        # Coverage overrides to avoid shutdown after initial phase
        return True

    def update_with_data(self, force_update: bool = False) -> None:
        if not self.data_X and not force_update and self.current_waypoint < len(self.waypoints):
            return

        enough_buffer = len(self.data_X) > self.data_buffer_size
        mission_complete = self.current_waypoint >= len(self.waypoints)

        if not (enough_buffer or mission_complete or force_update):
            return
        
        # Skip processing data in the initial phase of coverage missions
        if getattr(self, "mission_type", "") == "Coverage" and getattr(self, "coverage_phase", "") == "initial":
            return

        with self.data_lock:
            data_X = np.array(self.data_X).reshape(-1, 2)
            data_y = np.array(self.data_y).reshape(-1, 1)
            self.data_X = []
            self.data_y = []

        update_waypoint = -1
        new_waypoints = None

        if self._should_update_models() and self.current_waypoint < len(self.waypoints):
            start_time = self.get_clock().now().nanoseconds
            new_waypoints, update_waypoint = self._update_models_and_waypoints(data_X, data_y)
            end_time = self.get_clock().now().nanoseconds
            self.runtime_est = (end_time - start_time) / 1e9
            self.get_logger().info(f"Model update time: {self.runtime_est:.3f} secs")

        if update_waypoint != -1 and new_waypoints is not None:
            with self.waypoints_lock:
                if self.current_waypoint < update_waypoint:
                    self.waypoints = new_waypoints
                    lat_lon_waypoints = self.X_scaler.inverse_transform(new_waypoints)
                    self.distances = haversine(lat_lon_waypoints[1:], lat_lon_waypoints[:-1])
                    num_changed = len(self.waypoints) - (update_waypoint + 1)
                    self.get_logger().info(
                        f"Updated path from waypoint {update_waypoint + 1} onward "
                        f"({num_changed} waypoints modified)"
                    )

        if data_X.size > 0:
            self.dset_X.resize(self.dset_X.shape[0] + len(data_X), axis=0)
            self.dset_X[-len(data_X):] = data_X
            self.dset_y.resize(self.dset_y.shape[0] + len(data_y), axis=0)
            self.dset_y[-len(data_y):] = data_y

        current_waypoint_idx = self.current_waypoint if self.current_waypoint > -1 else 0
        if getattr(self, "mission_type", "") == "Coverage" and getattr(self, "coverage_phase", "") == "coverage":
            current_waypoint_idx += int(self.num_waypoints)

        fname = f"waypoints_{current_waypoint_idx}-{strftime('%H-%M-%S', gmtime())}"
        X_data_plot = self.X_scaler.transform(data_X) if data_X.size > 0 else None

        self.plot_paths(fname, self.waypoints, X_data=X_data_plot, update_waypoint=update_waypoint)

        if self.current_waypoint >= len(self.waypoints):
            if not force_update and self.data_X:
                self.update_with_data(force_update=True)

            self.get_logger().info("Finished mission, requesting shutdown of online planner")
            self.request_shutdown("Mission complete")

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
        plt.figure()
        ax = plt.gca()
        ax.set_aspect("equal")
        plt.xlabel("X")
        plt.ylabel("Y")

        plt.scatter(self.X_objective[:, 0], self.X_objective[:, 1], marker=".", s=1, label="Candidates")
        plt.plot(waypoints[:, 0], waypoints[:, 1], label="Path", marker="o", c="r")

        if update_waypoint is not None and update_waypoint >= 0 and update_waypoint < len(waypoints):
            plt.scatter(
                waypoints[update_waypoint, 0],
                waypoints[update_waypoint, 1],
                label="Update Waypoint",
                zorder=2,
                c="g",
            )

        if X_data is not None and X_data.size > 0:
            plt.scatter(X_data[:, 0], X_data[:, 1], label="Data", c="b", marker="x", zorder=3, s=1)

        if inducing_pts is not None:
            plt.scatter(
                inducing_pts[:, 0],
                inducing_pts[:, 1],
                label="Inducing Pts",
                marker=".",
                c="g",
                zorder=4,
                s=2,
            )

        plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
        plt.savefig(os.path.join(self.data_folder, f"{fname}.png"), bbox_inches="tight")
        plt.close()
