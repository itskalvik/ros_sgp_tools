#! /usr/bin/env python3
from time import gmtime, strftime
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Point

import os
import pickle
import gpflow
from gpflow.config import default_float

from sgptools.methods import get_method
from sgptools.kernels import get_kernel
from sgptools.utils.tsp import run_tsp
from sgptools.utils.misc import project_waypoints, get_inducing_pts
from sgptools.utils.gpflow import get_model_params

from base_planner import BasePathPlanner
from utils import *  # LatLonStandardScaler, get_mission_plan, haversine, RunningStats, etc.


class CoveragePathPlanner(BasePathPlanner):
    MISSION_TYPE = "Coverage"

    def _init_mission_models_and_waypoints(self) -> None:
        # Coverage state
        self.coverage_phase: str = "initial"   # "initial" -> "coverage"
        self.coverage_planned: bool = False
        self.coverage_waypoints = None
        self.coverage_model = None
        self.coverage_fovs = None

        coverage_cfg = self.config.get("ipp_model", {})
        self.num_waypoints = coverage_cfg.get("num_waypoints", 20)

        self.get_logger().info(
            f"Coverage mission: generating initial exploratory path with {self.num_waypoints} points."
        )

        X_init = get_inducing_pts(self.X_objective, num_inducing=self.num_waypoints, seed=self.seed)
        self.get_logger().info("Running TSP solver to get initial coverage path...")
        X_init, _ = run_tsp(X_init, start_nodes=self.start_location, **self.config.get("tsp", {}))
        X_init = np.array(X_init)[0]
        X_init = project_waypoints(X_init, self.X_objective)

        self.waypoints = X_init
        if self.waypoints is None:
            raise RuntimeError("Coverage mission failed to initialize initial waypoints.")

    def _should_request_shutdown_on_completion(self) -> bool:
        # Don't shutdown when the initial phase ends; only after coverage phase ends.
        return self.coverage_phase != "initial"

    def plan_coverage_from_data(self) -> None:
        if self.coverage_planned:
            return

        if len(self.data_X) == 0:
            raise RuntimeError(
                "No data collected for coverage kernel fitting."
            )
        else:
            X_train = np.array(self.data_X).reshape(-1, 2)
            y_train = np.array(self.data_y).reshape(-1, 1)
            self.data_X = []
            self.data_y = []

        # Store copy of init data in hdf5
        self.data_file.create_dataset(
            "X_init",
            X_train.shape,
            dtype=np.float64,
            data=X_train,
        )
        self.data_file.create_dataset(
            "y_init",
            y_train.shape,
            dtype=np.float64,
            data=y_train,
        )

        X_train_scaled = self.X_scaler.transform(X_train)
        y_train_scaled = (y_train - np.mean(y_train, axis=0)) / (np.std(y_train, axis=0) + 1e-6)

        fname = f"waypoints_{self.current_waypoint}-{strftime('%H-%M-%S', gmtime())}"
        self.plot_paths(fname, self.waypoints, X_data=X_train_scaled, update_waypoint=0)

        hyper_cfg = self.config["hyperparameters"]
        self.kernel_name = hyper_cfg["kernel_function"]
        kernel_kwargs = hyper_cfg.get("kernel", {})
        base_kernel = get_kernel(self.kernel_name)(**kernel_kwargs)

        self.get_logger().info(
            f"Fitting GP kernel '{self.kernel_name}' on {X_train_scaled.shape[0]} samples for coverage..."
        )

        _, noise_variance, kernel, init_model = get_model_params(
            X_train=X_train_scaled.astype(default_float()),
            y_train=y_train_scaled.astype(default_float()),
            optimizer='tf.Adam',
            kernel=base_kernel,
            return_model=True,
            verbose=False,
        )

        _, prior_var = init_model.predict_f(self.X_objective)
        max_prior_var = float(prior_var.numpy().max())
        self.get_logger().info(f"Max prior variance on objective set: {max_prior_var:.4f}")

        coverage_cfg = self.config.get("ipp_model", {})
        method_name = coverage_cfg.get("method", "HexCoverage")
        num_sensing = len(self.X_objective)
        optimizer_kwargs = coverage_cfg.get("optimizer", {})

        var_ratio = coverage_cfg.get("variance_ratio")
        if var_ratio is not None:
            post_var_threshold = max_prior_var * float(var_ratio)
            optimizer_kwargs["post_var_threshold"] = post_var_threshold
            self.get_logger().info(f"Using post_var_threshold: {post_var_threshold:.4f}; computed from variance_ratio: {var_ratio:.4f}")

        coverage_model_cls = get_method(method_name)
        self.coverage_model = coverage_model_cls(
            num_sensing=num_sensing,
            X_objective=self.X_objective,
            kernel=kernel,
            noise_variance=noise_variance,
            pbounds=self.fence_vertices_local,
        )

        self.get_logger().info(f"Running coverage planner optimize() with method={method_name}...")
        X_sol, fovs = self.coverage_model.optimize(
            return_fovs=True,
            X_warm_start=X_train_scaled.astype(default_float()),
            start_nodes=self.waypoints[None, -1],
            **optimizer_kwargs,
        )
        X_sol = np.array(X_sol)[0]
        X_sol = project_waypoints(X_sol, self.X_objective)

        self.coverage_fovs = fovs
        self.coverage_waypoints = X_sol
        self.coverage_planned = True

        fname = f"coverage_path-{strftime('%H-%M-%S', gmtime())}"
        self.data_file.create_dataset(
            fname,
            self.coverage_waypoints.shape,
            dtype=np.float64,
            data=self.X_scaler.inverse_transform(self.coverage_waypoints),
        )
        self.plot_paths(
            fname,
            self.coverage_waypoints,
            update_waypoint=0)
        self.get_logger().info(f"Coverage planner produced {len(self.coverage_waypoints)-1} waypoints.")

        # Save learned GP model
        fname = os.path.join(self.data_folder, f"model_params.pkl")
        params_kernel = gpflow.utilities.parameter_dict(init_model.kernel)
        params_likelihood = gpflow.utilities.parameter_dict(init_model.likelihood)
        params = {'kernel': params_kernel, 
                  'likelihood': params_likelihood}
        with open(fname, 'wb') as handle:
            pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.get_logger().info(f"Saved GP model hyperparameters.")

    def waypoint_service_callback(self, request, response):
        # Phase-aware waypoint service (same behavior as your original)
        if not request.ok:
            self.get_logger().error("Path follower failed; requesting shutdown.")
            self.request_shutdown("Path follower error")
            response.new_waypoint = False
            return response

        self.current_waypoint += 1

        if self.coverage_phase == "initial" and self.current_waypoint >= len(self.waypoints):
            self.get_logger().info(
                "Initial coverage path complete; fitting kernel and planning coverage path..."
            )
            self.plan_coverage_from_data()

            self.coverage_phase = "coverage"
            self.current_waypoint = 0

            with self.waypoints_lock:
                self.waypoints = self.coverage_waypoints
                abs_waypoints = self.X_scaler.inverse_transform(self.waypoints)
                self.distances = haversine(abs_waypoints[1:], abs_waypoints[:-1])
                waypoint = abs_waypoints[self.current_waypoint]

            response.new_waypoint = True
            response.waypoint = Point(x=float(waypoint[0]), y=float(waypoint[1]))
            return response

        if self.coverage_phase == "coverage" and self.current_waypoint >= len(self.waypoints):
            self.get_logger().info("Coverage mission complete.")
            response.new_waypoint = False
            return response

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


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CoveragePathPlanner()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        while rclpy.ok() and not node.shutdown_requested:
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
