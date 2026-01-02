#! /usr/bin/env python3
import rclpy
from rclpy.executors import MultiThreadedExecutor

from sgptools.kernels import get_kernel

from base_planner import BasePathPlanner


class AdaptiveIPPPathPlanner(BasePathPlanner):
    MISSION_TYPE = "AdaptiveIPP"

    def _init_mission_models_and_waypoints(self) -> None:
        hyper_cfg = self.config["hyperparameters"]
        self.kernel_name = hyper_cfg["kernel_function"]
        kernel_kwargs = hyper_cfg.get("kernel", {})
        kernel = get_kernel(self.kernel_name)(**kernel_kwargs)
        noise_variance = float(hyper_cfg["noise_variance"])

        self.get_logger().info("AdaptiveIPP mission: initializing IPP + parameter model.")
        self.init_models(
            init_ipp_model=True,
            init_param_model=True,
            kernel=kernel,
            noise_variance=noise_variance,
        )

        if self.waypoints is None:
            raise RuntimeError("AdaptiveIPP mission failed to initialize waypoints.")

    def _should_update_models(self) -> bool:
        return True

    def _update_models_and_waypoints(self, data_X, data_y):
        # Same behavior as your original: update param model, then update IPP waypoints
        start_time = self.get_clock().now().nanoseconds
        finished = self.update_param(data_X, data_y)
        mid_time = self.get_clock().now().nanoseconds

        if finished:
            new_waypoints, update_waypoint = self.update_waypoints()
        end_time = self.get_clock().now().nanoseconds

        param_runtime = (mid_time - start_time) / 1e9
        ipp_runtime = (end_time - mid_time) / 1e9
        self.get_logger().info(f"Param update time: {param_runtime:.3f} secs")
        self.get_logger().info(f"IPP update time: {ipp_runtime:.3f} secs")
        self.runtime_est = param_runtime + ipp_runtime

        return new_waypoints, update_waypoint


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AdaptiveIPPPathPlanner()
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
