#! /usr/bin/env python3
import rclpy
from rclpy.executors import MultiThreadedExecutor

from sgptools.kernels import get_kernel

from base_planner import BasePathPlanner


class IPPPathPlanner(BasePathPlanner):
    MISSION_TYPE = "IPP"

    def _init_mission_models_and_waypoints(self) -> None:
        hyper_cfg = self.config["hyperparameters"]
        self.kernel_name = hyper_cfg["kernel_function"]
        kernel_kwargs = hyper_cfg.get("kernel", {})
        kernel = get_kernel(self.kernel_name)(**kernel_kwargs)
        noise_variance = float(hyper_cfg["noise_variance"])

        self.get_logger().info("IPP mission: initializing IPP model (no param model).")
        self.init_models(
            init_ipp_model=True,
            init_param_model=False,
            kernel=kernel,
            noise_variance=noise_variance,
        )

        if self.waypoints is None:
            raise RuntimeError("IPP mission failed to initialize waypoints.")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = IPPPathPlanner()
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
