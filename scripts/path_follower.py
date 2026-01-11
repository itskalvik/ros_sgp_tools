#! /usr/bin/env python3
import os
import sys
import argparse
import importlib
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from ros_sgp_tools.srv import Waypoint

import shapely
from shapely.geometry import Polygon, Point as ShapelyPoint
from extremitypathfinder.extremitypathfinder import PolygonEnvironment

from utils import get_mission_plan, LatLonStandardScaler
from sgptools.utils.misc import polygon2candidates
import numpy as np


class WaypointServiceClient(Node):
    """Get a new waypoint from the Waypoint Service"""
    def __init__(self):
        super().__init__('waypoint_service_client')
        self.client = self.create_client(Waypoint, 'waypoint')
        while not self.client.wait_for_service(timeout_sec=1.0):
            rclpy.spin_once(self, timeout_sec=1.0)
        self.request = Waypoint.Request()

    def get_waypoint(self, ok=True):
        self.request.ok = ok
        future = self.client.call_async(self.request)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result.new_waypoint:
            return [result.waypoint.x, result.waypoint.y]
        else: 
            self.get_logger().info('Mission complete')
            return


def _load_controller_class(controller_name: str):
    """
    Dynamically import the requested controller base class.
    """
    if controller_name == "aqua2":
        mod = importlib.import_module("aqua2_control.controller")
        return getattr(mod, "Controller")
    elif controller_name == "mavros":
        mod = importlib.import_module("mavros_control.controller")
        return getattr(mod, "Controller")
    else:
        raise ValueError(f"Unknown controller '{controller_name}'")


def _parse_args(argv):
    """
    Parse only our custom args; leave the rest for rclpy/ROS args.
    """
    parser = argparse.ArgumentParser(
        description="Waypoint path follower (select controller backend)."
    )
    parser.add_argument(
        "--controller",
        choices=["mavros", "aqua2"],
        default="mavros",
        help="Controller backend to use.",
    )
    # Keep unknown args (e.g. --ros-args ...) to pass to rclpy.init
    return parser.parse_known_args(argv)


def build_path_follower_node(controller_name: str):
    """
    Create a PathFollower class that inherits from the selected Controller base.
    """
    ControllerBase = _load_controller_class(controller_name)

    class PathFollower(ControllerBase):
        def __init__(self):
            # Match each original controller's constructor arguments
            if controller_name == "aqua2":
                super().__init__()
            else:  # mavros
                super().__init__(
                    navigation_type=0,
                    start_mission=False,
                )

            self._shutdown_requested: bool = False
            self.waypoint_service = WaypointServiceClient()

            self._get_fence()
            self._init_planner()

            self.mission()

        def _get_fence(self) -> None:
            default_plan_fname = os.path.join(
                get_package_share_directory("ros_sgp_tools"),
                "launch",
                "data",
                "mission.plan",
            )
            self.declare_parameter("geofence_plan", default_plan_fname)
            plan_fname: str = self.get_parameter("geofence_plan").value
            self.get_logger().info(f"GeoFence Plan File: {plan_fname}")
            if not os.path.exists(plan_fname):
                raise FileNotFoundError(f"Geofence plan file not found: {plan_fname}")
            fence_vertices, start_location, _ = get_mission_plan(plan_fname)

            X_objective = polygon2candidates(fence_vertices, num_samples=2500)
            X_objective = np.array(X_objective).reshape(-1, 2)

            if controller_name == "aqua2":
                origin = start_location
                self.force_origin = True
            else:
                origin = None
                self.force_origin = False

            self.X_scaler = LatLonStandardScaler()
            self.X_scaler.fit(X_objective, origin=origin)

            fence_vertices_scaled = self.X_scaler.transform(
                fence_vertices, force_origin=self.force_origin
            )
            self.fence = Polygon(fence_vertices_scaled)

        def _init_planner(self) -> None:
            """
            ExtremityPathfinder environment built once from the (scaled) fence polygon.
            """
            try:
                hull = shapely.convex_hull(self.fence)
                boundary = [(float(x), float(y)) for (x, y) in list(hull.exterior.coords)[:-1]]
                holes_xy = []
                for ring in hull.difference(self.fence).geoms:
                    holes_xy.append([(float(x), float(y)) for (x, y) in list(ring.exterior.coords)[:-1]])
                env = PolygonEnvironment()
                env.store(boundary, holes_xy)
                env.prepare()
                self.planner = env
                self.get_logger().info("ExtremityPathfinder environment prepared.")
            except Exception as e:
                self.get_logger().error(f"Failed to initialize ExtremityPathfinder: {e}. Falling back to direct go2waypoint.")
                self.planner = None

        def _plan_waypoints(self, goal_xy: List[float]) -> List[List[float]]:
            """
            Returns a list of [x,y] waypoints INCLUDING the goal, excluding the start.
            Uses ExtremityPathfinder in SCALED space, then converts back to ORIGINAL space.
            """
            if self.planner is None:
                return [goal_xy]

            start_raw = (self.vehicle_position[0], self.vehicle_position[1])
            goal_raw = (goal_xy[0], goal_xy[1])

            # --- REQUIRED: use X_scaler for transforms ---
            start_scaled = self.X_scaler.transform(
                np.array(start_raw).reshape(1, -1), force_origin=self.force_origin
            )[0]
            goal_scaled = self.X_scaler.transform(
                np.array(goal_raw).reshape(1, -1), force_origin=self.force_origin
            )[0]

            # Ensure points are inside the (scaled) fence
            if not self.fence.covers(ShapelyPoint(float(start_scaled[0]), float(start_scaled[1]))):
                self.get_logger().warn("Start is outside scaled fence; using direct go2waypoint.")
                return [goal_xy]
            if not self.fence.covers(ShapelyPoint(float(goal_scaled[0]), float(goal_scaled[1]))):
                self.get_logger().warn("Goal is outside scaled fence; using direct go2waypoint.")
                return [goal_xy]

            try:
                # find_shortest_path returns (path, distance)
                path_scaled, _ = self.planner.find_shortest_path(
                    (float(start_scaled[0]), float(start_scaled[1])),
                    (float(goal_scaled[0]), float(goal_scaled[1])),
                )
                if not path_scaled or len(path_scaled) < 2:
                    return [goal_xy]

                # convert planned points back to ORIGINAL coords using X_scaler
                pts_raw = self.X_scaler.inverse_transform(np.array(path_scaled)).tolist()

                # exclude start, keep goal
                return pts_raw[1:] if len(pts_raw) > 1 else [goal_xy]

            except Exception as e:
                self.get_logger().warn(f"Planner failed for segment -> using direct goal. Error: {e}")
                return [goal_xy]

        def mission(self):
            """
            IPP mission:
              - MAVROS: GUIDED -> set home -> arm -> waypoints -> disarm
              - Aqua2: waypoints -> disarm
            """
            if controller_name == "mavros":
                self.get_logger().info("Engaging GUIDED mode")
                if self.set_mode("GUIDED"):
                    self.get_logger().info("GUIDED mode engaged")

                self.get_logger().info("Setting current position as home")
                if self.set_home(self.vehicle_position[0], self.vehicle_position[1]):
                    self.get_logger().info("Home position set")

                self.get_logger().info("Arming")
                if self.arm(True):
                    self.get_logger().info("Armed")

            while rclpy.ok():
                waypoint = self.waypoint_service.get_waypoint()
                if waypoint is None:
                    break

                self.get_logger().info(f"Received waypoint: {waypoint[0]} {waypoint[1]}")

                planned_wps = self._plan_waypoints([waypoint[0], waypoint[1]])
                self.get_logger().info(f"Planned {len(planned_wps)} segment waypoints (incl. goal).")

                for i, wp in enumerate(planned_wps, start=1):
                    self.get_logger().info(f"Visiting planned wp {i}/{len(planned_wps)}: {wp[0]} {wp[1]}")
                    if self.go2waypoint([wp[0], wp[1]]):
                        self.get_logger().info("Reached planned waypoint")
                    else:
                        self.get_logger().warn("Failed to reach planned waypoint; continuing to next waypoint.")
                        break

            if controller_name == "mavros":
                self.get_logger().info("Disarming")
                if self.arm(False):
                    self.get_logger().info("Disarmed")

            self.request_shutdown("Mission complete")

        @property
        def shutdown_requested(self) -> bool:
            return self._shutdown_requested

        def request_shutdown(self, reason: str = "") -> None:
            if reason:
                self.get_logger().info(f"Shutdown requested: {reason}")
            self._shutdown_requested = True

    return PathFollower


def main(args=None):
    argv = sys.argv[1:] if args is None else args
    parsed, ros_args = _parse_args(argv)

    # Initialize ROS with the remaining args (including --ros-args ...)
    rclpy.init(args=ros_args)

    try:
        PathFollower = build_path_follower_node(parsed.controller)
        node = PathFollower()

        while rclpy.ok() and not node.shutdown_requested:
            rclpy.spin_once(node, timeout_sec=0.1)

    except KeyboardInterrupt:
        # If node exists, log; otherwise just exit
        try:
            node.get_logger().info("Keyboard interrupt, shutting down")
        except Exception:
            pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()
