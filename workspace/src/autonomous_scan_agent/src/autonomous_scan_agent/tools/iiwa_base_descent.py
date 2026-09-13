#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small Cartesian descent along base -Z (position control), then switch to impedance.

Used after whole-body capture (joint home hold) and before scan/contact motion.
"""

from __future__ import annotations

import copy
import math
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped

try:
    from iiwa_msgs.msg import CartesianPose, ControlMode, CartesianQuantity
    from iiwa_msgs.srv import ConfigureControlMode, ConfigureControlModeRequest

    _HAVE_IIWA = True
except ImportError:
    CartesianPose = None  # type: ignore
    ControlMode = None  # type: ignore
    CartesianQuantity = None  # type: ignore
    ConfigureControlMode = None  # type: ignore
    ConfigureControlModeRequest = None  # type: ignore
    _HAVE_IIWA = False


def _load_cartesian_values(prefix: str, defaults: Tuple[float, ...]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    fields = ("x", "y", "z", "a", "b", "c")
    for i, field in enumerate(fields):
        values[field] = float(rospy.get_param(f"{prefix}_{field}", defaults[i]))
    return values


def _to_cartesian_quantity(values: Dict[str, float]) -> Any:
    cq = CartesianQuantity()
    cq.x = values.get("x", 0.0)
    cq.y = values.get("y", 0.0)
    cq.z = values.get("z", 0.0)
    cq.a = values.get("a", 0.0)
    cq.b = values.get("b", 0.0)
    cq.c = values.get("c", 0.0)
    return cq


class _BaseDescentRunner(object):
    def __init__(
        self,
        *,
        base_frame: str,
        ns: str,
        linear_step_size: float,
        linear_rate: float,
        pos_tol: float,
        ang_tol_rad: float,
        reach_timeout: float,
    ) -> None:
        self.base_frame = str(base_frame or "iiwa_link_0")
        ns_clean = str(ns or "/iiwa").strip("/")
        ns_prefix = f"/{ns_clean}" if ns_clean else ""
        self.pose_topic = rospy.get_param("~cartesian_state_topic", f"{ns_prefix}/state/CartesianPose")
        self.pose_cmd_topic = rospy.get_param("~cartesian_cmd_topic", f"{ns_prefix}/command/CartesianPoseLin")
        self.impedance_service = rospy.get_param(
            "~impedance_service", f"{ns_prefix}/configuration/ConfigureControlMode"
        )

        self.linear_step_size = float(linear_step_size)
        self.linear_rate = float(linear_rate)
        self.pos_tol = float(pos_tol)
        self.ang_tol_rad = float(ang_tol_rad)
        self.reach_timeout = float(reach_timeout)

        self._pose_lock = threading.Lock()
        self._current_pose: Optional[PoseStamped] = None
        self._ctrl_client = None

        if _HAVE_IIWA and CartesianPose is not None:
            rospy.Subscriber(self.pose_topic, CartesianPose, self._cartesian_cb, queue_size=1)
        else:
            rospy.Subscriber(self.pose_topic, PoseStamped, self._pose_cb, queue_size=1)
        self.pose_pub = rospy.Publisher(self.pose_cmd_topic, PoseStamped, queue_size=1)

    def _pose_cb(self, msg: PoseStamped) -> None:
        with self._pose_lock:
            self._current_pose = msg

    def _cartesian_cb(self, msg: Any) -> None:
        with self._pose_lock:
            self._current_pose = msg.poseStamped

    def _wait_for_current_pose(self, timeout: float = 5.0) -> None:
        start = time.time()
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            with self._pose_lock:
                if self._current_pose is not None:
                    return
            if time.time() - start > timeout:
                raise RuntimeError(f"timeout waiting for {self.pose_topic}")
            rate.sleep()

    def _get_current_pose(self) -> PoseStamped:
        with self._pose_lock:
            if self._current_pose is None:
                raise RuntimeError(f"no Cartesian state on {self.pose_topic}")
            return copy.deepcopy(self._current_pose)

    def _ensure_ctrl_client(self) -> None:
        if not _HAVE_IIWA or ConfigureControlMode is None:
            raise RuntimeError("ConfigureControlMode unavailable")
        if self._ctrl_client is None:
            rospy.wait_for_service(self.impedance_service, timeout=5.0)
            self._ctrl_client = rospy.ServiceProxy(self.impedance_service, ConfigureControlMode)

    def _configure_position(self) -> None:
        self._ensure_ctrl_client()
        req = ConfigureControlModeRequest()
        req.control_mode = ControlMode.POSITION_CONTROL
        resp = self._ctrl_client(req)
        if not getattr(resp, "success", False):
            raise RuntimeError(f"position control rejected: {getattr(resp, 'error', 'unknown')}")
        rospy.loginfo("[base_descent] switched to position control")

    def _configure_impedance(self) -> None:
        self._ensure_ctrl_client()
        stiff = _load_cartesian_values("/path_execute_server/impedance_stiff", (1200.0, 1200.0, 400.0, 50.0, 50.0, 150.0))
        damp = _load_cartesian_values("/path_execute_server/impedance_damp", (0.8, 0.8, 0.8, 0.8, 0.8, 0.8))
        req = ConfigureControlModeRequest()
        req.control_mode = ControlMode.CARTESIAN_IMPEDANCE
        req.cartesian_impedance.cartesian_stiffness = _to_cartesian_quantity(stiff)
        req.cartesian_impedance.cartesian_damping = _to_cartesian_quantity(damp)
        req.cartesian_impedance.nullspace_stiffness = float(
            rospy.get_param("/path_execute_server/impedance_nullspace_stiffness", 200.0)
        )
        req.cartesian_impedance.nullspace_damping = float(
            rospy.get_param("/path_execute_server/impedance_nullspace_damping", 1.0)
        )
        max_dev = _load_cartesian_values("/path_execute_server/impedance_max_path_deviation", (-1.0,) * 6)
        max_vel = _load_cartesian_values("/path_execute_server/impedance_max_cartesian_velocity", (-1.0,) * 6)
        max_force = _load_cartesian_values("/path_execute_server/impedance_max_control_force", (-1.0,) * 6)
        req.limits.max_path_deviation = _to_cartesian_quantity(max_dev)
        req.limits.max_cartesian_velocity = _to_cartesian_quantity(max_vel)
        req.limits.max_control_force = _to_cartesian_quantity(max_force)
        req.limits.max_control_force_stop = bool(
            rospy.get_param("/path_execute_server/impedance_max_control_force_stop", False)
        )
        resp = self._ctrl_client(req)
        if not getattr(resp, "success", False):
            raise RuntimeError(f"impedance rejected: {getattr(resp, 'error', 'unknown')}")
        rospy.loginfo("[base_descent] switched to cartesian impedance")

    def _pose_error(self, reference: PoseStamped, actual: PoseStamped) -> Tuple[float, float]:
        ref_pos = np.array(
            [reference.pose.position.x, reference.pose.position.y, reference.pose.position.z],
            dtype=np.float64,
        )
        act_pos = np.array(
            [actual.pose.position.x, actual.pose.position.y, actual.pose.position.z],
            dtype=np.float64,
        )
        err_xyz = float(np.linalg.norm(act_pos - ref_pos))

        ref_q = np.array(
            [
                reference.pose.orientation.x,
                reference.pose.orientation.y,
                reference.pose.orientation.z,
                reference.pose.orientation.w,
            ],
            dtype=np.float64,
        )
        act_q = np.array(
            [
                actual.pose.orientation.x,
                actual.pose.orientation.y,
                actual.pose.orientation.z,
                actual.pose.orientation.w,
            ],
            dtype=np.float64,
        )
        ref_q /= np.linalg.norm(ref_q) or 1.0
        act_q /= np.linalg.norm(act_q) or 1.0
        dot = float(np.clip(np.dot(ref_q, act_q), -1.0, 1.0))
        ang_err = 2.0 * math.acos(abs(dot))
        if ang_err > math.pi:
            ang_err = 2.0 * math.pi - ang_err
        return err_xyz, float(ang_err)

    def _wait_until_reached(self, target_pose: PoseStamped) -> None:
        start = time.time()
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            cur = self._get_current_pose()
            err_xyz, ang_err = self._pose_error(target_pose, cur)
            if err_xyz <= self.pos_tol and ang_err <= self.ang_tol_rad:
                return
            if time.time() - start > self.reach_timeout:
                raise RuntimeError(
                    f"descent reach timeout: err_xyz={err_xyz:.4f} m, ang={math.degrees(ang_err):.2f} deg"
                )
            rate.sleep()

    def _publish_pose(self, pose: PoseStamped) -> None:
        msg = copy.deepcopy(pose)
        msg.header.stamp = rospy.Time.now()
        self.pose_pub.publish(msg)

    def _execute_linear_motion(self, target_pose: PoseStamped) -> PoseStamped:
        current_pose = self._get_current_pose()
        current_vec = np.array(
            [current_pose.pose.position.x, current_pose.pose.position.y, current_pose.pose.position.z],
            dtype=np.float64,
        )
        target_vec = np.array(
            [target_pose.pose.position.x, target_pose.pose.position.y, target_pose.pose.position.z],
            dtype=np.float64,
        )
        distance = float(np.linalg.norm(target_vec - current_vec))
        if distance < 1e-6:
            return target_pose

        num_steps = max(1, int(math.ceil(distance / self.linear_step_size)))
        direction = (target_vec - current_vec) / distance
        rate = rospy.Rate(self.linear_rate)
        for i in range(num_steps + 1):
            alpha = float(i) / float(num_steps)
            interp_pos = current_vec + direction * distance * alpha
            step_pose = PoseStamped()
            step_pose.header.frame_id = target_pose.header.frame_id
            step_pose.header.stamp = rospy.Time.now()
            step_pose.pose.position.x = float(interp_pos[0])
            step_pose.pose.position.y = float(interp_pos[1])
            step_pose.pose.position.z = float(interp_pos[2])
            step_pose.pose.orientation = target_pose.pose.orientation
            self._publish_pose(step_pose)
            if i < num_steps:
                rate.sleep()
        self._wait_until_reached(target_pose)
        return target_pose

    def run(self, descent_m: float, settle_sec: float) -> Dict[str, Any]:
        descent_m = float(descent_m)
        if descent_m <= 0.0:
            self._configure_impedance()
            return {"status": "success", "message": "skipped_nonpositive_descent", "descent_m": descent_m}

        self._wait_for_current_pose()
        current = self._get_current_pose()
        frame_id = str(current.header.frame_id or self.base_frame)
        if frame_id != self.base_frame:
            rospy.logwarn(
                "[base_descent] current frame_id=%s != expected %s; using current frame",
                frame_id,
                self.base_frame,
            )

        start_z = float(current.pose.position.z)
        target = copy.deepcopy(current)
        target.header.frame_id = frame_id
        target.pose.position.z = start_z - descent_m

        rospy.loginfo(
            "[base_descent] position move along %s -Z: z %.4f -> %.4f (delta=%.4f m)",
            frame_id,
            start_z,
            float(target.pose.position.z),
            descent_m,
        )

        self._configure_position()
        rospy.sleep(0.1)
        self._execute_linear_motion(target)
        if settle_sec > 0.0:
            rospy.sleep(float(settle_sec))
        self._configure_impedance()

        return {
            "status": "success",
            "message": "home_descent_complete",
            "base_frame": frame_id,
            "descent_m": descent_m,
            "start_z": start_z,
            "target_z": float(target.pose.position.z),
        }


def descent_base_minus_z(
    descent_m: float = 0.02,
    *,
    base_frame: str = "iiwa_link_0",
    ns: str = "/iiwa",
    linear_step_size: float = 0.01,
    linear_rate: float = 10.0,
    pos_tol: float = 0.005,
    ang_tol_deg: float = 3.0,
    reach_timeout: float = 30.0,
    settle_sec: float = 0.3,
) -> Dict[str, Any]:
    """
    Move TCP downward by `descent_m` along base -Z in position control, then enable impedance.
    """
    if not _HAVE_IIWA:
        return {"status": "failed", "message": "iiwa_msgs_not_available"}
    try:
        runner = _BaseDescentRunner(
            base_frame=base_frame,
            ns=ns,
            linear_step_size=linear_step_size,
            linear_rate=linear_rate,
            pos_tol=pos_tol,
            ang_tol_rad=math.radians(float(ang_tol_deg)),
            reach_timeout=reach_timeout,
        )
        return runner.run(descent_m=descent_m, settle_sec=settle_sec)
    except Exception as exc:
        rospy.logerr("[base_descent] failed: %s", str(exc))
        return {"status": "failed", "message": str(exc)}
