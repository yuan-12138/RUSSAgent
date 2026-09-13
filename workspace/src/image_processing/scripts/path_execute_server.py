#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
path_execute_server.py (ROS1)

常驻路径执行器：
- 订阅 ~execute_yaml(std_msgs/String)：收到 YAML 路径后开始执行（若正在运行可选择取消并切换）
- 服务 ~wait(std_srvs/Trigger)：阻塞等待当前执行结束，返回 success/message
- 服务 ~cancel(std_srvs/Trigger)：请求取消当前执行
- 发布 ~state(std_msgs/String)：idle/running/completed/failed/cancelled

目的：
1) 避免每次 rosrun path_execute.py 重新拉起进程造成卡顿
2) 为“动态执行/在线纠偏”提供一个稳定的执行后端

注意：
- 仍然通过 /iiwa/command/CartesianPoseLin 发布目标
- 控制模式切换逻辑沿用 path_execute.py（可通过参数关闭）
"""

import os
import yaml
import json
import math
import time
import copy
import signal
import subprocess
import threading
import traceback
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse

try:
    from iiwa_msgs.msg import CartesianPose, ControlMode, CartesianQuantity
    from iiwa_msgs.srv import ConfigureControlMode, ConfigureControlModeRequest
    _HAVE_CARTESIAN = True
    _HAVE_CTRL = True
except ImportError:
    try:
        from iiwa_msgs.msg import CartesianPose
        _HAVE_CARTESIAN = True
    except ImportError:
        _HAVE_CARTESIAN = False
        CartesianPose = None
    ControlMode = None
    CartesianQuantity = None
    ConfigureControlMode = None
    ConfigureControlModeRequest = None
    _HAVE_CTRL = False


class Cancelled(Exception):
    pass


def _parse_execute_request(raw: str) -> Tuple[str, Dict[str, Any]]:
    """
    Accept plain YAML path or JSON: {"yaml": "...", "overrides": {...}}.
    Overrides travel with the request so Tool6 point0 settings apply atomically.
    """
    s = (raw or "").strip()
    if not s:
        return "", {}
    if s.startswith("{"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                path_yaml = str(obj.get("yaml") or obj.get("path_yaml") or "").strip()
                overrides = obj.get("overrides") or obj.get("executor_overrides") or {}
                if isinstance(overrides, dict):
                    return path_yaml, overrides
        except Exception:
            pass
    return s, {}


class PathExecuteCore(object):
    """
    将 path_execute.py 的“一次性执行”逻辑改造成可复用核心：
    - execute(path_yaml) 可被多次调用
    - 支持 cancel_event 取消
    """

    def __init__(self, cancel_event: threading.Event):
        self._cancel_event = cancel_event
        # Track the last control mode we requested (best-effort).
        # This is NOT a ground-truth query of the robot controller, but helps avoid redundant switches.
        self._last_mode: Optional[str] = None  # "position" | "impedance" | None

        # 参数（与 path_execute.py 同名，便于复用）
        self.linear_step_size = float(rospy.get_param("~linear_step_size", 0.01))
        self.linear_rate = float(rospy.get_param("~linear_rate", 10.0))
        # 默认：xy 放宽到 2cm（更贴近你现场需求）
        self.error_position_tolerance = float(rospy.get_param("~error_position_tolerance", 0.02))
        # 默认：忽略 Z 方向的到位误差（Z 更容易受接触/软组织变形影响）
        self.ignore_z_error = bool(rospy.get_param("~ignore_z_error", True))
        self.error_position_tolerance_xy = float(rospy.get_param("~error_position_tolerance_xy", self.error_position_tolerance))
        self.error_position_tolerance_z = float(
            rospy.get_param("~error_position_tolerance_z", (1e9 if self.ignore_z_error else self.error_position_tolerance))
        )
        self.error_angle_tolerance = float(rospy.get_param("~error_angle_tolerance", math.radians(3.0)))
        self.error_timeout = float(rospy.get_param("~error_timeout", 60.0))
        # 默认：lookahead 同步放宽到 2cm，避免因微小误差卡住推进
        self.lookahead_distance = float(rospy.get_param("~lookahead_distance", 0.02))
        self.lookahead_distance_xy = float(rospy.get_param("~lookahead_distance_xy", self.lookahead_distance))
        self.lookahead_distance_z = float(
            rospy.get_param("~lookahead_distance_z", (1e9 if self.ignore_z_error else self.lookahead_distance))
        )
        self.lookahead_publish_delay = float(rospy.get_param("~lookahead_publish_delay", 0.0))
        # 常驻执行器默认不做“预接近停顿”，避免反复抬起/等待造成卡顿
        self.pre_stage_pause = float(rospy.get_param("~pre_stage_pause", 0.0))
        # 常驻执行器默认跳过预接近阶段（只在需要时显式开启）
        self.enable_pre_stage = bool(rospy.get_param("~enable_pre_stage", False))
        # Pre-stage 默认在阻抗下完成（不切 position），从 pre_pose 开始就保持柔顺接触能力
        self.use_position_for_pre_stage = bool(rospy.get_param("~use_position_for_pre_stage", False))

        # Full-path behavior:
        # - For multi-frame paths (full_path), pre-stage (lift/approach) often causes unnecessary mode switches
        #   and "lift-then-press" behavior after point0 is already verified.
        # - Default: skip pre-stage for multi-frame paths to keep impedance continuous.
        self.skip_pre_stage_for_multi_frame = bool(rospy.get_param("~skip_pre_stage_for_multi_frame", True))

        # Optional re-scan point0 sequence (post-scan adjust):
        # 1) move to elevated approach pose (+Z lift, e.g. +8cm in start.final_pose)
        # 2) descend to contact point0 (frames[0] minus the same lift)
        # 3) when near contact (XY + Z tolerances), keep commanding point0 for hold_sec
        # 4) then begin path tracking from frames[1]
        self.point0_approach_enabled = bool(rospy.get_param("~point0_approach_enabled", False))
        self.point0_approach_z_lift_m = float(rospy.get_param("~point0_approach_z_lift_m", 0.08))
        self.point0_approach_xy_tol = float(rospy.get_param("~point0_approach_xy_tol", 0.02))
        self.point0_approach_z_tol = float(rospy.get_param("~point0_approach_z_tol", 0.02))
        self.point0_contact_xy_tol = float(rospy.get_param("~point0_contact_xy_tol", 0.02))
        self.point0_contact_z_tol = float(rospy.get_param("~point0_contact_z_tol", 0.03))
        self.point0_contact_hold_sec = float(rospy.get_param("~point0_contact_hold_sec", 4.0))

        # 控制模式切换（可关闭，避免频繁 position<->impedance）
        # 推荐策略：不切回 position（避免“抬起/再接近”），但确保在执行前进入阻抗模式
        self.auto_configure_position = bool(rospy.get_param("~auto_configure_position", False))
        self.auto_configure_impedance = bool(rospy.get_param("~auto_configure_impedance", True))
        self.impedance_service = rospy.get_param("~impedance_service", "/iiwa/configuration/ConfigureControlMode")

        self.impedance_stiff_vals = self._load_cartesian_values("~impedance_stiff", (1200.0, 1200.0, 400.0, 50.0, 50.0, 150.0))
        self.impedance_damp_vals = self._load_cartesian_values("~impedance_damp", (0.8, 0.8, 0.8, 0.8, 0.8, 0.8))
        self.impedance_nullspace_stiffness = float(rospy.get_param("~impedance_nullspace_stiffness", 200.0))
        self.impedance_nullspace_damping = float(rospy.get_param("~impedance_nullspace_damping", 1.0))
        self.impedance_max_path_dev_vals = self._load_cartesian_values("~impedance_max_path_deviation", (-1.0,) * 6)
        self.impedance_max_cart_vel_vals = self._load_cartesian_values("~impedance_max_cartesian_velocity", (-1.0,) * 6)
        self.impedance_max_control_force_vals = self._load_cartesian_values("~impedance_max_control_force", (-1.0,) * 6)
        self.impedance_max_control_force_stop = bool(rospy.get_param("~impedance_max_control_force_stop", False))

        self.pose_pub = rospy.Publisher("/iiwa/command/CartesianPoseLin", PoseStamped, queue_size=1)
        self._current_pose_lock = threading.Lock()
        self._current_pose: Optional[PoseStamped] = None
        self._impedance_client = None

        # --- optional auto recording (one-click sync bag) ---
        # We keep this in the executor server so recording can be started
        # immediately AFTER impedance becomes active (and after the English prompt is printed).
        self._record_proc: Optional[subprocess.Popen] = None
        self._record_started_for_yaml: str = ""

        if _HAVE_CARTESIAN:
            rospy.Subscriber("/iiwa/state/CartesianPose", CartesianPose, self._cartesian_cb, queue_size=1)
        else:
            rospy.Subscriber("/iiwa/state/CartesianPose", PoseStamped, self._pose_cb, queue_size=1)

    def _reload_runtime_params(self) -> None:
        """Re-read private params so Tool4 executor_overrides take effect on a live server."""
        self.enable_pre_stage = bool(rospy.get_param("~enable_pre_stage", self.enable_pre_stage))
        self.use_position_for_pre_stage = bool(
            rospy.get_param("~use_position_for_pre_stage", self.use_position_for_pre_stage)
        )
        self.skip_pre_stage_for_multi_frame = bool(
            rospy.get_param("~skip_pre_stage_for_multi_frame", self.skip_pre_stage_for_multi_frame)
        )
        self.auto_configure_position = bool(
            rospy.get_param("~auto_configure_position", self.auto_configure_position)
        )
        self.auto_configure_impedance = bool(
            rospy.get_param("~auto_configure_impedance", self.auto_configure_impedance)
        )
        self.pre_stage_pause = float(rospy.get_param("~pre_stage_pause", self.pre_stage_pause))
        self.point0_approach_enabled = bool(
            rospy.get_param("~point0_approach_enabled", self.point0_approach_enabled)
        )
        self.point0_approach_z_lift_m = float(
            rospy.get_param("~point0_approach_z_lift_m", self.point0_approach_z_lift_m)
        )
        self.point0_approach_xy_tol = float(
            rospy.get_param("~point0_approach_xy_tol", self.point0_approach_xy_tol)
        )
        self.point0_approach_z_tol = float(
            rospy.get_param("~point0_approach_z_tol", self.point0_approach_z_tol)
        )
        self.point0_contact_xy_tol = float(
            rospy.get_param("~point0_contact_xy_tol", self.point0_contact_xy_tol)
        )
        self.point0_contact_z_tol = float(
            rospy.get_param("~point0_contact_z_tol", self.point0_contact_z_tol)
        )
        self.point0_contact_hold_sec = float(
            rospy.get_param("~point0_contact_hold_sec", self.point0_contact_hold_sec)
        )
        self.ignore_z_error = bool(rospy.get_param("~ignore_z_error", self.ignore_z_error))
        self.error_position_tolerance_xy = float(
            rospy.get_param("~error_position_tolerance_xy", self.error_position_tolerance_xy)
        )
        self.error_position_tolerance_z = float(
            rospy.get_param(
                "~error_position_tolerance_z",
                (1e9 if self.ignore_z_error else self.error_position_tolerance),
            )
        )
        self.error_angle_tolerance = float(
            rospy.get_param("~error_angle_tolerance", self.error_angle_tolerance)
        )
        self.error_timeout = float(rospy.get_param("~error_timeout", self.error_timeout))

    def _apply_runtime_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply per-request overrides bundled in execute_yaml (takes precedence over ROS params)."""
        if not overrides:
            return
        bool_keys = (
            "enable_pre_stage",
            "use_position_for_pre_stage",
            "skip_pre_stage_for_multi_frame",
            "auto_configure_position",
            "auto_configure_impedance",
            "point0_approach_enabled",
            "ignore_z_error",
        )
        float_keys = (
            "pre_stage_pause",
            "point0_approach_z_lift_m",
            "point0_approach_xy_tol",
            "point0_approach_z_tol",
            "point0_contact_xy_tol",
            "point0_contact_z_tol",
            "point0_contact_hold_sec",
            "error_position_tolerance_xy",
            "error_position_tolerance_z",
            "error_angle_tolerance",
            "error_timeout",
        )
        for key in bool_keys:
            if key in overrides:
                setattr(self, key, bool(overrides[key]))
        for key in float_keys:
            if key in overrides:
                setattr(self, key, float(overrides[key]))
        if overrides.get("revert_to_position_on_finish") is not None:
            self._runtime_revert_to_position_on_finish = bool(overrides["revert_to_position_on_finish"])

    def _log_execute_profile(self, *, is_multi_frame: bool) -> None:
        rospy.loginfo(
            "[path_execute_server] execute profile: point0_approach=%s multi_frame=%s "
            "z_lift=%.3fm hold=%.2fs ignore_z=%s z_tol=%.4f pre_stage=%s",
            self.point0_approach_enabled,
            is_multi_frame,
            self.point0_approach_z_lift_m,
            self.point0_contact_hold_sec,
            self.ignore_z_error,
            self.error_position_tolerance_z,
            self.enable_pre_stage,
        )
        use_point0 = bool(self.point0_approach_enabled and is_multi_frame)
        if use_point0:
            rospy.loginfo("[path_execute_server] will run point0 approach sequence before path tracking")
        elif self.point0_approach_enabled and not is_multi_frame:
            rospy.logwarn("[path_execute_server] point0_approach enabled but path is single-frame; sequence skipped")

    def _record_is_running(self) -> bool:
        return bool(self._record_proc is not None and self._record_proc.poll() is None)

    def _record_start(self, path_yaml: str) -> None:
        """
        Start roslaunch autonomous_scan_agent record_sync_bag.launch.
        Best-effort: should not crash robot execution if recording fails.
        """
        try:
            if self._record_is_running():
                return
            pkg = str(rospy.get_param("~record_launch_pkg", "autonomous_scan_agent")).strip() or "autonomous_scan_agent"
            launch_file = str(rospy.get_param("~record_launch_file", "record_sync_bag.launch")).strip() or "record_sync_bag.launch"
            out_dir = str(rospy.get_param("~record_out_dir", "")).strip()
            bag_prefix = str(rospy.get_param("~record_bag_prefix", "")).strip()

            cmd = ["roslaunch", pkg, launch_file]
            if out_dir:
                cmd.append(f"out_dir:={out_dir}")
            if bag_prefix:
                cmd.append(f"bag_prefix:={bag_prefix}")

            # Quiet mode: suppress roslaunch spam; keep only our own concise English logs.
            record_quiet = bool(rospy.get_param("~record_quiet", True))
            stdout = subprocess.DEVNULL if record_quiet else None
            stderr = subprocess.DEVNULL if record_quiet else None

            rospy.loginfo("Recording started.")
            # Create a new process group so we can stop the whole roslaunch tree via SIGINT.
            self._record_proc = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=stdout, stderr=stderr)
            self._record_started_for_yaml = str(path_yaml or "")
        except Exception as exc:
            rospy.logwarn("[path_execute_server] auto-record: start failed (ignored): %s", str(exc))
            self._record_proc = None
            self._record_started_for_yaml = ""

    def _record_stop(self) -> None:
        """Stop roslaunch recording process (best-effort)."""
        proc = self._record_proc
        self._record_proc = None
        self._record_started_for_yaml = ""
        if proc is None:
            return
        try:
            if proc.poll() is None:
                rospy.logwarn("[path_execute_server] auto-record: stopping (pid=%s)...", proc.pid)
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                except Exception:
                    proc.send_signal(signal.SIGINT)
        except Exception:
            return
        try:
            proc.wait(timeout=10.0)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                pass
        rospy.loginfo("Recording stopped.")

    def _check_cancel(self) -> None:
        if self._cancel_event.is_set():
            raise Cancelled()

    def _pose_cb(self, msg: PoseStamped) -> None:
        with self._current_pose_lock:
            self._current_pose = msg

    def _cartesian_cb(self, msg: CartesianPose) -> None:
        with self._current_pose_lock:
            self._current_pose = msg.poseStamped

    def _load_cartesian_values(self, prefix: str, defaults: Tuple[float, ...]) -> dict:
        values = {}
        fields = ("x", "y", "z", "a", "b", "c")
        for i, field in enumerate(fields):
            values[field] = float(rospy.get_param(f"{prefix}_{field}", defaults[i]))
        return values

    def _to_cartesian_quantity(self, values: dict):
        cq = CartesianQuantity()
        cq.x = values.get("x", 0.0)
        cq.y = values.get("y", 0.0)
        cq.z = values.get("z", 0.0)
        cq.a = values.get("a", 0.0)
        cq.b = values.get("b", 0.0)
        cq.c = values.get("c", 0.0)
        return cq

    def _ensure_control_client(self) -> bool:
        if not _HAVE_CTRL or ConfigureControlMode is None:
            rospy.logwarn("[path_execute_server] 未找到 ConfigureControlMode 接口，跳过控制模式切换")
            return False
        if self._impedance_client is None:
            try:
                self._impedance_client = rospy.ServiceProxy(self.impedance_service, ConfigureControlMode)
            except Exception as exc:
                rospy.logwarn("[path_execute_server] 建立控制模式服务失败: %s", str(exc))
                self._impedance_client = None
                return False
        return True

    def _maybe_configure_position(self) -> None:
        if not self.auto_configure_position:
            return
        # Avoid redundant mode switches (service calls are slow and can fail transiently)
        if self._last_mode == "position":
            return
        if not self._ensure_control_client():
            return
        req = ConfigureControlModeRequest()
        req.control_mode = ControlMode.POSITION_CONTROL
        try:
            resp = self._impedance_client(req)
        except Exception as exc:
            # Some transports raise IOError/OSError (e.g. Errno 5) instead of rospy.ServiceException.
            # Do NOT crash the whole execution; log and continue.
            rospy.logwarn("[path_execute_server] 切换位置控制失败(忽略继续执行): %s", str(exc))
            return
        if not resp.success:
            rospy.logwarn("[path_execute_server] 切换位置控制失败: %s", resp.error)
            return
        self._last_mode = "position"
        rospy.loginfo("[path_execute_server] ✓ 已切换到位置控制模式")

    def _maybe_configure_impedance(self, *, required: bool = True) -> None:
        """
        Activate Cartesian impedance before scan/contact motion.

        Always calls ConfigureControlMode (no _last_mode short-circuit) because other
        nodes (reset/MoveIt) may switch the arm back to position without updating us.
        """
        if not self.auto_configure_impedance:
            if required:
                raise RuntimeError(
                    "auto_configure_impedance is disabled but impedance is required for path_execute motion"
                )
            return
        if not self._ensure_control_client():
            msg = "ConfigureControlMode service unavailable; cannot activate impedance"
            rospy.logerr("[path_execute_server] %s", msg)
            if required:
                raise RuntimeError(msg)
            return
        req = ConfigureControlModeRequest()
        req.control_mode = ControlMode.CARTESIAN_IMPEDANCE
        req.cartesian_impedance.cartesian_stiffness = self._to_cartesian_quantity(self.impedance_stiff_vals)
        req.cartesian_impedance.cartesian_damping = self._to_cartesian_quantity(self.impedance_damp_vals)
        req.cartesian_impedance.nullspace_stiffness = self.impedance_nullspace_stiffness
        req.cartesian_impedance.nullspace_damping = self.impedance_nullspace_damping
        req.limits.max_path_deviation = self._to_cartesian_quantity(self.impedance_max_path_dev_vals)
        req.limits.max_cartesian_velocity = self._to_cartesian_quantity(self.impedance_max_cart_vel_vals)
        req.limits.max_control_force = self._to_cartesian_quantity(self.impedance_max_control_force_vals)
        req.limits.max_control_force_stop = self.impedance_max_control_force_stop
        try:
            resp = self._impedance_client(req)
        except Exception as exc:
            msg = f"ConfigureControlMode (impedance) failed: {exc}"
            rospy.logerr("[path_execute_server] %s", msg)
            if required:
                raise RuntimeError(msg) from exc
            return
        if not resp.success:
            msg = f"ConfigureControlMode (impedance) rejected: {getattr(resp, 'error', 'unknown')}"
            rospy.logerr("[path_execute_server] %s", msg)
            if required:
                raise RuntimeError(msg)
            return
        self._last_mode = "impedance"
        # Operator-facing prompt tied to the moment impedance control becomes active.
        # Keep it bilingual for easy debugging/log review.
        rospy.loginfo("Deep breath and hold. We are about to begin scanning now.")

    def _wait_for_current_pose(self, timeout: float = 5.0) -> None:
        start = time.time()
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            self._check_cancel()
            with self._current_pose_lock:
                if self._current_pose is not None:
                    return
            if time.time() - start > timeout:
                raise RuntimeError("等待 /iiwa/state/CartesianPose 超时")
            rate.sleep()

    def _dict_to_pose(self, data: dict) -> PoseStamped:
        pose = PoseStamped()
        frame_id = data.get("frame_id", "iiwa_link_0")
        pose.header.frame_id = frame_id
        pose.pose.position.x = float(data["position"]["x"])
        pose.pose.position.y = float(data["position"]["y"])
        pose.pose.position.z = float(data["position"]["z"])
        pose.pose.orientation.x = float(data["orientation"]["x"])
        pose.pose.orientation.y = float(data["orientation"]["y"])
        pose.pose.orientation.z = float(data["orientation"]["z"])
        pose.pose.orientation.w = float(data["orientation"]["w"])
        return pose

    def _publish_pose(self, pose: PoseStamped) -> None:
        self._check_cancel()
        msg = copy.deepcopy(pose)
        msg.header.stamp = rospy.Time.now()
        self.pose_pub.publish(msg)

    def _compute_pose_error_components(
        self, reference: PoseStamped, actual: Optional[PoseStamped]
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        if actual is None:
            with self._current_pose_lock:
                actual = copy.deepcopy(self._current_pose)
        if actual is None:
            return (None, None, None, None)
        ref_pos = np.array([reference.pose.position.x, reference.pose.position.y, reference.pose.position.z], dtype=np.float64)
        act_pos = np.array([actual.pose.position.x, actual.pose.position.y, actual.pose.position.z], dtype=np.float64)
        d = act_pos - ref_pos
        err_xy = float(math.hypot(d[0], d[1]))
        err_z = float(abs(d[2]))
        err_xyz = float(np.linalg.norm(d))

        ref_q = np.array(
            [reference.pose.orientation.x, reference.pose.orientation.y, reference.pose.orientation.z, reference.pose.orientation.w], dtype=np.float64
        )
        act_q = np.array([actual.pose.orientation.x, actual.pose.orientation.y, actual.pose.orientation.z, actual.pose.orientation.w], dtype=np.float64)
        ref_q /= np.linalg.norm(ref_q) or 1.0
        act_q /= np.linalg.norm(act_q) or 1.0
        dot = float(np.clip(np.dot(ref_q, act_q), -1.0, 1.0))
        ang_err = 2.0 * math.acos(abs(dot))
        if ang_err > math.pi:
            ang_err = 2.0 * math.pi - ang_err
        return err_xy, err_z, err_xyz, ang_err

    def _wait_until_reached(self, target_pose: PoseStamped) -> None:
        start = time.time()
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            self._check_cancel()
            err_xy, err_z, _err_xyz, ang_err = self._compute_pose_error_components(target_pose, None)
            if err_xy is not None and ang_err is not None:
                if (err_xy <= self.error_position_tolerance_xy) and (err_z <= self.error_position_tolerance_z) and (ang_err <= self.error_angle_tolerance):
                    return
            if time.time() - start > self.error_timeout:
                return
            rate.sleep()

    def _execute_linear_motion(self, target_pose: PoseStamped, label: str, wait_for_completion: bool) -> PoseStamped:
        with self._current_pose_lock:
            current_pose = copy.deepcopy(self._current_pose)

        target_vec = np.array([target_pose.pose.position.x, target_pose.pose.position.y, target_pose.pose.position.z], dtype=np.float64)

        if current_pose is None:
            self._publish_pose(target_pose)
            if wait_for_completion:
                self._wait_until_reached(target_pose)
            return target_pose

        current_vec = np.array([current_pose.pose.position.x, current_pose.pose.position.y, current_pose.pose.position.z], dtype=np.float64)
        distance = np.linalg.norm(target_vec - current_vec)
        if distance < 1e-6:
            return target_pose

        num_steps = max(1, int(np.ceil(distance / self.linear_step_size)))
        direction = (target_vec - current_vec) / distance
        rate = rospy.Rate(self.linear_rate)

        for i in range(num_steps + 1):
            self._check_cancel()
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

        if wait_for_completion:
            self._wait_until_reached(target_pose)
        return target_pose

    def _wait_until_pose_reached(
        self,
        target_pose: PoseStamped,
        xy_tol: float,
        z_tol: float,
        ang_tol: float,
        *,
        label: str,
        fail_on_timeout: bool = True,
    ) -> None:
        start = time.time()
        rate = rospy.Rate(50)
        last_log_wall = 0.0
        while not rospy.is_shutdown():
            self._check_cancel()
            err_xy, err_z, _err_xyz, ang_err = self._compute_pose_error_components(target_pose, None)
            if err_xy is not None and err_z is not None and ang_err is not None:
                if (err_xy <= xy_tol) and (err_z <= z_tol) and (ang_err <= ang_tol):
                    rospy.loginfo(
                        "[path_execute_server] %s reached (err_xy=%.4f err_z=%.4f ang=%.2fdeg)",
                        label,
                        err_xy,
                        err_z,
                        math.degrees(ang_err),
                    )
                    return
                now = time.time()
                if (now - last_log_wall) > 2.0:
                    rospy.logwarn(
                        "[path_execute_server] waiting %s: err_xy=%.4f/%.3f err_z=%.4f/%.3f ang=%.2fdeg",
                        label,
                        err_xy,
                        xy_tol,
                        err_z,
                        z_tol,
                        math.degrees(ang_err),
                    )
                    last_log_wall = now
            if time.time() - start > self.error_timeout:
                msg = f"{label} not reached within {self.error_timeout:.0f}s"
                if fail_on_timeout:
                    raise RuntimeError(msg)
                rospy.logwarn("[path_execute_server] %s (continuing)", msg)
                return
            rate.sleep()

    def _hold_pose_command(
        self,
        target_pose: PoseStamped,
        hold_sec: float,
        *,
        xy_tol: Optional[float] = None,
        z_tol: Optional[float] = None,
        label: str = "hold",
    ) -> None:
        if hold_sec <= 0.0:
            return
        if xy_tol is not None and z_tol is not None:
            self._wait_until_pose_reached(
                target_pose,
                xy_tol,
                z_tol,
                self.error_angle_tolerance,
                label=f"{label}_pre",
                fail_on_timeout=True,
            )
        rate = rospy.Rate(50)
        end_time = time.time() + float(hold_sec)
        while not rospy.is_shutdown() and time.time() < end_time:
            self._check_cancel()
            self._publish_pose(target_pose)
            rate.sleep()
        # Skip post-hold reach check: impedance contact drifts in Z during hold (expected).

    def _execute_point0_approach_sequence(
        self,
        approach_pose: PoseStamped,
        contact_pose: PoseStamped,
    ) -> None:
        lift_m = float(self.point0_approach_z_lift_m)
        rospy.loginfo(
            "[path_execute_server] point0 sequence: (1) move to +%.3fm approach pose",
            lift_m,
        )
        self._execute_linear_motion(approach_pose, "point0_approach", wait_for_completion=False)
        self._wait_until_pose_reached(
            approach_pose,
            self.point0_approach_xy_tol,
            self.point0_approach_z_tol,
            self.error_angle_tolerance,
            label="point0_approach",
        )

        rospy.loginfo("[path_execute_server] point0 sequence: (2) descend to contact point0")
        self._execute_linear_motion(contact_pose, "point0_contact", wait_for_completion=False)
        self._wait_until_pose_reached(
            contact_pose,
            self.point0_contact_xy_tol,
            self.point0_contact_z_tol,
            self.error_angle_tolerance,
            label="point0_contact",
        )

        rospy.loginfo(
            "[path_execute_server] point0 sequence: (3) hold contact command for %.2fs (XY+Z reach required)",
            self.point0_contact_hold_sec,
        )
        self._hold_pose_command(
            contact_pose,
            self.point0_contact_hold_sec,
            xy_tol=self.point0_contact_xy_tol,
            z_tol=self.point0_contact_z_tol,
            label="point0_contact_hold",
        )
        self._wait_until_pose_reached(
            contact_pose,
            self.point0_contact_xy_tol,
            self.point0_contact_z_tol,
            self.error_angle_tolerance,
            label="point0_contact_before_path",
            fail_on_timeout=False,
        )

    def _execute_path_frames(self, path_frames: List[PoseStamped]) -> None:
        if len(path_frames) < 2:
            return

        current_index = 1
        max_index = len(path_frames) - 1
        self._publish_pose(path_frames[current_index])

        if self.lookahead_publish_delay > 0.0:
            rospy.sleep(self.lookahead_publish_delay)

        # Watchdog should measure "no progress" instead of "total path duration".
        # If we advance current_index (i.e., publish next frame), we reset the watchdog timer.
        last_progress_wall = time.time()
        last_log_wall = 0.0
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            self._check_cancel()
            # watchdog：若长时间没有推进到下一帧，认为无进展
            now = time.time()
            if (now - last_progress_wall) > float(self.error_timeout):
                raise RuntimeError(f"path tracking timeout (no progress) at index={current_index}/{max_index}")
            target_pose = path_frames[current_index]
            err_xy, err_z, _err_xyz, ang_err = self._compute_pose_error_components(target_pose, None)
            if err_xy is None or ang_err is None:
                rate.sleep()
                continue
            # periodic debug to help field diagnosis
            if (now - last_log_wall) > 2.0:
                rospy.logwarn(
                    "[path_execute_server] tracking idx=%d/%d err_xy=%.4f err_z=%.4f lookahead_xy=%.3f tol_xy=%.3f",
                    current_index,
                    max_index,
                    float(err_xy),
                    float(err_z),
                    float(self.lookahead_distance_xy),
                    float(self.error_position_tolerance_xy),
                )
                last_log_wall = now
            if current_index < max_index and (err_xy <= self.lookahead_distance_xy) and (err_z <= self.lookahead_distance_z):
                current_index += 1
                self._publish_pose(path_frames[current_index])
                last_progress_wall = time.time()
                continue
            if current_index == max_index and (err_xy <= self.error_position_tolerance_xy) and (
                ang_err <= self.error_angle_tolerance
            ):
                break
            rate.sleep()

    def execute(self, path_yaml: str, runtime_overrides: Optional[Dict[str, Any]] = None) -> None:
        self._cancel_event.clear()
        self._reload_runtime_params()
        if runtime_overrides:
            self._apply_runtime_overrides(runtime_overrides)
        self._runtime_revert_to_position_on_finish = None
        self._wait_for_current_pose()

        path_yaml = os.path.expanduser(path_yaml)
        if not os.path.exists(path_yaml):
            raise RuntimeError(f"路径 YAML 不存在: {path_yaml}")
        with open(path_yaml, "r") as f:
            data = yaml.safe_load(f) or {}

        start = data.get("start")
        if not start or "pre_pose" not in start or "final_pose" not in start:
            raise RuntimeError("YAML 缺少 start/pre_pose 或 start/final_pose")
        start_pre_pose = self._dict_to_pose(start["pre_pose"])
        start_final_pose = self._dict_to_pose(start["final_pose"])

        frames = data.get("frames", [])
        if not frames:
            raise RuntimeError("YAML 缺少 frames")
        path_frames = [self._dict_to_pose(fr) for fr in frames]
        is_multi_frame = len(path_frames) > 1
        self._log_execute_profile(is_multi_frame=is_multi_frame)
        # Auto recording is intended for full scan (multi-frame full_path), NOT for point0.
        # Recording is disabled by default because it is outside the released agent pipeline.
        # Enable explicitly if a compatible recording subsystem is available:
        #   rosparam set /path_execute_server/auto_record_sync true
        auto_record_sync = bool(rospy.get_param("~auto_record_sync", False))
        # Stay in impedance after scan; reset_to_capture_pose (home) switches to position via MoveIt.
        revert_to_position_on_finish = bool(
            self._runtime_revert_to_position_on_finish
            if self._runtime_revert_to_position_on_finish is not None
            else rospy.get_param("~revert_to_position_on_finish", False)
        )
        record_started = False

        # --- stage1: optional pre approach (position) ---
        # IMPORTANT: For online contact retry, we export point YAML with direct_move=True,
        # which sets pre_pose == final_pose. In that case, do NOT switch to position control
        # and do NOT run the pre-stage. This matches the intended behavior:
        # - first attempt: pre-approach (lift/descend) is allowed
        # - retries: move directly to the point without lifting (no extra control switching)
        do_pre_stage = False
        if self.enable_pre_stage:
            try:
                # Compare pre/final pose: if essentially identical, skip pre-stage.
                err_xy, err_z, err_xyz, ang_err = self._compute_pose_error_components(start_pre_pose, start_final_pose)
                pos_err = float(err_xyz or 0.0)
                ang_err = float(ang_err or 0.0)
                # thresholds: 0.5mm / 1deg
                if (pos_err > 5e-4) or (ang_err > math.radians(1.0)):
                    do_pre_stage = True
                else:
                    rospy.loginfo("[path_execute_server] pre_pose ~= final_pose (direct_move). Skip pre-stage and position-control switch.")
            except Exception:
                # Be conservative: if we cannot compare, keep original behavior.
                do_pre_stage = True

        # Key improvement: for full_path (multi-frame), skip pre-stage by default to keep impedance continuous.
        # This prevents an extra POSITION->IMPEDANCE switch after point0 verification.
        if is_multi_frame and self.skip_pre_stage_for_multi_frame:
            if do_pre_stage:
                rospy.loginfo("[path_execute_server] multi-frame path: skip pre-stage to keep impedance (skip_pre_stage_for_multi_frame=true)")
            do_pre_stage = False

        # Impedance before any motion toward scan start (pre_pose and/or final_pose).
        # Pre-stage also runs in impedance unless use_position_for_pre_stage is explicitly true.
        if (not do_pre_stage) or (not self.use_position_for_pre_stage):
            self._maybe_configure_impedance(required=True)

        if do_pre_stage:
            if self.use_position_for_pre_stage:
                self._maybe_configure_position()
            self._execute_linear_motion(start_pre_pose, "pre", wait_for_completion=True)
            if self.pre_stage_pause > 0.0:
                rospy.sleep(self.pre_stage_pause)

        try:
            # Legacy optional path: position pre-stage then impedance for final descent.
            if do_pre_stage and self.use_position_for_pre_stage:
                self._maybe_configure_impedance(required=True)
            # Start recording only after impedance is active and the English prompt is printed.
            if is_multi_frame and auto_record_sync:
                self._record_start(path_yaml=path_yaml)
                record_started = True

            use_point0_sequence = bool(self.point0_approach_enabled and is_multi_frame)
            is_point0_only = len(path_frames) == 1
            if use_point0_sequence:
                contact_pose = copy.deepcopy(path_frames[0])
                contact_pose.pose.position.z -= float(self.point0_approach_z_lift_m)
                self._execute_point0_approach_sequence(start_final_pose, contact_pose)
                path_frames[0] = copy.deepcopy(contact_pose)
            elif is_point0_only:
                self._execute_linear_motion(start_final_pose, "point0_final", wait_for_completion=False)
                self._wait_until_pose_reached(
                    start_final_pose,
                    self.point0_contact_xy_tol,
                    self.point0_contact_z_tol,
                    self.error_angle_tolerance,
                    label="point0_final",
                    fail_on_timeout=True,
                )
            else:
                self._execute_linear_motion(start_final_pose, "final", wait_for_completion=True)

            # --- path ---
            if not use_point0_sequence:
                # 确保首帧与 start_final_pose 对齐
                first = path_frames[0]
                err_xy, err_z, err_xyz, ang_err = self._compute_pose_error_components(first, start_final_pose)
                pos_err = err_xy if self.ignore_z_error else err_xyz
                if (pos_err or 0.0) > 1e-3 or (ang_err or 0.0) > math.radians(1):
                    path_frames.insert(0, start_final_pose)
            self._execute_path_frames(path_frames)
        finally:
            if record_started:
                self._record_stop()
            # For full scan (multi-frame), revert to position control so subsequent reset is not in impedance mode.
            # Best-effort: if service not available, just log and continue.
            if is_multi_frame and revert_to_position_on_finish:
                try:
                    self._maybe_configure_position()
                except Exception:
                    pass


class PathExecuteServerNode(object):
    def __init__(self) -> None:
        rospy.init_node("path_execute_server", anonymous=False)

        self.cancel_on_new = bool(rospy.get_param("~cancel_on_new", True))
        self._cancel_event = threading.Event()
        self._core = PathExecuteCore(self._cancel_event)
        rospy.on_shutdown(self._core._record_stop)

        self._state_lock = threading.Lock()
        self._state = "idle"
        # idle 不是“成功执行完成”，wait 在 idle 时应返回失败，避免上层误判“已执行”
        self._last_result: Tuple[bool, str] = (False, "idle")
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._current_yaml: str = ""

        self.pub_state = rospy.Publisher("~state", String, queue_size=1, latch=True)
        self.sub_execute = rospy.Subscriber("~execute_yaml", String, self._on_execute_yaml, queue_size=1)
        self.srv_wait = rospy.Service("~wait", Trigger, self._on_wait)
        self.srv_cancel = rospy.Service("~cancel", Trigger, self._on_cancel)

        self._set_state("idle")
        rospy.loginfo("[path_execute_server] ready. Publish YAML path to ~execute_yaml; wait via ~wait; cancel via ~cancel.")
        rospy.spin()

    def _set_state(self, s: str) -> None:
        with self._state_lock:
            self._state = s
        self.pub_state.publish(String(data=s))

    def _is_running(self) -> bool:
        with self._state_lock:
            return bool(self._running)

    def _join_thread(self, timeout: float = 1.0) -> None:
        t = self._thread
        if t is None:
            return
        try:
            t.join(timeout=timeout)
        except Exception:
            pass

    def _run(self, path_yaml: str, runtime_overrides: Optional[Dict[str, Any]] = None) -> None:
        try:
            self._set_state("running")
            self._core.execute(path_yaml, runtime_overrides=runtime_overrides)
            with self._state_lock:
                self._last_result = (True, "completed")
            self._set_state("completed")
        except Cancelled:
            with self._state_lock:
                self._last_result = (False, "cancelled")
            self._set_state("cancelled")
        except Exception as e:
            # Print full traceback to rosout for debugging (Errno 5 often hides the real call site).
            rospy.logerr("[path_execute_server] execution failed with exception: %s", repr(e))
            rospy.logerr("[path_execute_server] traceback:\n%s", traceback.format_exc())
            with self._state_lock:
                self._last_result = (False, f"failed: {e}")
            self._set_state(f"failed: {e}")
        finally:
            with self._state_lock:
                self._running = False
                self._current_yaml = ""

    def _sync_public_state_if_not_running(self) -> None:
        """Fix latched ~/state when internal flags say we are idle (prevents wait() returning 'idle' while topic shows 'running')."""
        with self._state_lock:
            if self._running:
                return
            ok, msg = self._last_result
            pub = str(self._state or "").strip()
            if pub.startswith("running"):
                if ok:
                    self._set_state("completed")
                elif msg and msg != "idle":
                    self._set_state(msg if msg.startswith("failed") or msg.startswith("cancelled") else f"failed: {msg}")
                else:
                    self._set_state("idle")

    def _on_execute_yaml(self, msg: String) -> None:
        path_yaml, runtime_overrides = _parse_execute_request(msg.data or "")
        path_yaml = path_yaml.strip()
        if not path_yaml:
            return

        # 若正在运行：按配置取消并切换
        if self._is_running():
            # 相同 YAML 的重复请求：直接忽略，避免 cancel 自己（常见于 latch/重连）
            if path_yaml == (self._current_yaml or ""):
                rospy.logwarn("[path_execute_server] duplicate request ignored: %s", path_yaml)
                return
            if self.cancel_on_new:
                rospy.logwarn("[path_execute_server] cancel on new request")
                self._cancel_event.set()
                self._join_thread(timeout=0.5)

        if self._is_running():
            rospy.logwarn("[path_execute_server] still running; ignore new request: %s", path_yaml)
            return

        self._sync_public_state_if_not_running()

        with self._state_lock:
            self._running = True
            self._current_yaml = path_yaml

        t = threading.Thread(
            target=self._run,
            args=(path_yaml, runtime_overrides or None),
            daemon=True,
        )
        self._thread = t
        t.start()
        rospy.loginfo("[path_execute_server] started: %s", path_yaml)

    def _on_cancel(self, _req) -> TriggerResponse:
        if not self._is_running():
            return TriggerResponse(success=True, message="not running")
        self._cancel_event.set()
        return TriggerResponse(success=True, message="cancel requested")

    def _on_wait(self, _req) -> TriggerResponse:
        # 阻塞直到不在运行
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if not self._is_running():
                break
            rate.sleep()
        self._sync_public_state_if_not_running()
        ok, msg = self._last_result
        return TriggerResponse(success=bool(ok), message=str(msg))


if __name__ == "__main__":
    try:
        PathExecuteServerNode()
    except rospy.ROSInterruptException:
        pass


