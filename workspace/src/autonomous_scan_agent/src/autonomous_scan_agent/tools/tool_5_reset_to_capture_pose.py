#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Move the robot to a predefined joint pose.

Poses are loaded from ~/.ros/autonomous_scan_agent/capture_poses.yaml.
This is intended as a non-interactive "go home/reset" at the end of scanning.

Implementation note:
- This tool uses MoveIt (MoveGroupCommander) just like capture_local_mover.py (plan + execute),
  because it is more reliable in this setup than directly calling the iiwa action.
"""

import os
from autonomous_scan_agent.russagent_paths import path_preview_yaml, workspace_root, asa_pkg_root, repo_root
from typing import Any, Dict, List, Optional

import rospy
import yaml

try:
    import moveit_commander
    from moveit_commander import RobotCommander, MoveGroupCommander
    _HAVE_MOVEIT = True
except Exception:
    moveit_commander = None
    RobotCommander = None
    MoveGroupCommander = None
    _HAVE_MOVEIT = False

from .tool_base import BaseTool

try:
    from iiwa_msgs.msg import ControlMode  # type: ignore
    from iiwa_msgs.srv import ConfigureControlMode, ConfigureControlModeRequest  # type: ignore
    _HAVE_IIWA_CTRL = True
except Exception:
    ControlMode = None
    ConfigureControlMode = None
    ConfigureControlModeRequest = None
    _HAVE_IIWA_CTRL = False


DEFAULT_YAML = os.path.expanduser("~/.ros/autonomous_scan_agent/capture_poses.yaml")
DEFAULT_JOINT_NAMES = [
    "iiwa_joint_1",
    "iiwa_joint_2",
    "iiwa_joint_3",
    "iiwa_joint_4",
    "iiwa_joint_5",
    "iiwa_joint_6",
    "iiwa_joint_7",
]


def _reorder(joint_names_order: List[str], joints_dict: Dict[str, Any]) -> List[float]:
    missing = [j for j in joint_names_order if j not in joints_dict]
    if missing:
        raise ValueError(f"YAML missing joints: {missing}")
    return [float(joints_dict[j]) for j in joint_names_order]


def _resolve_existing_yaml_path(yaml_path: str) -> Optional[str]:
    """
    LLM/handbook sometimes passes a non-existent yaml_path.
    We try a small set of common locations and return the first existing one.
    """
    candidates = []
    if yaml_path:
        candidates.append(os.path.expanduser(yaml_path))
    candidates.append(DEFAULT_YAML)
    # historical/explicit absolute form (same as DEFAULT_YAML but helps readability)
    candidates.append("$HOME/.ros/autonomous_scan_agent/capture_poses.yaml")
    # sometimes users keep a copy under workspace/config
    candidates.append(os.path.join(workspace_root(), 'config', 'capture_poses.yaml'))

    for p in candidates:
        try:
            if p and os.path.exists(p):
                return p
        except Exception:
            continue
    return None


class ResetToCapturePoseTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="reset_to_capture_pose_tool",
            description="Move robot to a specified pose index from capture_poses.yaml using MoveIt (plan+execute).",
        )
        self._inited = False
        self._group = None
        self._robot = None
        self._ns_clean = ""
        self._desc_key = ""
        self._group_name = ""

    def _ensure_moveit(self, ns: str, group: str, vel_scale: float, acc_scale: float) -> Optional[str]:
        if not _HAVE_MOVEIT:
            return "moveit_not_available"
        # In this repo, MoveIt is commonly launched at ROOT namespace (e.g. /robot_description),
        # while some components use /iiwa namespace. We auto-detect which one is available.
        ns_clean = (ns or "").strip("/")
        # Candidate namespaces to probe, ordered by preference:
        # 1) user-provided ns
        # 2) root (empty)
        # 3) iiwa (common fallback)
        candidates = []
        if ns_clean:
            candidates.append(ns_clean)
        candidates.append("")  # root
        if "iiwa" not in candidates:
            candidates.append("iiwa")

        chosen_ns_clean = None
        chosen_desc_key = None

        # wait for moveit parameters (same as capture_local_mover.py)
        t0 = rospy.Time.now()
        while not rospy.is_shutdown():
            for cand in candidates:
                ns_prefix = ("/" + cand) if cand else ""
                desc_key = (ns_prefix + "/robot_description") if ns_prefix else "/robot_description"
                sem_key = (ns_prefix + "/robot_description_semantic") if ns_prefix else "/robot_description_semantic"
                if rospy.has_param(desc_key) and rospy.has_param(sem_key):
                    chosen_ns_clean = cand
                    chosen_desc_key = desc_key
                    break
            if chosen_desc_key:
                break
            if (rospy.Time.now() - t0).to_sec() > 5.0:
                return "moveit_params_not_ready"
            rospy.sleep(0.1)

        ns_clean = chosen_ns_clean or ""
        desc_key = chosen_desc_key or "/robot_description"

        # re-init if ns/group changed
        if self._inited and (self._ns_clean == ns_clean) and (self._group_name == group) and (self._desc_key == desc_key):
            try:
                self._group.set_max_velocity_scaling_factor(float(vel_scale))
                self._group.set_max_acceleration_scaling_factor(float(acc_scale))
            except Exception:
                pass
            return None

        try:
            moveit_commander.roscpp_initialize([])
            self._robot = RobotCommander(robot_description=desc_key.lstrip("/"))

            # Validate/repair group name (LLM sometimes passes "iiwa" which is NOT a planning group).
            available_groups = []
            try:
                available_groups = list(self._robot.get_group_names() or [])
            except Exception:
                available_groups = []

            chosen_group = str(group or "").strip() or "manipulator"
            if available_groups and chosen_group not in available_groups:
                # Prefer common group name if present
                if "manipulator" in available_groups:
                    chosen_group = "manipulator"
                else:
                    # fallback to first available
                    chosen_group = str(available_groups[0])

            self._group = MoveGroupCommander(chosen_group, robot_description=desc_key.lstrip("/"), ns=ns_clean)
            self._group.set_max_velocity_scaling_factor(float(vel_scale))
            self._group.set_max_acceleration_scaling_factor(float(acc_scale))
        except Exception as e:
            # Provide actionable hint for field debugging
            try:
                groups = list(self._robot.get_group_names() or []) if self._robot is not None else []
            except Exception:
                groups = []
            return f"moveit_init_failed:{e}; available_groups={groups}"

        self._inited = True
        self._ns_clean = ns_clean
        self._desc_key = desc_key
        try:
            self._group_name = str(self._group.get_name())
        except Exception:
            self._group_name = group
        return None

    def execute(
        self,
        # For sim/real consistency: finish flag indicates whether this reset ends the session.
        # Real mode does not inherently have "session end", but the runner may choose to exit when finish=true.
        finish: bool = True,
        # Match rgbpair_skel_pipeline capture home: ~pose_indices[0] (default [3,4] -> capture_03).
        pose_index: int = 3,
        yaml_path: str = DEFAULT_YAML,
        ns: str = "/iiwa",
        group: str = "manipulator",
        # 速度调小：capture_local_mover 默认 0.1；之前默认 0.05，再减半到 0.025；现在再减半到 0.0125
        vel_scale: float = 0.0125,
        acc_scale: float = 0.0125,
        wait_timeout_sec: float = 90.0,
        settle_sec: float = 0.5,
        force_position_control: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        _ = kwargs
        # IMPORTANT: this tool should never throw; ToolRegistry wraps exceptions as status="error"
        # which makes debugging harder. We convert all exceptions into status="failed" with hints.
        try:
            err = self._ensure_moveit(ns=ns, group=group, vel_scale=vel_scale, acc_scale=acc_scale)
            if err:
                return {"status": "failed", "message": err, "hint": "Ensure move_group is running and MoveIt params are available."}

            requested_yaml_path = os.path.expanduser(yaml_path or DEFAULT_YAML)
            resolved_yaml_path = _resolve_existing_yaml_path(requested_yaml_path)
            if not resolved_yaml_path:
                return {
                    "status": "failed",
                    "message": f"yaml_not_found:{requested_yaml_path}",
                    "hint": "capture_poses.yaml not found. Generate it via capture_pose.py, or put it under ~/.ros/autonomous_scan_agent/capture_poses.yaml.",
                    "searched": [
                        requested_yaml_path,
                        DEFAULT_YAML,
                        "$HOME/.ros/autonomous_scan_agent/capture_poses.yaml",
                        os.path.join(workspace_root(), 'config', 'capture_poses.yaml'),
                    ],
                }
            yaml_path = resolved_yaml_path

            try:
                with open(yaml_path, "r") as f:
                    poses = (yaml.safe_load(f) or {}).get("poses", [])
            except Exception as e:
                return {"status": "failed", "message": "yaml_load_failed", "error": str(e), "yaml_path": yaml_path}
            if not poses:
                return {"status": "failed", "message": "yaml_no_poses", "yaml_path": yaml_path}

            try:
                idx = int(pose_index)
            except Exception:
                idx = 0
            if idx < 0 or idx >= len(poses):
                return {"status": "failed", "message": "pose_index_out_of_range", "pose_index": idx, "num_poses": len(poses)}

            pose = poses[idx] or {}
            name = str(pose.get("name", f"pose_{idx}"))
            # MoveIt expects dict in active joint order. Use YAML joints mapping directly.
            joints = pose.get("joints", {}) or {}
            try:
                target_vals = _reorder(self._group.get_active_joints(), joints)
            except Exception as e:
                return {"status": "failed", "message": "pose_joints_invalid", "error": str(e), "pose_index": idx, "pose_name": name}

            rospy.logwarn(
                "[reset_to_capture_pose_tool] MoveIt reset: pose=%s idx=%d vel_scale=%.3f acc_scale=%.3f",
                name, idx, float(vel_scale), float(acc_scale)
            )
            try:
                self._group.set_joint_value_target(target_vals)
                plan = self._group.plan()
                traj = plan if hasattr(plan, "joint_trajectory") else (plan[1] if isinstance(plan, (list, tuple)) and len(plan) > 1 else None)
                if not traj or (not hasattr(traj, "joint_trajectory")) or len(traj.joint_trajectory.points) < 1:
                    return {"status": "failed", "message": "moveit_plan_failed", "pose_index": idx, "pose_name": name}
                ok = self._group.execute(traj, wait=True)
                self._group.stop()
                self._group.clear_pose_targets()
                if not ok:
                    return {"status": "failed", "message": "moveit_execute_failed", "pose_index": idx, "pose_name": name}
            except Exception as e:
                return {"status": "failed", "message": "moveit_execute_exception", "error": str(e), "pose_index": idx, "pose_name": name}

            rospy.sleep(float(settle_sec))
            # After reset, force position control mode (best-effort) so subsequent teleop/MoveIt moves are safe.
            if force_position_control and _HAVE_IIWA_CTRL and ConfigureControlMode is not None:
                try:
                    ns_clean = (ns or "").strip("/")
                    srv = f"/{ns_clean}/configuration/ConfigureControlMode" if ns_clean else "/iiwa/configuration/ConfigureControlMode"
                    cli = rospy.ServiceProxy(srv, ConfigureControlMode)
                    req = ConfigureControlModeRequest()
                    req.control_mode = ControlMode.POSITION_CONTROL
                    resp = cli(req)
                    if getattr(resp, "success", False):
                        rospy.loginfo("[reset_to_capture_pose_tool] ✓ switched to position control mode")
                    else:
                        rospy.logwarn("[reset_to_capture_pose_tool] failed to switch to position control: %s", getattr(resp, "error", "unknown"))
                except Exception as e:
                    rospy.logwarn("[reset_to_capture_pose_tool] switch to position control failed (ignored): %s", str(e))

            return {
                "status": "success",
                "pose_index": idx,
                "pose_name": name,
                "yaml_path": yaml_path,
                "finish": bool(finish),
                "nl_observation": (
                    "Robot reset completed. Scan session finished (sim)."
                    if bool(finish)
                    else (
                        "Robot reset completed. Session continues (finish=false). "
                        "Robot is at the capture pose. Ready for the next scan segment or verification (sim)."
                    )
                ),
            }
        except Exception as e:
            return {"status": "failed", "message": "exception", "error": str(e)}

    def _get_parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "pose_index": {"type": "integer", "description": "Index in capture_poses.yaml (0-based). Default 3 = capture_03 (rgbpair capture home)."},
                "yaml_path": {"type": "string", "description": "Path to capture_poses.yaml"},
                "ns": {"type": "string", "description": "Robot namespace, e.g. /iiwa"},
                "group": {"type": "string", "description": "MoveIt group name (default manipulator)"},
                "vel_scale": {"type": "number", "description": "MoveIt velocity scaling factor (default 0.0125)"},
                "acc_scale": {"type": "number", "description": "MoveIt acceleration scaling factor (default 0.0125)"},
                "wait_timeout_sec": {"type": "number"},
                "settle_sec": {"type": "number"},
                "force_position_control": {"type": "boolean", "description": "If true, best-effort switch to iiwa position control mode after reset."},
            },
            "required": [],
        }

