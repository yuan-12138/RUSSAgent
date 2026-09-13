#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from typing import Any, Dict, List, Optional

import rospy
import yaml

from .tool_base import BaseTool

_PENDING_EXECUTE_PARAM = "/autonomous_scan_agent/point0_pending_execute"
_PENDING_CONTACT_PARAM = "/autonomous_scan_agent/point0_pending_contact"
_PENDING_YAML_PARAM = "/autonomous_scan_agent/point0_pending_yaml"


def _set_guard(pending_execute: Optional[bool] = None, pending_contact: Optional[bool] = None, pending_yaml: Optional[str] = None) -> None:
    try:
        if pending_execute is not None:
            rospy.set_param(_PENDING_EXECUTE_PARAM, bool(pending_execute))
        if pending_contact is not None:
            rospy.set_param(_PENDING_CONTACT_PARAM, bool(pending_contact))
        if pending_yaml is not None:
            rospy.set_param(_PENDING_YAML_PARAM, str(pending_yaml))
    except Exception:
        # Best-effort guardrails; never crash the tool because of param server.
        pass


def _get_guard() -> Dict[str, Any]:
    try:
        return {
            "pending_execute": bool(rospy.get_param(_PENDING_EXECUTE_PARAM, False)),
            "pending_contact": bool(rospy.get_param(_PENDING_CONTACT_PARAM, False)),
            "pending_yaml": str(rospy.get_param(_PENDING_YAML_PARAM, "")),
        }
    except Exception:
        return {"pending_execute": False, "pending_contact": False, "pending_yaml": ""}


def _get_pos(d: Dict[str, Any]) -> List[float]:
    p = d.get("position", {}) or {}
    return [float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(p.get("z", 0.0))]


def _set_pos(d: Dict[str, Any], xyz: List[float]) -> None:
    d.setdefault("position", {})
    d["position"]["x"] = float(xyz[0])
    d["position"]["y"] = float(xyz[1])
    d["position"]["z"] = float(xyz[2])


def _get_quat(d: Dict[str, Any]) -> List[float]:
    q = d.get("orientation", {}) or {}
    return [float(q.get("x", 0.0)), float(q.get("y", 0.0)), float(q.get("z", 0.0)), float(q.get("w", 1.0))]


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> List[List[float]]:
    # Standard quaternion -> rotation matrix
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return [
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ]


def _z_axis_from_pose_dict(d: Dict[str, Any]) -> List[float]:
    """Tool +Z axis from pose orientation quaternion."""
    qx, qy, qz, qw = _get_quat(d)
    rot = _quat_to_rot(qx, qy, qz, qw)
    z = [float(rot[0][2]), float(rot[1][2]), float(rot[2][2])]
    n = (z[0] * z[0] + z[1] * z[1] + z[2] * z[2]) ** 0.5
    if n < 1e-9:
        return [0.0, 0.0, 1.0]
    return [z[0] / n, z[1] / n, z[2] / n]


def _standoff_direction_from_tool_z(z_axis: List[float]) -> List[float]:
    """Outward standoff direction (above surface); matches legacy +base Z when tool Z points body-side."""
    if float(z_axis[2]) <= 0.0:
        return [-z_axis[0], -z_axis[1], -z_axis[2]]
    return list(z_axis)


class PathYamlManagerTool(BaseTool):
    """
    Tool3：管理/编辑 Tool2 生成的 path_preview1.yaml，用于“逐点观测+纠正”。

    关键能力：
    - load: 加载 YAML（可等待文件生成）
    - export_point_yaml: 导出“单点执行”的 YAML（start/pre+final + frames>=1），给 Tool4 执行
    - export_residual_yaml: 导出从指定 index 开始的 residual YAML（用于 batch）
    - modify_from_index: 从某个 index 起对所有后续点做全局偏移（offset_normal/offset_y）
    - modify_specific_points: 仅修改某些 indices（比如离线 rib shadow 反馈）

    约定：
    - points 用 frames 作为 0-based 索引：point_i_final := frames[i]
    - pre_pose 由 final_pose 沿外向 standoff 方向（tool +Z 的反向，与 legacy +base Z 一致）抬高 (pre_approach_offset-approach_offset)
    - offset_normal/offset_y 也在 target_frame 下施加：+Z / +Y
    """

    def __init__(self):
        super().__init__(
            name="path_yaml_tool",
            description="Load/edit/export path_preview1.yaml for step-by-step verification and correction.",
        )
        self._path_yaml: Optional[str] = None
        self._data: Optional[Dict[str, Any]] = None
        # Initialize guardrail params for this process/session.
        _set_guard(pending_execute=False, pending_contact=False, pending_yaml="")

    def execute(self, action: str, **kwargs) -> dict:
        # Tool-level guardrails (v3-style) implemented via ROS param server:
        # If point0 has been exported but not executed yet, do not allow further YAML operations
        # (export/modify/save/...) that would cause the LLM to loop. Force "execute point0" next.
        if action != "load":
            g = _get_guard()
            if g.get("pending_execute") is True:
                msg = "must_execute_point0_next"
                hint = 'You already exported point0. Next step MUST be path_execute_server_tool(path_id="~/.ros/path_point0.yaml") before any further YAML changes.'
                rospy.logwarn("[path_yaml_tool] %s: %s", msg, hint)
                return {"status": "failed", "message": msg, "hint": hint, "pending_yaml": g.get("pending_yaml", "")}

        if action == "load":
            return self._action_load(**kwargs)
        if action == "export_point_yaml":
            return self._action_export_point_yaml(**kwargs)
        if action == "export_residual_yaml":
            return self._action_export_residual_yaml(**kwargs)
        if action == "modify_from_index":
            return self._action_modify_from_index(**kwargs)
        if action == "modify_specific_points":
            return self._action_modify_specific_points(**kwargs)
        if action == "save":
            return self._action_save(**kwargs)
        return {"status": "failed", "message": f"Unknown action: {action}"}

    def _get_parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "load",
                        "save",
                        "export_point_yaml",
                        "export_residual_yaml",
                        "modify_from_index",
                        "modify_specific_points",
                    ],
                },
                "path_yaml": {"type": "string"},
                "wait_timeout_sec": {"type": "number"},
                "index": {"type": "integer"},
                "from_index": {"type": "integer"},
                "out_yaml": {"type": "string"},
                "direct_move": {"type": "boolean", "description": "If true: export with pre_pose == final_pose (no lift)."},
                "offset_normal": {"type": "number"},
                "offset_y": {"type": "number"},
                "recompute_start_pre_pose": {"type": "boolean", "description": "If true, recompute start.pre_pose from frames[0]. Default false to avoid lift during online correction."},
                "indices": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["action"],
        }

    def _ensure_loaded(self) -> Dict[str, Any]:
        """
        IMPORTANT CONTRACT (LLM + runner friendly):
        - You MUST call action="load" explicitly before export/modify/save.
        - We DO NOT auto-load ~/.ros/path_preview1.yaml here, because auto-load would bypass the
          runner's env_state update (runner only flips path_preview_yaml_loaded when action="load" succeeds).
        """
        if self._data is None or self._path_yaml is None:
            raise RuntimeError("YAML not loaded yet (call action=load first)")
        return self._data

    def _action_load(self, **kwargs) -> dict:
        path_yaml = os.path.expanduser(kwargs.get("path_yaml", "~/.ros/path_preview1.yaml"))
        wait_timeout = float(kwargs.get("wait_timeout_sec", 20.0))
        t0 = time.time()
        while not os.path.exists(path_yaml) and (time.time() - t0) < wait_timeout and not rospy.is_shutdown():
            rospy.sleep(0.2)
        if not os.path.exists(path_yaml):
            return {"status": "failed", "message": f"path_yaml not found: {path_yaml}"}
        with open(path_yaml, "r") as f:
            data = yaml.safe_load(f) or {}
        frames = data.get("frames") or []
        if len(frames) < 1:
            return {"status": "failed", "message": f"invalid YAML (frames empty): {path_yaml}"}
        self._path_yaml = path_yaml
        self._data = data
        return {"status": "success", "path_yaml": path_yaml, "num_frames": len(frames)}

    def _action_save(self, **kwargs) -> dict:
        try:
            data = self._ensure_loaded()
        except Exception as e:
            return {
                "status": "failed",
                "message": "yaml_not_loaded",
                "hint": 'Call path_yaml_tool with action="load" first (e.g., path_yaml="~/.ros/path_preview1.yaml").',
                "error": str(e),
            }
        path_yaml = os.path.expanduser(kwargs.get("path_yaml", self._path_yaml or "~/.ros/path_preview1.yaml"))
        if not path_yaml:
            return {"status": "failed", "message": "path_yaml empty"}
        os.makedirs(os.path.dirname(path_yaml), exist_ok=True)
        with open(path_yaml, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        self._path_yaml = path_yaml
        return {"status": "success", "path_yaml": path_yaml}

    def _offsets(self) -> float:
        data = self._ensure_loaded()
        pre = float(data.get("pre_approach_offset", 0.30))
        appr = float(data.get("approach_offset", 0.11))
        return pre - appr

    def _make_pre_from_final(self, final_pose: Dict[str, Any]) -> Dict[str, Any]:
        dz = self._offsets()
        pre_pose = yaml.safe_load(yaml.safe_dump(final_pose))  # deep copy
        xyz = _get_pos(pre_pose)
        z_axis = _z_axis_from_pose_dict(final_pose)
        outward = _standoff_direction_from_tool_z(z_axis)
        _set_pos(
            pre_pose,
            [
                float(xyz[0] + outward[0] * dz),
                float(xyz[1] + outward[1] * dz),
                float(xyz[2] + outward[2] * dz),
            ],
        )
        return pre_pose

    def _get_frames(self) -> List[Dict[str, Any]]:
        data = self._ensure_loaded()
        frames = data.get("frames") or []
        if not isinstance(frames, list) or len(frames) < 1:
            raise RuntimeError("frames invalid/empty")
        return frames

    def _action_export_point_yaml(self, **kwargs) -> dict:
        try:
            data = self._ensure_loaded()
        except Exception as e:
            return {
                "status": "failed",
                "message": "yaml_not_loaded",
                "hint": 'Call path_yaml_tool with action="load" first (e.g., path_yaml="~/.ros/path_preview1.yaml").',
                "error": str(e),
            }
        idx = int(kwargs.get("index", 0))
        out_yaml = os.path.expanduser(kwargs.get("out_yaml", "~/.ros/path_point.yaml"))
        direct_move = bool(kwargs.get("direct_move", False))

        # Reuse runner is designed for point0 verification loop only.
        # Prevent the LLM from looping on exporting other indices (e.g., index=1) without executing full_path.
        allow_nonzero = bool(rospy.get_param("~allow_export_nonzero_point", False))
        if idx != 0 and not allow_nonzero:
            msg = "reuse_runner_point0_only"
            hint = (
                "This runner only supports point0 verification. Do NOT export other indices. "
                "Next step should be executing full_path via path_execute_server_tool(path_id='~/.ros/path_preview1.yaml') "
                "after point0_contact is good."
            )
            rospy.logwarn("[path_yaml_tool] %s: idx=%d. %s", msg, idx, hint)
            return {"status": "failed", "message": msg, "hint": hint, "index": idx, "out_yaml": out_yaml}

        # Tool-level guardrail (v3-style): after exporting point0, you MUST execute it before exporting again.
        # This prevents the LLM from looping on export_point_yaml.
        if idx == 0:
            g = _get_guard()
            if g.get("pending_execute") is True:
                msg = "must_execute_point0_next"
                hint = 'You already exported point0. Next step MUST be path_execute_server_tool(path_id="~/.ros/path_point0.yaml") before exporting again.'
                rospy.logwarn("[path_yaml_tool] %s: %s", msg, hint)
                return {
                    "status": "failed",
                    "message": msg,
                    "hint": hint,
                    "pending_yaml": g.get("pending_yaml", ""),
                }

        frames = self._get_frames()
        if idx < 0 or idx >= len(frames):
            return {"status": "failed", "message": f"index out of range: {idx} (len={len(frames)})"}

        final_pose = frames[idx]
        pre_pose = final_pose if direct_move else self._make_pre_from_final(final_pose)

        # High-signal info: what exactly are we exporting?
        rospy.loginfo(f"[PathYamlManager] Exporting point index {idx} to {out_yaml}")
        rospy.loginfo(f"[PathYamlManager]   - direct_move: {direct_move}")
        rospy.loginfo(f"[PathYamlManager]   - Pre Pose Pos: {_get_pos(pre_pose)}")
        rospy.loginfo(f"[PathYamlManager]   - Final Pose Pos: {_get_pos(final_pose)}")

        out_data = {
            "pre_approach_offset": data.get("pre_approach_offset", 0.30),
            "approach_offset": data.get("approach_offset", 0.11),
            "start": {
                "pre_pose": pre_pose,
                "final_pose": final_pose,
            },
            "frames": [final_pose],
        }
        os.makedirs(os.path.dirname(out_yaml), exist_ok=True)
        with open(out_yaml, "w") as f:
            yaml.safe_dump(out_data, f, sort_keys=False)
        if idx == 0:
            # Export done -> require execute next; contact is no longer "current".
            _set_guard(pending_execute=True, pending_contact=False, pending_yaml=out_yaml)
        return {"status": "success", "out_yaml": out_yaml, "index": idx, "direct_move": direct_move}

    def _action_export_residual_yaml(self, **kwargs) -> dict:
        try:
            data = self._ensure_loaded()
        except Exception as e:
            return {
                "status": "failed",
                "message": "yaml_not_loaded",
                "hint": 'Call path_yaml_tool with action="load" first (e.g., path_yaml="~/.ros/path_preview1.yaml").',
                "error": str(e),
            }
        from_index = int(kwargs.get("from_index", 0))
        out_yaml = os.path.expanduser(kwargs.get("out_yaml", "~/.ros/path_residual.yaml"))
        direct_move = bool(kwargs.get("direct_move", True))
        frames = self._get_frames()
        if from_index < 0 or from_index >= len(frames):
            return {"status": "failed", "message": f"from_index out of range: {from_index} (len={len(frames)})"}

        final_pose = frames[from_index]
        pre_pose = final_pose if direct_move else self._make_pre_from_final(final_pose)
        out_frames = frames[from_index:]
        out_data = {
            "pre_approach_offset": data.get("pre_approach_offset", 0.30),
            "approach_offset": data.get("approach_offset", 0.11),
            "start": {"pre_pose": pre_pose, "final_pose": final_pose},
            "frames": out_frames,
        }
        os.makedirs(os.path.dirname(out_yaml), exist_ok=True)
        with open(out_yaml, "w") as f:
            yaml.safe_dump(out_data, f, sort_keys=False)
        return {"status": "success", "out_yaml": out_yaml, "from_index": from_index, "num_frames": len(out_frames), "direct_move": direct_move}

    def _apply_delta_to_pose(self, pose: Dict[str, Any], dy: float, dz: float) -> None:
        """
        Apply offsets in target_frame:
        - offset_y (dy): along +Y of target_frame (lateral)
        - offset_normal (dz): along the pose's LOCAL +Z axis (tool normal) expressed in target_frame
          This matches the clinical meaning of "press along probe normal".
        """
        xyz = _get_pos(pose)
        # lateral in frame +Y
        xyz[1] += float(dy)

        # normal along pose local +Z axis
        if abs(float(dz)) > 0.0:
            qx, qy, qz, qw = _get_quat(pose)
            R = _quat_to_rot(qx, qy, qz, qw)
            # local +Z axis in target_frame = 3rd column of R
            nz = (R[0][2], R[1][2], R[2][2])
            xyz[0] += float(dz) * float(nz[0])
            xyz[1] += float(dz) * float(nz[1])
            xyz[2] += float(dz) * float(nz[2])

        _set_pos(pose, xyz)

    def _action_modify_from_index(self, **kwargs) -> dict:
        # Any modification invalidates previous contact; allow export again and force a new execute+contact cycle.
        _set_guard(pending_execute=False, pending_contact=False)
        try:
            _ = self._ensure_loaded()
        except Exception as e:
            return {
                "status": "failed",
                "message": "yaml_not_loaded",
                "hint": 'Call path_yaml_tool with action="load" first (e.g., path_yaml="~/.ros/path_preview1.yaml").',
                "error": str(e),
            }
        idx = int(kwargs.get("from_index", kwargs.get("index", 0)))
        dz = float(kwargs.get("offset_normal", 0.0))
        dy = float(kwargs.get("offset_y", 0.0))
        recompute_start_pre_pose = bool(kwargs.get("recompute_start_pre_pose", False))
        # v3-aligned: allow flipping press direction by rosparam.
        # Default 1.0 keeps existing behavior unless the user sets it.
        # Default to -1.0 to match v3/v1 convention: negative sign means "press" along probe normal.
        sign = float(rospy.get_param("~offset_normal_sign", -1.0))
        dz_eff = dz * sign
        frames = self._get_frames()
        if idx < 0 or idx >= len(frames):
            return {"status": "failed", "message": f"from_index out of range: {idx} (len={len(frames)})"}

        rospy.loginfo(
            "[PathYamlManager] modify_from_index: from=%d offset_y=%.4f offset_normal=%.4f sign=%.2f eff=%.4f",
            idx, dy, dz, sign, dz_eff
        )
        for i in range(idx, len(frames)):
            self._apply_delta_to_pose(frames[i], dy=dy, dz=dz_eff)

        # 同步 start/end 到 frames[0]/frames[-1]（避免前后不一致）
        data = self._ensure_loaded()
        try:
            start = data.get("start") or {}
            start["final_pose"] = frames[0]
            # 在线纠偏时不强制改 pre_pose，避免“抬起再压下”的体验（由 export_point_yaml(direct_move=True) 控制单点是否抬起）
            if recompute_start_pre_pose or ("pre_pose" not in start):
                start["pre_pose"] = self._make_pre_from_final(frames[0])
            data["start"] = start
        except Exception:
            pass
        try:
            end = data.get("end") or {}
            end["final_pose"] = frames[-1]
            data["end"] = end
        except Exception:
            pass

        return {
            "status": "success",
            "from_index": idx,
            "offset_y": dy,
            "offset_normal": dz,
            "offset_normal_sign": sign,
            "offset_normal_effective": dz_eff,
        }

    def _action_modify_specific_points(self, **kwargs) -> dict:
        try:
            _ = self._ensure_loaded()
        except Exception as e:
            return {
                "status": "failed",
                "message": "yaml_not_loaded",
                "hint": 'Call path_yaml_tool with action="load" first (e.g., path_yaml="~/.ros/path_preview1.yaml").',
                "error": str(e),
            }
        indices = kwargs.get("indices", []) or []
        dy = float(kwargs.get("offset_y", 0.0))
        dz = float(kwargs.get("offset_normal", 0.0))
        # Default to -1.0 to match v3/v1 convention: negative sign means "press" along probe normal.
        sign = float(rospy.get_param("~offset_normal_sign", -1.0))
        dz_eff = dz * sign
        frames = self._get_frames()
        count = 0
        for i in indices:
            try:
                ii = int(i)
            except Exception:
                continue
            if 0 <= ii < len(frames):
                self._apply_delta_to_pose(frames[ii], dy=dy, dz=dz_eff)
                count += 1
        return {
            "status": "success",
            "modified": count,
            "offset_y": dy,
            "offset_normal": dz,
            "offset_normal_sign": sign,
            "offset_normal_effective": dz_eff,
        }


