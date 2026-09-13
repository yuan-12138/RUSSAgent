#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from autonomous_scan_agent.russagent_paths import path_preview_yaml, workspace_root, asa_pkg_root, repo_root
import sys
from typing import Any, Dict
import subprocess
import rospy

from .tool_base import BaseTool

# Resolve the project-owned RGB-pair pipeline.
_HAMLYN_RGBPAIR_PIPELINE = os.path.join(asa_pkg_root(), 'scripts', 'rgbpair_skel_pipeline.py')


def _normalize_organ_name(organ: str) -> str:
    s = str(organ or "").strip().lower()
    s = s.replace(" ", "_")
    s = s.replace("-", "_")
    return s


def _norm_point_key(x: Any) -> str:
    s = str(x or "").strip()
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("-", "_").replace(" ", "_")
    return s.lower()


_POINT_NAME_TO_INDEX: Dict[str, int] = {
    _norm_point_key("xiphoid_process"): 1,
    _norm_point_key("xiphoid"): 1,
    _norm_point_key("xiphoid_process"): 1,
    _norm_point_key("right_costal_margin"): 2,
    _norm_point_key("right costal margin"): 2,
    _norm_point_key("T12_left"): 3,
    _norm_point_key("T12(left)"): 3,
    _norm_point_key("left_inferior_border_of_12th_rib_endpoint"): 4,
    _norm_point_key("T12_right"): 5,
    _norm_point_key("T12(right)"): 5,
    _norm_point_key("right_inferior_border_of_12th_rib_endpoint"): 6,
    _norm_point_key("T12"): 7,
    _norm_point_key("L5"): 8,
    _norm_point_key("C5_left"): 9,
    _norm_point_key("C7_left"): 10,
    _norm_point_key("C5_right"): 11,
    _norm_point_key("C7_right"): 12,
    _norm_point_key("C5(left)"): 9,
    _norm_point_key("C7(left)"): 10,
    _norm_point_key("C5(right)"): 11,
    _norm_point_key("C7(right)"): 12,
    _norm_point_key("L_ASIS"): 13,
    _norm_point_key("L_pubic_tubercle"): 14,
    _norm_point_key("R_ASIS"): 15,
    _norm_point_key("R_pubic_tubercle"): 16,
    _norm_point_key("C2_left"): 21,
    _norm_point_key("C6_left"): 22,
    _norm_point_key("C2_right"): 23,
    _norm_point_key("C6_right"): 24,
    _norm_point_key("C2(left)"): 21,
    _norm_point_key("C6(left)"): 22,
    _norm_point_key("C2(right)"): 23,
    _norm_point_key("C6(right)"): 24,
}


def _point_index(x: Any) -> int:
    if x is None:
        return 0
    if isinstance(x, bool):
        return 0
    if isinstance(x, int):
        return int(x)
    s = str(x).strip()
    if s.isdigit():
        try:
            return int(s)
        except Exception:
            return 0
    return int(_POINT_NAME_TO_INDEX.get(_norm_point_key(s), 0))


def _default_start_end_points_real(organ: str, side: str) -> Dict[str, int]:
    """
    Real-mode mapping from (organ, side) -> (startpoint, endpoint).
    This is kept consistent with sim for the shared tasks.

      - gallbladder: 1 -> 2
      - kidney left: 3 -> 4
      - kidney right: 5 -> 6
      - spine: 7 -> 8
    """
    org = _normalize_organ_name(organ)
    sd = str(side or "").strip().lower()
    if sd not in ("left", "right"):
        sd = ""

    if org == "gallbladder":
        return {"startpoint": 1, "endpoint": 2}
    if org == "kidney":
        if sd != "left":
            return {"startpoint": 5, "endpoint": 6}
        return {"startpoint": 3, "endpoint": 4}
    if org == "spine":
        return {"startpoint": 7, "endpoint": 8}
    return {"startpoint": 0, "endpoint": 0}


def _organ_side_from_points_real(startpoint: int, endpoint: int) -> Dict[str, str]:
    """
    Reverse mapping: (startpoint, endpoint) -> {organ, side}.
    Real mode currently supports gallbladder/kidney/spine only.
    """
    try:
        sp = int(startpoint)
        ep = int(endpoint)
    except Exception:
        return {}

    table = {
        (1, 2): ("gallbladder", ""),
        (3, 4): ("kidney", "left"),
        (5, 6): ("kidney", "right"),
        (7, 8): ("spine", ""),
    }
    hit = table.get((sp, ep))
    if not hit:
        return {}
    organ, side = hit
    return {"organ": organ, "side": side}


class CliffSkelRiblineTool(BaseTool):
    """
    Tool1：调用 rgbpair_skel_pipeline.py 完成：
    - 两张图采集/拼接
    - CLIFF + SKEL
    - 生成肋缘线（并导出 upper 坐标系 uv：ribline_uv_upper_shifted_fit.npy）
    输出写入固定目录（默认 rgbpair_latest，每次覆盖）。
    """

    def __init__(self):
        super().__init__(
            name="cliff_skel_ribline_tool",
            description="Capture 2 RGB images, stitch, run CLIFF+SKEL, and export a task trajectory UV (upper image coordinates) into fixed output dir.",
        )
        self.fixed_out_dir = os.path.expanduser(
            rospy.get_param("~ribline_out_dir", os.path.join(workspace_root(), 'input_images', 'rgbpair_latest'))
        )

    def execute(self, **kwargs) -> dict:
        # 允许外部覆盖输出目录/是否覆盖
        fixed_out_dir = os.path.expanduser(kwargs.get("out_dir", self.fixed_out_dir))
        overwrite = bool(kwargs.get("overwrite", True))
        traj_kind = str(kwargs.get("traj_kind", "gallbladder")).strip().lower()
        # In practice we most often scan the right kidney; default to "right" to reduce LLM burden.
        kidney_side = str(kwargs.get("kidney_side", "right")).strip().lower()
        if kidney_side not in ("right", "left"):
            kidney_side = "right"
        kidney_start_mode = str(kwargs.get("kidney_start_mode", "thorax_spine_lowest")).strip().lower()
        if kidney_start_mode not in ("legacy", "thorax_spine_lowest"):
            kidney_start_mode = "thorax_spine_lowest"

        cmd = [
            sys.executable,
            _HAMLYN_RGBPAIR_PIPELINE,
            f"_fixed_out_dir:={fixed_out_dir}",
            f"_overwrite_fixed_out_dir:={'true' if overwrite else 'false'}",
            "_skip_projection:=true",  # Critical for RAG agent control
            f"_traj_kind:={traj_kind}",
            f"_kidney_side:={kidney_side}",
            f"_kidney_start_mode:={kidney_start_mode}",
        ]
        rospy.loginfo("[Tool1] running: %s", " ".join(cmd))
        try:
            subprocess.check_call(cmd)
        except Exception as e:
            return {"status": "failed", "message": str(e), "out_dir": fixed_out_dir, "traj_kind": traj_kind}

        # 返回关键输出（统一用 ribline_uv_upper_shifted_fit_npy 作为“下游投影用 UV 轨迹”字段名）
        uv_upper = os.path.join(fixed_out_dir, "ribline_uv_upper_shifted_fit.npy")
        uv_upper_fit = os.path.join(fixed_out_dir, "ribline_uv_upper_fit.npy")
        upper_overlay = os.path.join(fixed_out_dir, "upper_only_skel_rib_both.png")

        if traj_kind == "spine":
            # Exported by cliff_skel_trajectory.py in UPPER image coordinates
            uv_upper = os.path.join(fixed_out_dir, "spine_curve_uv_upper.npy")
            upper_overlay = os.path.join(fixed_out_dir, "rgb_stitched_upright_720x1280_demo_skel_overlay_spine_curve_only.png")
        elif traj_kind == "kidney":
            # Required: kidney offset-only line (the one shown in *_kidney_offset_only.png)
            uv_upper = os.path.join(fixed_out_dir, "kidney_line_uv_upper_shifted.npy")
            upper_overlay = os.path.join(fixed_out_dir, "rgb_stitched_upright_720x1280_demo_skel_overlay_kidney_offset_only.png")

        ok = os.path.exists(uv_upper)
        res = {
            "status": "success" if ok else "failed",
            "out_dir": fixed_out_dir,
            "ribline_uv_upper_shifted_fit_npy": uv_upper,
            "ribline_uv_upper_fit_npy": uv_upper_fit,
            "upper_only_overlay_png": upper_overlay,
            "thorax_spine_lowest_overlay_png": os.path.join(
                fixed_out_dir, "rgb_stitched_upright_720x1280_demo_thorax_spine_lowest.png"
            ),
            "lumbar_spine_highest_overlay_png": os.path.join(
                fixed_out_dir, "rgb_stitched_upright_720x1280_demo_lumbar_spine_highest.png"
            ),
            "thorax_mesh_left_lowest_overlay_png": os.path.join(
                fixed_out_dir, "rgb_stitched_upright_720x1280_demo_thorax_mesh_left_lowest.png"
            ),
            "traj_kind": traj_kind,
            "kidney_side": kidney_side,
            "kidney_start_mode": kidney_start_mode,
            "message": "done" if ok else f"expected output not found: {uv_upper}",
        }
        if ok:
            try:
                rospy.set_param("/autonomous_scan_agent/latest_ribline_uv_npy", uv_upper)
                rospy.set_param("/autonomous_scan_agent/latest_traj_kind", traj_kind)
                rospy.set_param("/autonomous_scan_agent/latest_traj_out_dir", fixed_out_dir)
                if traj_kind == "kidney":
                    rospy.set_param("/autonomous_scan_agent/latest_kidney_side", kidney_side)
            except Exception:
                pass
            # Align with sim_nl_standalone tool1 NL observations (exact phrasing matters for LLM policy transfer).
            if traj_kind == "kidney":
                res["nl_observation"] = (
                    f"Kidney trajectory acquired successfully ({kidney_side} side, "
                    f"start_mode={kidney_start_mode})."
                )
            elif traj_kind == "spine":
                res["nl_observation"] = "Spine trajectory acquired successfully."
            else:
                res["nl_observation"] = "Gallbladder trajectory acquired successfully."
        return res

    def _get_parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "out_dir": {"type": "string", "description": "Fixed output directory (will be overwritten by default)."},
                "overwrite": {"type": "boolean", "description": "Whether to overwrite the fixed output directory."},
                "traj_kind": {"type": "string", "description": "Trajectory kind: gallbladder | kidney | spine"},
                "kidney_side": {"type": "string", "description": "For traj_kind=kidney: right | left"},
                "kidney_start_mode": {
                    "type": "string",
                    "description": "For traj_kind=kidney: legacy | thorax_spine_lowest (left kidney only).",
                },
            },
        }


class CliffSkelSpineTool(CliffSkelRiblineTool):
    def __init__(self):
        super().__init__()
        self.name = "cliff_skel_spine_tool"
        self.description = "Capture 2 RGB images, stitch, run CLIFF+SKEL, and export spine centerline curve UV (upper image coordinates)."

    def execute(self, **kwargs) -> dict:
        kwargs = dict(kwargs or {})
        kwargs["traj_kind"] = "spine"
        return super().execute(**kwargs)


class CliffSkelKidneyTool(CliffSkelRiblineTool):
    def __init__(self):
        super().__init__()
        self.name = "cliff_skel_kidney_tool"
        self.description = "Capture 2 RGB images, stitch, run CLIFF+SKEL, and export kidney scan line UV (upper image coordinates)."

    def execute(self, **kwargs) -> dict:
        kwargs = dict(kwargs or {})
        kwargs["traj_kind"] = "kidney"
        return super().execute(**kwargs)


class AcquireTrajectoryTool(BaseTool):
    """
    Unified trajectory acquisition tool for real workflow.
    Routes organ requests to existing Tool1 implementations.
    """

    def __init__(self):
        super().__init__(
            name="acquire_trajectory_tool",
            description="Unified trajectory acquisition. organ selects target trajectory; side is mainly for kidney workflow; overwrite controls refresh behavior.",
        )
        self._gall = CliffSkelRiblineTool()
        self._kidney = CliffSkelKidneyTool()
        self._spine = CliffSkelSpineTool()

    def execute(
        self,
        organ: str = "",
        side: str = "",
        overwrite: bool = True,
        startpoint: Any = 0,
        endpoint: Any = 0,
        **kwargs,
    ) -> dict:
        # REQUIRED calling style:
        # Always provide organ + startpoint + endpoint (+ overwrite).
        # For side-specific organs (kidney), also provide side.
        org_in = str(organ or "").strip().lower()
        side_in = str(side or "").strip().lower()
        # Gallbladder/spine are NOT side-specific. Ignore any provided side to avoid unnecessary failures/loops.
        if org_in in ("gallbladder", "spine"):
            side_in = ""
        org = org_in
        sd = side_in
        if not org_in:
            return {
                "status": "failed",
                "message": "missing_organ",
                "nl_observation": "Trajectory acquisition failed: missing required parameter 'organ'. Please re-enter the task.",
            }
        if sd not in ("left", "right"):
            sd = "right"

        base_kwargs = dict(kwargs or {})
        base_kwargs["overwrite"] = bool(overwrite)
        # For schema alignment with sim: accept startpoint/endpoint and echo them back.
        # Real tool currently does not use these indices internally.
        sp = _point_index(startpoint)
        ep = _point_index(endpoint)

        if sp <= 0 or ep <= 0:
            return {
                "status": "failed",
                "message": "missing_keypoints",
                "organ": org_in,
                "startpoint": sp,
                "endpoint": ep,
                "nl_observation": (
                    "Trajectory acquisition failed: missing required keypoints (startpoint/endpoint). "
                    f"Please re-enter the task (organ={org_in})."
                ),
            }

        resolved = _organ_side_from_points_real(sp, ep)
        if resolved:
                resolved_org = str(resolved.get("organ") or "").strip().lower()
                resolved_side = str(resolved.get("side") or "").strip().lower()
                # For non-side-specific organs (gallbladder/spine), mapping returns side='' and we must keep it empty.
                if resolved_side and (resolved_side not in ("left", "right")):
                    resolved_side = "right"

                # If organ/side were provided, validate they match the keypoints mapping.
                if org_in != resolved_org:
                    return {
                        "status": "failed",
                        "message": "organ_keypoints_mismatch",
                        "organ": org_in,
                        "startpoint": sp,
                        "endpoint": ep,
                        "nl_observation": (
                            "Trajectory acquisition failed: organ does not match the provided keypoints "
                            f"(organ={org_in}, startpoint={sp}, endpoint={ep}). "
                            "Please re-enter the task."
                        ),
                    }
                if resolved_side and (side_in != resolved_side):
                    return {
                        "status": "failed",
                        "message": "side_keypoints_mismatch",
                        "organ": org_in,
                        "side": side_in,
                        "startpoint": sp,
                        "endpoint": ep,
                        "nl_observation": (
                            "Trajectory acquisition failed: side does not match the provided keypoints "
                            f"(side={side_in}, startpoint={sp}, endpoint={ep}). "
                            "Please re-enter the task."
                        ),
                    }

                org = resolved_org or org
                sd = resolved_side or sd
        else:
            return {
                "status": "failed",
                "message": "unsupported_points",
                "organ": org_in,
                "startpoint": sp,
                "endpoint": ep,
                "nl_observation": (
                    "Trajectory acquisition failed: unsupported startpoint/endpoint mapping "
                    f"(startpoint={sp}, endpoint={ep}). Please re-enter the task."
                ),
            }

        # Side requirement for kidney in real mode
        if org == "kidney":
            if side_in not in ("left", "right"):
                return {
                    "status": "failed",
                    "message": "missing_side",
                    "organ": org_in,
                    "startpoint": sp,
                    "endpoint": ep,
                    "nl_observation": (
                        "Trajectory acquisition failed: missing required parameter 'side' ('left' or 'right') "
                        "for organ=kidney. Please re-enter the task."
                    ),
                }
        # For gallbladder/spine, side is ignored above; do not fail on side to reduce LLM loops.

        # overwrite policy: if overwrite is explicitly false, enforce "reuse-only" behavior.
        # In sim/real alignment policy, overwrite=false means do NOT acquire a new trajectory.
        # If the caller sets overwrite=false, we fail with an explicit English observation about missing existing trajectory.
        if not bool(overwrite):
            return {
                "status": "failed",
                "message": "existing_trajectory_not_found",
                "organ": org_in,
                "side": side_in,
                "startpoint": sp,
                "endpoint": ep,
                "nl_observation": (
                    "Existing trajectory not found (overwrite=false). "
                    "Trajectory acquisition failed: cannot reuse a trajectory because no previous trajectory was found. "
                    "Please re-enter the task."
                ),
            }

        base_kwargs["startpoint"] = sp
        base_kwargs["endpoint"] = ep

        if org == "gallbladder":
            res = self._gall.execute(**base_kwargs)
            res["startpoint"] = sp
            res["endpoint"] = ep
            return res
        if org == "kidney":
            base_kwargs["kidney_side"] = sd
            if kwargs.get("kidney_start_mode") is not None:
                base_kwargs["kidney_start_mode"] = str(kwargs.get("kidney_start_mode"))
            res = self._kidney.execute(**base_kwargs)
            res["startpoint"] = sp
            res["endpoint"] = ep
            return res
        if org == "spine":
            res = self._spine.execute(**base_kwargs)
            res["startpoint"] = sp
            res["endpoint"] = ep
            return res

        return {
            "status": "failed",
            "message": f"unsupported_organ: {org}",
            "nl_observation": "Trajectory acquisition failed: unsupported organ. Supported organs in real mode are gallbladder, kidney, spine.",
        }

    def _get_parameters_schema(self) -> dict:
        # Keep the tool schema aligned with the public API catalog.
        return {
            "type": "object",
            "properties": {
                "organ": {"type": "string", "description": "Target organ/trajectory kind (e.g., gallbladder/kidney/spine)."},
                "side": {"type": "string", "description": "For side-specific organs (e.g., kidney): left | right."},
                "overwrite": {"type": "boolean", "description": "Whether to overwrite an existing trajectory (default true)."},
                "startpoint": {"type": "integer", "description": "Start keypoint index (see handbook mapping)."},
                "endpoint": {"type": "integer", "description": "End keypoint index (see handbook mapping)."},
            },
            "required": [],
        }
