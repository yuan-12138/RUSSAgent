#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from autonomous_scan_agent.russagent_paths import path_preview_yaml, workspace_root, asa_pkg_root, repo_root
from typing import Any, Dict

import rospy
import yaml

from .tool_base import BaseTool
from .tool_4_path_execution import PathExecuteServerTool


class PostScanAdjustTool(BaseTool):
    """
    Real Tool6:
    - One-time post-scan result verification.
    - If target is off-center, shift path x by +/-0.01 m and re-execute full path ONCE.
    - No second verification loop inside this tool.
    """

    def __init__(self):
        super().__init__(
            name="post_scan_adjust_tool",
            description=(
                "One-time post-scan verification and optional path x adjustment. "
                "Input centered/-x/+x. If off-center, adjust YAML and re-execute once."
            ),
        )
        self.default_yaml = path_preview_yaml()

    def _load_yaml(self, path_yaml: str) -> Dict[str, Any]:
        with open(path_yaml, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _save_yaml(self, path_yaml: str, data: Dict[str, Any]) -> None:
        with open(path_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)

    def _apply_x_offset(self, data: Dict[str, Any], offset_x: float) -> int:
        count = 0
        frames = data.get("frames", [])
        for frame in frames:
            pos = frame.get("position", {})
            if "x" in pos:
                pos["x"] = float(pos["x"]) + float(offset_x)
                count += 1

        if "start" in data:
            for key in ("surface_point", "pre_pose", "final_pose"):
                item = data["start"].get(key, {})
                if "position" in item and "x" in item["position"]:
                    item["position"]["x"] = float(item["position"]["x"]) + float(offset_x)
                elif "x" in item:
                    item["x"] = float(item["x"]) + float(offset_x)

        if "end" in data:
            for key in ("surface_point", "final_pose"):
                item = data["end"].get(key, {})
                if "position" in item and "x" in item["position"]:
                    item["position"]["x"] = float(item["position"]["x"]) + float(offset_x)
                elif "x" in item:
                    item["x"] = float(item["x"]) + float(offset_x)

        return count

    def _apply_first_point_z_offset(self, data: Dict[str, Any], offset_z: float) -> int:
        """
        Apply a Z offset ONLY to the first point used for the re-scan.
        We adjust:
        - frames[0].position.z (first scan frame), if present
        - start.final_pose.position.z (approach contact pose), if present
        This is intentionally NOT applied to all frames.
        """
        count = 0
        # frames[0]
        try:
            frames = data.get("frames", [])
            if isinstance(frames, list) and len(frames) > 0 and isinstance(frames[0], dict):
                pos = frames[0].get("position", {}) or {}
                if isinstance(pos, dict) and ("z" in pos):
                    pos["z"] = float(pos["z"]) + float(offset_z)
                    count += 1
        except Exception:
            pass

        # start.final_pose
        try:
            start = data.get("start", {}) or {}
            if isinstance(start, dict):
                fp = start.get("final_pose", {}) or {}
                if isinstance(fp, dict):
                    if "position" in fp and isinstance(fp.get("position"), dict) and ("z" in fp["position"]):
                        fp["position"]["z"] = float(fp["position"]["z"]) + float(offset_z)
                        count += 1
                    elif "z" in fp:
                        fp["z"] = float(fp["z"]) + float(offset_z)
                        count += 1
        except Exception:
            pass

        return count

    def execute(self, **kwargs) -> Dict[str, Any]:
        path_yaml = os.path.expanduser(str(kwargs.get("path_yaml") or "").strip() or self.default_yaml)

        print("\n" + "=" * 60)
        print("SCAN RESULT VERIFICATION")
        print("Please check the ultrasound image.")
        print("Is the scan target centered? (Enter 'centered', '-x', or '+x')")
        print("  - '-x': Target is shifted toward -x in image (adjust path -x)")
        print("  - '+x': Target is shifted toward +x in image (adjust path +x)")
        print("  - You may also type 'left'/'right' as synonyms for '-x' / '+x'")
        print("  - 'centered' / 'good': Scan is good")
        print("=" * 60)

        try:
            ans = input("> ").strip().lower()
        except Exception:
            ans = ""

        if ans in ("centered", "good", "ok", "yes"):
            return {
                "status": "success",
                "message": "verified_good",
                "nl_observation": (
                    "Scan result verified: Target is centered. "
                    "This verification corresponds to the most recently executed scan segment. "
                    "No adjustment was applied. "
                    "Robot is at the end of the scan path (not at the capture pose). "
                    "Verification complete."
                ),
            }

        if ("-x" in ans) or ("minus" in ans) or ("left" in ans):
            offset_x = -0.015
            obs_desc = "off-center (shifted -x)"
        elif ("+x" in ans) or ("plus" in ans) or ("right" in ans):
            offset_x = 0.015
            obs_desc = "off-center (shifted +x)"
        else:
            rospy.logwarn("[post_scan_adjust_tool] Unknown input '%s', assuming centered.", ans)
            return {
                "status": "success",
                "message": "verified_good_implicit",
                "nl_observation": "Scan result verification: input unclear; assuming centered. Verification complete.",
            }

        if not os.path.exists(path_yaml):
            return {
                "status": "failed",
                "message": "path_yaml_not_found",
                "path_yaml": path_yaml,
                "nl_observation": f"Scan result verified: Target was {obs_desc}. YAML not found: {path_yaml}.",
            }

        try:
            data = self._load_yaml(path_yaml)
            count = self._apply_x_offset(data, offset_x)
            # Extra rule for re-scan (left/right): apply a Z offset to the first point only.
            # This is intended to be the opposite direction of the "poor contact" adjustment in point0_auto_verify_tool
            # (which moves "down/deeper"). Here we move "up/shallower" by default.
            z0 = float(kwargs.get("first_point_z_offset_m", 0.05))
            z_offset = abs(z0)  # enforce "+" direction by default
            z_count = self._apply_first_point_z_offset(data, z_offset)
            self._save_yaml(path_yaml, data)
            rospy.loginfo(
                "[post_scan_adjust_tool] Adjusted YAML x by %.4f on %d frames; first-point z by +%.4f on %d items: %s",
                offset_x,
                count,
                z_offset,
                z_count,
                path_yaml,
            )
        except Exception as e:
            return {
                "status": "failed",
                "message": f"yaml_adjust_failed: {e}",
                "path_yaml": path_yaml,
                "nl_observation": f"Scan result verified: Target was {obs_desc}. Failed to adjust YAML.",
            }

        # Re-execute once after adjustment. Keep call simple and reuse existing Tool4.
        exec_res = PathExecuteServerTool().execute(
            path_id=path_yaml,
            wait_timeout_sec=float(kwargs.get("wait_timeout_sec", 120.0)),
            breath_instruction_on_impedance=bool(kwargs.get("breath_instruction_on_impedance", False)),
            executor_enable_pre_stage=True,
            executor_use_position_for_pre_stage=False,
            executor_auto_configure_position=False,
            executor_auto_configure_impedance=True,
            executor_revert_to_position_on_finish=False,
            executor_point0_approach_enabled=True,
            executor_point0_approach_z_lift_m=z_offset,
            executor_point0_contact_hold_sec=float(kwargs.get("point0_contact_hold_sec", 4.0)),
            executor_point0_contact_z_tol=float(kwargs.get("point0_contact_z_tol", 0.03)),
        )

        if str(exec_res.get("status") or "").lower() == "success":
            return {
                "status": "success",
                "message": "adjusted_and_reexecuted",
                "path_yaml": path_yaml,
                "nl_observation": (
                    f"Scan result verified: Target was {obs_desc}. "
                    "This verification corresponds to the most recently executed scan segment. "
                    f"Path adjusted by {offset_x}m. First point z adjusted by +{z_offset}m. "
                    "Repeat scan executed once using the adjusted path. "
                    "Robot is at the end of the scan path (not at the capture pose). "
                    "Verification complete."
                ),
            }

        return {
            "status": "failed",
            "message": f"reexecute_failed: {exec_res.get('message')}",
            "path_yaml": path_yaml,
            "nl_observation": (
                f"Scan result verified: Target was {obs_desc}. "
                f"Path adjusted by {offset_x}m but repeat scan execution failed."
            ),
        }

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path_yaml": {"type": "string"},
                "wait_timeout_sec": {"type": "number"},
                "breath_instruction_on_impedance": {"type": "boolean"},
                "point0_contact_hold_sec": {"type": "number"},
                "point0_contact_z_tol": {"type": "number"},
            },
        }

