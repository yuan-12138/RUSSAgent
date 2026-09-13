#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict
import rospy
from autonomous_scan_agent.russagent_paths import path_preview_yaml, workspace_root, asa_pkg_root, repo_root
import time

from .tool_base import BaseTool
# Internal dependencies (do NOT use tool+number module names here)
from .path_yaml_manager import PathYamlManagerTool
from .tool_4_path_execution import PathExecuteServerTool
from .contact_query import ContactQueryTool
from .iiwa_base_descent import descent_base_minus_z


class Point0AutoVerifyTool(BaseTool):
    """
    Tool: Auto Point0 Verification & Correction Loop.
    
    Logic:
    1. Export Point0 from current YAML.
    2. Execute Point0.
    3. Ask operator for contact quality.
    4. If 'good': Apply accumulated offset to ALL points (batch modify), then return Success.
    5. If 'poor'/'none': Apply offset (e.g. -5mm normal) to Point0 only, and RETRY loop from step 1.
    """

    def __init__(self):
        super().__init__(
            name="point0_auto_verify_tool",
            description="Automated loop: Export Point0 -> Execute -> Check Contact -> (Modify & Retry if poor). Once good, apply correction to full path.",
        )
        self.yaml_tool = PathYamlManagerTool()
        self.exec_tool = PathExecuteServerTool()
        self.contact_tool = ContactQueryTool()

    def execute(self, offset_step_m: float = -0.005, **kwargs) -> Dict[str, Any]:
        """
        offset_step_m: adjustment step per retry (negative = down/deeper). Default -5mm.
        """
        # Step 0: Explicitly load the preview YAML to ensure we are working on the latest file.
        # This removes the burden from the LLM to call 'load' separately.
        rospy.loginfo("[AutoVerify] Explicitly loading path_preview1.yaml (wait up to 60s)...")
        res_load = self.yaml_tool.execute(action="load", path_yaml=path_preview_yaml(), wait_timeout_sec=60.0)
        if res_load.get("status") != "success":
             return {"status": "failed", "message": f"Internal Load failed: {res_load.get('message')}"}

        # Normalize to "down/deeper" step.
        # Some callers may mistakenly pass a positive value; treat it as magnitude and apply downward.
        offset_step_in = float(offset_step_m)
        offset_step = -abs(offset_step_in)
        max_retries = int(kwargs.get("max_retries", 10))
        total_offset = 0.0
        
        rospy.loginfo("[AutoVerify] Starting Point0 verification loop...")

        # After trajectory projection: leave capture/home in position control, descend along base -Z,
        # then switch to impedance before the first point0 approach.
        if bool(kwargs.get("enable_home_descent", True)):
            descent_m = float(kwargs.get("home_descent_m", 0.02))
            rospy.loginfo("[AutoVerify] Home descent along base -Z: %.4f m", descent_m)
            res_descent = descent_base_minus_z(
                descent_m=descent_m,
                base_frame=str(kwargs.get("base_frame", "iiwa_link_0")),
                ns=str(kwargs.get("ns", "/iiwa")),
                settle_sec=float(kwargs.get("home_descent_settle_sec", 0.3)),
            )
            if res_descent.get("status") != "success":
                return {
                    "status": "failed",
                    "message": f"Home descent failed: {res_descent.get('message')}",
                }

        for attempt in range(max_retries):
            # 1. Export Point0
            # Note: We always export index 0. If we modified it previously, the YAML on disk is already updated.
            res_yaml = self.yaml_tool.execute(
                action="export_point_yaml", 
                index=0, 
                out_yaml="os.path.join(os.path.dirname(path_preview_yaml()), 'path_point0.yaml')",
                # First attempt: use pre-pose then final, both in impedance control.
                # Subsequent attempts: stay in impedance and do a direct move (pre_pose == final_pose),
                # so we do NOT switch back to position repeatedly.
                direct_move=(attempt > 0),
            )
            if res_yaml.get("status") != "success":
                return {"status": "failed", "message": f"Export failed: {res_yaml.get('message')}"}

            # 2. Execute Point0
            res_exec = self.exec_tool.execute(
                path_id="os.path.join(os.path.dirname(path_preview_yaml()), 'path_point0.yaml')",
                wait_timeout_sec=float(kwargs.get("wait_timeout_sec", 120.0)),
                # First attempt: approach from pre_pose (30cm) in impedance, then descend to final (12cm).
                # Retries: direct_move keeps pre_pose == final_pose; stay in impedance.
                executor_enable_pre_stage=(attempt == 0),
                executor_use_position_for_pre_stage=False,
                executor_auto_configure_position=False,
                executor_auto_configure_impedance=True,
            )
            if res_exec.get("status") != "success":
                return {"status": "failed", "message": f"Execution failed: {res_exec.get('message')}"}
            
            # 3. Query Contact
            print("\n" + "="*40)
            print(f"[AutoVerify] Attempt {attempt+1}. Point0 Reached.")
            print("Please check contact quality (enter: 'good', 'poor', or 'none').")
            res_contact = self.contact_tool.execute(prompt="Contact quality? (good/poor/none)")
            quality = str(res_contact.get("contact_quality", "unknown"))
            print("="*40 + "\n")

            if quality == "good":
                rospy.loginfo("[AutoVerify] Contact GOOD. Total offset applied: %.4fm", total_offset)
                # Since we used 'modify_from_index(from_index=0)' in the loop,
                # the WHOLE path (including remaining points) has already been shifted.
                # No further action needed.
                return {
                    "status": "success",
                    "message": "contact_good",
                    "final_contact": "good",
                    "nl_observation": "Contact quality is GOOD. Point0 verification finished.",
                }

            elif quality in ("poor", "none"):
                rospy.loginfo("[AutoVerify] Contact %s. Adjusting down by %.4fm...", quality, abs(offset_step))
                
                # Modify YAML: Shift EVERYTHING from index 0 down.
                # This ensures the whole path follows the correction, so when Point0 is good, the whole path is good.
                res_mod = self.yaml_tool.execute(
                    action="modify_from_index",
                    from_index=0,
                    offset_normal=offset_step 
                    # Assuming normal is roughly 'down' into body. 
                    # If 'offset_z' (impedance z) is preferred, we need to support it.
                    # Usually offset_normal is safer for curved paths.
                )
                if res_mod.get("status") != "success":
                    return {"status": "failed", "message": "Modification failed"}

                # CRITICAL: persist the modification back to path_preview1.yaml on disk.
                # Otherwise, only the exported point0 YAML changes, while the subsequent full path execution
                # still uses the old ${RUSSAGENT_PATH_PREVIEW:-$HOME/.ros/russagent/path_preview1.yaml}.
                res_save = self.yaml_tool.execute(action="save", path_yaml=path_preview_yaml())
                if res_save.get("status") != "success":
                    return {"status": "failed", "message": f"Save failed after modification: {res_save.get('message')}"}
                
                total_offset += offset_step
                time.sleep(0.5)
            
            else:
                 # Unknown input, retry asking?
                 rospy.logwarn("[AutoVerify] Unknown quality '%s'. Retrying...", quality)

        return {
            "status": "failed",
            "message": "max_retries_reached",
            # Align with sim_nl_standalone Tool3 nl_observation (keep exact phrase)
            "nl_observation": "Point0 verification failed (sim): max retries reached.",
        }

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "offset_step_m": {"type": "number", "description": "Adjustment step per retry in meters (negative means down/deeper; default -0.005m)"},
                "max_retries": {"type": "integer", "description": "Maximum number of retry attempts (default 10)"},
                "enable_home_descent": {
                    "type": "boolean",
                    "description": "If true (default), before first point0 move: descend along base -Z from capture pose in position control, then switch to impedance.",
                },
                "home_descent_m": {
                    "type": "number",
                    "description": "Descent distance along base -Z in meters (default 0.02 = 2 cm).",
                },
                "base_frame": {"type": "string", "description": "Base frame for -Z descent (default iiwa_link_0)."},
                "home_descent_settle_sec": {"type": "number", "description": "Pause after descent before impedance switch (default 0.3s)."},
            }
        }
