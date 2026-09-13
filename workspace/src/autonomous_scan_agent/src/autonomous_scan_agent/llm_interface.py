#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from autonomous_scan_agent.russagent_paths import path_preview_yaml, workspace_root, asa_pkg_root, repo_root
from .tools.tool_4_path_execution import PathExecuteServerTool
from .tools.tool_5_reset_to_capture_pose import ResetToCapturePoseTool
from .tools.tool_1_trajectory_acquisition import AcquireTrajectoryTool
from .tools.tool_2_trajectory_projection import RiblineProjectionPreviewTool
from .tools.tool_0_operator_io import OperatorIOTool
from .tools.tool_3_contact_verification import Point0AutoVerifyTool
from .tools.tool_6_post_scan_adjustment import PostScanAdjustTool

# Internal tools (not exposed to LLM but imported for type checking or internal use if needed)
# (Tool17 uses these internally; they should not be exposed to the LLM tool list)
# from .tools.path_yaml_manager import PathYamlManagerTool
# from .tools.contact_query import ContactQueryTool

class ToolRegistry:
    def __init__(self, include_tools: list = None, exclude_tools: list = None):
        """
        include_tools: 仅注册这些 tool.name（None 表示全注册）
        exclude_tools: 黑名单（优先级高于 include）
        """
        self.tools = {}
        self._include = set(include_tools or [])
        self._exclude = set(exclude_tools or [])
        self._register_tools()

    def _register_tools(self):
        """
        关键点：**先过滤、再实例化**。
        """

        tool_specs = [
            # Tool0/1/2/3/4 (LLM runner)
            ("operator_io_tool", OperatorIOTool),
            ("acquire_trajectory_tool", AcquireTrajectoryTool),
            ("ribline_projection_tool", RiblineProjectionPreviewTool),
            # ("path_yaml_tool", PathYamlManagerTool), # Internal use by Tool17
            # ("path_execute_tool", PathExecuteTool), # Deprecated
            ("path_execute_server_tool", PathExecuteServerTool),
            # Dedicated operator interaction tools (safer schemas than operator_io_tool)
            # ("wait_ok_tool", WaitOkTool), # Integrated into Tool2
            # ("contact_query_tool", ContactQueryTool), # Integrated into Tool17
            ("point0_auto_verify_tool", Point0AutoVerifyTool),
            ("post_scan_adjust_tool", PostScanAdjustTool),
            # legacy /其他能力（按需启用）
            # ... removed deprecated tools ...
            ("reset_to_capture_pose_tool", ResetToCapturePoseTool),
        ]

        for name, cls in tool_specs:
            if name in self._exclude:
                continue
            if self._include and name not in self._include:
                continue
            # 仅在需要时才实例化，避免副作用
            self.register(cls())

    def register(self, tool):
        name = getattr(tool, "name", None)
        if not name:
            return
        if name in self._exclude:
            return
        if self._include and name not in self._include:
            return
        self.tools[name] = tool

    def get_tool(self, name):
        return self.tools.get(name)

    def get_all_definitions(self):
        return [tool.get_definition() for tool in self.tools.values()]

    def execute_tool(self, name, **kwargs):
        tool = self.get_tool(name)
        if not tool:
            return {"status": "error", "message": f"Tool {name} not found", "nl_observation": f"{name} execution failed: tool not found"}
        try:
            res = tool.execute(**kwargs)
            # Ensure result is a dict
            if not isinstance(res, dict):
                return {"status": "error", "message": "tool_return_not_dict", "nl_observation": f"{name} execution failed: return value is not dict"}
            
            # Inject Natural Language Observation if not present
            if "nl_observation" not in res or not str(res.get("nl_observation") or "").strip():
                status = str(res.get("status") or "").strip().lower()
                msg = str(res.get("message") or "").strip()
                
                # Default mapping for success cases
                _NL_BY_TOOL = {
                    "operator_io_tool": "Task information acquisition finished",
                    "acquire_trajectory_tool": "Trajectory acquisition finished",
                    "point0_auto_verify_tool": "Point0 verification and path correction finished",
                    "post_scan_adjust_tool": "Post-scan verification and optional adjustment finished",
                    "ribline_projection_tool": "Projection finished",
                    "path_execute_server_tool": "Trajectory execution finished",
                    "reset_to_capture_pose_tool": "Reset execution finished",
                }
                
                if status in ("error", "failed", "failure"):
                     res["nl_observation"] = f"{name} execution failed: {msg or status}"
                else:
                    if name == "operator_io_tool":
                        ans = str(res.get("answer") or "").strip()
                        interp = str(res.get("interpretation") or "").strip()
                        if ans:
                            res["nl_observation"] = f"Operator replied: '{ans}' ({interp})"
                        else:
                            res["nl_observation"] = "Instruction delivered. No input required."
                    elif name == "acquire_trajectory_tool":
                        traj_kind = str(res.get("traj_kind") or "").strip().lower()
                        if not traj_kind:
                            # For unified tool, prefer explicit organ if present.
                            traj_kind = str(kwargs.get("organ") or "").strip().lower() or "gallbladder"
                        if traj_kind == "kidney":
                            res["nl_observation"] = "Kidney trajectory acquired successfully."
                        elif traj_kind == "spine":
                            res["nl_observation"] = "Spine trajectory acquired successfully."
                        else:
                            res["nl_observation"] = "Gallbladder trajectory acquired successfully."
                    elif name == "ribline_projection_tool":
                        out_yaml = str(res.get("output_yaml") or path_preview_yaml())
                        res["nl_observation"] = f"Projection finished successfully. Path YAML is ready at {out_yaml}."
                    elif name == "point0_auto_verify_tool":
                        final_contact = str(res.get("final_contact") or "").strip().lower()
                        if final_contact == "good" or status == "success":
                            res["nl_observation"] = "Contact quality is GOOD. Point0 verification finished."
                        else:
                            res["nl_observation"] = f"Point0 verification finished. final_contact={final_contact or 'unknown'}."
                    elif name == "path_execute_server_tool":
                        path_yaml = str(res.get("path_yaml") or path_preview_yaml())
                        base = os.path.basename(path_yaml) if path_yaml else ""
                        if base == "path_point0.yaml":
                            # In the real pipeline, LLM typically should not call point0 YAML directly (Tool17 handles it).
                            res["nl_observation"] = f"Point0 execution completed. Executed: {base or path_yaml}."
                        else:
                            res["nl_observation"] = (
                                f"Trajectory execution completed (sim). Executed: {base or path_yaml}. "
                                "Scan segment executed. Scan result has not been verified yet. "
                                "Robot is at the end of the scan path (not at the capture pose). "
                                "If a one-time scan result verification step is available, do it before deciding the final reset."
                            )
                    else:
                        res["nl_observation"] = _NL_BY_TOOL.get(name, f"{name} execution finished")
            
            return res
        except Exception as e:
            return {"status": "error", "message": str(e), "nl_observation": f"{name} execution failed: {e}"}
