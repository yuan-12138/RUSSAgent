#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from autonomous_scan_agent.russagent_paths import path_preview_yaml, workspace_root, asa_pkg_root, repo_root
import json
import subprocess
from typing import Any, Dict, Optional

import rospy
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .tool_base import BaseTool

_PENDING_EXECUTE_PARAM = "/autonomous_scan_agent/point0_pending_execute"
_PENDING_CONTACT_PARAM = "/autonomous_scan_agent/point0_pending_contact"
_PENDING_YAML_PARAM = "/autonomous_scan_agent/point0_pending_yaml"

_PATH_EXEC_NS = "/path_execute_server"
_EXECUTOR_PARAM_DEFAULTS = {
    "enable_pre_stage": False,
    "skip_pre_stage_for_multi_frame": True,
    "auto_configure_position": False,
    "auto_configure_impedance": True,
    "use_position_for_pre_stage": False,
    "revert_to_position_on_finish": False,
    "pre_stage_pause": 0.0,
    "point0_approach_enabled": False,
    "point0_approach_z_lift_m": 0.08,
    "point0_contact_hold_sec": 4.0,
    "point0_contact_z_tol": 0.03,
    "ignore_z_error": True,
    "error_position_tolerance_z": 1e9,
}


def _set_guard(pending_execute: Optional[bool] = None, pending_contact: Optional[bool] = None, pending_yaml: Optional[str] = None) -> None:
    try:
        if pending_execute is not None:
            rospy.set_param(_PENDING_EXECUTE_PARAM, bool(pending_execute))
        if pending_contact is not None:
            rospy.set_param(_PENDING_CONTACT_PARAM, bool(pending_contact))
        if pending_yaml is not None:
            rospy.set_param(_PENDING_YAML_PARAM, str(pending_yaml))
    except Exception:
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


class PathExecuteServerTool(BaseTool):
    """
    新 Tool4（常驻执行器版）：
    - 若 path_execute_server 未启动：可选自动启动
    - 发布 YAML 到 /path_execute_server/execute_yaml
    - 调用 /path_execute_server/wait 等待执行结束并返回结果

    保留旧 Tool4（path_execute_tool，rosrun path_execute.py）作为备用，不删除不修改。
    """

    def __init__(self):
        super().__init__(
            name="path_execute_server_tool",
            description="Execute path YAML via persistent executor node (image_processing/path_execute_server.py) to reduce startup latency.",
        )
        self._pub: Optional[rospy.Publisher] = None

    def execute(self, path_id: str = "", **kwargs) -> Dict[str, Any]:
        """
        Keep Yiping-style control flow (publish YAML to /path_execute_server and wait),
        but harden path handling for our current pipeline:
        - default full-path yaml: ~/.ros/path_preview1.yaml
        - map relative names to ~/.ros
        NOTE: we intentionally do NOT support the old name path_preview.yaml anymore.
        """
        if not isinstance(path_id, str) or not path_id.strip():
            path_id = path_preview_yaml()

        path_yaml = os.path.expanduser(kwargs.get("path_yaml", os.path.expanduser(path_id)))
        # Normalize common relative YAML names to ~/.ros (LLM often omits absolute path).
        if isinstance(path_yaml, str) and (not os.path.isabs(path_yaml)):
            base = os.path.basename(path_yaml)
            if base == "path_preview.yaml":
                return {
                    "status": "failed",
                    "message": "deprecated_path_name",
                    "hint": "Use ${RUSSAGENT_PATH_PREVIEW:-$HOME/.ros/russagent/path_preview1.yaml} (or path_preview1.yaml). Old name path_preview.yaml is not supported.",
                }
            if base in ("path_preview1.yaml", "path_point0.yaml"):
                path_yaml = os.path.join(os.path.dirname(path_preview_yaml()), base)
        # Reject old absolute name as well
        if isinstance(path_yaml, str) and os.path.basename(path_yaml) == "path_preview.yaml":
            return {
                "status": "failed",
                "message": "deprecated_path_name",
                "hint": "Use ${RUSSAGENT_PATH_PREVIEW:-$HOME/.ros/russagent/path_preview1.yaml}. Old name path_preview.yaml is not supported.",
                "path_yaml": path_yaml,
            }
        auto_start = bool(kwargs.get("auto_start", False))
        wait_timeout = float(kwargs.get("wait_timeout_sec", 300.0))
        breath_instruction_on_impedance = bool(kwargs.get("breath_instruction_on_impedance", False))

        execute_topic = str(kwargs.get("execute_topic", "/path_execute_server/execute_yaml"))
        wait_service = str(kwargs.get("wait_service", "/path_execute_server/wait"))
        state_topic = str(kwargs.get("state_topic", "/path_execute_server/state"))
        start_wait_sec = float(kwargs.get("start_wait_sec", 8.0))
        # spawn 参数：默认不切回 position，但确保 impedance（避免“上抬再下压”，同时保证命令能被控制器执行）
        spawn_auto_configure_position = bool(kwargs.get("spawn_auto_configure_position", False))
        spawn_auto_configure_impedance = bool(kwargs.get("spawn_auto_configure_impedance", True))
        spawn_pre_stage_pause = float(kwargs.get("spawn_pre_stage_pause", 0.0))
        spawn_enable_pre_stage = bool(kwargs.get("spawn_enable_pre_stage", False))
        spawn_cancel_on_new = bool(kwargs.get("spawn_cancel_on_new", False))

        # INTERNAL override (not meant for LLM):
        # Force full-path execution to do a pre-stage in POSITION_CONTROL -> then IMPEDANCE.
        # This is useful for post-scan adjustment re-execution where we want a stable approach from pre_pose.
        force_pre_stage = bool(kwargs.get("force_pre_stage", False))

        def _get_param(name: str, default: Any) -> Any:
            try:
                return rospy.get_param(f"{_PATH_EXEC_NS}/{name}", default)
            except Exception:
                return default

        def _set_param(name: str, value: Any) -> None:
            try:
                rospy.set_param(f"{_PATH_EXEC_NS}/{name}", value)
            except Exception:
                pass

        # Best-effort: temporarily override executor params and restore after wait returns.
        old_params: Optional[Dict[str, Any]] = None

        def _maybe_save_executor_overrides(overrides: Dict[str, Any]) -> None:
            nonlocal old_params
            if not overrides:
                return
            if old_params is None:
                old_params = {}
            for key, value in overrides.items():
                if key not in old_params:
                    old_params[key] = _get_param(key, _EXECUTOR_PARAM_DEFAULTS.get(key))
                _set_param(key, value)

        executor_overrides: Dict[str, Any] = {}
        if kwargs.get("executor_enable_pre_stage") is not None:
            executor_overrides["enable_pre_stage"] = bool(kwargs.get("executor_enable_pre_stage"))
        if kwargs.get("executor_use_position_for_pre_stage") is not None:
            executor_overrides["use_position_for_pre_stage"] = bool(kwargs.get("executor_use_position_for_pre_stage"))
        if kwargs.get("executor_auto_configure_position") is not None:
            executor_overrides["auto_configure_position"] = bool(kwargs.get("executor_auto_configure_position"))
        if kwargs.get("executor_auto_configure_impedance") is not None:
            executor_overrides["auto_configure_impedance"] = bool(kwargs.get("executor_auto_configure_impedance"))
        if kwargs.get("executor_revert_to_position_on_finish") is not None:
            executor_overrides["revert_to_position_on_finish"] = bool(
                kwargs.get("executor_revert_to_position_on_finish")
            )
        if kwargs.get("executor_point0_approach_enabled") is not None:
            executor_overrides["point0_approach_enabled"] = bool(kwargs.get("executor_point0_approach_enabled"))
        if kwargs.get("executor_point0_approach_z_lift_m") is not None:
            executor_overrides["point0_approach_z_lift_m"] = float(kwargs.get("executor_point0_approach_z_lift_m"))
        if kwargs.get("executor_point0_contact_hold_sec") is not None:
            executor_overrides["point0_contact_hold_sec"] = float(kwargs.get("executor_point0_contact_hold_sec"))
        if kwargs.get("executor_point0_contact_z_tol") is not None:
            executor_overrides["point0_contact_z_tol"] = float(kwargs.get("executor_point0_contact_z_tol"))
        if bool(kwargs.get("executor_point0_approach_enabled")):
            z_tol = float(
                kwargs.get("executor_point0_contact_z_tol")
                or _EXECUTOR_PARAM_DEFAULTS.get("point0_contact_z_tol", 0.03)
            )
            executor_overrides["ignore_z_error"] = False
            executor_overrides["error_position_tolerance_z"] = z_tol
        if executor_overrides:
            _maybe_save_executor_overrides(executor_overrides)
            rospy.loginfo("[path_execute_server_tool] executor overrides for this run: %s", executor_overrides)

        if force_pre_stage:
            _maybe_save_executor_overrides({
                "enable_pre_stage": True,
                "skip_pre_stage_for_multi_frame": False,
                "auto_configure_position": False,
                "auto_configure_impedance": True,
                "use_position_for_pre_stage": False,
                "pre_stage_pause": 0.0,
            })
            rospy.logwarn(
                "[path_execute_server_tool] force_pre_stage enabled (impedance pre-approach only; will restore params after)."
            )

        # Tool-level guardrail (v3-style):
        # - After executing point0, contact must be queried before executing point0 again.
        # This prevents repeated execute spam and guides the LLM back to contact_query_tool.
        is_point0 = isinstance(path_yaml, str) and path_yaml.endswith("path_point0.yaml")
        if is_point0:
            g = _get_guard()
            if g.get("pending_contact") is True:
                msg = "must_query_contact_next"
                hint = "Point0 was executed. Next step MUST be contact_query_tool before executing again."
                rospy.logwarn("[path_execute_server_tool] %s: %s", msg, hint)
                return {
                    "status": "failed",
                    "message": msg,
                    "hint": hint,
                    "path_yaml": path_yaml,
                }

        rospy.loginfo("[PathExecuteServerTool] request execute: %s", path_yaml)

        if auto_start:
            try:
                rospy.wait_for_service(wait_service, timeout=0.2)
            except Exception:
                cmd = [
                    "rosrun",
                    "image_processing",
                    "path_execute_server.py",
                    f"_auto_configure_position:={'true' if spawn_auto_configure_position else 'false'}",
                    f"_auto_configure_impedance:={'true' if spawn_auto_configure_impedance else 'false'}",
                    f"_pre_stage_pause:={spawn_pre_stage_pause}",
                    f"_enable_pre_stage:={'true' if spawn_enable_pre_stage else 'false'}",
                    f"_cancel_on_new:={'true' if spawn_cancel_on_new else 'false'}",
                    "_revert_to_position_on_finish:=false",
                    "_use_position_for_pre_stage:=false",
                    # 默认误差/推进阈值：xy=2cm，忽略 z（避免 residual 卡在误差判定）
                    f"_ignore_z_error:=true",
                    f"_error_position_tolerance_xy:=0.02",
                    f"_lookahead_distance_xy:=0.02",
                ]
                rospy.logwarn("[PathExecuteServerTool] server not found, spawning: %s", " ".join(cmd))
                try:
                    subprocess.Popen(cmd)
                except Exception as e:
                    return {"status": "failed", "message": f"spawn server failed: {e}", "path_yaml": path_yaml}

        # wait server ready
        try:
            rospy.loginfo("[PathExecuteServerTool] waiting service: %s", wait_service)
            rospy.wait_for_service(wait_service, timeout=20.0)
        except Exception as e:
            return {"status": "failed", "message": f"wait service not available: {e}", "path_yaml": path_yaml}

        if self._pub is None:
            # 这里用 latch=True：避免发布瞬间还没连上导致消息丢失（会导致上层误以为执行完成但机器人不动）
            self._pub = rospy.Publisher(execute_topic, String, queue_size=1, latch=True)
            rospy.sleep(0.1)

        if not os.path.exists(path_yaml):
            return {"status": "failed", "message": f"path_yaml not found: {path_yaml}", "path_yaml": path_yaml}

        # 等待至少一个连接，降低“消息丢失”概率
        t0 = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            if self._pub.get_num_connections() > 0:
                break
            if rospy.Time.now().to_sec() - t0 > 2.0:
                break
            rospy.sleep(0.05)

        # In real execution, the breath instruction is printed BEFORE publishing the execute request.
        if breath_instruction_on_impedance:
            rospy.loginfo("[PathExecuteServerTool] Instructing patient to take a deep breath and hold before contact during scanning.")
            print("Deep breath and hold. We are about to begin scanning now.")

        # publish execute request（latch=True + 等连接，避免丢消息；不要重复 publish，避免 server cancel_on_new）
        exec_payload = path_yaml
        if executor_overrides:
            exec_payload = json.dumps(
                {"yaml": path_yaml, "overrides": executor_overrides},
                separators=(",", ":"),
            )
        rospy.loginfo("[PathExecuteServerTool] publish to %s (connections=%d)", execute_topic, self._pub.get_num_connections())
        self._pub.publish(String(data=exec_payload))

        # 等待 server 进入 running（握手），否则 wait 可能直接返回 idle 导致误判
        started = False
        t1 = rospy.Time.now().to_sec()
        last_state = ""
        while not rospy.is_shutdown():
            if rospy.Time.now().to_sec() - t1 > start_wait_sec:
                break
            try:
                msg = rospy.wait_for_message(state_topic, String, timeout=0.5)
                last_state = (msg.data or "").strip()
                if last_state.startswith("running"):
                    started = True
                    break
                # 若立刻 completed/failed/cancelled 也算“有响应”
                if last_state.startswith("completed") or last_state.startswith("failed") or last_state.startswith("cancelled"):
                    started = True
                    break
            except Exception:
                pass

        if not started:
            return {
                "status": "failed",
                "message": f"executor did not start (state={last_state!r}). If server just spawned, retry.",
                "path_yaml": path_yaml,
            }
        rospy.loginfo("[PathExecuteServerTool] executor started (state=%s), waiting completion...", last_state)

        # wait completion
        try:
            wait = rospy.ServiceProxy(wait_service, Trigger)
            # Trigger 无参数；wait 会阻塞直到 server 不在 running
            # 注意：这是“逻辑超时”而非 socket 超时；如果需要严格超时，建议升级为 actionlib
            start_wall = rospy.Time.now().to_sec()
            while not rospy.is_shutdown():
                if rospy.Time.now().to_sec() - start_wall > wait_timeout:
                    return {"status": "failed", "message": "wait timeout", "path_yaml": path_yaml}
                resp = wait()
                rospy.loginfo("[PathExecuteServerTool] wait returned: success=%s message=%s", resp.success, resp.message)
                out = {
                    "status": "success" if resp.success else "failed",
                    "message": resp.message,
                    "path_yaml": path_yaml,
                }
                # Provide a sim-style NL observation to guide the next tool decision.
                # (Exact phrasing alignment with sim_nl_standalone is intentional for policy transfer.)
                if resp.success:
                    out["nl_observation"] = (
                        f"Trajectory execution completed (sim). Executed: {os.path.basename(path_yaml)}. "
                        "Scan segment executed. Scan result has not been verified yet. "
                        "Robot is at the end of the scan path (not at the capture pose). "
                        "If a one-time scan result verification step is available, do it before deciding the final reset."
                    )
                if is_point0 and resp.success:
                    # Execute done -> require contact next; clear pending_execute.
                    _set_guard(pending_execute=False, pending_contact=True, pending_yaml=path_yaml)
                return out
        except Exception as e:
            return {"status": "failed", "message": f"wait call failed: {e}", "path_yaml": path_yaml}
        finally:
            if old_params is not None:
                for key, value in old_params.items():
                    _set_param(key, value)
                rospy.logwarn("[path_execute_server_tool] restored executor params.")

        return {"status": "failed", "message": "wait timeout", "path_yaml": path_yaml}

    def _get_parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "path_id": {"type": "string"},
                "path_yaml": {"type": "string", "description": "YAML path to execute (defaults to path_id)."},
                "auto_start": {"type": "boolean"},
                "wait_timeout_sec": {"type": "number"},
                "execute_topic": {"type": "string"},
                "wait_service": {"type": "string"},
                "state_topic": {"type": "string"},
                "start_wait_sec": {"type": "number"},
                "spawn_auto_configure_position": {"type": "boolean"},
                "spawn_auto_configure_impedance": {"type": "boolean"},
                "spawn_pre_stage_pause": {"type": "number"},
                "spawn_enable_pre_stage": {"type": "boolean"},
                "spawn_cancel_on_new": {"type": "boolean"},
                "executor_enable_pre_stage": {"type": "boolean"},
                "executor_use_position_for_pre_stage": {"type": "boolean"},
                "executor_auto_configure_position": {"type": "boolean"},
                "executor_auto_configure_impedance": {"type": "boolean"},
                "executor_revert_to_position_on_finish": {"type": "boolean"},
                "executor_point0_approach_enabled": {"type": "boolean"},
                "executor_point0_approach_z_lift_m": {"type": "number"},
                "executor_point0_contact_hold_sec": {"type": "number"},
                "executor_point0_contact_z_tol": {"type": "number"},
                "breath_instruction_on_impedance": {"type": "boolean"},
            },
            "required": ["path_id"],
        }


