#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from autonomous_scan_agent.russagent_paths import path_preview_yaml, workspace_root, asa_pkg_root, repo_root
import sys
import subprocess
import time
import yaml
import rospy

from .tool_base import BaseTool
from ..reasoning_tts import speak_guidance_sync

# Resolve the project-owned trajectory projection script.
_HAMLYN_PROJECTION_SCRIPT = os.path.join(repo_root(), 'workspace', 'src', 'image_processing', 'scripts', 'path_stroke_preview_uv.py')


def _resolve_projection_python() -> str:
    """
    Pick a Python interpreter for the projection node. It needs the ROS bridge
    (PyKDL/tf2_geometry_msgs) AND, in DepthCloud mode, open3d.

    The agent runs under the `Russ_agent` conda environment, which
    has been set up with PyKDL/tf2 (system .so symlinked) PLUS open3d. So we prefer the
    CURRENT interpreter (sys.executable) first. The system python3 ships PyKDL/tf2 but
    has NO open3d, so it is only a last resort.

    The capability probe requires open3d as well, which makes system python3 be skipped
    automatically when it lacks open3d (avoids the "Open3D 未安装" runtime crash).

    Override with env ASA_PROJECTION_PYTHON if needed.
    """
    override = os.environ.get("ASA_PROJECTION_PYTHON", "").strip()
    candidates = [override] if override else []
    candidates += [sys.executable, "/usr/bin/python3", "/usr/bin/python3.8"]
    probe = "import PyKDL, tf2_geometry_msgs, open3d"
    seen = set()
    for py in candidates:
        if not py or py in seen or not os.path.exists(py):
            continue
        seen.add(py)
        try:
            subprocess.check_call(
                [py, "-c", probe],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return py
        except Exception:
            continue
    # Last resort: relax the probe to ROS-only (no open3d), preferring current interpreter.
    for py in candidates:
        if not py or not os.path.exists(py):
            continue
        try:
            subprocess.check_call(
                [py, "-c", "import PyKDL, tf2_geometry_msgs"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return py
        except Exception:
            continue
    # Absolute fallback: current interpreter.
    return sys.executable


def _yaml_has_frames(path: str) -> bool:
    try:
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        frames = data.get("frames")
        return isinstance(frames, list) and len(frames) > 0
    except Exception:
        return False


def _wait_for_output_yaml(path: str, timeout_sec: float) -> bool:
    t0 = time.time()
    while (time.time() - t0) < float(timeout_sec) and not rospy.is_shutdown():
        if _yaml_has_frames(path):
            return True
        rospy.sleep(0.2)
    return _yaml_has_frames(path)


def _wait_operator_trajectory_validation(out_yaml: str) -> None:
    print("\n" + "=" * 60, flush=True)
    print("3D SCANNING TRAJECTORY READY — OPERATOR VALIDATION", flush=True)
    print(f"Path YAML: {out_yaml}", flush=True)
    print("Please double-check the 3D trajectory in RViz (markers/poses).", flush=True)
    print("Press Enter to confirm and start robot scanning.", flush=True)
    print("=" * 60, flush=True)
    speak_guidance_sync(
        "Three D scanning trajectory is ready. Please check the trajectory in R viz. "
        "Press Enter to confirm and start robot scanning."
    )
    try:
        input()
    except Exception:
        pass
    rospy.loginfo("[Tool2] Operator confirmed 3D trajectory validation.")


class RiblineProjectionPreviewTool(BaseTool):
    """
    Tool2：启动 path_stroke_preview_uv.py（无GUI），订阅实时 depth+camera_info，
    自动读取 ribline_uv_npy 并生成 YAML，同时发布 RViz marker/pose（latch）。

    说明：
    - 本工具通过启动一个 ROS 节点子进程来完成（避免在同一进程内重复 rospy.init_node）。
    - 生成后该节点可保持运行（以便 RViz 随时显示 marker），也可以选择短暂运行后退出（当前默认保持）。
    """

    def __init__(self):
        super().__init__(
            name="ribline_projection_tool",
            description="Project ribline uv to 3D using live depth+camera_info, generate path_preview1.yaml, and publish RViz markers/poses.",
        )
        self.default_out_dir = os.path.expanduser(
            rospy.get_param("~ribline_out_dir", os.path.join(workspace_root(), 'input_images', 'rgbpair_latest'))
        )
        self.default_uv = os.path.expanduser(
            rospy.get_param(
                "~ribline_uv_default",
                os.path.join(self.default_out_dir, "ribline_uv_upper_shifted_fit.npy"),
            )
        )
        self.default_yaml = os.path.expanduser(rospy.get_param("~path_yaml_default", path_preview_yaml()))

    def _get_rostopic_list(self) -> list:
        """
        Best-effort query. If rostopic is unavailable or ROS master is down, return empty list.
        """
        try:
            out = subprocess.check_output(["rostopic", "list"], stderr=subprocess.STDOUT).decode("utf-8", errors="ignore")
            return [ln.strip() for ln in out.splitlines() if ln.strip().startswith("/")]
        except Exception:
            return []

    def _pick_existing_topic(self, candidates: list, topics: list) -> str:
        for t in candidates or []:
            if t in topics:
                return t
        return str(candidates[0]) if candidates else ""

    def execute(self, **kwargs) -> dict:
        uv_npy = os.path.expanduser(kwargs.get("ribline_uv_npy", self.default_uv))
        requested_output_yaml = kwargs.get("output_yaml", None)
        output_yaml = os.path.expanduser(requested_output_yaml) if isinstance(requested_output_yaml, str) else self.default_yaml
        # Force projection output YAML to a stable location used by Tool17/Tool4, to prevent LLM hallucinated paths
        # (e.g., "/path/to/path_preview1.yaml") and avoid mismatches between projection output and execution input.
        # The whole pipeline expects: ${RUSSAGENT_PATH_PREVIEW:-$HOME/.ros/russagent/path_preview1.yaml}
        if not isinstance(output_yaml, str) or (not output_yaml.startswith(os.path.expanduser("~/.ros/"))) or (os.path.basename(output_yaml) != "path_preview1.yaml"):
            output_yaml = self.default_yaml
        target_frame = str(kwargs.get("target_frame", rospy.get_param("~target_frame", "iiwa_link_0")))
        # If caller does not specify, choose a safe default based on current task_kind.
        # This reduces LLM decision burden and aligns with handbook: gallbladder requires breath-hold, kidney/spine not.
        if "instruct_and_wait" in kwargs:
            instruct_and_wait = bool(kwargs.get("instruct_and_wait", False))
        else:
            try:
                task_kind = str(rospy.get_param("/autonomous_scan_agent/task_kind", "") or "").strip().lower()
            except Exception:
                task_kind = ""
            instruct_and_wait = True if ("gallbladder" in task_kind) else False

        # Get traj_kind/organ for UV file selection (all organs use the same -90deg axis rotation in projection)
        traj_kind = str(kwargs.get("traj_kind", "") or kwargs.get("organ", "")).strip().lower()
        if not traj_kind:
            try:
                task_kind = str(rospy.get_param("/autonomous_scan_agent/task_kind", "") or "").strip().lower()
                if "spine" in task_kind:
                    traj_kind = "spine"
                elif "kidney" in task_kind:
                    traj_kind = "kidney"
                elif "gallbladder" in task_kind or "gb" in task_kind:
                    traj_kind = "gallbladder"
            except Exception:
                pass

        # Check UV existence BEFORE breath instruction to avoid unnecessary prompts.
        # If ribline_uv_npy was not explicitly provided, select the correct file based on traj_kind
        if not os.path.exists(uv_npy) or uv_npy == self.default_uv:
            # Select trajectory file based on traj_kind (matching tool1 output)
            if traj_kind == "spine":
                candidate = os.path.join(self.default_out_dir, "spine_curve_uv_upper.npy")
                if os.path.exists(candidate):
                    uv_npy = candidate
            elif traj_kind == "kidney":
                candidate = os.path.join(self.default_out_dir, "kidney_line_uv_upper_shifted.npy")
                if os.path.exists(candidate):
                    uv_npy = candidate
                else:
                    # fallback: stitched 坐标（若未做 upper 逆变换）
                    cand_st = os.path.join(self.default_out_dir, "kidney_line_uv_shifted.npy")
                    if os.path.exists(cand_st):
                        uv_npy = cand_st
            else:
                # Default to gallbladder trajectory files
                cand1 = os.path.join(self.default_out_dir, "ribline_uv_upper_shifted_fit.npy")
                cand2 = os.path.join(self.default_out_dir, "ribline_uv_upper_fit.npy")
                if os.path.exists(cand1):
                    uv_npy = cand1
                elif os.path.exists(cand2):
                    uv_npy = cand2

        # Final check: if file still doesn't exist, return error
        if not os.path.exists(uv_npy):
            return {
                "status": "failed",
                "message": f"ribline_uv_npy not found: {uv_npy} (traj_kind={traj_kind}). Re-run acquire_trajectory_tool for {traj_kind}.",
            }

        if instruct_and_wait:
            print("\n" + "="*60)
            print("INSTRUCTION: Please instruct the patient to take a deep breath and hold it.")
            print("Waiting for confirmation (type 'ok' and Enter)...")
            print("="*60)
            try:
                # Simple blocking input
                input("> ")
            except Exception:
                pass
            print("Breath hold confirmed. Generating projection...")

        # Topics: prefer Azure Kinect topics in your deployment (easy_handeye bringup uses azure_kinect_ros_driver).
        # Also protect against LLM passing empty/invalid strings.
        available_topics = self._get_rostopic_list()
        # Candidate order: Azure Kinect topics only by default.
        # (If you want RealSense later, pass explicit topics or override via ROS params.)
        rgb_candidates = ["/rgb/image_raw"]
        depth_candidates = ["/depth_to_rgb/image_raw"]
        caminfo_candidates = ["/rgb/camera_info"]

        rgb_topic_req = str(kwargs.get("rgb_topic", "")).strip()
        depth_topic_req = str(kwargs.get("depth_topic", "")).strip()
        caminfo_topic_req = str(kwargs.get("camera_info_topic", "")).strip()

        # Only accept requested topics if they actually exist (prevents `_camera_info_topic:=` or wrong driver).
        rgb_topic = rgb_topic_req if (rgb_topic_req and (rgb_topic_req in available_topics)) else "/rgb/image_raw"
        depth_topic = depth_topic_req if (depth_topic_req and (depth_topic_req in available_topics)) else "/depth_to_rgb/image_raw"
        camera_info_topic = caminfo_topic_req if (caminfo_topic_req and (caminfo_topic_req in available_topics)) else "/rgb/camera_info"

        auto_delay = float(kwargs.get("auto_generate_delay_sec", 4.0))
        no_gui = True

        # uv_npy existence already checked above.

        # CRITICAL: avoid stale preview YAML from previous runs.
        # Projection runs asynchronously; if output_yaml already exists, Tool17 may load the old file
        # before the new projection overwrites it, causing point0.yaml to be exported from stale frames.
        try:
            if isinstance(output_yaml, str) and os.path.exists(output_yaml):
                rospy.logwarn("[Tool2] removing stale output_yaml before projection: %s", output_yaml)
                os.remove(output_yaml)
        except Exception as e:
            rospy.logwarn("[Tool2] failed to remove stale output_yaml (ignored): %s", str(e))

        # Avoid duplicate node name conflict: path_stroke_preview_uv uses a fixed node name.
        # If an old node exists, kill it first so we get a clean start.
        try:
            nodes = subprocess.check_output(["rosnode", "list"]).decode().split()
            if "/path_stroke_preview_uv" in nodes:
                rospy.logwarn("[Tool2] found existing /path_stroke_preview_uv, killing it to avoid name conflict")
                subprocess.check_call(["rosnode", "kill", "/path_stroke_preview_uv"])
                rospy.sleep(0.3)
        except Exception:
            pass

        cmd = [
            _resolve_projection_python(),
            _HAMLYN_PROJECTION_SCRIPT,
            f"_ribline_uv_npy:={uv_npy}",
            f"_output_yaml:={output_yaml}",
            f"_target_frame:={target_frame}",
            f"_rgb_topic:={rgb_topic}",
            f"_depth_topic:={depth_topic}",
            f"_camera_info_topic:={camera_info_topic}",
            f"_use_depthcloud:=true",
            f"_no_gui:={'true' if no_gui else 'false'}",
            # Make RViz preview positions match the actual YAML poses (i.e., include approach/pre-approach offsets).
            f"_viz_no_offset:=false",
            f"_auto_generate_delay_sec:={auto_delay}",
        ]
        # Pass traj_kind so projection picks the correct UV file (axis rotation is unified at -90deg)
        if traj_kind:
            cmd.append(f"_traj_kind:={traj_kind}")
        if traj_kind == "kidney":
            cmd.append("_kidney_truncate_on_height_jump:=true")
            cmd.append("_kidney_height_jump_threshold_m:=0.03")
            kidney_side = str(kwargs.get("kidney_side", "") or "").strip().lower()
            if kidney_side not in ("left", "right"):
                try:
                    kidney_side = str(
                        rospy.get_param("/autonomous_scan_agent/latest_kidney_side", "") or ""
                    ).strip().lower()
                except Exception:
                    kidney_side = ""
            if kidney_side not in ("left", "right"):
                try:
                    task_kind = str(rospy.get_param("/autonomous_scan_agent/task_kind", "") or "").strip().lower()
                    if "left" in task_kind:
                        kidney_side = "left"
                    elif "right" in task_kind:
                        kidney_side = "right"
                except Exception:
                    kidney_side = ""
            if kidney_side in ("left", "right"):
                cmd.append(f"_kidney_side:={kidney_side}")
            if kidney_side == "left":
                cmd.append("_kidney_left_base_x_offset_m:=0.02")
                cmd.append("_kidney_left_base_y_offset_m:=0.015")
        if traj_kind == "spine":
            cmd.append("_spine_filter_outliers:=true")
            cmd.append("_spine_truncate_to_visible_half:=true")
            cmd.append("_spine_base_x_offset_m:=0.025")
            cmd.append("_spine_use_endpoint_line:=true")

        rospy.loginfo("[Tool2] spawning: %s", " ".join(cmd))
        try:
            # 让节点常驻（用于 RViz 预览）；不阻塞整个 agent：建议上层以异步/分进程方式运行 agent
            p = subprocess.Popen(cmd)
        except Exception as e:
            return {"status": "failed", "message": str(e)}

        yaml_ready_timeout = float(
            kwargs.get(
                "yaml_ready_timeout_sec",
                rospy.get_param("~yaml_ready_timeout_sec", float(os.environ.get("REAL_YAML_READY_TIMEOUT_SEC", "20.0"))),
            )
        )
        require_validation = bool(
            kwargs.get("require_trajectory_validation", rospy.get_param("~require_trajectory_validation", True))
        )

        if not _wait_for_output_yaml(output_yaml, yaml_ready_timeout):
            return {
                "status": "failed",
                "message": "projection_yaml_not_ready",
                "output_yaml": output_yaml,
                "nl_observation": (
                    f"Projection node started but YAML is not ready (frames empty): {output_yaml}. "
                    "Please retry ribline_projection_tool."
                ),
            }

        if require_validation:
            _wait_operator_trajectory_validation(output_yaml)

        return {
            "status": "success",
            "message": "projection ready; operator validated trajectory" if require_validation else "projection ready",
            "pid": p.pid,
            "ribline_uv_npy": uv_npy,
            "output_yaml": output_yaml,
            "target_frame": target_frame,
            "topics": {
                "rgb": rgb_topic,
                "depth": depth_topic,
                "camera_info": camera_info_topic,
            },
            "requested_output_yaml": requested_output_yaml,
            "trajectory_validated": bool(require_validation),
            "nl_observation": (
                f"Projection finished. Operator confirmed 3D trajectory at {output_yaml}. "
                "Proceed to point0 verification and scanning."
                if require_validation
                else f"Projection finished successfully. Path YAML is ready at {output_yaml}."
            ),
        }

    def _get_parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "ribline_uv_npy": {"type": "string"},
                "output_yaml": {"type": "string"},
                "target_frame": {"type": "string"},
                "instruct_and_wait": {"type": "boolean", "description": "If true, print breath hold instruction and wait for user 'ok' before projecting."},
                "rgb_topic": {"type": "string"},
                "depth_topic": {"type": "string"},
                "camera_info_topic": {"type": "string"},
                "auto_generate_delay_sec": {"type": "number"},
                "yaml_ready_timeout_sec": {"type": "number", "description": "Max seconds to wait for path_preview1.yaml frames after projection starts."},
                "require_trajectory_validation": {"type": "boolean", "description": "If true (default), block until operator presses Enter to confirm the 3D trajectory in RViz."},
                "traj_kind": {"type": "string", "description": "Trajectory kind (gallbladder/kidney/spine). Selects UV file; 3D pose uses unified -90deg rotation."},
                "organ": {"type": "string", "description": "Organ type (alternative to traj_kind). Selects UV file; 3D pose uses unified -90deg rotation."},
            },
        }


