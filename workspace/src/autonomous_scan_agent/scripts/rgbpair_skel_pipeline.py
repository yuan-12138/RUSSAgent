#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rgbpair_skel_pipeline.py

一键串联流程：
1) 执行两个关节位姿（iiwa action），分别抓拍两张 RGB(+可选 CameraInfo)，保存到 out_dir
2) 对两张图做旋转/拼接/letterbox，生成 stitched_upright 图
3) 调用 cliff_skel_trajectory.py 跑 CLIFF+SKEL，输出 ribline + upper_only 结果

目标：把你目前的手动三条命令简化为一条 rosrun 命令。
"""

import os
from autonomous_scan_agent.russagent_paths import path_preview_yaml, workspace_root, asa_pkg_root, repo_root
import sys
import glob
import yaml
import rospy
import datetime
import subprocess
import rosgraph
import select

import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
import tf2_ros

try:
    import open3d as o3d  # type: ignore
except Exception:
    o3d = None

import shutil

try:
    import actionlib
    from iiwa_msgs.msg import JointPosition, ControlMode
    from iiwa_msgs.msg import MoveToJointPositionAction, MoveToJointPositionGoal
    from iiwa_msgs.srv import ConfigureControlMode, ConfigureControlModeRequest
    _HAVE_IIWA_ACTION = True
    _HAVE_IIWA_CTRL = True
except Exception:
    actionlib = None
    JointPosition = None
    ControlMode = None
    MoveToJointPositionAction = None
    MoveToJointPositionGoal = None
    ConfigureControlMode = None
    ConfigureControlModeRequest = None
    _HAVE_IIWA_ACTION = False
    _HAVE_IIWA_CTRL = False


try:
    import moveit_commander
    from moveit_commander import MoveGroupCommander, RobotCommander
    _HAVE_MOVEIT = True
except Exception:
    moveit_commander = None
    MoveGroupCommander = None
    RobotCommander = None
    _HAVE_MOVEIT = False


DEFAULT_YAML = os.path.expanduser("~/.ros/autonomous_scan_agent/capture_poses.yaml")
DEFAULT_OUT_ROOT = os.path.join(workspace_root(), 'input_images')
DEFAULT_FIXED_OUT_DIR = os.path.join(workspace_root(), 'input_images', 'rgbpair_latest')
DEFAULT_JOINT_NAMES = [
    "iiwa_joint_1",
    "iiwa_joint_2",
    "iiwa_joint_3",
    "iiwa_joint_4",
    "iiwa_joint_5",
    "iiwa_joint_6",
    "iiwa_joint_7",
]


def skippable_delay(sec: float, *, label: str = "rgbpair_skel_pipeline", skippable: bool = True) -> None:
    """Wait up to `sec` seconds; press Enter to skip immediately (TTY only)."""
    sec = float(sec)
    if sec <= 0:
        return
    if skippable:
        print(
            f"\n[{label}] There will now be a {sec:.0f}-second waiting period "
            f"before the robot moves to capture poses.\n"
            f"Press Enter to skip and continue immediately.\n",
            flush=True,
        )
        rospy.loginfo("[%s] skippable delay %.1fs (Enter to skip)", label, sec)
        try:
            if sys.stdin.isatty():
                ready, _, _ = select.select([sys.stdin], [], [], sec)
                if ready:
                    sys.stdin.readline()
                    rospy.loginfo("[%s] delay skipped by user", label)
                    return
                rospy.loginfo("[%s] delay completed (%.1fs)", label, sec)
                return
        except Exception as exc:
            rospy.logwarn("[%s] skippable delay unavailable, falling back to sleep: %s", label, exc)
    rospy.loginfo("[%s] waiting %.1fs ...", label, sec)
    rospy.sleep(sec)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def reorder(joint_names_order, joints_dict):
    missing = [j for j in joint_names_order if j not in joints_dict]
    if missing:
        raise ValueError("YAML 缺少关节: {}".format(missing))
    return [float(joints_dict[j]) for j in joint_names_order]


def _max_abs_joint_error(cur, target):
    if not cur or not target or len(cur) != len(target):
        return float("inf")
    return max(abs(float(a) - float(b)) for a, b in zip(cur, target))


def wait_until_reached(get_current_joints_fn, target_joints, joint_tol=0.01, timeout=30.0, poll_hz=25.0):
    t0 = rospy.Time.now()
    rate = rospy.Rate(poll_hz)
    last_err = None
    while not rospy.is_shutdown():
        cur = get_current_joints_fn()
        err = _max_abs_joint_error(cur, target_joints)
        last_err = err
        if err <= float(joint_tol):
            return True, err
        if (rospy.Time.now() - t0).to_sec() > float(timeout):
            return False, last_err
        rate.sleep()
    return False, last_err


def ensure_position_control(ns: str, *, required: bool = True) -> bool:
    """Joint capture moves require position control (not cartesian impedance from prior scan)."""
    if not _HAVE_IIWA_CTRL or ConfigureControlMode is None or ControlMode is None:
        msg = "[rgbpair] ConfigureControlMode unavailable"
        if required:
            raise RuntimeError(msg)
        rospy.logwarn(msg)
        return False
    ns_clean = str(ns or "/iiwa").strip("/")
    srv_name = "/{}/configuration/ConfigureControlMode".format(ns_clean) if ns_clean else "/iiwa/configuration/ConfigureControlMode"
    try:
        rospy.wait_for_service(srv_name, timeout=5.0)
        cli = rospy.ServiceProxy(srv_name, ConfigureControlMode)
        req = ConfigureControlModeRequest()
        req.control_mode = ControlMode.POSITION_CONTROL
        resp = cli(req)
        if getattr(resp, "success", False):
            rospy.loginfo("[rgbpair] switched to position control")
            return True
        err = getattr(resp, "error", "unknown")
        if required:
            raise RuntimeError("ConfigureControlMode rejected: {}".format(err))
        rospy.logwarn("[rgbpair] ConfigureControlMode rejected (ignored): %s", err)
        return False
    except Exception as exc:
        if required:
            raise RuntimeError("failed to switch to position control: {}".format(exc))
        rospy.logwarn("[rgbpair] switch to position control failed (ignored): %s", str(exc))
        return False


class _MoveItCaptureHelper(object):
    def __init__(self, ns: str, group: str, vel_scale: float, acc_scale: float):
        if not _HAVE_MOVEIT:
            raise RuntimeError("moveit_commander not available")
        moveit_commander.roscpp_initialize([])
        ns_clean = str(ns or "").strip("/")
        candidates = [ns_clean, "", "iiwa"]
        seen = set()
        desc_key = None
        chosen_ns = ""
        for cand in candidates:
            if cand in seen:
                continue
            seen.add(cand)
            prefix = ("/" + cand) if cand else ""
            key = (prefix + "/robot_description") if prefix else "/robot_description"
            if rospy.has_param(key):
                desc_key = key
                chosen_ns = cand
                break
        if not desc_key:
            raise RuntimeError("MoveIt robot_description not found")
        self._robot = RobotCommander(robot_description=desc_key.lstrip("/"))
        group_name = str(group or "manipulator")
        available = list(self._robot.get_group_names() or [])
        if available and group_name not in available:
            group_name = "manipulator" if "manipulator" in available else available[0]
        self._group = MoveGroupCommander(group_name, robot_description=desc_key.lstrip("/"), ns=chosen_ns)
        self._group.set_max_velocity_scaling_factor(float(vel_scale))
        self._group.set_max_acceleration_scaling_factor(float(acc_scale))
        self._joint_order = list(self._group.get_active_joints())

    def goto_joints(self, joints_dict, name, joint_tol, reach_timeout, settle_sec, get_current_joints_fn):
        target = reorder(self._joint_order, joints_dict)
        err0 = _max_abs_joint_error(get_current_joints_fn(), target)
        rospy.loginfo("[%s] MoveIt: max|Δjoint|=%.4f rad (tol=%.4f)", name, err0, joint_tol)
        if err0 <= float(joint_tol):
            rospy.loginfo("[%s] already at target joints; skip motion", name)
            rospy.sleep(settle_sec)
            return name
        self._group.set_joint_value_target(target)
        plan = self._group.plan()
        traj = plan if hasattr(plan, "joint_trajectory") else (plan[1] if isinstance(plan, (list, tuple)) and len(plan) > 1 else None)
        if not traj or not hasattr(traj, "joint_trajectory") or len(traj.joint_trajectory.points) < 1:
            raise RuntimeError("[{}] MoveIt plan failed".format(name))
        ok = self._group.execute(traj, wait=True)
        self._group.stop()
        self._group.clear_pose_targets()
        if not ok:
            raise RuntimeError("[{}] MoveIt execute failed".format(name))
        reached, err = wait_until_reached(get_current_joints_fn, target, joint_tol=joint_tol, timeout=reach_timeout)
        if not reached:
            raise RuntimeError("[{}] MoveIt wait timeout: max|Δjoint|={:.4f} rad".format(name, err))
        rospy.loginfo("[%s] MoveIt 到位，等待稳定 %.2fs", name, settle_sec)
        rospy.sleep(settle_sec)
        return name


def caminfo_to_dict(msg: CameraInfo):
    return {
        "header": {"frame_id": msg.header.frame_id, "stamp": {"secs": msg.header.stamp.secs, "nsecs": msg.header.stamp.nsecs}},
        "height": int(msg.height),
        "width": int(msg.width),
        "distortion_model": str(msg.distortion_model),
        "D": [float(x) for x in msg.D],
        "K": [float(x) for x in msg.K],
        "R": [float(x) for x in msg.R],
        "P": [float(x) for x in msg.P],
        "binning_x": int(msg.binning_x),
        "binning_y": int(msg.binning_y),
        "roi": {
            "x_offset": int(msg.roi.x_offset),
            "y_offset": int(msg.roi.y_offset),
            "height": int(msg.roi.height),
            "width": int(msg.roi.width),
            "do_rectify": bool(msg.roi.do_rectify),
        },
    }


def caminfo_to_intrinsics_yaml(msg: CameraInfo) -> dict:
    fx = float(msg.K[0])
    fy = float(msg.K[4])
    cx = float(msg.K[2])
    cy = float(msg.K[5])
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": int(msg.width),
        "height": int(msg.height),
    }


def depth_msg_to_meters(depth_msg: Image, bridge: CvBridge) -> np.ndarray:
    depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
    if depth.dtype == np.uint16:
        depth = depth.astype(np.float32) / 1000.0
    else:
        depth = depth.astype(np.float32)
    return depth


def depth_to_cloud(depth_m: np.ndarray, intr: dict) -> np.ndarray:
    fx = float(intr["fx"])
    fy = float(intr.get("fy", fx))
    cx = float(intr["cx"])
    cy = float(intr["cy"])
    h, w = depth_m.shape[:2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth_m.astype(np.float32)
    valid = (Z > 0.05) & (Z < 3.0) & np.isfinite(Z)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    pts = np.stack((X[valid], Y[valid], Z[valid]), axis=1).astype(np.float32)
    return pts


def apply_tf(points: np.ndarray, t_xyz: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in q_xyzw]
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    out = (pts @ R.T) + np.asarray(t_xyz, dtype=np.float64).reshape(1, 3)
    return out.astype(np.float32)

def capture_and_save(rgb_topic, caminfo_topic, out_png, out_caminfo_yaml=None, timeout=10.0):
    bridge = CvBridge()
    rospy.loginfo("等待 RGB: %s (timeout=%.1fs)", rgb_topic, timeout)
    img_msg = rospy.wait_for_message(rgb_topic, Image, timeout=timeout)
    cv_bgr = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
    ensure_dir(os.path.dirname(out_png))
    ok = cv2.imwrite(out_png, cv_bgr)
    if not ok:
        raise RuntimeError("cv2.imwrite 失败: {}".format(out_png))
    rospy.loginfo("已保存: %s", out_png)

    if out_caminfo_yaml:
        rospy.loginfo("等待 CameraInfo: %s (timeout=%.1fs)", caminfo_topic, timeout)
        ci_msg = rospy.wait_for_message(caminfo_topic, CameraInfo, timeout=timeout)
        with open(out_caminfo_yaml, "w") as f:
            yaml.safe_dump(caminfo_to_dict(ci_msg), f, sort_keys=False)
        rospy.loginfo("已保存: %s", out_caminfo_yaml)


def _maybe_rotate(img, rotate: str):
    rotate = (rotate or "none").lower().strip()
    if rotate == "none":
        return img
    if rotate in ("cw", "90cw", "90"):
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rotate in ("ccw", "90ccw", "-90"):
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotate in ("180", "flip"):
        return cv2.rotate(img, cv2.ROTATE_180)
    raise ValueError("unknown rotate: {}".format(rotate))


def _resize_b_to_match(img_a, img_b, mode: str):
    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]
    mode = mode.lower().strip()
    if mode == "horizontal":
        if ha == hb:
            return img_a, img_b
        new_wb = int(round(wb * (float(ha) / float(hb))))
        img_b2 = cv2.resize(img_b, (new_wb, ha), interpolation=cv2.INTER_AREA)
        return img_a, img_b2
    if mode == "vertical":
        if wa == wb:
            return img_a, img_b
        new_hb = int(round(hb * (float(wa) / float(wb))))
        img_b2 = cv2.resize(img_b, (wa, new_hb), interpolation=cv2.INTER_AREA)
        return img_a, img_b2
    raise ValueError("unknown mode: {}".format(mode))


def stitch_two_images(a_path, b_path, out_path, rotate_a="cw", rotate_b="cw",
                      mode="vertical", order="a_then_b", overlap_ratio=0.55,
                      letterbox_w=720, letterbox_h=1280):
    img_a = cv2.imread(a_path)
    img_b = cv2.imread(b_path)
    if img_a is None:
        raise RuntimeError("读取失败: {}".format(a_path))
    if img_b is None:
        raise RuntimeError("读取失败: {}".format(b_path))

    img_a = _maybe_rotate(img_a, rotate_a)
    img_b = _maybe_rotate(img_b, rotate_b)
    if order == "b_then_a":
        img_a, img_b = img_b, img_a

    img_a, img_b = _resize_b_to_match(img_a, img_b, mode)

    r = float(overlap_ratio)
    r = max(0.0, min(0.95, r))
    if mode == "vertical":
        ov = int(round(img_b.shape[0] * r))
        ov = max(0, min(img_b.shape[0] - 1, ov))
        img_b = img_b[ov:, :, :]
        stitched = cv2.vconcat([img_a, img_b])
    else:
        ov = int(round(img_b.shape[1] * r))
        ov = max(0, min(img_b.shape[1] - 1, ov))
        img_b = img_b[:, ov:, :]
        stitched = cv2.hconcat([img_a, img_b])

    # letterbox
    out_w = int(letterbox_w)
    out_h = int(letterbox_h)
    if out_w > 0 and out_h > 0:
        h, w = stitched.shape[:2]
        s = min(float(out_w) / float(w), float(out_h) / float(h))
        new_w = max(1, int(round(w * s)))
        new_h = max(1, int(round(h * s)))
        resized = cv2.resize(stitched, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = (128 * (np.ones((out_h, out_w, 3), dtype=resized.dtype)))
        pad_x = (out_w - new_w) // 2
        pad_y = (out_h - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w, :] = resized
        stitched = canvas

    ensure_dir(os.path.dirname(out_path))
    ok = cv2.imwrite(out_path, stitched)
    if not ok:
        raise RuntimeError("保存拼接图失败: {}".format(out_path))
    return out_path


def run_demo_with_skel(
    out_dir: str,
    stitched_path: str,
    upper_img: str,
    lower_img: str,
    rib_shift_px: float,
    stitch_rotate: str,
    stitch_mode: str,
    stitch_overlap_ratio: float,
    stitch_letterbox_w: int,
    stitch_letterbox_h: int,
    traj_kind: str = "gallbladder",
    kidney_side: str = "right",
    kidney_start_mode: str = "thorax_spine_lowest",
):
    """
    Run CLIFF+SKEL and export task-specific UV trajectories into out_dir.

    traj_kind:
    - gallbladder: costal margin ribline (cliff_skel_trajectory.py) -> ribline_uv_upper_shifted_fit.npy
    - kidney:      kidney offset line / kidney traj (kidney_spine_geometry.py) -> kidney_line_uv_shifted.npy (and kidney_traj_*_uv.npy)
    - spine:       spine centerline curve (kidney_spine_geometry.py) -> spine_curve_uv.npy
    """
    # IMPORTANT: keep CLiFF+SKEL identical for all tasks.
    # Only trajectory extraction differs (handled inside cliff_skel_trajectory.py via --traj_kind).
    kind = (traj_kind or "gallbladder").strip().lower()
    demo_py = os.path.join(asa_pkg_root(), 'src', 'autonomous_scan_agent', 'tools', 'cliff_repo', 'cliff_skel_trajectory.py')
    cmd = [
        sys.executable,
        demo_py,
        "--input_path",
        stitched_path,
        "--rib_shift_px",
        str(float(rib_shift_px)),
        "--export_upper_only",
        "--upper_img",
        upper_img,
        "--lower_img",
        lower_img,
        "--stitch_rotate",
        stitch_rotate,
        "--stitch_mode",
        stitch_mode,
        "--stitch_overlap_ratio",
        str(float(stitch_overlap_ratio)),
        "--stitch_letterbox_w",
        str(int(stitch_letterbox_w)),
        "--stitch_letterbox_h",
        str(int(stitch_letterbox_h)),
        "--traj_kind",
        ("gallbladder" if kind not in ("kidney", "spine") else kind),
        "--kidney_side",
        str(kidney_side or "right"),
        "--kidney_start_mode",
        str(kidney_start_mode or "thorax_spine_lowest"),
    ]
    rospy.loginfo("Running: %s", " ".join(cmd))
    # 通过 env 传参：不改 demo 脚本 CLI，直接控制 SKEL 的迭代/日志
    env = os.environ.copy()
    env.setdefault("SKEL_QUIET", "1")
    env.setdefault("SKEL_PRINT_EVERY", "100")
    env.setdefault("SKEL_ITER_SCALE", "0.4")

    # 过滤 demo 输出：吞掉 [export] Saved/Failed... 这类文件导出日志，其它仍保留
    p = subprocess.Popen(
        cmd,
        cwd=os.path.dirname(demo_py),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert p.stdout is not None
    for line in p.stdout:
        s = (line or "").rstrip("\n")
        if not s:
            continue
        if "[export]" in s:
            continue
        # 仍然打印一些关键进度（保留原味，便于排障）
        rospy.loginfo("%s", s)
    ret = p.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)


def run_uv_to_yaml(snapshot_dir: str, target_frame: str, uv_npy: str,
                   frame_spacing: float, neighbor_radius: float, num_neighbors: int,
                   approach_offset: float, pre_approach_offset: float, snap_to_cloud: bool):
    tool = os.path.join(repo_root(), 'workspace', 'src', 'image_processing', 'scripts', 'ribline_snapshot_to_path_yaml.py')
    cmd = [
        sys.executable, tool,
        "--snapshot_dir", snapshot_dir,
        "--target_frame", target_frame,
        "--rib_uv_npy", uv_npy,
        "--frame_spacing", str(float(frame_spacing)),
        "--neighbor_radius", str(float(neighbor_radius)),
        "--num_neighbors", str(int(num_neighbors)),
        "--approach_offset", str(float(approach_offset)),
        "--pre_approach_offset", str(float(pre_approach_offset)),
    ]
    if snap_to_cloud:
        cmd.append("--snap_to_cloud")
    rospy.loginfo("Running: %s", " ".join(cmd))
    subprocess.check_call(cmd, cwd=os.path.dirname(tool))


def main():
    # Fail-fast if ROS master is not online (otherwise rospy.init_node may appear "stuck")
    try:
        if not rosgraph.is_master_online():
            sys.stderr.write("[rgbpair_skel_pipeline] ROS master not online. Please start `roscore` first.\n")
            return
    except Exception:
        # If rosgraph check fails for any reason, fall back to rospy behavior
        pass

    rospy.init_node("rgbpair_skel_pipeline", anonymous=True)

    if not _HAVE_IIWA_ACTION and JointPosition is None:
        raise RuntimeError("未安装/无法导入 iiwa_msgs，无法读取关节状态。")

    capture_motion_backend = str(rospy.get_param("~capture_motion_backend", "moveit")).strip().lower()
    moveit_group = str(rospy.get_param("~moveit_group", "manipulator"))
    moveit_vel_scale = float(rospy.get_param("~moveit_vel_scale", 0.05))
    moveit_acc_scale = float(rospy.get_param("~moveit_acc_scale", 0.05))
    if capture_motion_backend == "moveit" and not _HAVE_MOVEIT:
        rospy.logwarn("[rgbpair] moveit unavailable; falling back to iiwa action")
        capture_motion_backend = "action"
    if capture_motion_backend == "action" and not _HAVE_IIWA_ACTION:
        raise RuntimeError("iiwa action unavailable and moveit disabled")
    rospy.loginfo("[rgbpair] capture motion backend: %s", capture_motion_backend)

    # ---- capture params ----
    start_delay_sec = float(rospy.get_param("~start_delay_sec", 10.0))
    start_delay_skippable = bool(rospy.get_param("~start_delay_skippable", True))
    ns = rospy.get_param("~ns", "/iiwa")
    yaml_path = rospy.get_param("~yaml", DEFAULT_YAML)
    pose_indices = rospy.get_param("~pose_indices", [3, 4])
    rgb_topic = rospy.get_param("~rgb_topic", "/rgb/image_raw")
    depth_topic = rospy.get_param("~depth_topic", "/depth_to_rgb/image_raw")
    caminfo_topic = rospy.get_param("~camera_info_topic", "/rgb/camera_info")
    save_camera_info = bool(rospy.get_param("~save_camera_info", True))
    msg_timeout = float(rospy.get_param("~msg_timeout", 10.0))
    settle_sec = float(rospy.get_param("~settle_sec", 0.7))
    joint_tol = float(rospy.get_param("~joint_tol", 0.01))
    reach_timeout = float(rospy.get_param("~reach_timeout", 30.0))
    out_root = rospy.get_param("~out_root", DEFAULT_OUT_ROOT)
    tag = rospy.get_param("~tag", "rgbpair")
    return_to_first_pose = bool(rospy.get_param("~return_to_first_pose", True))
    target_frame = rospy.get_param("~target_frame", "iiwa_link_0")
    fixed_out_dir = rospy.get_param("~fixed_out_dir", DEFAULT_FIXED_OUT_DIR)
    overwrite_fixed_out_dir = bool(rospy.get_param("~overwrite_fixed_out_dir", True))

    # ---- stitch params (defaults match your “good” config) ----
    rotate_a = rospy.get_param("~stitch_rotate_a", "cw")
    rotate_b = rospy.get_param("~stitch_rotate_b", "cw")
    stitch_mode = rospy.get_param("~stitch_mode", "vertical")
    stitch_order = rospy.get_param("~stitch_order", "a_then_b")
    stitch_overlap_ratio = float(rospy.get_param("~stitch_overlap_ratio", 0.55))
    stitch_letterbox_w = int(rospy.get_param("~stitch_letterbox_w", 720))
    stitch_letterbox_h = int(rospy.get_param("~stitch_letterbox_h", 1280))

    # ---- skel params ----
    rib_shift_px = float(rospy.get_param("~rib_shift_px", 12.0))
    traj_kind = str(rospy.get_param("~traj_kind", "gallbladder")).strip().lower()
    kidney_side = str(rospy.get_param("~kidney_side", "right")).strip().lower()
    kidney_start_mode = str(rospy.get_param("~kidney_start_mode", "thorax_spine_lowest")).strip().lower()

    # ---- 2D->3D->poses params (match image_processing logic) ----
    posegen_frame_spacing = float(rospy.get_param("~posegen_frame_spacing", 0.01))
    posegen_neighbor_radius = float(rospy.get_param("~posegen_neighbor_radius", 0.03))
    posegen_num_neighbors = int(rospy.get_param("~posegen_num_neighbors", 100))
    posegen_approach_offset = float(rospy.get_param("~posegen_approach_offset", 0.0))
    posegen_pre_approach_offset = float(rospy.get_param("~posegen_pre_approach_offset", 0.05))
    posegen_snap_to_cloud = bool(rospy.get_param("~posegen_snap_to_cloud", True))
    
    # New param to skip projection (for agentic workflow where agent handles projection separately)
    skip_projection = bool(rospy.get_param("~skip_projection", False))

    # Derived topics
    ns_clean = str(ns).strip("/")
    ns_prefix = ("/" + ns_clean) if ns_clean else ""
    iiwa_state_topic = rospy.get_param("~iiwa_state_topic", (ns_prefix + "/state/JointPosition"))
    iiwa_action_name = rospy.get_param("~iiwa_action_name", (ns_prefix + "/action/move_to_joint_position"))

    if start_delay_sec > 0:
        skippable_delay(start_delay_sec, skippable=start_delay_skippable)

    if not os.path.exists(yaml_path):
        raise RuntimeError("位姿库不存在: {}".format(yaml_path))
    poses = (yaml.safe_load(open(yaml_path, "r")) or {}).get("poses", [])
    if not poses:
        raise RuntimeError("YAML 中没有 poses: {}".format(yaml_path))
    if not isinstance(pose_indices, (list, tuple)) or len(pose_indices) != 2:
        raise ValueError("~pose_indices 必须是长度为 2 的数组，例如 [3,4]")
    i0, i1 = int(pose_indices[0]), int(pose_indices[1])
    if i0 < 0 or i0 >= len(poses) or i1 < 0 or i1 >= len(poses):
        raise ValueError("pose_indices 超出范围：共有 {} 个 pose，给的是 {}".format(len(poses), pose_indices))
    p0 = poses[i0]
    p1 = poses[i1]
    n0 = p0.get("name", "pose_{}".format(i0))
    n1 = p1.get("name", "pose_{}".format(i1))

    # 输出目录：默认固定写死（每次覆盖），避免每次找最新时间戳目录
    out_dir = str(fixed_out_dir).strip() or ""
    if out_dir:
        out_dir = os.path.abspath(os.path.expanduser(out_dir))
        if overwrite_fixed_out_dir and os.path.exists(out_dir):
            shutil.rmtree(out_dir, ignore_errors=True)
        ensure_dir(out_dir)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(out_root, "{}_{}".format(tag, ts))
        ensure_dir(out_dir)

    # record meta
    meta = {
        "yaml": yaml_path,
        "pose_indices": [i0, i1],
        "pose_names": [n0, n1],
        "ns": ns_prefix or "/(root)",
        "iiwa_state_topic": iiwa_state_topic,
        "iiwa_action_name": iiwa_action_name,
        "rgb_topic": rgb_topic,
        "camera_info_topic": caminfo_topic,
        "depth_topic": depth_topic,
        "save_camera_info": save_camera_info,
        "return_to_first_pose": return_to_first_pose,
        "stitch": {
            "rotate_a": rotate_a,
            "rotate_b": rotate_b,
            "mode": stitch_mode,
            "order": stitch_order,
            "overlap_ratio": stitch_overlap_ratio,
            "letterbox_w": stitch_letterbox_w,
            "letterbox_h": stitch_letterbox_h,
        },
        "rib_shift_px": rib_shift_px,
        "target_frame": target_frame,
        "fixed_out_dir": out_dir,
        "posegen": {
            "frame_spacing": posegen_frame_spacing,
            "neighbor_radius": posegen_neighbor_radius,
            "num_neighbors": posegen_num_neighbors,
            "approach_offset": posegen_approach_offset,
            "pre_approach_offset": posegen_pre_approach_offset,
            "snap_to_cloud": posegen_snap_to_cloud,
        },
    }
    with open(os.path.join(out_dir, "meta.yaml"), "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    # subscribe joint position
    latest = {"vals": None}

    def _iiwa_state_cb(msg: JointPosition):
        latest["vals"] = [msg.position.a1, msg.position.a2, msg.position.a3, msg.position.a4, msg.position.a5, msg.position.a6, msg.position.a7]

    rospy.Subscriber(iiwa_state_topic, JointPosition, _iiwa_state_cb, queue_size=1)
    rospy.loginfo("等待关节状态: %s", iiwa_state_topic)
    t0 = rospy.Time.now()
    while not rospy.is_shutdown() and latest["vals"] is None:
        if (rospy.Time.now() - t0).to_sec() > 5.0:
            raise RuntimeError("等待 {} 超时（无数据）".format(iiwa_state_topic))
        rospy.sleep(0.05)

    cli = None
    moveit_helper = None
    if capture_motion_backend == "action":
        cli = actionlib.SimpleActionClient(iiwa_action_name, MoveToJointPositionAction)
        rospy.loginfo("等待 action server: %s", iiwa_action_name)
        if not cli.wait_for_server(rospy.Duration(5.0)):
            raise RuntimeError("action server 不存在/不可达: {}".format(iiwa_action_name))
    else:
        moveit_helper = _MoveItCaptureHelper(ns, moveit_group, moveit_vel_scale, moveit_acc_scale)
        rospy.loginfo("[rgbpair] MoveIt capture helper ready (group=%s)", moveit_group)

    def goto_pose_action(pose, label):
        ensure_position_control(ns, required=True)
        name = pose.get("name", label)
        target = reorder(DEFAULT_JOINT_NAMES, pose.get("joints", {}))
        cur = latest["vals"]
        err0 = _max_abs_joint_error(cur, target)
        rospy.loginfo("[%s] MoveToJointPosition: max|Δjoint|=%.4f rad (tol=%.4f)", name, err0, joint_tol)
        if err0 <= float(joint_tol):
            rospy.loginfo("[%s] already at target joints; skip motion", name)
            rospy.sleep(settle_sec)
            return name

        st = cli.get_state()
        if st in (actionlib.GoalStatus.ACTIVE, actionlib.GoalStatus.PENDING, actionlib.GoalStatus.PREEMPTING):
            rospy.logwarn("[%s] cancelling stale action goal (state=%s) before new move", name, st)
            cli.cancel_goal()
            cli.wait_for_result(rospy.Duration(2.0))

        rospy.loginfo("[%s] 发送 MoveToJointPosition goal…", name)
        jp = JointPosition()
        jp.position.a1 = float(target[0])
        jp.position.a2 = float(target[1])
        jp.position.a3 = float(target[2])
        jp.position.a4 = float(target[3])
        jp.position.a5 = float(target[4])
        jp.position.a6 = float(target[5])
        jp.position.a7 = float(target[6])

        goal = MoveToJointPositionGoal()
        goal.joint_position = jp
        cli.send_goal(goal)
        if not cli.wait_for_result(rospy.Duration(float(reach_timeout))):
            cli.cancel_goal()
            raise RuntimeError("[{}] action 等待超时（>{}s）".format(name, reach_timeout))

        state = cli.get_state()
        result = cli.get_result()
        if state != actionlib.GoalStatus.SUCCEEDED or result is None or not bool(getattr(result, "success", False)):
            err_msg = getattr(result, "error", "") if result is not None else ""
            raise RuntimeError("[{}] action 失败: state={} success={} error={}".format(
                name, state, getattr(result, "success", None), err_msg))

        reached, err = wait_until_reached(lambda: latest["vals"], target, joint_tol=joint_tol, timeout=reach_timeout)
        if not reached:
            raise RuntimeError("[{}] 等待到位超时：max|Δjoint|={:.4f} rad > tol {:.4f}".format(name, err, joint_tol))
        rospy.loginfo("[%s] 到位，等待稳定 %.2fs", name, settle_sec)
        rospy.sleep(settle_sec)
        return name

    def goto_pose(pose, label):
        name = pose.get("name", label)
        if capture_motion_backend == "moveit":
            ensure_position_control(ns, required=False)
            return moveit_helper.goto_joints(
                pose.get("joints", {}),
                name,
                joint_tol,
                reach_timeout,
                settle_sec,
                lambda: latest["vals"],
            )
        return goto_pose_action(pose, label)

    # TF buffer (used to save snapshot TF at pose0)
    tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
    _ = tf2_ros.TransformListener(tf_buffer)

    # capture pose0 (upper / first RGB)
    goto_pose(p0, "pose0")
    rgb1 = os.path.join(out_dir, "rgb_01_{}.png".format(n0))
    ci1 = os.path.join(out_dir, "camera_info_01_{}.yaml".format(n0)) if save_camera_info else None
    capture_and_save(rgb_topic, caminfo_topic, rgb1, ci1, timeout=msg_timeout)

    # Move to second pose immediately after first RGB (do not block on depth/cloud here).
    rospy.loginfo("First RGB saved; moving to second capture pose: %s", n1)
    goto_pose(p1, "pose1")
    rgb2 = os.path.join(out_dir, "rgb_02_{}.png".format(n1))
    ci2 = os.path.join(out_dir, "camera_info_02_{}.yaml".format(n1)) if save_camera_info else None
    capture_and_save(rgb_topic, caminfo_topic, rgb2, ci2, timeout=msg_timeout)

    # Depth / point cloud must be captured at pose0 for 3D projection.
    rospy.loginfo("Both RGB captured; returning to pose0 (%s) for depth/point-cloud snapshot", n0)
    goto_pose(p0, "return_pose0_for_depth")

    bridge = CvBridge()
    rospy.loginfo("等待 Depth: %s (timeout=%.1fs)", depth_topic, msg_timeout)
    depth_msg = rospy.wait_for_message(depth_topic, Image, timeout=msg_timeout)
    depth_m = depth_msg_to_meters(depth_msg, bridge)
    depth_path = os.path.join(out_dir, "depth_m.npy")
    np.save(depth_path, depth_m.astype(np.float32))
    rospy.loginfo("已保存: %s", depth_path)

    rospy.loginfo("等待 CameraInfo: %s (timeout=%.1fs)", caminfo_topic, msg_timeout)
    caminfo_msg = rospy.wait_for_message(caminfo_topic, CameraInfo, timeout=msg_timeout)
    intr = caminfo_to_intrinsics_yaml(caminfo_msg)
    intr_path = os.path.join(out_dir, "camera_info.yaml")
    with open(intr_path, "w") as f:
        yaml.safe_dump(intr, f, sort_keys=False)
    rospy.loginfo("已保存: %s", intr_path)

    cam_frame = str(depth_msg.header.frame_id)
    try:
        tf_stamped = tf_buffer.lookup_transform(target_frame, cam_frame, rospy.Time(0), rospy.Duration(1.0))
        t = tf_stamped.transform.translation
        q = tf_stamped.transform.rotation
        tf_path = os.path.join(out_dir, f"tf_{cam_frame}_to_{target_frame}.yaml".replace("/", "_"))
        with open(tf_path, "w") as f:
            yaml.safe_dump(
                {
                    "child_frame_id": cam_frame,
                    "transform": {
                        "translation": {"x": float(t.x), "y": float(t.y), "z": float(t.z)},
                        "rotation": {"x": float(q.x), "y": float(q.y), "z": float(q.z), "w": float(q.w)},
                    },
                },
                f,
                sort_keys=False,
            )
        rospy.loginfo("已保存: %s", tf_path)
    except Exception as e:
        raise RuntimeError("TF lookup failed ({} -> {}): {}".format(cam_frame, target_frame, e))

    # build cloud in target_frame (pose0)
    cloud_cam = depth_to_cloud(depth_m, intr)
    if cloud_cam.shape[0] < 100:
        rospy.logwarn("depth->cloud 点数过少：%d", int(cloud_cam.shape[0]))
    # apply tf
    t_xyz = np.array([float(t.x), float(t.y), float(t.z)], dtype=np.float32)
    q_xyzw = np.array([float(q.x), float(q.y), float(q.z), float(q.w)], dtype=np.float32)
    cloud_target = apply_tf(cloud_cam, t_xyz=t_xyz, q_xyzw=q_xyzw)
    cloud_npy = os.path.join(out_dir, f"cloud_{str(target_frame).replace('/', '_')}.npy")
    np.save(cloud_npy, cloud_target.astype(np.float32))
    rospy.loginfo("已保存: %s", cloud_npy)
    if o3d is not None:
        try:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud_target))
            ply = os.path.join(out_dir, f"cloud_{str(target_frame).replace('/', '_')}.ply")
            o3d.io.write_point_cloud(ply, pcd, write_ascii=False)
            rospy.loginfo("已保存: %s", ply)
        except Exception as e:
            rospy.logwarn("open3d 保存 ply 失败(忽略): %s", str(e))

    # stitch
    stitched_path = os.path.join(out_dir, "rgb_stitched_upright_{}x{}.png".format(stitch_letterbox_w, stitch_letterbox_h))
    # NOTE: rotate values for upper/lower should match; for your case both cw
    stitch_two_images(
        rgb1,
        rgb2,
        stitched_path,
        rotate_a=rotate_a,
        rotate_b=rotate_b,
        mode=stitch_mode,
        order=stitch_order,
        overlap_ratio=stitch_overlap_ratio,
        letterbox_w=stitch_letterbox_w,
        letterbox_h=stitch_letterbox_h,
    )
    rospy.loginfo("已生成 stitched: %s", stitched_path)

    # run skel
    run_demo_with_skel(
        out_dir=out_dir,
        stitched_path=stitched_path,
        upper_img=rgb1,
        lower_img=rgb2,
        rib_shift_px=rib_shift_px,
        stitch_rotate=str(rotate_a),
        stitch_mode=str(stitch_mode),
        stitch_overlap_ratio=float(stitch_overlap_ratio),
        stitch_letterbox_w=int(stitch_letterbox_w),
        stitch_letterbox_h=int(stitch_letterbox_h),
        traj_kind=traj_kind,
        kidney_side=kidney_side,
        kidney_start_mode=kidney_start_mode,
    )

    if not skip_projection:
        # --- Breath-hold prompt before 2D->3D projection / scan path generation ---
        breath_prompt = str(rospy.get_param("~breath_hold_prompt", "正在生成扫描轨迹，请深呼吸并摒住呼吸。"))
        breath_wait_sec = float(rospy.get_param("~breath_hold_wait_sec", 5.0))
        if breath_prompt:
            rospy.loginfo("%s %.1f 秒后开始投影到点云并生成轨迹。", breath_prompt, breath_wait_sec)
            if breath_wait_sec > 0:
                rospy.sleep(breath_wait_sec)

        # generate path_preview.yaml from red trajectory in pose0 snapshot
        uv_red = os.path.join(out_dir, "ribline_uv_shifted_fit.npy")
        if not os.path.exists(uv_red):
            # fallback to shifted (non-fit)
            uv_red = os.path.join(out_dir, "ribline_uv_shifted.npy")
        run_uv_to_yaml(
            snapshot_dir=out_dir,
            target_frame=str(target_frame),
            uv_npy=uv_red,
            frame_spacing=posegen_frame_spacing,
            neighbor_radius=posegen_neighbor_radius,
            num_neighbors=posegen_num_neighbors,
            approach_offset=posegen_approach_offset,
            pre_approach_offset=posegen_pre_approach_offset,
            snap_to_cloud=posegen_snap_to_cloud,
        )
    else:
        rospy.loginfo("[rgbpair_skel_pipeline] Skipping projection and breath hold (skip_projection=True)")

    rospy.loginfo("✓ pipeline 完成。输出目录: %s", out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()


