#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
path_stroke_preview_uv.py (ROS1)

严格 1:1 复用 path_stroke_preview.py 的“2D像素->3D->TF->法向->姿态->YAML”逻辑，
但像素轨迹不来自手绘，而是来自离线保存的 ribline_uv.npy。

用法：
1) 先离线跑 demo_with_skel.py 得到 /.../rgbshot_xxx/ribline_uv.npy
2) 启动本节点，订阅 Depth + CameraInfo（相机可以此时再开）
3) 调用服务 /path_stroke_preview_uv/generate_from_file 生成 YAML

不会修改/依赖原版 path_stroke_preview.py（留底）。
"""

from __future__ import annotations

import os
import math
import copy
import threading
from collections import deque
from typing import List, Tuple, Optional

import cv2
import numpy as np
import rospy
import tf2_ros
import tf2_geometry_msgs
import yaml
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from scipy.spatial import KDTree

try:
    from trajectory_outlier_filter import filter_trajectory_points
except ImportError:
    import sys

    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if _SCRIPT_DIR not in sys.path:
        sys.path.insert(0, _SCRIPT_DIR)
    from trajectory_outlier_filter import filter_trajectory_points

from geometry_msgs.msg import PointStamped, PoseStamped, Point as GeoPoint
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from sensor_msgs import point_cloud2 as pc2
from std_srvs.srv import Trigger, TriggerResponse
from visualization_msgs.msg import Marker, MarkerArray

try:
    import open3d as o3d  # type: ignore
except Exception:
    o3d = None


class PathStrokePreviewUvNode(object):
    def __init__(self) -> None:
        rospy.init_node("path_stroke_preview_uv", anonymous=False)

        # 坐标系配置
        self.camera_frame = rospy.get_param("~camera_frame", "camera_base")
        self.target_frame = rospy.get_param("~target_frame", "iiwa_link_0")
        self.visualization_frame = rospy.get_param("~visualization_frame", "camera_base")

        # 轨迹采样参数
        self.frame_spacing = float(rospy.get_param("~frame_spacing", 0.02))
        self.pre_approach_offset = float(rospy.get_param("~pre_approach_offset", 0.30))
        self.approach_offset = float(rospy.get_param("~approach_offset", 0.11))

        # 2D 像素轨迹加密：UV 轨迹（尤其是肾线只有起点/终点 2 个点）在投影到 3D 之前，
        # 先沿 2D 折线按固定像素步长插值出密集像素点，逐点查深度 → 3D 轨迹贴合体表，
        # 也避免“只有 2 个端点、某端点深度无效就跌到 <2 点”导致投影失败。
        self.densify_step_px = float(rospy.get_param("~densify_step_px", 3.0))
        self.densify_max_points = int(rospy.get_param("~densify_max_points", 600))
        self.kidney_truncate_on_height_jump = bool(rospy.get_param("~kidney_truncate_on_height_jump", True))
        self.kidney_height_jump_threshold_m = float(rospy.get_param("~kidney_height_jump_threshold_m", 0.03))

        self.output_yaml = os.path.expanduser(rospy.get_param("~output_yaml", "~/.ros/path_preview.yaml"))

        # 从文件读取像素轨迹
        self.ribline_uv_npy = os.path.expanduser(rospy.get_param("~ribline_uv_npy", ""))
        if not self.ribline_uv_npy:
            rospy.logwarn("[path_stroke_preview_uv] ~ribline_uv_npy is empty; service will fail until set.")

        # 器官类型（gallbladder / kidney / spine）：2D UV 轨迹来源不同；3D 姿态统一绕法向 -90deg (CW)
        self.traj_kind = str(rospy.get_param("~traj_kind", "")).strip().lower()

        # Spine-only: repair depth/projection outliers on the 3D polyline before pose generation.
        self.spine_filter_outliers = bool(rospy.get_param("~spine_filter_outliers", True))
        self.spine_outlier_lateral_m = float(rospy.get_param("~spine_outlier_lateral_m", 0.035))
        self.spine_outlier_step_factor = float(rospy.get_param("~spine_outlier_step_factor", 2.5))
        self.spine_outlier_z_jump_m = float(rospy.get_param("~spine_outlier_z_jump_m", 0.04))
        self.spine_outlier_window = int(rospy.get_param("~spine_outlier_window", 5))
        self.spine_outlier_max_passes = int(rospy.get_param("~spine_outlier_max_passes", 3))
        # Spine scan path: match *_demo_skel_overlay_spine_curve_only.png (drop first len//2 points).
        self.spine_truncate_to_visible_half = bool(rospy.get_param("~spine_truncate_to_visible_half", True))
        self.spine_full_min_points_for_truncate = int(rospy.get_param("~spine_full_min_points_for_truncate", 18))
        # After 3D path generation: shift scan poses along target_frame +X / +Y.
        self.spine_base_x_offset_m = float(rospy.get_param("~spine_base_x_offset_m", 0.025))
        self.kidney_side = str(rospy.get_param("~kidney_side", "") or "").strip().lower()
        self.kidney_left_base_x_offset_m = float(rospy.get_param("~kidney_left_base_x_offset_m", 0.02))
        self.kidney_left_base_y_offset_m = float(rospy.get_param("~kidney_left_base_y_offset_m", 0.015))
        # After visible-half truncation, keep start/end UV only (same cardinality as kidney_line_uv).
        # _densify_pixel_path resamples along that segment at densify_step_px before depth projection.
        self.spine_use_endpoint_line = bool(rospy.get_param("~spine_use_endpoint_line", True))

        # 点云/深度相关（必须用 depthcloud 模式）
        self.use_depthcloud = rospy.get_param("~use_depthcloud", True)
        if self.use_depthcloud and o3d is None:
            raise RuntimeError("Open3D 未安装，无法在 DepthCloud 模式下生成点云")

        self.rgb_topic = rospy.get_param("~rgb_topic", "/rgb/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/depth_to_rgb/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/rgb/camera_info")
        self.pointcloud_topic = rospy.get_param("~pointcloud_topic", "/points2_down")
        rospy.loginfo(
            "[path_stroke_preview_uv] topics: rgb=%s depth=%s camera_info=%s",
            self.rgb_topic,
            self.depth_topic,
            self.camera_info_topic,
        )

        self.neighbor_search_radius = float(rospy.get_param("~neighbor_search_radius", 0.03))
        self.num_neighbors = int(rospy.get_param("~num_neighbors", 100))
        self.frame_buffer_size = int(rospy.get_param("~frame_buffer_size", 1))
        self.min_frames_for_detection = int(rospy.get_param("~min_frames_for_detection", 1))
        self.voxel_size = float(rospy.get_param("~voxel_size", 0))
        self.max_points = int(rospy.get_param("~max_points", 50000))

        self.pixel_sampling_step = float(rospy.get_param("~pixel_sampling_step", 2.0))
        self.depth_search_window = int(rospy.get_param("~depth_search_window", 5))
        self.max_stroke_points = int(rospy.get_param("~max_stroke_points", 5000))

        # TF & 数据缓存
        self._tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)
        self._bridge = CvBridge()

        self._camera_info: Optional[CameraInfo] = None
        self._depth_buffer: deque = deque(maxlen=max(self.frame_buffer_size, 1))
        self._pointcloud_lock = threading.Lock()
        self._cloud_points: Optional[np.ndarray] = None
        self._kdtree: Optional[KDTree] = None
        self._cloud_frame: Optional[str] = None

        self._image_lock = threading.Lock()
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._depth_frame_id: Optional[str] = None

        self._stroke_lock = threading.Lock()
        self._stroke_pixels: List[Tuple[int, int]] = []

        self._surface_points_resampled: List[np.ndarray] = []
        self._interpolated_frames: List[PoseStamped] = []
        self._start_axes: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self._start_normal: Optional[np.ndarray] = None
        self._end_axes: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self._end_normal: Optional[np.ndarray] = None

        # 可视化
        self.pub_path_markers = rospy.Publisher("~path_markers", MarkerArray, queue_size=1, latch=True)
        self.pub_path_markers_surface = rospy.Publisher(
            "~path_markers_surface", MarkerArray, queue_size=1, latch=True
        )
        self.pub_path_line = rospy.Publisher("~path_line", Marker, queue_size=1, latch=True)
        self.pub_path_points = rospy.Publisher("~path_points", Marker, queue_size=1, latch=True)
        self.pub_start_pre_pose = rospy.Publisher("~start_pre_pose", PoseStamped, queue_size=1, latch=True)
        self.pub_start_final_pose = rospy.Publisher("~start_final_pose", PoseStamped, queue_size=1, latch=True)
        self.pub_end_final_pose = rospy.Publisher("~end_final_pose", PoseStamped, queue_size=1, latch=True)

        # 订阅
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self._on_camera_info, queue_size=1)
        if self.use_depthcloud:
            rgb_sub = Subscriber(self.rgb_topic, Image)
            depth_sub = Subscriber(self.depth_topic, Image)
            self._sync = ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=5, slop=0.05)
            self._sync.registerCallback(self._on_rgb_depth)
        else:
            rospy.Subscriber(self.rgb_topic, Image, self._on_rgb_only, queue_size=1)
            rospy.Subscriber(self.pointcloud_topic, PointCloud2, self._on_pointcloud, queue_size=1)

        # 服务
        self.srv_generate_from_file = rospy.Service("~generate_from_file", Trigger, self._srv_generate_from_file_cb)

        # OpenCV 窗口（可选：仅用于查看）
        self.window_name = rospy.get_param("~window_name", "PathStrokePreviewUV")
        self.no_gui = bool(rospy.get_param("~no_gui", False))
        if not self.no_gui:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # 启动后自动生成（可选）
        self.auto_generate_delay_sec = float(rospy.get_param("~auto_generate_delay_sec", 0.0))
        # Auto-generate retry policy (important for headless mode where camera_info/depth may arrive later)
        self.auto_generate_retry_period_sec = float(rospy.get_param("~auto_generate_retry_period_sec", 1.0))
        self.auto_generate_max_retries = int(rospy.get_param("~auto_generate_max_retries", 30))
        self._auto_generate_attempts = 0
        if self.auto_generate_delay_sec > 0.0:
            rospy.loginfo("[path_stroke_preview_uv] auto-generate enabled: delay=%.1fs", self.auto_generate_delay_sec)
            rospy.Timer(rospy.Duration(self.auto_generate_delay_sec), self._auto_generate_timer_cb, oneshot=True)

        rospy.on_shutdown(self._on_shutdown)
        rospy.loginfo("[path_stroke_preview_uv] ▶ 调用服务生成：rosservice call /path_stroke_preview_uv/generate_from_file")

        if not self.no_gui:
            rate = rospy.Rate(30)
            while not rospy.is_shutdown():
                self._update_display()
                rate.sleep()
        else:
            rospy.loginfo("[path_stroke_preview_uv] no_gui=True, spinning...")
            rospy.spin()

    def _auto_generate_timer_cb(self, _evt) -> None:
        """
        Auto-generate may run before camera_info/depth are ready. In headless pipelines,
        we retry a few times to avoid the agent falling into a projection/verify loop.
        """
        self._auto_generate_attempts += 1
        try:
            resp = self._srv_generate_from_file_cb(None)
            if resp.success:
                rospy.loginfo(
                    "[path_stroke_preview_uv] ✓ auto-generate success (attempt %d/%d): %s",
                    self._auto_generate_attempts,
                    self.auto_generate_max_retries,
                    self.output_yaml,
                )
                return

            msg = str(resp.message)
            rospy.logwarn(
                "[path_stroke_preview_uv] auto-generate failed (attempt %d/%d): %s",
                self._auto_generate_attempts,
                self.auto_generate_max_retries,
                msg,
            )

            retryable = ("相机内参未就绪" in msg) or ("深度图尚未就绪" in msg)
            if retryable and (self._auto_generate_attempts < self.auto_generate_max_retries):
                rospy.Timer(
                    rospy.Duration(max(self.auto_generate_retry_period_sec, 0.2)),
                    self._auto_generate_timer_cb,
                    oneshot=True,
                )
        except Exception as e:
            rospy.logwarn(
                "[path_stroke_preview_uv] auto-generate exception (attempt %d/%d): %s",
                self._auto_generate_attempts,
                self.auto_generate_max_retries,
                str(e),
            )
            if self._auto_generate_attempts < self.auto_generate_max_retries:
                rospy.Timer(
                    rospy.Duration(max(self.auto_generate_retry_period_sec, 0.2)),
                    self._auto_generate_timer_cb,
                    oneshot=True,
                )

    # -------- subscribers --------
    def _on_camera_info(self, msg: CameraInfo) -> None:
        if self._camera_info is None:
            self._camera_info = msg
            rospy.loginfo("[path_stroke_preview_uv] 已接收相机内参：%dx%d", msg.width, msg.height)

    def _on_rgb_depth(self, rgb_msg: Image, depth_msg: Image) -> None:
        if self._camera_info is None:
            return
        try:
            cv_rgb = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn("[path_stroke_preview_uv] 转换 RGB 图像失败: %s", str(e))
            return
        try:
            cv_depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as e:
            rospy.logwarn("[path_stroke_preview_uv] 转换深度图失败: %s", str(e))
            return

        if cv_depth.dtype == np.uint16:
            cv_depth = cv_depth.astype(np.float32) / 1000.0
        else:
            cv_depth = cv_depth.astype(np.float32)

        with self._image_lock:
            self._latest_rgb = cv_rgb
            self._latest_depth = cv_depth
            self._depth_frame_id = depth_msg.header.frame_id

        with self._pointcloud_lock:
            self._depth_buffer.append(depth_msg)

    def _on_rgb_only(self, _msg: Image) -> None:
        return

    def _on_pointcloud(self, _msg: PointCloud2) -> None:
        return

    # -------- display --------
    def _update_display(self) -> None:
        if getattr(self, "no_gui", False):
            return
        with self._image_lock:
            img = None if self._latest_rgb is None else self._latest_rgb.copy()
        if img is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for RGB...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow(self.window_name, blank)
            cv2.waitKey(1)
            return

        with self._stroke_lock:
            stroke = copy.copy(self._stroke_pixels)
        if stroke:
            pts = np.array(stroke, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=False, color=(255, 0, 0), thickness=2)
        cv2.putText(img, "call /path_stroke_preview_uv/generate_from_file", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(self.window_name, img)
        cv2.waitKey(1)

    # -------- service --------
    def _srv_generate_from_file_cb(self, _req) -> TriggerResponse:
        if not self.ribline_uv_npy or not os.path.exists(self.ribline_uv_npy):
            return TriggerResponse(success=False, message=f"ribline_uv_npy 不存在: {self.ribline_uv_npy}")
        uv = np.load(self.ribline_uv_npy).astype(np.float32).reshape(-1, 2)
        uv = self._prepare_spine_uv_for_scan(uv)
        pixels = [(int(round(u)), int(round(v))) for u, v in uv.tolist()]
        with self._stroke_lock:
            self._stroke_pixels = pixels

        # 严格复用：走同一条 generate 流水线（等价于手绘后点 generate）
        return self._srv_generate_cb_like()

    def _densify_pixel_path(self, pixels: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """沿 2D 折线按固定像素步长 densify_step_px 重采样（与左肾扫描相同）。

        肾线 / 脊柱扫描 UV 通常只有起点、终点 2 个点；投影前沿直线（或折线）
        每隔 step 像素取一个采样点，再逐点查深度生成 3D 轨迹。
        """
        pts = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        if len(pts) < 2:
            return [(int(round(x)), int(round(y))) for x, y in pts.tolist()]
        seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total = float(seg_len.sum())
        if total < 1e-9:
            xy = (int(round(pts[0, 0])), int(round(pts[0, 1])))
            return [xy, xy]
        step = max(0.5, float(self.densify_step_px))
        max_pts = max(2, int(self.densify_max_points))
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])

        def _interp_at_arc(s: float) -> Tuple[float, float]:
            s_clamped = min(max(0.0, s), total)
            j = int(np.searchsorted(cum, s_clamped, side="right") - 1)
            j = max(0, min(j, len(pts) - 2))
            seg = cum[j + 1] - cum[j]
            r = 0.0 if seg <= 1e-9 else (s_clamped - cum[j]) / seg
            p = pts[j] + r * (pts[j + 1] - pts[j])
            return float(p[0]), float(p[1])

        dense: List[Tuple[int, int]] = []
        last = None
        s = 0.0
        while s < total - 1e-9 and len(dense) < max_pts:
            x, y = _interp_at_arc(s)
            xy = (int(round(x)), int(round(y)))
            if xy != last:
                dense.append(xy)
                last = xy
            s += step
        x, y = _interp_at_arc(total)
        xy_end = (int(round(x)), int(round(y)))
        if xy_end != last and len(dense) < max_pts:
            dense.append(xy_end)
        if len(dense) < 2:
            dense = [(int(round(pts[0, 0])), int(round(pts[0, 1]))),
                     (int(round(pts[-1, 0])), int(round(pts[-1, 1])))]
        return dense

    def _truncate_kidney_path_on_height_jump(self, points: List[np.ndarray]) -> List[np.ndarray]:
        """Kidney-only guard: cut the tail when projected points jump off the body in target-frame Z."""
        if self.traj_kind != "kidney" or not self.kidney_truncate_on_height_jump or len(points) < 3:
            return points
        threshold = max(0.0, float(self.kidney_height_jump_threshold_m))
        if threshold <= 0.0:
            return points
        zs = [float(np.asarray(p, dtype=np.float64).reshape(3)[2]) for p in points]
        ref_z = float(np.median(zs[: min(3, len(zs))]))
        body_dev_limit = max(threshold * 2.5, 0.07)
        for i in range(1, len(points)):
            dz = abs(zs[i] - zs[i - 1])
            dz_body = abs(zs[i] - ref_z)
            # 相邻点高度突变，或末端相对体表参考高度偏离过大 → 截断剩余轨迹
            if dz > threshold or (i >= 2 and dz_body > body_dev_limit):
                cut = max(2, i)
                rospy.logwarn(
                    "[path_stroke_preview_uv] kidney path height anomaly dz=%.3fm body_dev=%.3fm at point %d/%d; truncating tail to %d points",
                    dz,
                    dz_body,
                    i + 1,
                    len(points),
                    cut,
                )
                return points[:cut]
        return points

    def _filter_spine_trajectory_outliers(self, points: List[np.ndarray]) -> List[np.ndarray]:
        if self.traj_kind != "spine" or not self.spine_filter_outliers or len(points) < 3:
            return points
        filtered, outlier_idx = filter_trajectory_points(
            points,
            lateral_threshold_m=self.spine_outlier_lateral_m,
            step_spike_factor=self.spine_outlier_step_factor,
            z_jump_threshold_m=self.spine_outlier_z_jump_m,
            window=self.spine_outlier_window,
            max_passes=self.spine_outlier_max_passes,
        )
        if outlier_idx:
            rospy.logwarn(
                "[path_stroke_preview_uv] spine outlier repair: %d point(s) interpolated at indices %s",
                len(outlier_idx),
                outlier_idx,
            )
        return [filtered[i, :].astype(np.float64) for i in range(filtered.shape[0])]

    def _maybe_truncate_spine_uv_for_scan(self, uv: np.ndarray) -> np.ndarray:
        """
        Use the same visible segment as cliff_skel_trajectory spine_curve_only overlay:
        start index = len(uv) // 2 (hide upper thoracic half).
        """
        if self.traj_kind != "spine" or not self.spine_truncate_to_visible_half:
            return uv
        pts = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 4:
            return pts
        if len(pts) < int(self.spine_full_min_points_for_truncate):
            rospy.loginfo(
                "[path_stroke_preview_uv] spine UV has %d points (< %d); keeping as-is (already visible half)",
                len(pts),
                int(self.spine_full_min_points_for_truncate),
            )
            return pts
        half_idx = len(pts) // 2
        visible = pts[half_idx:]
        if len(visible) < 2:
            return pts
        rospy.loginfo(
            "[path_stroke_preview_uv] spine scan UV: visible half [%d:] (%d/%d pts, matches overlay)",
            half_idx,
            len(visible),
            len(pts),
        )
        return visible

    def _prepare_spine_uv_for_scan(self, uv: np.ndarray) -> np.ndarray:
        """Spine 2D：可见半段 → 首尾 2 点（与 kidney_line_uv 相同），固定像素步长重采样在 _densify_pixel_path。"""
        if self.traj_kind != "spine":
            return uv
        pts = self._maybe_truncate_spine_uv_for_scan(uv)
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 2:
            return pts
        if not self.spine_use_endpoint_line:
            return pts
        if len(pts) > 2:
            rospy.loginfo(
                "[path_stroke_preview_uv] spine UV: %d pts -> 2 endpoints (kidney-style); "
                "fixed %.1fpx resample before projection",
                len(pts),
                self.densify_step_px,
            )
            return np.stack([pts[0], pts[-1]], axis=0).astype(np.float32)
        return pts

    def _srv_generate_cb_like(self) -> TriggerResponse:
        with self._stroke_lock:
            pixel_path = copy.copy(self._stroke_pixels)
        if len(pixel_path) < 2:
            return TriggerResponse(success=False, message="像素轨迹点过少")
        if self._camera_info is None:
            return TriggerResponse(success=False, message="相机内参未就绪")

        # 投影前先在 2D 上加密（起点/终点 -> 沿线密集像素点）
        pixel_path_dense = self._densify_pixel_path(pixel_path)
        rospy.loginfo("[path_stroke_preview_uv] densify 2D path: %d -> %d pixels (step=%.1fpx)",
                      len(pixel_path), len(pixel_path_dense), self.densify_step_px)
        pixel_path = pixel_path_dense

        with self._image_lock:
            depth = None if self._latest_depth is None else self._latest_depth.copy()
            depth_frame = self._depth_frame_id
        if depth is None or depth_frame is None:
            return TriggerResponse(success=False, message="深度图尚未就绪")

        surface_points_camera = self._pixels_to_camera_points(pixel_path, depth, self._camera_info)
        if len(surface_points_camera) < 2:
            return TriggerResponse(success=False, message="像素轨迹对应的深度点过少")
        points_target = self._transform_points_array(np.asarray(surface_points_camera, dtype=np.float32), depth_frame, self.target_frame)
        if points_target is None or points_target.shape[0] < 2:
            return TriggerResponse(success=False, message="点转换到目标坐标系失败")
        surface_points = [points_target[i, :].astype(np.float64) for i in range(points_target.shape[0])]
        surface_points = self._truncate_kidney_path_on_height_jump(surface_points)
        surface_points = self._filter_spine_trajectory_outliers(surface_points)
        if len(surface_points) < 2:
            return TriggerResponse(success=False, message="肾脏轨迹高度突变后有效点过少")

        if not self._ensure_point_cloud():
            return TriggerResponse(success=False, message="点云尚未就绪或转换失败")

        success = self._generate_path(surface_points)
        if success:
            return TriggerResponse(success=True, message="轨迹生成完成，已保存到 YAML")
        return TriggerResponse(success=False, message="轨迹生成失败，详情见日志")

    # ===== 以下函数为 path_stroke_preview.py 的原始实现（保持一致）=====
    # （为节省篇幅，这里直接复制了关键函数；其余工具函数也一并保留）

    @staticmethod
    def _slerp_unit_vectors(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between two unit vectors."""
        t = float(np.clip(t, 0.0, 1.0))
        a = np.asarray(v0, dtype=np.float64).reshape(3)
        b = np.asarray(v1, dtype=np.float64).reshape(3)
        a = a / (np.linalg.norm(a) or 1.0)
        b = b / (np.linalg.norm(b) or 1.0)
        dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
        if dot > 0.9995:
            out = (1.0 - t) * a + t * b
            return out / (np.linalg.norm(out) or 1.0)
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        if sin_theta < 1e-9:
            return a.copy()
        w0 = math.sin((1.0 - t) * theta) / sin_theta
        w1 = math.sin(t * theta) / sin_theta
        out = w0 * a + w1 * b
        return out / (np.linalg.norm(out) or 1.0)

    def _generate_path(self, surface_points: List[np.ndarray]) -> bool:
        points_resampled = self._resample_path(surface_points, self.frame_spacing)
        points_resampled = self._filter_spine_trajectory_outliers(points_resampled)
        if len(points_resampled) < 2:
            rospy.logwarn("[path_stroke_preview_uv] 轨迹点过少，无法生成插值")
            return False

        self._surface_points_resampled = points_resampled
        self._interpolated_frames = []
        prev_axes = None
        prev_normal = None

        # Kidney: first point surface normal (+ -90deg on frame 0); later points lock Z to that normal.
        # Spine: same first-point frame; later points SLERP Z from first normal -> base -Z; Y toward start.
        use_kidney_style_pose = self.traj_kind == "kidney"
        is_spine = self.traj_kind == "spine"
        first_z_axis: Optional[np.ndarray] = None
        spine_z_start: Optional[np.ndarray] = None
        spine_z_end = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        n_pts = len(points_resampled)
        first_surface_pt = np.asarray(points_resampled[0], dtype=np.float64)

        for idx, surface_point in enumerate(points_resampled):
            use_y_toward_first = False
            z_axis: Optional[np.ndarray] = None
            if use_kidney_style_pose and idx >= 1 and first_z_axis is not None:
                use_y_toward_first = True
                z_axis = first_z_axis / (np.linalg.norm(first_z_axis) or 1.0)
            elif is_spine and idx >= 1 and spine_z_start is not None:
                use_y_toward_first = True
                t = float(idx) / float(n_pts - 1) if n_pts > 1 else 0.0
                z_axis = self._slerp_unit_vectors(spine_z_start, spine_z_end, t)

            if use_y_toward_first and z_axis is not None:
                dir_to_first = first_surface_pt - np.asarray(surface_point, dtype=np.float64)
                y_axis = dir_to_first - np.dot(dir_to_first, z_axis) * z_axis
                if np.linalg.norm(y_axis) < 1e-6:
                    # Degenerate (point coincides with first point or aligned with Z): reuse previous Y.
                    y_ref = prev_axes[1] if prev_axes is not None else np.array([0.0, 1.0, 0.0], dtype=np.float64)
                    y_axis = y_ref - np.dot(y_ref, z_axis) * z_axis
                    if np.linalg.norm(y_axis) < 1e-6:
                        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                y_axis = y_axis / (np.linalg.norm(y_axis) or 1.0)
                x_axis = np.cross(y_axis, z_axis)
                if np.linalg.norm(x_axis) < 1e-6:
                    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                x_axis = x_axis / (np.linalg.norm(x_axis) or 1.0)
                # Re-orthogonalize Y to guarantee a right-handed orthonormal frame.
                y_axis = np.cross(z_axis, x_axis)
                y_axis = y_axis / (np.linalg.norm(y_axis) or 1.0)

                axes = (x_axis, y_axis, z_axis)
                prev_axes = axes
                prev_normal = z_axis.copy()

                position = self._offset_along_normal(surface_point, z_axis, self.approach_offset)
                qx, qy, qz, qw = self._axes_to_quaternion(axes)
                pose = PoseStamped()
                pose.header.frame_id = self.target_frame
                pose.header.stamp = rospy.Time.now()
                pose.pose.position.x = float(position[0])
                pose.pose.position.y = float(position[1])
                pose.pose.position.z = float(position[2])
                pose.pose.orientation.x = qx
                pose.pose.orientation.y = qy
                pose.pose.orientation.z = qz
                pose.pose.orientation.w = qw
                self._interpolated_frames.append(pose)

                if idx == len(points_resampled) - 1:
                    self._end_axes = axes
                    self._end_normal = z_axis.copy()
                continue

            normal = self._estimate_normal(surface_point)
            if normal is None:
                if prev_normal is None:
                    rospy.logwarn("[path_stroke_preview_uv] 第 %d 个轨迹点法向量计算失败，放弃生成", idx + 1)
                    return False
                normal = prev_normal.copy()

            tangent = self._compute_tangent(points_resampled, idx)
            forward = tangent - np.dot(tangent, normal) * normal
            if np.linalg.norm(forward) < 1e-6:
                forward = tangent.copy()

            normal = normal / (np.linalg.norm(normal) or 1.0)
            forward = forward / (np.linalg.norm(forward) or 1.0)

            if idx == len(points_resampled) - 1:
                if len(points_resampled) >= 2:
                    prev_vec = points_resampled[idx - 1] - surface_point
                    if np.linalg.norm(prev_vec) > 1e-6:
                        x_axis = prev_vec
                    else:
                        x_axis = -forward
                else:
                    x_axis = -forward
            else:
                x_axis = -forward

            x_axis = x_axis - np.dot(x_axis, normal) * normal
            if np.linalg.norm(x_axis) < 1e-6:
                x_axis = -forward - np.dot(-forward, normal) * normal
            if np.linalg.norm(x_axis) < 1e-6:
                x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            x_axis = x_axis / (np.linalg.norm(x_axis) or 1.0)

            y_axis = np.cross(normal, x_axis)
            if np.linalg.norm(y_axis) < 1e-6:
                if prev_axes is not None:
                    y_axis = prev_axes[1].copy()
                else:
                    y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                y_axis = y_axis - np.dot(y_axis, normal) * normal
                if np.linalg.norm(y_axis) < 1e-6:
                    y_axis = np.cross(normal, x_axis)
            y_axis = y_axis / (np.linalg.norm(y_axis) or 1.0)

            x_axis = np.cross(y_axis, normal)
            x_axis = x_axis / (np.linalg.norm(x_axis) or 1.0)

            # Rotate tool axes around surface normal (+Z): -90deg (CW), x' = -y, y' = x
            x_axis_rot = (-y_axis).copy()
            y_axis_rot = x_axis.copy()

            prev_axes = (x_axis_rot, y_axis_rot, normal.copy())
            prev_normal = normal.copy()

            position = self._offset_along_normal(surface_point, normal, self.approach_offset)
            qx, qy, qz, qw = self._axes_to_quaternion(prev_axes)

            pose = PoseStamped()
            pose.header.frame_id = self.target_frame
            pose.header.stamp = rospy.Time.now()
            pose.pose.position.x = float(position[0])
            pose.pose.position.y = float(position[1])
            pose.pose.position.z = float(position[2])
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            self._interpolated_frames.append(pose)

            if idx == 0:
                self._start_axes = prev_axes
                self._start_normal = normal.copy()
                if use_kidney_style_pose:
                    first_z_axis = normal.copy()
                if is_spine:
                    spine_z_start = normal.copy()
            if idx == len(points_resampled) - 1:
                self._end_axes = prev_axes
                self._end_normal = normal.copy()

        if self._trajectory_has_base_offset():
            self._apply_trajectory_base_offsets_to_frames()

        self._publish_start_end_poses()
        self._publish_path_markers()
        self._save_yaml()
        rospy.loginfo("[path_stroke_preview_uv] ✓ 轨迹生成成功，共 %d 个坐标系", len(self._interpolated_frames))
        rospy.loginfo("[path_stroke_preview_uv] ✓ 结果写入 %s", self.output_yaml)
        return True

    def _standoff_direction(self, normal: np.ndarray) -> np.ndarray:
        """
        Direction from the surface toward the robot (legacy standoff used +base Z).

        Tool-frame +Z is `normal` (often body-side, normal.z <= 0). Standoff is opposite
        to tool +Z when that axis points into the body, so the path sits above the surface.
        """
        n = np.asarray(normal, dtype=np.float64)
        n = n / (np.linalg.norm(n) or 1.0)
        base_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if float(np.dot(n, base_z)) <= 0.0:
            return -n
        return n

    def _offset_along_normal(self, surface_point: np.ndarray, normal: np.ndarray, offset_m: float) -> np.ndarray:
        """Offset from the surface along the outward standoff direction (above the surface)."""
        direction = self._standoff_direction(normal)
        return np.asarray(surface_point, dtype=np.float64) + direction * float(offset_m)

    def _resample_path(self, points: List[np.ndarray], spacing: float) -> List[np.ndarray]:
        points_np = [np.array(p, dtype=np.float64) for p in points]
        if len(points_np) < 2:
            return points_np
        cumulative = [0.0]
        for i in range(1, len(points_np)):
            dist = np.linalg.norm(points_np[i] - points_np[i - 1])
            cumulative.append(cumulative[-1] + dist)

        total_length = cumulative[-1]
        if total_length < 1e-6:
            return [points_np[0].copy(), points_np[-1].copy()]

        num_samples = max(2, int(math.ceil(total_length / spacing)) + 1)
        sample_dists = np.linspace(0.0, total_length, num_samples)

        sampled_points: List[np.ndarray] = []
        for s in sample_dists:
            for i in range(1, len(cumulative)):
                if s <= cumulative[i]:
                    ratio = 0.0
                    if cumulative[i] > cumulative[i - 1]:
                        ratio = (s - cumulative[i - 1]) / (cumulative[i] - cumulative[i - 1])
                    point = points_np[i - 1] + ratio * (points_np[i] - points_np[i - 1])
                    sampled_points.append(point)
                    break
        if np.linalg.norm(sampled_points[-1] - points_np[-1]) > 1e-6:
            sampled_points[-1] = points_np[-1].copy()
        return sampled_points

    def _compute_tangent(self, points: List[np.ndarray], idx: int) -> np.ndarray:
        if idx < len(points) - 1:
            tangent = points[idx + 1] - points[idx]
        else:
            tangent = points[idx] - points[idx - 1]
        if np.linalg.norm(tangent) < 1e-9:
            tangent = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return tangent / (np.linalg.norm(tangent) or 1.0)

    def _estimate_normal(self, point: np.ndarray) -> Optional[np.ndarray]:
        with self._pointcloud_lock:
            kdtree = self._kdtree
            cloud_points = self._cloud_points
            cloud_frame = self._cloud_frame
        if kdtree is None or cloud_points is None or cloud_frame != self.target_frame:
            return None
        try:
            distances, indices = kdtree.query(point, k=self.num_neighbors + 1, distance_upper_bound=self.neighbor_search_radius)
        except Exception:
            return None
        valid_mask = np.isfinite(distances)
        valid_indices = indices[valid_mask]
        if len(valid_indices) < 3:
            return None
        neighbors = cloud_points[valid_indices]
        centroid = neighbors.mean(axis=0)
        neighbors_centered = neighbors - centroid
        _, _, Vt = np.linalg.svd(neighbors_centered, full_matrices=False)
        normal = Vt[-1, :]
        if np.linalg.norm(normal) < 1e-10:
            return None
        normal = normal / np.linalg.norm(normal)
        if normal[2] > 0:
            normal = -normal
        return normal

    def _ensure_point_cloud(self) -> bool:
        if not self.use_depthcloud:
            with self._pointcloud_lock:
                return self._kdtree is not None and self._cloud_frame == self.target_frame
        if self._camera_info is None:
            rospy.logwarn("[path_stroke_preview_uv] 尚未收到 CameraInfo")
            return False
        if len(self._depth_buffer) < max(self.min_frames_for_detection, 1):
            rospy.logwarn("[path_stroke_preview_uv] 深度帧数量不足 (%d)", len(self._depth_buffer))
            return False
        frames = list(self._depth_buffer)[-self.frame_buffer_size:]
        aggregated = []
        for depth_msg in frames:
            points, frame = self._depth_to_cloud(depth_msg, self._camera_info)
            if points is None or points.shape[0] == 0:
                continue
            points_target = self._transform_points_array(points, frame, self.target_frame)
            if points_target is None or points_target.shape[0] == 0:
                continue
            aggregated.append(points_target)
        if not aggregated:
            rospy.logwarn("[path_stroke_preview_uv] 聚合点云失败，无有效点")
            return False
        points_all = np.vstack(aggregated)
        if self.voxel_size > 0.0 and o3d is not None:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_all))
            pcd = pcd.voxel_down_sample(self.voxel_size)
            points_all = np.asarray(pcd.points)
        if points_all.shape[0] > self.max_points:
            step = max(1, points_all.shape[0] // self.max_points)
            points_all = points_all[::step]
        if points_all.shape[0] < 30:
            rospy.logwarn("[path_stroke_preview_uv] 聚合点云过少 (%d)", points_all.shape[0])
            return False
        with self._pointcloud_lock:
            self._cloud_points = points_all.astype(np.float32)
            self._kdtree = KDTree(self._cloud_points)
            self._cloud_frame = self.target_frame
        return True

    def _depth_to_cloud(self, depth_msg: Image, cam_info: CameraInfo) -> Tuple[Optional[np.ndarray], Optional[str]]:
        depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0
        else:
            depth = depth.astype(np.float32)
        K = np.array(cam_info.K).reshape(3, 3)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        Z = depth
        valid = (Z > 0.05) & (Z < 3.0) & np.isfinite(Z)
        if not np.any(valid):
            return None, None
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        points = np.stack((X[valid], Y[valid], Z[valid]), axis=1).astype(np.float32)
        return points, depth_msg.header.frame_id

    def _transform_points_array(self, points: np.ndarray, from_frame: str, to_frame: str) -> Optional[np.ndarray]:
        try:
            transform = self._tf_buffer.lookup_transform(to_frame, from_frame, rospy.Time(0), rospy.Duration(0.5))
        except Exception as e:
            rospy.logwarn("[path_stroke_preview_uv] TF 转换失败 (%s -> %s): %s", from_frame, to_frame, str(e))
            return None
        q = transform.transform.rotation
        qw, qx, qy, qz = q.w, q.x, q.y, q.z
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
        ], dtype=np.float64)
        t = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ], dtype=np.float64)
        points_out = (R @ points.T).T + t
        return points_out.astype(np.float32)

    def _pixels_to_camera_points(self, pixels: List[Tuple[int, int]], depth: np.ndarray, cam_info: CameraInfo) -> List[np.ndarray]:
        K = np.array(cam_info.K).reshape(3, 3)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        h, w = depth.shape
        points: List[np.ndarray] = []
        for (x, y) in pixels:
            if x < 0 or x >= w or y < 0 or y >= h:
                continue
            z = depth[y, x]
            if not np.isfinite(z) or z <= 0.05 or z > 3.0:
                z = self._search_valid_depth(depth, x, y)
                if not np.isfinite(z) or z <= 0.05 or z > 3.0:
                    continue
            X = (x - cx) * z / fx
            Y = (y - cy) * z / fy
            points.append(np.array([X, Y, z], dtype=np.float64))
        return points

    def _search_valid_depth(self, depth: np.ndarray, x: int, y: int) -> float:
        h, w = depth.shape
        window = max(1, self.depth_search_window)
        for r in range(1, window + 1):
            x_min = max(0, x - r)
            x_max = min(w - 1, x + r)
            y_min = max(0, y - r)
            y_max = min(h - 1, y + r)
            patch = depth[y_min:y_max + 1, x_min:x_max + 1]
            valid = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 3.0)]
            if valid.size > 0:
                return float(np.median(valid))
        return float("nan")

    def _build_path_markers_marker_array(self, use_surface_positions: bool) -> MarkerArray:
        """Build axis markers; surface positions omit approach_offset (on-body 3D points)."""
        path_markers = MarkerArray()
        raw_pts = self._surface_points_resampled if use_surface_positions else None
        for idx, frame_pose in enumerate(self._interpolated_frames):
            if raw_pts is not None and idx < len(raw_pts):
                pose = PoseStamped()
                pose.header.frame_id = self.target_frame
                pose.header.stamp = rospy.Time.now()
                pose.pose.position.x = float(raw_pts[idx][0])
                pose.pose.position.y = float(raw_pts[idx][1])
                pose.pose.position.z = float(raw_pts[idx][2])
                pose.pose.orientation = frame_pose.pose.orientation
            else:
                pose = frame_pose
            R = self._quaternion_to_matrix(
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w,
            )
            axes = (R[:, 0], R[:, 1], R[:, 2])
            ns = f"path_frame_surface_{idx}" if use_surface_positions else f"path_frame_{idx}"
            path_markers.markers.extend(self._build_axes_markers(
                pose,
                axes,
                namespace=ns,
                colors=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
            ))
        return path_markers

    def _publish_path_markers(self) -> None:
        if not self._interpolated_frames:
            return
        # Executed poses (with approach_offset); viz_no_offset can still override legacy path_markers.
        use_raw = bool(rospy.get_param("~viz_no_offset", False))
        self.pub_path_markers.publish(self._build_path_markers_marker_array(use_surface_positions=use_raw))
        # Always publish on-body surface points (before approach_offset).
        if self._surface_points_resampled:
            self.pub_path_markers_surface.publish(self._build_path_markers_marker_array(use_surface_positions=True))

        line = Marker()
        line.header.frame_id = self.target_frame
        line.header.stamp = rospy.Time.now()
        line.ns = "path_line"
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.004
        line.color.r = 0.2
        line.color.g = 0.8
        line.color.b = 0.2
        line.color.a = 0.9
        if use_raw and self._surface_points_resampled:
            for p3 in self._surface_points_resampled:
                p = GeoPoint()
                p.x = float(p3[0])
                p.y = float(p3[1])
                p.z = float(p3[2])
                line.points.append(p)
        else:
            for pose in self._interpolated_frames:
                p = GeoPoint()
                p.x = pose.pose.position.x
                p.y = pose.pose.position.y
                p.z = pose.pose.position.z
                line.points.append(p)
        self.pub_path_line.publish(line)

    def _publish_start_end_poses(self) -> None:
        if not self._surface_points_resampled or self._start_axes is None or self._start_normal is None:
            return
        start_surface = self._surface_points_resampled[0]
        # Visualization only: if true, show raw surface points (no approach/pre-approach offsets).
        # Default False so RViz preview matches the actual executed poses written into YAML.
        use_raw = bool(rospy.get_param("~viz_no_offset", False))
        if use_raw:
            pre_position = start_surface
            final_position = start_surface
        else:
            start_n = self._start_normal if self._start_normal is not None else np.array([0.0, 0.0, 1.0], dtype=np.float64)
            pre_position = self._offset_along_normal(start_surface, start_n, self.pre_approach_offset)
            final_position = self._offset_along_normal(start_surface, start_n, self.approach_offset)
        pre_position = self._shift_trajectory_base_offsets(pre_position)
        final_position = self._shift_trajectory_base_offsets(final_position)
        qx, qy, qz, qw = self._axes_to_quaternion(self._start_axes)

        pose_pre = PoseStamped()
        pose_pre.header.frame_id = self.target_frame
        pose_pre.header.stamp = rospy.Time.now()
        pose_pre.pose.position.x = float(pre_position[0])
        pose_pre.pose.position.y = float(pre_position[1])
        pose_pre.pose.position.z = float(pre_position[2])
        pose_pre.pose.orientation.x = qx
        pose_pre.pose.orientation.y = qy
        pose_pre.pose.orientation.z = qz
        pose_pre.pose.orientation.w = qw
        self.pub_start_pre_pose.publish(pose_pre)

        pose_final = PoseStamped()
        pose_final.header.frame_id = self.target_frame
        pose_final.header.stamp = rospy.Time.now()
        pose_final.pose.position.x = float(final_position[0])
        pose_final.pose.position.y = float(final_position[1])
        pose_final.pose.position.z = float(final_position[2])
        pose_final.pose.orientation.x = qx
        pose_final.pose.orientation.y = qy
        pose_final.pose.orientation.z = qz
        pose_final.pose.orientation.w = qw
        self.pub_start_final_pose.publish(pose_final)

        if self._end_axes is None or self._end_normal is None:
            return
        end_surface = self._surface_points_resampled[-1]
        if use_raw:
            end_position = end_surface
        else:
            end_n = self._end_normal if self._end_normal is not None else np.array([0.0, 0.0, 1.0], dtype=np.float64)
            end_position = self._offset_along_normal(end_surface, end_n, self.approach_offset)
        end_position = self._shift_trajectory_base_offsets(end_position)
        qx_e, qy_e, qz_e, qw_e = self._axes_to_quaternion(self._end_axes)
        pose_end = PoseStamped()
        pose_end.header.frame_id = self.target_frame
        pose_end.header.stamp = rospy.Time.now()
        pose_end.pose.position.x = float(end_position[0])
        pose_end.pose.position.y = float(end_position[1])
        pose_end.pose.position.z = float(end_position[2])
        pose_end.pose.orientation.x = qx_e
        pose_end.pose.orientation.y = qy_e
        pose_end.pose.orientation.z = qz_e
        pose_end.pose.orientation.w = qw_e
        self.pub_end_final_pose.publish(pose_end)

    def _save_yaml(self) -> None:
        if not self._surface_points_resampled or self._start_axes is None or self._start_normal is None \
                or self._end_axes is None or self._end_normal is None:
            return

        start_surface = self._surface_points_resampled[0]
        end_surface = self._surface_points_resampled[-1]

        start_n = self._start_normal if self._start_normal is not None else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        end_n = self._end_normal if self._end_normal is not None else start_n
        start_pre_pos = self._offset_along_normal(start_surface, start_n, self.pre_approach_offset)
        start_final_pos = self._offset_along_normal(start_surface, start_n, self.approach_offset)
        start_pre_pos = self._shift_trajectory_base_offsets(start_pre_pos)
        start_final_pos = self._shift_trajectory_base_offsets(start_final_pos)
        qx_s, qy_s, qz_s, qw_s = self._axes_to_quaternion(self._start_axes)

        end_final_pos = self._offset_along_normal(end_surface, end_n, self.approach_offset)
        end_final_pos = self._shift_trajectory_base_offsets(end_final_pos)
        qx_e, qy_e, qz_e, qw_e = self._axes_to_quaternion(self._end_axes)

        frames_data = []
        for idx, pose in enumerate(self._interpolated_frames):
            frames_data.append({
                "id": int(idx + 1),
                "frame_id": pose.header.frame_id,
                "position": {
                    "x": float(pose.pose.position.x),
                    "y": float(pose.pose.position.y),
                    "z": float(pose.pose.position.z),
                },
                "orientation": {
                    "x": float(pose.pose.orientation.x),
                    "y": float(pose.pose.orientation.y),
                    "z": float(pose.pose.orientation.z),
                    "w": float(pose.pose.orientation.w),
                },
            })

        data = {
            "pre_approach_offset": self.pre_approach_offset,
            "approach_offset": self.approach_offset,
            "start": {
                "surface_point": {
                    "x": float(start_surface[0]),
                    "y": float(start_surface[1]),
                    "z": float(start_surface[2]),
                },
                "pre_pose": {
                    "frame_id": self.target_frame,
                    "position": {
                        "x": float(start_pre_pos[0]),
                        "y": float(start_pre_pos[1]),
                        "z": float(start_pre_pos[2]),
                    },
                    "orientation": {
                        "x": qx_s,
                        "y": qy_s,
                        "z": qz_s,
                        "w": qw_s,
                    },
                },
                "final_pose": {
                    "frame_id": self.target_frame,
                    "position": {
                        "x": float(start_final_pos[0]),
                        "y": float(start_final_pos[1]),
                        "z": float(start_final_pos[2]),
                    },
                    "orientation": {
                        "x": qx_s,
                        "y": qy_s,
                        "z": qz_s,
                        "w": qw_s,
                    },
                },
            },
            "end": {
                "surface_point": {
                    "x": float(end_surface[0]),
                    "y": float(end_surface[1]),
                    "z": float(end_surface[2]),
                },
                "final_pose": {
                    "frame_id": self.target_frame,
                    "position": {
                        "x": float(end_final_pos[0]),
                        "y": float(end_final_pos[1]),
                        "z": float(end_final_pos[2]),
                    },
                    "orientation": {
                        "x": qx_e,
                        "y": qy_e,
                        "z": qz_e,
                        "w": qw_e,
                    },
                },
            },
            "frames": frames_data,
        }

        target_dir = os.path.dirname(self.output_yaml)
        if target_dir and not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        with open(self.output_yaml, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False)

    # ---- math utils copied from original ----
    def _build_axes_markers(self, pose: PoseStamped, axes: Tuple[np.ndarray, np.ndarray, np.ndarray],
                            namespace: str, colors: List[Tuple[float, float, float]]) -> List[Marker]:
        markers: List[Marker] = []
        for idx, (axis_vec, color) in enumerate(zip(axes, colors)):
            marker = Marker()
            marker.header.frame_id = pose.header.frame_id
            marker.header.stamp = rospy.Time.now()
            marker.ns = namespace
            marker.id = idx
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.lifetime = rospy.Duration(0)

            start_point = GeoPoint()
            start_point.x = pose.pose.position.x
            start_point.y = pose.pose.position.y
            start_point.z = pose.pose.position.z

            end_point = GeoPoint()
            end_point.x = pose.pose.position.x + axis_vec[0] * 0.06
            end_point.y = pose.pose.position.y + axis_vec[1] * 0.06
            end_point.z = pose.pose.position.z + axis_vec[2] * 0.06

            marker.points = [start_point, end_point]
            marker.scale.x = 0.003
            marker.scale.y = 0.0075
            marker.scale.z = 0.0
            marker.color.a = 1.0
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            markers.append(marker)
        return markers

    def _axes_to_quaternion(self, axes: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[float, float, float, float]:
        R = np.array([axes[0], axes[1], axes[2]], dtype=np.float64).T
        return self._rotation_matrix_to_quaternion(R)

    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> Tuple[float, float, float, float]:
        trace = np.trace(R)
        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        return (float(x), float(y), float(z), float(w))

    def _trajectory_base_x_offset_m(self) -> float:
        if self.traj_kind == "spine":
            return float(self.spine_base_x_offset_m)
        if self.traj_kind == "kidney" and self.kidney_side == "left":
            return float(self.kidney_left_base_x_offset_m)
        return 0.0

    def _trajectory_base_y_offset_m(self) -> float:
        if self.traj_kind == "kidney" and self.kidney_side == "left":
            return float(self.kidney_left_base_y_offset_m)
        return 0.0

    def _trajectory_has_base_offset(self) -> bool:
        return abs(self._trajectory_base_x_offset_m()) > 1e-9 or abs(self._trajectory_base_y_offset_m()) > 1e-9

    def _shift_trajectory_base_offsets(self, position: np.ndarray) -> np.ndarray:
        """Shift position along target_frame +X / +Y when organ-specific offsets are active."""
        dx = self._trajectory_base_x_offset_m()
        dy = self._trajectory_base_y_offset_m()
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return np.asarray(position, dtype=np.float64)
        out = np.asarray(position, dtype=np.float64).copy()
        out[0] += dx
        out[1] += dy
        return out

    def _apply_trajectory_base_offsets_to_frames(self) -> None:
        dx = self._trajectory_base_x_offset_m()
        dy = self._trajectory_base_y_offset_m()
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return
        for pose in self._interpolated_frames:
            pose.pose.position.x += dx
            pose.pose.position.y += dy
        label = self.traj_kind
        if self.traj_kind == "kidney":
            label = f"kidney ({self.kidney_side})"
        parts = []
        if abs(dx) > 1e-9:
            parts.append(f"+{dx * 1000.0:.1f} mm {self.target_frame} +X")
        if abs(dy) > 1e-9:
            parts.append(f"+{dy * 1000.0:.1f} mm {self.target_frame} +Y")
        rospy.loginfo(
            "[path_stroke_preview_uv] %s: shifted %d path frames (%s)",
            label,
            len(self._interpolated_frames),
            ", ".join(parts),
        )

    def _quaternion_to_matrix(self, qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        x2 = qx + qx
        y2 = qy + qy
        z2 = qz + qz
        xx = qx * x2
        xy = qx * y2
        xz = qx * z2
        yy = qy * y2
        yz = qy * z2
        zz = qz * z2
        wx = qw * x2
        wy = qw * y2
        wz = qw * z2
        return np.array([
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ], dtype=np.float64)

    def _on_shutdown(self) -> None:
        if not getattr(self, "no_gui", False):
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == "__main__":
    PathStrokePreviewUvNode()


