#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于鼠标轨迹绘制的轨迹预览节点

使用方式：
1. 启动节点后会弹出 OpenCV 窗口，实时显示 RGB 图像。
2. 按住鼠标左键在窗口内绘制轨迹，松开左键结束绘制；按 `c` 键可清空轨迹。
3. 绘制完成后，调用 `rosservice call /path_stroke_preview/generate` 触发行生成。
   节点会沿轨迹均匀采样，估算法向量并构建姿态，让 X 轴指向前进方向的反方向，Z 轴贴合表面法向量。
4. 生成结果会以 Marker 形式发布，并保存到 YAML（默认 `~/.ros/path_preview.yaml`），可直接供 `path_execute.py` 使用。
"""

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

from geometry_msgs.msg import PointStamped, PoseStamped, Point as GeoPoint
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from sensor_msgs import point_cloud2 as pc2
from std_srvs.srv import Trigger, TriggerResponse
from visualization_msgs.msg import Marker, MarkerArray

try:
	import open3d as o3d  # type: ignore
except Exception:
	o3d = None


class PathStrokePreviewNode(object):
	def __init__(self) -> None:
		rospy.init_node("path_stroke_preview", anonymous=False)

		# 坐标系配置
		self.camera_frame = rospy.get_param("~camera_frame", "camera_base")
		self.target_frame = rospy.get_param("~target_frame", "iiwa_link_0")
		self.visualization_frame = rospy.get_param("~visualization_frame", "camera_base")

		# 轨迹采样参数
		self.frame_spacing = float(rospy.get_param("~frame_spacing", 0.02))
		self.pre_approach_offset = float(rospy.get_param("~pre_approach_offset", 0.15))
		self.approach_offset = float(rospy.get_param("~approach_offset", 0.08))

		self.output_yaml = os.path.expanduser(rospy.get_param("~output_yaml", "~/.ros/path_preview.yaml"))

		# 点云/深度相关
		self.use_depthcloud = rospy.get_param("~use_depthcloud", True)
		if self.use_depthcloud and o3d is None:
			raise RuntimeError("Open3D 未安装，无法在 DepthCloud 模式下生成点云")

		# 注意：很多环境只有 /rgb/image_raw，没有 /rgb/image_rect_color。
		# 若订阅不到图像，_update_display() 不会调用 waitKey()，窗口会看起来“卡死”。
		self.rgb_topic = rospy.get_param("~rgb_topic", "/rgb/image_raw")
		self.depth_topic = rospy.get_param("~depth_topic", "/depth_to_rgb/image_raw")
		self.camera_info_topic = rospy.get_param("~camera_info_topic", "/rgb/camera_info")
		self.pointcloud_topic = rospy.get_param("~pointcloud_topic", "/points2_down")

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
		self._drawing: bool = False
		self._current_stroke: List[Tuple[int, int]] = []
		self._stroke_pixels: List[Tuple[int, int]] = []

		self._surface_points_resampled: List[np.ndarray] = []
		self._interpolated_frames: List[PoseStamped] = []
		self._start_axes: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
		self._start_normal: Optional[np.ndarray] = None
		self._end_axes: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
		self._end_normal: Optional[np.ndarray] = None

		# 可视化
		self.pub_path_markers = rospy.Publisher("~path_markers", MarkerArray, queue_size=1, latch=True)
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
		self.srv_generate = rospy.Service("~generate", Trigger, self._srv_generate_cb)

		# OpenCV 窗口
		self.window_name = rospy.get_param("~window_name", "PathStrokePreview")
		cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
		cv2.setMouseCallback(self.window_name, self._mouse_cb)

		rospy.on_shutdown(self._on_shutdown)

		rospy.loginfo("[path_stroke_preview] ▶ 按住左键在窗口内绘制轨迹，完成后调用服务: rosservice call /path_stroke_preview/generate")

		rate = rospy.Rate(30)
		while not rospy.is_shutdown():
			self._update_display()
			rate.sleep()

	# --------------------------------------------------------------------------
	# 订阅回调

	def _on_camera_info(self, msg: CameraInfo) -> None:
		if self._camera_info is None:
			self._camera_info = msg
			rospy.loginfo("[path_stroke_preview] 已接收相机内参：%dx%d", msg.width, msg.height)

	def _on_rgb_depth(self, rgb_msg: Image, depth_msg: Image) -> None:
		if self._camera_info is None:
			return
		try:
			cv_rgb = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
		except Exception as e:
			rospy.logwarn("[path_stroke_preview] 转换 RGB 图像失败: %s", str(e))
			return
		try:
			cv_depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
		except Exception as e:
			rospy.logwarn("[path_stroke_preview] 转换深度图失败: %s", str(e))
			return

		if cv_depth.dtype == np.uint16:
			cv_depth = cv_depth.astype(np.float32) / 1000.0
		else:
			cv_depth = cv_depth.astype(np.float32)

		with self._image_lock:
			self._latest_rgb = cv_rgb
			self._latest_depth = cv_depth
			self._depth_frame_id = depth_msg.header.frame_id

		self._depth_buffer.append(depth_msg)

	def _on_rgb_only(self, rgb_msg: Image) -> None:
		try:
			cv_rgb = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
		except Exception as e:
			rospy.logwarn("[path_stroke_preview] 转换 RGB 图像失败: %s", str(e))
			return
		with self._image_lock:
			self._latest_rgb = cv_rgb

	def _on_pointcloud(self, msg: PointCloud2) -> None:
		points = []
		for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
			points.append([p[0], p[1], p[2]])
		if len(points) < 10:
			return
		with self._pointcloud_lock:
			self._cloud_points = np.array(points, dtype=np.float32)
			self._kdtree = KDTree(self._cloud_points)
			self._cloud_frame = msg.header.frame_id

	# --------------------------------------------------------------------------
	# 鼠标 & 显示

	def _mouse_cb(self, event: int, x: int, y: int, flags: int, param) -> None:
		with self._stroke_lock:
			if event == cv2.EVENT_LBUTTONDOWN:
				self._drawing = True
				self._current_stroke = [(x, y)]
			elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
				self._append_point_locked(x, y)
			elif event == cv2.EVENT_LBUTTONUP:
				if self._drawing:
					self._append_point_locked(x, y, force=True)
					self._drawing = False
					if len(self._current_stroke) >= 2:
						self._stroke_pixels = self._current_stroke[:self.max_stroke_points]
						rospy.loginfo("[path_stroke_preview] 捕捉到轨迹，共 %d 个像素点", len(self._stroke_pixels))
					else:
						rospy.logwarn("[path_stroke_preview] 轨迹长度不足，已忽略")
					self._current_stroke = []

	def _append_point_locked(self, x: int, y: int, force: bool = False) -> None:
		if not self._current_stroke:
			self._current_stroke.append((x, y))
			return
		px, py = self._current_stroke[-1]
		if force or math.hypot(x - px, y - py) >= self.pixel_sampling_step:
			self._current_stroke.append((x, y))
			if len(self._current_stroke) > self.max_stroke_points:
				self._current_stroke = self._current_stroke[-self.max_stroke_points :]

	def _update_display(self) -> None:
		with self._image_lock:
			if self._latest_rgb is None:
				# 没有图像也要跑事件循环，否则 OpenCV 窗口会卡住
				blank = np.zeros((480, 640, 3), dtype=np.uint8)
				cv2.putText(blank, f"Waiting for RGB on: {self.rgb_topic}",
					(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
				cv2.putText(blank, "Tip: set _rgb_topic:=/rgb/image_raw (or your actual topic)",
					(10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
				cv2.imshow(self.window_name, blank)
				cv2.waitKey(1)
				return
			image = self._latest_rgb.copy()
		with self._stroke_lock:
			current = copy.copy(self._current_stroke)
			stroke = copy.copy(self._stroke_pixels)
			drawing = self._drawing

		if stroke:
			pts = np.array(stroke, dtype=np.int32).reshape((-1, 1, 2))
			cv2.polylines(image, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
		if drawing and current:
			pts_curr = np.array(current, dtype=np.int32).reshape((-1, 1, 2))
			cv2.polylines(image, [pts_curr], isClosed=False, color=(0, 255, 0), thickness=2)

		cv2.putText(image, "Draw with LButton, press 'c' to clear, call service to generate",
			(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

		cv2.imshow(self.window_name, image)
		key = cv2.waitKey(1) & 0xFF
		if key == ord('c'):
			with self._stroke_lock:
				self._stroke_pixels = []
				self._current_stroke = []
				self._drawing = False
			rospy.loginfo("[path_stroke_preview] 已清空轨迹")

	# --------------------------------------------------------------------------
	# 服务回调

	def _srv_generate_cb(self, req) -> TriggerResponse:
		with self._stroke_lock:
			pixel_path = copy.copy(self._stroke_pixels)
		if len(pixel_path) < 2:
			return TriggerResponse(success=False, message="尚未绘制有效轨迹")

		if self._camera_info is None:
			return TriggerResponse(success=False, message="相机内参未就绪")

		if self.use_depthcloud:
			with self._image_lock:
				depth = None if self._latest_depth is None else self._latest_depth.copy()
				depth_frame = self._depth_frame_id
			if depth is None or depth_frame is None:
				return TriggerResponse(success=False, message="深度图尚未就绪")
			surface_points_camera = self._pixels_to_camera_points(pixel_path, depth, self._camera_info)
			if len(surface_points_camera) < 2:
				return TriggerResponse(success=False, message="像素轨迹对应的深度点过少")
			points_target = self._transform_points_array(
				np.asarray(surface_points_camera, dtype=np.float32),
				depth_frame,
				self.target_frame
			)
			if points_target is None or points_target.shape[0] < 2:
				return TriggerResponse(success=False, message="点转换到目标坐标系失败")
			surface_points = [points_target[i, :].astype(np.float64) for i in range(points_target.shape[0])]
		else:
			return TriggerResponse(success=False, message="当前节点仅在 use_depthcloud=True 时支持像素轨迹")

		if not self._ensure_point_cloud():
			return TriggerResponse(success=False, message="点云尚未就绪或转换失败")

		success = self._generate_path(surface_points)
		if success:
			return TriggerResponse(success=True, message="轨迹生成完成，已保存到 YAML")
		return TriggerResponse(success=False, message="轨迹生成失败，详情见日志")

	# --------------------------------------------------------------------------
	# 轨迹生成

	def _generate_path(self, surface_points: List[np.ndarray]) -> bool:
		points_resampled = self._resample_path(surface_points, self.frame_spacing)
		if len(points_resampled) < 2:
			rospy.logwarn("[path_stroke_preview] 轨迹点过少，无法生成插值")
			return False

		self._surface_points_resampled = points_resampled
		self._interpolated_frames = []
		prev_axes = None
		prev_normal = None

		for idx, surface_point in enumerate(points_resampled):
			normal = self._estimate_normal(surface_point)
			if normal is None:
				if prev_normal is None:
					rospy.logwarn("[path_stroke_preview] 第 %d 个轨迹点法向量计算失败，放弃生成", idx + 1)
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

			# 保证 X 轴与法向正交并归一化
			x_axis = x_axis - np.dot(x_axis, normal) * normal
			if np.linalg.norm(x_axis) < 1e-6:
				x_axis = -forward - np.dot(-forward, normal) * normal
			if np.linalg.norm(x_axis) < 1e-6:
				x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
			x_axis = x_axis / (np.linalg.norm(x_axis) or 1.0)

			# 构造 Y 轴并保持正交
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

			prev_axes = (x_axis.copy(), y_axis.copy(), normal.copy())
			prev_normal = normal.copy()

			# 平移 offset 改为沿 target_frame（通常 iiwa_link_0）的 +Z 方向，
			# 而不是沿表面法向 normal 方向。
			position = surface_point + np.array([0.0, 0.0, 1.0], dtype=np.float64) * self.approach_offset
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
			if idx == len(points_resampled) - 1:
				self._end_axes = prev_axes
				self._end_normal = normal.copy()

		self._publish_start_end_poses()
		self._publish_path_markers()
		self._save_yaml()
		rospy.loginfo("[path_stroke_preview] ✓ 轨迹生成成功，共 %d 个坐标系", len(self._interpolated_frames))
		rospy.loginfo("[path_stroke_preview] ✓ 结果写入 %s", self.output_yaml)
		return True

	# --------------------------------------------------------------------------
	# 轨迹采样与姿态构建

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
			distances, indices = kdtree.query(point, k=self.num_neighbors + 1,
				distance_upper_bound=self.neighbor_search_radius)
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

	# --------------------------------------------------------------------------
	# 可视化 & YAML

	def _publish_path_markers(self) -> None:
		if not self._interpolated_frames:
			return
		path_markers = MarkerArray()
		for idx, pose in enumerate(self._interpolated_frames):
			R = self._quaternion_to_matrix(
				pose.pose.orientation.x,
				pose.pose.orientation.y,
				pose.pose.orientation.z,
				pose.pose.orientation.w,
			)
			axes = (R[:, 0], R[:, 1], R[:, 2])
			path_markers.markers.extend(self._build_axes_markers(
				pose,
				axes,
				namespace=f"path_frame_{idx}",
				colors=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
			))
		self.pub_path_markers.publish(path_markers)

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
		base_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
		pre_position = start_surface + base_z * self.pre_approach_offset
		final_position = start_surface + base_z * self.approach_offset
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
		end_position = end_surface + base_z * self.approach_offset
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

		base_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
		start_pre_pos = start_surface + base_z * self.pre_approach_offset
		start_final_pos = start_surface + base_z * self.approach_offset
		qx_s, qy_s, qz_s, qw_s = self._axes_to_quaternion(self._start_axes)

		end_final_pos = end_surface + base_z * self.approach_offset
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

	# --------------------------------------------------------------------------
	# 点云辅助

	def _ensure_point_cloud(self) -> bool:
		if not self.use_depthcloud:
			with self._pointcloud_lock:
				return self._kdtree is not None and self._cloud_frame == self.target_frame

		if self._camera_info is None:
			rospy.logwarn("[path_stroke_preview] 尚未收到 CameraInfo")
			return False
		if len(self._depth_buffer) < max(self.min_frames_for_detection, 1):
			rospy.logwarn("[path_stroke_preview] 深度帧数量不足 (%d)", len(self._depth_buffer))
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
			rospy.logwarn("[path_stroke_preview] 聚合点云失败，无有效点")
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
			rospy.logwarn("[path_stroke_preview] 聚合点云过少 (%d)", points_all.shape[0])
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
			transform = self._tf_buffer.lookup_transform(
				to_frame,
				from_frame,
				rospy.Time(0),
				rospy.Duration(0.5)
			)
		except Exception as e:
			rospy.logwarn("[path_stroke_preview] TF 转换失败 (%s -> %s): %s", from_frame, to_frame, str(e))
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

	# --------------------------------------------------------------------------
	# 像素到三维点

	def _pixels_to_camera_points(self, pixels: List[Tuple[int, int]], depth: np.ndarray,
		cam_info: CameraInfo) -> List[np.ndarray]:
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

	# --------------------------------------------------------------------------
	# 坐标系与数学工具

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

	def _quaternion_to_matrix(self, qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
		x2 = qx + qx
		y2 = qy + qy
		z2 = qz + qz
		xqx = qx * x2
		xqy = qx * y2
		xqz = qx * z2
		yqy = qy * y2
		yqz = qy * z2
		zqz = qz * z2
		wqx = qw * x2
		wqy = qw * y2
		wqz = qw * z2
		return np.array([
			[1.0 - (yqy + zqz), xqy - wqz, xqz + wqy],
			[xqy + wqz, 1.0 - (xqx + zqz), yqz - wqx],
			[xqz - wqy, yqz + wqx, 1.0 - (xqx + yqy)]
		], dtype=np.float64)

	# --------------------------------------------------------------------------
	# 资源释放

	def _on_shutdown(self) -> None:
		try:
			cv2.destroyWindow(self.window_name)
		except Exception:
			pass


if __name__ == "__main__":
	try:
		PathStrokePreviewNode()
	except rospy.ROSInterruptException:
		pass

