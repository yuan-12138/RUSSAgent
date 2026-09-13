#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
路径执行节点

1. 读取 `path_preview.py` 生成的 YAML，获取起点 `pre/final` 姿态以及插值的路径坐标系。
2. 将当前机械臂从当前位置移动到 `start_pre_pose`（位置控制线性插补）。
3. 再移动到 `start_final_pose`（即路径的第 1 个坐标系）。
4. 按 lookahead 逻辑沿路径逐段发送目标姿态，仅在首段加速、末段减速，中间段保持匀速。

默认通过 /iiwa/command/CartesianPoseLin 控制机械臂，订阅 /iiwa/state/CartesianPose 获取反馈。
"""

import os
import yaml
import math
import time
import copy
import threading
from typing import List, Tuple, Optional

import numpy as np
import rospy

from geometry_msgs.msg import PoseStamped

try:
	from iiwa_msgs.msg import CartesianPose, ControlMode, CartesianQuantity
	from iiwa_msgs.srv import ConfigureControlMode, ConfigureControlModeRequest
	_HAVE_CARTESIAN = True
	_HAVE_CTRL = True
except ImportError:
	try:
		from iiwa_msgs.msg import CartesianPose
		_HAVE_CARTESIAN = True
	except ImportError:
		_HAVE_CARTESIAN = False
		CartesianPose = None
	ControlMode = None
	CartesianQuantity = None
	ConfigureControlMode = None
	ConfigureControlModeRequest = None
	_HAVE_CTRL = False


class PathExecuteNode(object):
	def __init__(self) -> None:
		rospy.init_node("path_execute", anonymous=False)

		self.path_yaml = os.path.expanduser(rospy.get_param("~path_yaml", "~/.ros/path_preview.yaml"))
		self.linear_step_size = float(rospy.get_param("~linear_step_size", 0.01))
		self.linear_rate = float(rospy.get_param("~linear_rate", 10.0))
		self.error_position_tolerance = float(rospy.get_param("~error_position_tolerance", 0.01))
		# 位置误差判定（可选：忽略/放宽 Z 方向）
		self.ignore_z_error = bool(rospy.get_param("~ignore_z_error", True))
		self.error_position_tolerance_xy = float(
			rospy.get_param("~error_position_tolerance_xy", 0.02)
		)
		self.error_position_tolerance_z = float(
			rospy.get_param(
				"~error_position_tolerance_z",
				(1e9 if self.ignore_z_error else self.error_position_tolerance),
			)
		)
		self.error_angle_tolerance = float(rospy.get_param("~error_angle_tolerance", math.radians(3.0)))
		self.error_timeout = float(rospy.get_param("~error_timeout", 60.0))
		self.lookahead_distance = float(rospy.get_param("~lookahead_distance", 0.01))
		# lookahead 也支持 Z 方向忽略/放宽
		self.lookahead_distance_xy = float(rospy.get_param("~lookahead_distance_xy", self.lookahead_distance))
		self.lookahead_distance_z = float(
			rospy.get_param(
				"~lookahead_distance_z",
				(1e9 if self.ignore_z_error else self.lookahead_distance),
			)
		)
		self.lookahead_publish_delay = float(rospy.get_param("~lookahead_publish_delay", 0.0))
		self.pre_stage_pause = float(rospy.get_param("~pre_stage_pause", 1.0))

		self.auto_configure_position = bool(rospy.get_param("~auto_configure_position", True))
		self.auto_configure_impedance = bool(rospy.get_param("~auto_configure_impedance", True))
		self.impedance_service = rospy.get_param("~impedance_service", "/iiwa/configuration/ConfigureControlMode")

		self.impedance_stiff_vals = self._load_cartesian_values("~impedance_stiff", (1200.0, 1200.0, 400.0, 50.0, 50.0, 150.0))
		self.impedance_damp_vals = self._load_cartesian_values("~impedance_damp", (0.8, 0.8, 0.8, 0.8, 0.8, 0.8))
		self.impedance_nullspace_stiffness = float(rospy.get_param("~impedance_nullspace_stiffness", 200.0))
		self.impedance_nullspace_damping = float(rospy.get_param("~impedance_nullspace_damping", 1.0))
		self.impedance_max_path_dev_vals = self._load_cartesian_values("~impedance_max_path_deviation", (-1.0,) * 6)
		self.impedance_max_cart_vel_vals = self._load_cartesian_values("~impedance_max_cartesian_velocity", (-1.0,) * 6)
		self.impedance_max_control_force_vals = self._load_cartesian_values("~impedance_max_control_force", (-1.0,) * 6)
		self.impedance_max_control_force_stop = bool(rospy.get_param("~impedance_max_control_force_stop", False))

		self.pose_pub = rospy.Publisher("/iiwa/command/CartesianPoseLin", PoseStamped, queue_size=1)
		self._current_pose_lock = threading.Lock()
		self._current_pose: Optional[PoseStamped] = None
		self._impedance_client = None

		if _HAVE_CARTESIAN:
			rospy.Subscriber("/iiwa/state/CartesianPose", CartesianPose, self._cartesian_cb, queue_size=1)
		else:
			rospy.Subscriber("/iiwa/state/CartesianPose", PoseStamped, self._pose_cb, queue_size=1)

		self._load_yaml()
		self._wait_for_current_pose()

		self._maybe_configure_position()
		self._execute_stage("第一阶段（预接近）", self.start_pre_pose)
		if self.pre_stage_pause > 0.0:
			rospy.sleep(self.pre_stage_pause)
		
		self._maybe_configure_impedance()
		self._execute_stage("第二阶段（起点最终）", self.start_final_pose)

		self._execute_path()
		rospy.loginfo("[path_execute] ✓ 路径执行完成")

	def _load_yaml(self) -> None:
		if not os.path.exists(self.path_yaml):
			raise RuntimeError(f"路径 YAML 不存在: {self.path_yaml}")
		with open(self.path_yaml, "r") as f:
			data = yaml.safe_load(f) or {}

		start = data.get("start")
		if not start or "pre_pose" not in start or "final_pose" not in start:
			raise RuntimeError("YAML 缺少 start/pre_pose 或 start/final_pose")
		self.start_pre_pose = self._dict_to_pose(start["pre_pose"])
		self.start_final_pose = self._dict_to_pose(start["final_pose"])

		frames = data.get("frames", [])
		if not frames:
			raise RuntimeError("YAML 缺少 frames")

		self.path_frames: List[PoseStamped] = []
		for frame in frames:
			self.path_frames.append(self._dict_to_pose(frame))

		# 确保路径首帧与 start_final_pose 对齐
		first = self.path_frames[0]
		pos_err, ang_err = self._compute_pose_error(first, self.start_final_pose)
		if pos_err > 1e-3 or ang_err > math.radians(1):
			rospy.logwarn("[path_execute] 注意：路径首帧与 start_final_pose 存在 %.4f m / %.2f° 差异",
				pos_err, math.degrees(ang_err))
			self.path_frames.insert(0, self.start_final_pose)

	def _dict_to_pose(self, data: dict) -> PoseStamped:
		pose = PoseStamped()
		frame_id = data.get("frame_id", "iiwa_link_0")
		pose.header.frame_id = frame_id
		pose.pose.position.x = float(data["position"]["x"])
		pose.pose.position.y = float(data["position"]["y"])
		pose.pose.position.z = float(data["position"]["z"])
		pose.pose.orientation.x = float(data["orientation"]["x"])
		pose.pose.orientation.y = float(data["orientation"]["y"])
		pose.pose.orientation.z = float(data["orientation"]["z"])
		pose.pose.orientation.w = float(data["orientation"]["w"])
		return pose

	def _load_cartesian_values(self, prefix: str, defaults: Tuple[float, ...]) -> dict:
		values = {}
		fields = ("x", "y", "z", "a", "b", "c")
		for i, field in enumerate(fields):
			values[field] = float(rospy.get_param(f"{prefix}_{field}", defaults[i]))
		return values

	def _to_cartesian_quantity(self, values: dict) -> CartesianQuantity:
		cq = CartesianQuantity()
		cq.x = values.get("x", 0.0)
		cq.y = values.get("y", 0.0)
		cq.z = values.get("z", 0.0)
		cq.a = values.get("a", 0.0)
		cq.b = values.get("b", 0.0)
		cq.c = values.get("c", 0.0)
		return cq

	def _ensure_control_client(self) -> bool:
		if not _HAVE_CTRL or ConfigureControlMode is None:
			rospy.logwarn("[path_execute] 未找到 ConfigureControlMode 接口，跳过控制模式切换")
			return False
		if self._impedance_client is None:
			try:
				self._impedance_client = rospy.ServiceProxy(self.impedance_service, ConfigureControlMode)
			except Exception as exc:
				rospy.logwarn("[path_execute] 建立控制模式服务失败: %s", str(exc))
				self._impedance_client = None
				return False
		return True

	def _maybe_configure_position(self) -> None:
		if not self.auto_configure_position:
			return
		if not self._ensure_control_client():
			return
		req = ConfigureControlModeRequest()
		req.control_mode = ControlMode.POSITION_CONTROL
		try:
			resp = self._impedance_client(req)
		except rospy.ServiceException as exc:
			rospy.logwarn("[path_execute] 切换位置控制失败: %s", str(exc))
			return
		if not resp.success:
			rospy.logwarn("[path_execute] 切换位置控制失败: %s", resp.error)
			return
		rospy.loginfo("[path_execute] ✓ 已切换到位置控制模式")

	def _maybe_configure_impedance(self) -> None:
		if not self.auto_configure_impedance:
			return
		if not self._ensure_control_client():
			return
		req = ConfigureControlModeRequest()
		req.control_mode = ControlMode.CARTESIAN_IMPEDANCE
		req.cartesian_impedance.cartesian_stiffness = self._to_cartesian_quantity(self.impedance_stiff_vals)
		req.cartesian_impedance.cartesian_damping = self._to_cartesian_quantity(self.impedance_damp_vals)
		req.cartesian_impedance.nullspace_stiffness = self.impedance_nullspace_stiffness
		req.cartesian_impedance.nullspace_damping = self.impedance_nullspace_damping
		req.limits.max_path_deviation = self._to_cartesian_quantity(self.impedance_max_path_dev_vals)
		req.limits.max_cartesian_velocity = self._to_cartesian_quantity(self.impedance_max_cart_vel_vals)
		req.limits.max_control_force = self._to_cartesian_quantity(self.impedance_max_control_force_vals)
		req.limits.max_control_force_stop = self.impedance_max_control_force_stop
		try:
			resp = self._impedance_client(req)
		except rospy.ServiceException as exc:
			rospy.logwarn("[path_execute] 配置笛卡尔阻抗失败: %s", str(exc))
			return
		if not resp.success:
			rospy.logwarn("[path_execute] 配置笛卡尔阻抗失败: %s", resp.error)
			return
		rospy.loginfo("[path_execute] ✓ 已切换到笛卡尔阻抗模式")

	def _pose_cb(self, msg: PoseStamped) -> None:
		with self._current_pose_lock:
			self._current_pose = msg

	def _cartesian_cb(self, msg: CartesianPose) -> None:
		with self._current_pose_lock:
			self._current_pose = msg.poseStamped

	def _wait_for_current_pose(self, timeout: float = 5.0) -> None:
		start = time.time()
		rate = rospy.Rate(50)
		while not rospy.is_shutdown():
			with self._current_pose_lock:
				if self._current_pose is not None:
					return
			if time.time() - start > timeout:
				raise RuntimeError("等待 /iiwa/state/CartesianPose 超时")
			rate.sleep()

	def _execute_stage(self, label: str, target_pose: PoseStamped) -> None:
		rospy.loginfo("[path_execute] >>> %s", label)
		self._execute_linear_motion(target_pose, label, wait_for_completion=True)

	def _execute_linear_motion(self, target_pose: PoseStamped, label: str, wait_for_completion: bool) -> PoseStamped:
		with self._current_pose_lock:
			current_pose = copy.deepcopy(self._current_pose)

		target_vec = np.array([
			target_pose.pose.position.x,
			target_pose.pose.position.y,
			target_pose.pose.position.z
		], dtype=np.float64)

		if current_pose is None:
			rospy.logwarn("[path_execute] 未获取当前位置，直接发布目标点 (%s)", label)
			self._publish_pose(target_pose)
			if wait_for_completion:
				self._wait_until_reached(target_pose)
			return target_pose

		current_vec = np.array([
			current_pose.pose.position.x,
			current_pose.pose.position.y,
			current_pose.pose.position.z
		], dtype=np.float64)

		distance = np.linalg.norm(target_vec - current_vec)
		if distance < 1e-6:
			rospy.loginfo("[path_execute] %s 已在目标附近，跳过", label)
			return target_pose

		num_steps = max(1, int(np.ceil(distance / self.linear_step_size)))
		direction = (target_vec - current_vec) / distance
		rate = rospy.Rate(self.linear_rate)

		rospy.loginfo("[path_execute] %s: 插值步数 %d，单步 %.6f m", label, num_steps, distance / num_steps)

		for i in range(num_steps + 1):
			alpha = float(i) / float(num_steps)
			interp_pos = current_vec + direction * distance * alpha
			step_pose = PoseStamped()
			step_pose.header.frame_id = target_pose.header.frame_id
			step_pose.header.stamp = rospy.Time.now()
			step_pose.pose.position.x = float(interp_pos[0])
			step_pose.pose.position.y = float(interp_pos[1])
			step_pose.pose.position.z = float(interp_pos[2])
			step_pose.pose.orientation = target_pose.pose.orientation
			self._publish_pose(step_pose)
			if i < num_steps:
				rate.sleep()

		if wait_for_completion:
			self._wait_until_reached(target_pose)
		return target_pose

	def _publish_pose(self, pose: PoseStamped) -> None:
		msg = copy.deepcopy(pose)
		msg.header.stamp = rospy.Time.now()
		self.pose_pub.publish(msg)

	def _wait_until_reached(self, target_pose: PoseStamped) -> None:
		start = time.time()
		best_xy = None
		best_z = None
		best_ang = None
		rate = rospy.Rate(50)
		while not rospy.is_shutdown():
			err_xy, err_z, _err_xyz, ang_err = self._compute_pose_error_components(target_pose, None)
			if err_xy is not None:
				if best_xy is None or err_xy < best_xy:
					best_xy = err_xy
					best_z = err_z
					best_ang = ang_err
				if (err_xy <= self.error_position_tolerance_xy) and (err_z <= self.error_position_tolerance_z) and (
					ang_err <= self.error_angle_tolerance
				):
					rospy.loginfo(
						"[path_execute] 到位误差: xy=%.4f m, z=%.4f m, %.2f°",
						err_xy,
						err_z,
						math.degrees(ang_err),
					)
					return
			if time.time() - start > self.error_timeout:
				if best_xy is not None:
					rospy.logwarn(
						"[path_execute] 等待超时，最小误差: xy=%.4f m, z=%.4f m, %.2f°",
						best_xy,
						(best_z or 0.0),
						math.degrees(best_ang or 0.0),
					)
				else:
					rospy.logwarn("[path_execute] 等待超时，未获取到反馈")
				return
			rate.sleep()

	def _compute_pose_error_components(
		self, reference: PoseStamped, actual: Optional[PoseStamped]
	) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
		if actual is None:
			with self._current_pose_lock:
				actual = copy.deepcopy(self._current_pose)
		if actual is None:
			return (None, None, None, None)
		ref_pos = np.array([
			reference.pose.position.x,
			reference.pose.position.y,
			reference.pose.position.z
		], dtype=np.float64)
		act_pos = np.array([
			actual.pose.position.x,
			actual.pose.position.y,
			actual.pose.position.z
		], dtype=np.float64)
		d = act_pos - ref_pos
		err_xy = float(math.hypot(d[0], d[1]))
		err_z = float(abs(d[2]))
		err_xyz = float(np.linalg.norm(d))

		ref_q = np.array([
			reference.pose.orientation.x,
			reference.pose.orientation.y,
			reference.pose.orientation.z,
			reference.pose.orientation.w
		], dtype=np.float64)
		act_q = np.array([
			actual.pose.orientation.x,
			actual.pose.orientation.y,
			actual.pose.orientation.z,
			actual.pose.orientation.w
		], dtype=np.float64)
		ref_q /= np.linalg.norm(ref_q) or 1.0
		act_q /= np.linalg.norm(act_q) or 1.0
		dot = float(np.clip(np.dot(ref_q, act_q), -1.0, 1.0))
		ang_err = 2.0 * math.acos(abs(dot))
		if ang_err > math.pi:
			ang_err = 2.0 * math.pi - ang_err
		return err_xy, err_z, err_xyz, ang_err

	def _compute_pose_error(self, reference: PoseStamped, actual: Optional[PoseStamped]) -> Tuple[Optional[float], Optional[float]]:
		"""
		兼容旧逻辑：返回一个“单值位置误差”+角度误差。
		- 默认：位置误差为 xyz 欧氏距离
		- ignore_z_error=true 时：位置误差为 xy 平面距离（更符合“忽略 Z”）
		"""
		err_xy, _err_z, err_xyz, ang_err = self._compute_pose_error_components(reference, actual)
		if err_xyz is None or ang_err is None:
			return (None, None)
		pos_err = err_xy if self.ignore_z_error else err_xyz
		return pos_err, ang_err

	def _execute_path(self) -> None:
		if len(self.path_frames) < 2:
			rospy.loginfo("[path_execute] 路径只有一个点，执行完成")
			return

		rospy.loginfo("[path_execute] >>> 开始路径跟踪，共 %d 个坐标系", len(self.path_frames))

		current_index = 1  # 目标索引
		max_index = len(self.path_frames) - 1

		# 首先发布第一个目标（索引1），索引0 已在第二阶段到达
		self._publish_pose(self.path_frames[current_index])
		rospy.loginfo("[path_execute] 发布路径坐标系 #%d", current_index + 1)

		if self.lookahead_publish_delay > 0.0:
			rospy.sleep(self.lookahead_publish_delay)

		rate = rospy.Rate(50)
		while not rospy.is_shutdown():
			target_pose = self.path_frames[current_index]
			err_xy, err_z, _err_xyz, ang_err = self._compute_pose_error_components(target_pose, None)
			if err_xy is None or ang_err is None:
				rate.sleep()
				continue

			if current_index < max_index and (err_xy <= self.lookahead_distance_xy) and (err_z <= self.lookahead_distance_z):
				current_index += 1
				self._publish_pose(self.path_frames[current_index])
				rospy.loginfo("[path_execute] 发布路径坐标系 #%d", current_index + 1)
				continue

			if current_index == max_index and (err_xy <= self.error_position_tolerance_xy) and (
				ang_err <= self.error_angle_tolerance
			):
				rospy.loginfo(
					"[path_execute] 路径终点误差: xy=%.4f m, z=%.4f m, %.2f°",
					err_xy,
					err_z,
					math.degrees(ang_err),
				)
				break

			rate.sleep()


if __name__ == "__main__":
	if os.environ.get("RUSSAGENT_ENABLE_ROBOT", "0").strip().lower() not in ("1", "true", "yes"):
		raise SystemExit("Robot execution disabled. Set RUSSAGENT_ENABLE_ROBOT=1 after safety validation.")
	try:
		PathExecuteNode()
	except rospy.ROSInterruptException:
		pass

