#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, yaml, time, rospy
import moveit_commander
from moveit_commander import RobotCommander, MoveGroupCommander
from moveit_msgs.msg import DisplayTrajectory
from std_srvs.srv import Trigger
from python_qt_binding import QtWidgets, QtCore

DEFAULT_YAML = "$HOME/.ros/easy_handeye/calib_poses.yaml"

def reorder(joint_names_order, joints_dict):
    missing = [j for j in joint_names_order if j not in joints_dict]
    if missing:
        raise ValueError("YAML 缺少关节: {}".format(missing))
    return [float(joints_dict[j]) for j in joint_names_order]

def call_trigger(service_ns, timeout=3.0):
    try:
        cli = rospy.ServiceProxy(service_ns, Trigger)
        cli.wait_for_service(timeout=timeout)
        res = cli()
        return res.success, res.message
    except Exception as e:
        return False, str(e)

def wait_for_keys(keys, poll_hz=10.0):
    rate = rospy.Rate(poll_hz)
    while not rospy.is_shutdown():
        ready = [k for k in keys if rospy.has_param(k)]
        if len(ready) == len(keys):
            return True
        rate.sleep()
    return False

class LocalMover(QtWidgets.QWidget):
    def __init__(self):
        super(LocalMover, self).__init__()
        rospy.init_node("calib_local_mover", anonymous=True)

        # ---- 参数 ----
        self.ns         = rospy.get_param("~ns", "")          # e.g. "/iiwa"
        self.yaml_path  = rospy.get_param("~yaml", DEFAULT_YAML)
        self.group_name = rospy.get_param("~group", "manipulator")
        self.vel_scale  = float(rospy.get_param("~vel_scale", 0.1))
        self.acc_scale  = float(rospy.get_param("~acc_scale", 0.1))
        self.sample_srv = rospy.get_param("~sample_service", "/iiwa_azure_eob_eye_on_base/take_sample")
        self.auto_sample= bool(rospy.get_param("~auto_sample", False))
        self.settle_sec = float(rospy.get_param("~settle_sec", 2))

        if not os.path.exists(self.yaml_path):
            raise RuntimeError("位姿库不存在：{}".format(self.yaml_path))

        # 规范化 ns
        ns = self.ns.strip("/")
        ns_prefix = ("/" + ns) if ns else ""
        rospy.loginfo("使用命名空间: '%s'", ns_prefix or "/(root)")

        # ---- 等待 MoveIt 语义模型（命名空间兼容）----
        desc_key = (ns_prefix + "/robot_description") if ns_prefix else "/robot_description"
        sem_key  = (ns_prefix + "/robot_description_semantic") if ns_prefix else "/robot_description_semantic"
        rospy.loginfo("等待参数: %s, %s", desc_key, sem_key)
        ok = wait_for_keys([desc_key, sem_key])
        if not ok:
            raise RuntimeError("等待 MoveIt 参数超时。")

        # ---- 初始化 MoveIt（绑定到 NS 的 move_group）----
        moveit_commander.roscpp_initialize(sys.argv)
        # RobotCommander 要的是不带前导斜杠的 param 名
        self.robot = RobotCommander(robot_description=desc_key.lstrip("/"))
        # MoveGroupCommander 的 ns 要不带前导斜杠（内部自己加）
        self.group = MoveGroupCommander(self.group_name,
                                        robot_description=desc_key.lstrip("/"),
                                        ns=ns)
        # 校验组名
        group_names = list(self.robot.get_group_names())
        if self.group_name not in group_names:
            rospy.logwarn("规划组 '%s' 不存在。可用组：%s",
                          self.group_name, ", ".join(group_names))
            self.group_name = "manipulator" if "manipulator" in group_names else group_names[0]
            rospy.logwarn("已自动切换为组 '%s'", self.group_name)
            self.group = MoveGroupCommander(self.group_name,
                                            robot_description=desc_key.lstrip("/"),
                                            ns=ns)

        self.group.set_max_velocity_scaling_factor(self.vel_scale)
        self.group.set_max_acceleration_scaling_factor(self.acc_scale)
        self.joint_order = self.group.get_active_joints()

        # 轨迹预览发布到 NS 下的 topic
        disp_topic = (ns_prefix + "/move_group/display_planned_path") if ns_prefix else "/move_group/display_planned_path"
        self.pub_disp = rospy.Publisher(disp_topic, DisplayTrajectory, queue_size=10)

        # ---- 读取位姿库 ----
        self.poses = (yaml.safe_load(open(self.yaml_path, "r")) or {}).get("poses", [])
        if not self.poses:
            raise RuntimeError("YAML 中没有 poses: {}".format(self.yaml_path))
        self.idx = 0
        self.last_plan = None

        self._build_ui()
        self.update_title()

    # ---------- UI ----------
    def _build_ui(self):
        self.setWindowTitle("&Local Mover")
        self.resize(600, 420)
        self.progress = QtWidgets.QProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0)
        self.status   = QtWidgets.QLabel("Ready"); self.status.setWordWrap(True)
        self.preview  = QtWidgets.QLabel(); self.preview.setMinimumHeight(160)
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self._set_preview_color("#444", "No plan")

        self.btn_check  = QtWidgets.QPushButton("Check starting pose")
        self.btn_next   = QtWidgets.QPushButton("Next Pose")
        self.btn_plan   = QtWidgets.QPushButton("Plan")
        self.btn_exec   = QtWidgets.QPushButton("Execute")
        self.btn_sample = QtWidgets.QPushButton("Take Sample")

        row = QtWidgets.QHBoxLayout()
        for b in (self.btn_check, self.btn_next, self.btn_plan, self.btn_exec, self.btn_sample):
            row.addWidget(b)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.progress); lay.addWidget(self.preview); lay.addWidget(self.status); lay.addLayout(row)

        self.btn_check.clicked.connect(self.on_check)
        self.btn_next.clicked.connect(self.on_next)
        self.btn_plan.clicked.connect(self.on_plan)
        self.btn_exec.clicked.connect(self.on_execute)
        self.btn_sample.clicked.connect(self.on_take_sample)

    def _set_preview_color(self, color_hex, text):
        self.preview.setStyleSheet(f"background-color: {color_hex}; color: #ddd;")
        self.preview.setText(text)

    def update_title(self):
        total = len(self.poses)
        self.setWindowTitle(f"&Local Mover  -  {self.idx+1}/{total}")
        self.progress.setValue(int(100.0 * (self.idx) / max(1, total)))

    # ---------- 动作 ----------
    def on_check(self):
        cur = dict(zip(self.joint_order, self.group.get_current_joint_values()))
        tgt = self.poses[0]["joints"]
        diffs = [abs(cur[j]-float(tgt[j])) for j in self.joint_order if j in tgt]
        avg = sum(diffs)/max(1,len(diffs))
        self.status.setText(f"Avg |Δjoint| = {avg:.4f} rad  (vs pose[0])")

    def on_next(self):
        self.idx = (self.idx + 1) % len(self.poses)
        self.last_plan = None
        self._set_preview_color("#444", "No plan")
        name = self.poses[self.idx].get('name', f"pose_{self.idx+1}")
        self.status.setText(f"Selected pose: {name}")
        self.update_title()

    def on_plan(self):
        pose = self.poses[self.idx]
        name = pose.get("name", f"pose_{self.idx+1}")
        try:
            target = reorder(self.joint_order, pose["joints"])
        except Exception as e:
            self.status.setText(f"[{name}] 关节重排失败: {e}")
            self._set_preview_color("#922", "Bad pose")
            return

        self.group.set_joint_value_target(target)
        plan = self.group.plan()

        traj = None
        if hasattr(plan, "joint_trajectory"):
            traj = plan
        else:
            try: traj = plan[1]
            except Exception: traj = None

        if traj and hasattr(traj, "joint_trajectory") and len(traj.joint_trajectory.points) > 0:
            self.last_plan = traj
            msg = DisplayTrajectory()
            msg.trajectory_start = self.robot.get_current_state()
            msg.trajectory.append(traj)
            self.pub_disp.publish(msg)
            self._set_preview_color("#2e7d32", "Good plan")
            self.status.setText(f"[{name}] 规划成功，已发布 RViz 预览。")
        else:
            self.last_plan = None
            self._set_preview_color("#922", "Plan failed")
            self.status.setText(f"[{name}] 规划失败。")

    def on_execute(self):
        if self.last_plan is None:
            self.status.setText("没有可执行的规划，请先 Plan")
            return
        name = self.poses[self.idx].get("name", f"pose_{self.idx+1}")
        ok = self.group.execute(self.last_plan, wait=True)
        self.group.stop(); self.group.clear_pose_targets()
        if ok:
            self._set_preview_color("#1565c0", "Executed")
            self.status.setText(f"[{name}] 执行完成，等待稳定 {self.settle_sec:.1f}s")
            rospy.sleep(self.settle_sec)
            if self.auto_sample: self.on_take_sample()
        else:
            self._set_preview_color("#922", "Exec failed")
            self.status.setText(f"[{name}] 执行失败。")

    def on_take_sample(self):
        succ, msg = call_trigger(self.sample_srv)
        self.status.setText(f"take_sample: {'OK' if succ else 'FAIL'} ({msg})")

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = LocalMover(); w.show()
    ret = app.exec_()
    moveit_commander.roscpp_shutdown()
    sys.exit(ret)

if __name__ == "__main__":
    main()
