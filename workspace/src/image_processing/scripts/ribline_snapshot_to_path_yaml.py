#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ribline_snapshot_to_path_yaml.py

离线脚本：把 demo_with_skel.py 导出的 ribline_uv.npy（像素轨迹）
按 path_stroke_preview.py 的思路转换为可执行的 path_preview.yaml：
1) ribline_uv (u,v) + depth_m.npy + camera_info.yaml -> 相机系 3D 点
2) 用 snapshot 保存的 TF (lookup_transform(to=target, from=camera_frame)) -> target_frame 3D 点
3) 可选：把轨迹点 snap 到 cloud_<target_frame>.ply 最近邻，确保落在点云表面
4) 用 cloud_<target_frame>.ply KDTree 估法向 + 构造姿态
5) 按 target_frame 的 +Z 方向施加 approach_offset / pre_approach_offset
6) 导出 path_preview.yaml（格式与 path_stroke_preview.py 一致，path_execute.py 可直接读取）
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import yaml
from scipy.spatial import KDTree

try:
    import open3d as o3d  # type: ignore
except Exception:
    o3d = None


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def _load_intrinsics(camera_info_yaml: str) -> Intrinsics:
    with open(camera_info_yaml, "r") as f:
        data = yaml.safe_load(f) or {}
    # 支持两种格式：
    # A) {fx, fy, cx, cy, width, height}  (pipeline 新格式)
    # B) ROS CameraInfo dump: {width,height,K:[...], ...} 或 {K:[...]}
    if "fx" in data and "cx" in data:
        fx = float(data["fx"])
        fy = float(data.get("fy", fx))
        cx = float(data["cx"])
        cy = float(data["cy"])
        width = int(data.get("width", 0))
        height = int(data.get("height", 0))
    else:
        K = data.get("K", None)
        if K is None:
            raise KeyError(f"camera_info.yaml missing fx/cx or K: {camera_info_yaml}")
        K = list(K)
        fx = float(K[0])
        fy = float(K[4])
        cx = float(K[2])
        cy = float(K[5])
        width = int(data.get("width", 0))
        height = int(data.get("height", 0))
    return Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)


def _depth_to_cloud_camera(depth_m: np.ndarray, intr: Intrinsics, max_points: int = 200000) -> np.ndarray:
    """
    从 depth_m + intrinsics 生成相机系点云（用于法向估计/可选 snap）。
    这等价于 path_stroke_preview.py 的 use_depthcloud 模式里的点云构建，只是离线化。
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    h, w = depth.shape[:2]
    fx, fy, cx, cy = float(intr.fx), float(intr.fy), float(intr.cx), float(intr.cy)
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    Z = depth
    valid = np.isfinite(Z) & (Z > 0.05) & (Z < 3.0)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy
    pts = np.stack((X[valid], Y[valid], Z[valid]), axis=1).astype(np.float32)
    if pts.shape[0] > int(max_points):
        step = max(1, pts.shape[0] // int(max_points))
        pts = pts[::step]
    return pts

def _search_valid_depth(depth_m: np.ndarray, x: int, y: int, window: int = 5) -> float:
    h, w = depth_m.shape[:2]
    x0 = max(0, x - window)
    x1 = min(w - 1, x + window)
    y0 = max(0, y - window)
    y1 = min(h - 1, y + window)
    patch = depth_m[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32)
    valid = np.isfinite(patch) & (patch > 0.05) & (patch < 3.0)
    if not np.any(valid):
        return float("nan")
    # 用中位数更稳健一点
    return float(np.median(patch[valid]))


def _pixels_to_camera_points(
    uv: np.ndarray,
    depth_m: np.ndarray,
    intr: Intrinsics,
    depth_search_window: int = 5,
) -> np.ndarray:
    pts: List[List[float]] = []
    h, w = depth_m.shape[:2]
    for u_f, v_f in uv.reshape(-1, 2):
        x = int(round(float(u_f)))
        y = int(round(float(v_f)))
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        z = float(depth_m[y, x])
        if (not np.isfinite(z)) or z <= 0.05 or z > 3.0:
            z = _search_valid_depth(depth_m, x, y, window=max(1, int(depth_search_window)))
        if (not np.isfinite(z)) or z <= 0.05 or z > 3.0:
            continue
        X = (x - intr.cx) * z / intr.fx
        Y = (y - intr.cy) * z / intr.fy
        pts.append([float(X), float(Y), float(z)])
    return np.asarray(pts, dtype=np.float32)


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    # 与 path_stroke_preview.py 的实现一致（右手系）
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _rot_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    # 稳健的 rotation matrix -> quaternion
    m00, m01, m02 = float(R[0, 0]), float(R[0, 1]), float(R[0, 2])
    m10, m11, m12 = float(R[1, 0]), float(R[1, 1]), float(R[1, 2])
    m20, m21, m22 = float(R[2, 0]), float(R[2, 1]), float(R[2, 2])
    tr = m00 + m11 + m22
    if tr > 0:
        S = (tr + 1.0) ** 0.5 * 2.0
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = (1.0 + m00 - m11 - m22) ** 0.5 * 2.0
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = (1.0 + m11 - m00 - m22) ** 0.5 * 2.0
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = (1.0 + m22 - m00 - m11) ** 0.5 * 2.0
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    # normalize
    n = float((qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5) or 1.0
    return qx / n, qy / n, qz / n, qw / n


def _load_tf_yaml_auto(snapshot_dir: str, target_frame: str) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    返回 (from_frame, R, t)，用于 points_out = (R @ points.T).T + t
    这里 R,t 对应 lookup_transform(to=target, from=from_frame) 的输出语义（与 path_stroke_preview.py 一致）。
    """
    cand = sorted(glob.glob(os.path.join(snapshot_dir, f"tf_*_to_{target_frame}.yaml")))
    if not cand:
        raise FileNotFoundError(f"TF yaml not found in snapshot_dir: tf_*_to_{target_frame}.yaml")
    tf_path = cand[0]
    with open(tf_path, "r") as f:
        data = yaml.safe_load(f) or {}
    # 支持两种格式：
    # A) {child_frame_id, transform:{translation:{}, rotation:{}}}  (demo 期望)
    # B) {from_frame, to_frame, translation:{}, rotation:{}}        (snapshot_capture.py 旧格式)
    child = str(data.get("child_frame_id", data.get("from_frame", ""))).strip()
    tr = data.get("transform", data) or {}
    t_data = tr.get("translation", {}) or {}
    r_data = tr.get("rotation", {}) or {}
    tx = float(t_data.get("x", 0.0))
    ty = float(t_data.get("y", 0.0))
    tz = float(t_data.get("z", 0.0))
    qx = float(r_data.get("x", 0.0))
    qy = float(r_data.get("y", 0.0))
    qz = float(r_data.get("z", 0.0))
    qw = float(r_data.get("w", 1.0))
    R = _quat_to_rot(qx, qy, qz, qw)
    t = np.array([tx, ty, tz], dtype=np.float64)
    if not child:
        raise ValueError(f"Invalid TF yaml (missing child_frame_id): {tf_path}")
    return child, R, t


def _apply_tf(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    out = (R @ pts.T).T + t.reshape(1, 3)
    return out.astype(np.float32)


def _load_cloud_target(snapshot_dir: str, target_frame: str) -> np.ndarray:
    ply = os.path.join(snapshot_dir, f"cloud_{target_frame}.ply")
    if not os.path.exists(ply):
        raise FileNotFoundError(ply)
    if o3d is None:
        raise RuntimeError("open3d not available; cannot read .ply")
    pcd = o3d.io.read_point_cloud(ply)
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pts.size == 0:
        raise RuntimeError(f"Empty point cloud: {ply}")
    return pts


def _resample_by_spacing(points: np.ndarray, spacing: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 2:
        return pts.astype(np.float32)
    out: List[np.ndarray] = [pts[0].copy()]
    acc = 0.0
    for i in range(1, pts.shape[0]):
        a = out[-1]
        b = pts[i]
        seg = b - a
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-9:
            continue
        while acc + seg_len >= spacing:
            r = (spacing - acc) / seg_len
            out.append(a + r * seg)
            a = out[-1]
            seg = b - a
            seg_len = float(np.linalg.norm(seg))
            acc = 0.0
            if seg_len < 1e-9:
                break
        acc += seg_len
    if np.linalg.norm(out[-1] - pts[-1]) > 1e-6:
        out.append(pts[-1].copy())
    return np.asarray(out, dtype=np.float32)


def _estimate_normal_from_cloud(
    kdtree: KDTree,
    cloud_points: np.ndarray,
    point: np.ndarray,
    num_neighbors: int,
    radius: float,
) -> Optional[np.ndarray]:
    try:
        distances, indices = kdtree.query(point, k=num_neighbors + 1, distance_upper_bound=radius)
    except Exception:
        return None
    valid = np.isfinite(distances)
    idx = indices[valid]
    if idx.size < 3:
        return None
    neigh = cloud_points[idx]
    centroid = neigh.mean(axis=0)
    X = neigh - centroid
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    normal = Vt[-1, :]
    n = float(np.linalg.norm(normal))
    if n < 1e-10:
        return None
    normal = (normal / n).astype(np.float64)
    # 与 path_stroke_preview.py 一致：强制 normal.z <= 0
    if normal[2] > 0:
        normal = -normal
    return normal.astype(np.float32)


def _compute_tangent(points: np.ndarray, idx: int) -> np.ndarray:
    if idx < points.shape[0] - 1:
        t = points[idx + 1] - points[idx]
    else:
        t = points[idx] - points[idx - 1]
    n = float(np.linalg.norm(t))
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return (t / n).astype(np.float64)


def _make_pose_frames(
    points: np.ndarray,
    kdtree: KDTree,
    cloud_points: np.ndarray,
    target_frame: str,
    num_neighbors: int,
    neighbor_radius: float,
    approach_offset: float,
    pre_approach_offset: float,
) -> Tuple[dict, List[dict]]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    frames: List[dict] = []
    prev_axes = None
    prev_normal = None
    z_axis_fixed = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # target_frame 的 +Z

    start_pre = None
    start_final = None

    for i in range(pts.shape[0]):
        p = pts[i]
        normal = _estimate_normal_from_cloud(kdtree, cloud_points, p, num_neighbors=num_neighbors, radius=neighbor_radius)
        if normal is None:
            if prev_normal is None:
                raise RuntimeError(f"Normal estimation failed at idx={i}")
            normal = prev_normal.copy()
        normal = normal.astype(np.float64)
        normal = normal / (np.linalg.norm(normal) or 1.0)

        tangent = _compute_tangent(pts, i)
        forward = tangent - np.dot(tangent, normal) * normal
        if np.linalg.norm(forward) < 1e-6:
            forward = tangent.copy()
        forward = forward / (np.linalg.norm(forward) or 1.0)

        # 跟 path_stroke_preview.py 一致：X 轴主要沿轨迹反方向（-forward）
        if i == pts.shape[0] - 1 and pts.shape[0] >= 2:
            prev_vec = pts[i - 1] - p
            x_axis = prev_vec if np.linalg.norm(prev_vec) > 1e-6 else -forward
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

        prev_axes = (x_axis.copy(), y_axis.copy(), normal.copy())
        prev_normal = normal.copy()

        # 位置 offset：沿 target_frame +Z（与你改过的 path_stroke_preview.py 一致）
        final_pos = p + approach_offset * z_axis_fixed

        R = np.stack([x_axis, y_axis, normal], axis=1)  # columns
        qx, qy, qz, qw = _rot_to_quat(R)

        frames.append(
            {
                "frame_id": target_frame,
                "position": {"x": float(final_pos[0]), "y": float(final_pos[1]), "z": float(final_pos[2])},
                "orientation": {"x": float(qx), "y": float(qy), "z": float(qz), "w": float(qw)},
            }
        )

        if i == 0:
            start_pre = p + pre_approach_offset * z_axis_fixed
            start_final = final_pos.copy()

    if start_pre is None or start_final is None:
        raise RuntimeError("Empty frames")

    start_block = {
        "pre_pose": {
            "frame_id": target_frame,
            "position": {"x": float(start_pre[0]), "y": float(start_pre[1]), "z": float(start_pre[2])},
            "orientation": frames[0]["orientation"],
        },
        "final_pose": {
            "frame_id": target_frame,
            "position": {"x": float(start_final[0]), "y": float(start_final[1]), "z": float(start_final[2])},
            "orientation": frames[0]["orientation"],
        },
    }
    return start_block, frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot_dir", required=True, help="snapshot_YYYYMMDD_HHMMSS directory")
    ap.add_argument("--target_frame", default="iiwa_link_0")
    ap.add_argument("--rib_uv_npy", default="ribline_uv.npy", help="uv npy filename inside snapshot_dir (e.g. ribline_uv_shifted_fit.npy)")
    ap.add_argument("--depth_search_window", type=int, default=5)
    ap.add_argument("--frame_spacing", type=float, default=0.01)
    ap.add_argument("--neighbor_radius", type=float, default=0.03)
    ap.add_argument("--num_neighbors", type=int, default=100)
    ap.add_argument("--approach_offset", type=float, default=0.0)
    ap.add_argument("--pre_approach_offset", type=float, default=0.05)
    ap.add_argument("--snap_to_cloud", action="store_true", help="snap ribline points to cloud surface (recommended)")
    ap.add_argument("--no_snap_to_cloud", action="store_true", help="disable snapping")
    ap.add_argument("--output_yaml", default="", help="output yaml path (default: <snapshot_dir>/path_preview.yaml)")
    args = ap.parse_args()

    snapshot_dir = os.path.abspath(args.snapshot_dir)
    target_frame = str(args.target_frame)

    camera_info_yaml = os.path.join(snapshot_dir, "camera_info.yaml")
    if not os.path.exists(camera_info_yaml):
        # fallback: camera_info_*.yaml (ROS dump)
        cands = sorted(glob.glob(os.path.join(snapshot_dir, "camera_info*.yaml")))
        if cands:
            camera_info_yaml = cands[0]
    depth_npy = os.path.join(snapshot_dir, "depth_m.npy")
    rib_uv_npy = os.path.join(snapshot_dir, str(args.rib_uv_npy))

    intr = _load_intrinsics(camera_info_yaml)
    depth_m = np.load(depth_npy).astype(np.float32)
    # 如果 camera_info yaml 没写 width/height，就用 depth shape 兜底
    if (intr.width <= 0 or intr.height <= 0) and depth_m.size > 0:
        intr = Intrinsics(
            fx=intr.fx, fy=intr.fy, cx=intr.cx, cy=intr.cy,
            width=int(depth_m.shape[1]), height=int(depth_m.shape[0]),
        )
    rib_uv = np.load(rib_uv_npy).astype(np.float32)

    pts_cam = _pixels_to_camera_points(rib_uv, depth_m, intr, depth_search_window=int(args.depth_search_window))
    if pts_cam.shape[0] < 2:
        raise RuntimeError("Too few valid 3D points from ribline_uv + depth")

    from_frame, R_tf, t_tf = _load_tf_yaml_auto(snapshot_dir, target_frame=target_frame)
    pts_target = _apply_tf(pts_cam, R_tf, t_tf)

    # cloud: prefer ply/npy. If missing, build from depth (no need to pre-save cloud).
    cloud_target = None
    try:
        cloud_target = _load_cloud_target(snapshot_dir, target_frame=target_frame)
    except Exception:
        npy = os.path.join(snapshot_dir, f"cloud_{target_frame}.npy")
        npy2 = os.path.join(snapshot_dir, f"cloud_{target_frame.replace('/', '_')}.npy")
        cand = npy if os.path.exists(npy) else npy2
        if os.path.exists(cand):
            cloud_target = np.load(cand).astype(np.float32)
        else:
            # Build camera cloud from depth, then TF into target frame
            cloud_cam = _depth_to_cloud_camera(depth_m, intr, max_points=200000)
            if cloud_cam.shape[0] < 50:
                raise RuntimeError("Cannot build point cloud from depth (too few points).")
            _, R_tf2, t_tf2 = _load_tf_yaml_auto(snapshot_dir, target_frame=target_frame)
            cloud_target = _apply_tf(cloud_cam, R_tf2, t_tf2)
    cloud_target = np.asarray(cloud_target, dtype=np.float32)
    kdtree_cloud = KDTree(cloud_target)

    do_snap = bool(args.snap_to_cloud) and (not bool(args.no_snap_to_cloud))
    if do_snap:
        _, nn_idx = kdtree_cloud.query(pts_target, k=1)
        pts_target = cloud_target[np.asarray(nn_idx, dtype=np.int64)]

    pts_resampled = _resample_by_spacing(pts_target, float(args.frame_spacing))
    if pts_resampled.shape[0] < 2:
        raise RuntimeError("Resampled path has too few points")

    start_block, frames = _make_pose_frames(
        pts_resampled,
        kdtree=kdtree_cloud,
        cloud_points=cloud_target,
        target_frame=target_frame,
        num_neighbors=int(args.num_neighbors),
        neighbor_radius=float(args.neighbor_radius),
        approach_offset=float(args.approach_offset),
        pre_approach_offset=float(args.pre_approach_offset),
    )

    out_yaml = args.output_yaml.strip() or os.path.join(snapshot_dir, "path_preview.yaml")
    out = {"start": start_block, "frames": frames}
    with open(out_yaml, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    np.save(os.path.join(snapshot_dir, "ribline_points_camera.npy"), pts_cam.astype(np.float32))
    np.save(os.path.join(snapshot_dir, "ribline_points_target.npy"), pts_target.astype(np.float32))
    np.save(os.path.join(snapshot_dir, "ribline_points_target_resampled.npy"), pts_resampled.astype(np.float32))

    print("OK")
    print(f"- snapshot_dir: {snapshot_dir}")
    print(f"- from_frame (TF child): {from_frame} -> target_frame: {target_frame}")
    print(f"- saved yaml: {out_yaml}")


if __name__ == "__main__":
    main()


