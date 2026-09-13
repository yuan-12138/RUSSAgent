#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A modified version of CLIFF demo.py that:
1. Supports single image input without strict directory structure.
2. Loads REAL intrinsics (focal length) from _intrinsics.json.
3. Integrates SKEL optimization to generate high-fidelity skeleton mesh.
"""

import os
import os.path as osp
import cv2
import json
import torch
import argparse
import numpy as np
import smplx
import torchgeometry as tgm
from tqdm import tqdm
from typing import Optional, Tuple
from pathlib import Path

# Add parent directories to path to find other modules if needed
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# CLIFF Imports
from models.cliff_hr48.cliff import CLIFF as cliff_hr48
from models.cliff_res50.cliff import CLIFF as cliff_res50
from common import constants
from common.utils import strip_prefix_if_present, cam_crop2full, estimate_focal_length
from common.renderer_pyrd import Renderer
from lib.yolov3_detector import HumanDetector
from lib.yolov3_dataset import DetectionDataset
from common.mocap_dataset import MocapDataset
from torch.utils.data import DataLoader

# Resolve SKEL from the configured workspace or this repository checkout.
WORKSPACE_ROOT = os.environ.get("RUSSAGENT_WS") or str(Path(__file__).resolve().parents[6])
SKEL_REPO_PATH = os.path.join(WORKSPACE_ROOT, "models/SKEL")
if SKEL_REPO_PATH not in sys.path:
    sys.path.insert(0, SKEL_REPO_PATH)

try:
    from skel.alignment.aligner import SkelFitter
except ImportError:
    print(f"Warning: Could not import SkelFitter from {SKEL_REPO_PATH}")

def load_intrinsics(img_path):
    """Try to load _intrinsics.json for the given image."""
    dir_name = os.path.dirname(img_path)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    json_path = os.path.join(dir_name, f"{base_name}_intrinsics.json")
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                # Return dict with fx, fy, cx, cy
                return data
        except Exception as e:
            print(f"Error loading intrinsics: {e}")
    return None


def project_points_to_uv(points_3d: np.ndarray, intrinsics: dict) -> np.ndarray:
    """
    把相机系 Nx3 轨迹点投影到像素系 Nx2 (float32)。
    不做裁剪（后续回投会自己做 round+边界检查）。
    """
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    pts = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    z = pts[:, 2].copy()
    z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = fx * (pts[:, 0] / z) + cx
    v = fy * (pts[:, 1] / z) + cy
    return np.stack([u, v], axis=1).astype(np.float32)


def _trim_head_uv(uv: np.ndarray, trim_head_ratio: float) -> np.ndarray:
    """Trim the beginning of the costal margin ribline (near the xiphoid region) to avoid a sharp corner in visualization."""
    uv = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
    if len(uv) < 3:
        return uv
    r = float(trim_head_ratio or 0.0)
    r = max(0.0, min(0.95, r))
    k = int(round(len(uv) * r))
    k = max(0, min(len(uv) - 2, k))
    return uv[k:]


def _trim_head_points(pts: np.ndarray, trim_head_ratio: float) -> np.ndarray:
    """与 _trim_head_uv 同语义的 3D 版：裁掉点序前 trim_head_ratio(剑突/上段)。pts: (N,3)。"""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 3:
        return pts
    r = float(trim_head_ratio or 0.0)
    r = max(0.0, min(0.95, r))
    k = int(round(len(pts) * r))
    k = max(0, min(len(pts) - 2, k))
    return pts[k:]


def _fit_straight_line_uv(uv: np.ndarray) -> np.ndarray:
    """
    用 PCA 拟合一条直线，输出 2 个端点（Nx2 -> 2x2）。
    这是纯 2D 操作：用来替代“折线”可视化/下游回投。
    """
    uv = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
    if len(uv) < 2:
        return uv.copy()
    mu = uv.mean(axis=0)
    x = uv - mu
    # PCA 主方向
    cov = (x.T @ x) / max(1, len(x))
    w, v = np.linalg.eigh(cov)
    d = v[:, int(np.argmax(w))].astype(np.float32)
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        # fallback：首尾方向
        d = (uv[-1] - uv[0]).astype(np.float32)
        n = float(np.linalg.norm(d)) or 1.0
    d = d / n
    t = (x @ d.reshape(2, 1))[:, 0]
    p0 = mu + d * float(np.min(t))
    p1 = mu + d * float(np.max(t))
    return np.stack([p0, p1], axis=0).astype(np.float32)


def _pca_dir_and_line_uv(uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回 (dir_unit, line_2pts)，其中：
    - line_2pts: 2x2 端点（PCA 直线）
    - dir_unit: 2D 单位方向向量（沿 PCA 主方向，不强制上下）
    """
    line = _fit_straight_line_uv(uv)
    if line.shape[0] < 2:
        return np.array([1.0, 0.0], dtype=np.float32), line
    d = (line[1] - line[0]).astype(np.float32)
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        d = np.array([1.0, 0.0], dtype=np.float32)
    else:
        d = d / n
    return d, line


def _normal_down_from_dir(d: np.ndarray) -> np.ndarray:
    """
    给定 PCA 方向 d，计算其法向量 n，并选择符号使其指向图像“向下”(v 增大)。
    n 与 d 垂直，且是直线方向（绿线）。
    """
    d = np.asarray(d, dtype=np.float32).reshape(2)
    n = np.array([-d[1], d[0]], dtype=np.float32)  # rotate +90
    nn = float(np.linalg.norm(n))
    if nn < 1e-6:
        n = np.array([0.0, 1.0], dtype=np.float32)
    else:
        n = n / nn
    # ensure pointing down in image (v+)
    if n[1] < 0:
        n = -n
    return n


def _pava_isotonic_increasing(y: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    """
    PAVA: 1D 单调回归（非减）。纯 numpy，兼容 python3.8。
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = len(y)
    if n <= 1:
        return y.astype(np.float32)
    if w is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        if len(w) != n:
            w = np.ones(n, dtype=np.float64)

    sum_w = []
    sum_wy = []
    start = []
    end = []
    for i in range(n):
        sum_w.append(w[i])
        sum_wy.append(w[i] * y[i])
        start.append(i)
        end.append(i)
        while len(sum_w) >= 2:
            j = len(sum_w) - 1
            m1 = sum_wy[j - 1] / sum_w[j - 1]
            m2 = sum_wy[j] / sum_w[j]
            if m1 <= m2:
                break
            sum_w[j - 1] += sum_w[j]
            sum_wy[j - 1] += sum_wy[j]
            end[j - 1] = end[j]
            sum_w.pop()
            sum_wy.pop()
            start.pop()
            end.pop()

    y_fit = np.zeros(n, dtype=np.float64)
    for sw, swy, s, e in zip(sum_w, sum_wy, start, end):
        m = swy / sw
        y_fit[s:e + 1] = m
    return y_fit.astype(np.float32)


def _fit_monotonic_curve_uv(uv: np.ndarray, d: np.ndarray, n_down: np.ndarray) -> np.ndarray:
    """
    把 2D 肋缘折线拟合成“单向曲线”（避免沿着肋缘走一半又反向上去）：
    - d: PCA 主方向（切向轴）
    - n_down: 与 d 正交、且指向“向下”的法向轴
    方法：在 (t,s) 坐标里对 s 做单调非减回归（PAVA），再还原到 uv。
    输出点数与输入一致，并按 t 从小到大排序（更单向）。
    """
    uv = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
    if len(uv) < 3:
        return uv.copy()

    d = np.asarray(d, dtype=np.float32).reshape(2)
    n_down = np.asarray(n_down, dtype=np.float32).reshape(2)
    d = d / (np.linalg.norm(d) + 1e-9)
    # 强制正交归一
    n_down = n_down - float(np.dot(n_down, d)) * d
    n_down = n_down / (np.linalg.norm(n_down) + 1e-9)

    t = (uv @ d.reshape(2, 1))[:, 0].astype(np.float32)
    s = (uv @ n_down.reshape(2, 1))[:, 0].astype(np.float32)
    order = np.argsort(t)
    t2 = t[order]
    s2 = s[order]

    # 关键：让曲线在法向坐标上单调（不再“上上下下”）
    s_fit = _pava_isotonic_increasing(s2)

    uv_fit = (d.reshape(1, 2) * t2.reshape(-1, 1) + n_down.reshape(1, 2) * s_fit.reshape(-1, 1)).astype(np.float32)
    return uv_fit


def _global_down_direction_uv(uv: np.ndarray) -> np.ndarray:
    """
    计算肋缘线的“大致发向量”（全局方向），并保证它在图像里是“向下”(v 增大)。
    纯 2D：用 PCA 主方向作为整体方向。
    """
    uv = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
    if len(uv) < 2:
        return np.array([0.0, 1.0], dtype=np.float32)
    mu = uv.mean(axis=0)
    x = uv - mu
    cov = (x.T @ x) / max(1, len(x))
    w, v = np.linalg.eigh(cov)
    d = v[:, int(np.argmax(w))].astype(np.float32)
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        d = (uv[-1] - uv[0]).astype(np.float32)
        n = float(np.linalg.norm(d)) or 1.0
    d = d / n
    # 让方向朝“向下”(v+)
    if d[1] < 0:
        d = -d
    return d


def _shift_polyline_global_down_uv(uv: np.ndarray, shift_px: float) -> np.ndarray:
    """
    把整条肋缘线沿“大致发向量”整体向下平移 shift_px（像素）。
    纯 2D（不会引入任何 3D）。
    """
    uv = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
    if len(uv) < 2 or float(shift_px) == 0.0:
        return uv.copy()
    d = _global_down_direction_uv(uv)
    return uv + d.reshape(1, 2) * float(shift_px)


def _draw_uv_polyline(img_bgr: np.ndarray, uv: np.ndarray, color, thickness: int = 3):
    uv = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
    if len(uv) < 2:
        return img_bgr
    out = img_bgr.copy()
    pts = np.round(uv).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(out, [pts], isClosed=False, color=color, thickness=int(thickness))
    return out


def _draw_uv_arrow(img_bgr: np.ndarray, p0: np.ndarray, p1: np.ndarray, color, thickness: int = 2):
    out = img_bgr.copy()
    a = tuple(np.round(np.asarray(p0, dtype=np.float32)).astype(np.int32).tolist())
    b = tuple(np.round(np.asarray(p1, dtype=np.float32)).astype(np.int32).tolist())
    cv2.arrowedLine(out, a, b, color=color, thickness=int(thickness), tipLength=0.2)
    return out

def _rotate_img(img: np.ndarray, rotate: str) -> np.ndarray:
    rotate = (rotate or "none").lower().strip()
    if rotate == "none":
        return img
    if rotate in ("cw", "90cw", "90"):
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rotate in ("ccw", "90ccw", "-90"):
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotate in ("180", "flip"):
        return cv2.rotate(img, cv2.ROTATE_180)
    raise ValueError(f"unknown rotate: {rotate}")


def _inv_rotate_img(img: np.ndarray, rotate: str) -> np.ndarray:
    rotate = (rotate or "none").lower().strip()
    if rotate == "none":
        return img
    if rotate in ("cw", "90cw", "90"):
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotate in ("ccw", "90ccw", "-90"):
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rotate in ("180", "flip"):
        return cv2.rotate(img, cv2.ROTATE_180)
    raise ValueError(f"unknown rotate: {rotate}")


def _compute_letterbox_params(content_w: int, content_h: int, out_w: int, out_h: int):
    """与 stitch_two_rgb.py 的 letterbox 保持一致：等比缩放+居中补边。返回 (new_w,new_h,pad_x,pad_y)."""
    s = min(float(out_w) / float(content_w), float(out_h) / float(content_h))
    new_w = max(1, int(round(content_w * s)))
    new_h = max(1, int(round(content_h * s)))
    pad_x = (out_w - new_w) // 2
    pad_y = (out_h - new_h) // 2
    return new_w, new_h, pad_x, pad_y


def _undo_letterbox(img_lb: np.ndarray, content_w: int, content_h: int, out_w: int, out_h: int) -> np.ndarray:
    """把 letterbox 后的图还原到 letterbox 前的 content 尺寸。"""
    new_w, new_h, pad_x, pad_y = _compute_letterbox_params(content_w, content_h, out_w, out_h)
    crop = img_lb[pad_y:pad_y + new_h, pad_x:pad_x + new_w]
    return cv2.resize(crop, (content_w, content_h), interpolation=cv2.INTER_AREA)


def _undo_letterbox_points(uv_lb: np.ndarray, content_w: int, content_h: int, out_w: int, out_h: int) -> np.ndarray:
    """
    把 letterbox 输出坐标系里的 uv 点，逆映射回 letterbox 前的 content 坐标系（与 _undo_letterbox 一致）。
    uv_lb: (N,2) in [0,out_w) x [0,out_h)
    """
    uv_lb = np.asarray(uv_lb, dtype=np.float32).reshape(-1, 2)
    if uv_lb.shape[0] == 0:
        return uv_lb
    s = min(float(out_w) / float(content_w), float(out_h) / float(content_h))
    new_w = max(1, int(round(content_w * s)))
    new_h = max(1, int(round(content_h * s)))
    pad_x = (out_w - new_w) // 2
    pad_y = (out_h - new_h) // 2
    u = (uv_lb[:, 0] - float(pad_x)) / float(s)
    v = (uv_lb[:, 1] - float(pad_y)) / float(s)
    return np.stack([u, v], axis=1).astype(np.float32)


def _inv_rotate_points_to_original(uv_rot: np.ndarray, rotate: str, orig_h: int, orig_w: int) -> np.ndarray:
    """
    将在旋转后的图像坐标系(upper_r)中的点，逆旋转回原图(upper0)坐标系。
    rotate: upper0 -> upper_r 使用的旋转方式；这里做 upper_r -> upper0 的逆变换。
    """
    rotate = (rotate or "none").lower().strip()
    uv_rot = np.asarray(uv_rot, dtype=np.float32).reshape(-1, 2)
    if uv_rot.shape[0] == 0:
        return uv_rot
    x = uv_rot[:, 0]
    y = uv_rot[:, 1]
    if rotate in ("cw", "90cw", "90"):
        # upper0(H,W) --cw--> upper_r(W,H)
        # x_rot = H-1-y0 ; y_rot = x0  => x0=y_rot ; y0=H-1-x_rot
        x0 = y
        y0 = float(orig_h - 1) - x
    elif rotate in ("ccw", "90ccw", "-90"):
        # upper0(H,W) --ccw--> upper_r(W,H)
        # x_rot = y0 ; y_rot = W-1-x0 => x0=W-1-y_rot ; y0=x_rot
        x0 = float(orig_w - 1) - y
        y0 = x
    elif rotate in ("180", "flip"):
        x0 = float(orig_w - 1) - x
        y0 = float(orig_h - 1) - y
    else:
        x0, y0 = x, y
    return np.stack([x0, y0], axis=1).astype(np.float32)

def run_skel_optimization(device, pred_rotmat, pred_betas, pred_trans, img_path):
    """
    Run SKEL optimization.
    pred_rotmat: (1, 24, 3, 3)
    pred_betas: (1, 10)
    pred_trans: (1, 3) - Global translation
    """
    try:
        # 1. Setup Environment
        official_data_dir = os.path.join(WORKSPACE_ROOT, "models/official_skel_data/data")
        if not os.path.exists(os.path.join(official_data_dir, "skel/skel_male.pkl")):
             # Fallback
             official_data_dir = os.path.join(SKEL_REPO_PATH, "data")
        
        os.environ["SKEL_DATA_DIR"] = official_data_dir
        os.environ['DISABLE_VIEWER'] = '1'
        
        # 2. Init Fitter
        fitter = SkelFitter(gender='male', device=device)
        
        # 3. Prepare Inputs
        # SKEL expects axis-angle pose: (1, 72)
        rot_pad = torch.tensor([0, 0, 1], dtype=torch.float32, device=device).view(1, 3, 1)
        rot_pad = rot_pad.expand(pred_rotmat.shape[0] * 24, -1, -1)
        rotmat_full = torch.cat((pred_rotmat.view(-1, 3, 3), rot_pad), dim=-1)
        pred_pose_aa = tgm.rotation_matrix_to_angle_axis(rotmat_full).contiguous().view(1, -1) # (1, 72)
        
        # 4. Run Fit
        print("Running SKEL Optimization...")
        # Convert to Numpy as SkelFitter expects arrays
        poses_in_np = pred_pose_aa.detach().cpu().numpy()
        betas_in_np = pred_betas.detach().cpu().numpy()
        trans_in_np = pred_trans.detach().cpu().numpy()
        
        res = fitter.run_fit(
            poses_in=poses_in_np,
            betas_in=betas_in_np,
            trans_in=trans_in_np,
        )
        
        # 5. Forward to get Mesh
        final_poses = torch.from_numpy(res['poses']).to(device)
        final_betas = torch.from_numpy(res['betas']).to(device)
        final_trans = torch.from_numpy(res['trans']).to(device)
        
        skel_out = fitter.skel.forward(
            poses=final_poses, 
            betas=final_betas, 
            trans=final_trans, 
            poses_type='skel', 
            skelmesh=True
        )
        
        skel_verts = skel_out.skel_verts[0].detach().cpu().numpy()
        skel_faces = fitter.skel.skel_f.detach().cpu().numpy()
        
        # 6. Save
        import trimesh
        out_obj = img_path.replace(".jpg", "_skeleton.obj").replace(".png", "_skeleton.obj")
        mesh = trimesh.Trimesh(skel_verts, skel_faces, process=False)
        mesh.export(out_obj)
        print(f"SKEL Mesh saved to: {out_obj}")

        # 7. Project and Draw (New Feature)
        # We need intrinsics here. We can assume intrinsics is available in scope or pass it.
        # Let's return skel_verts to main so main can handle drawing with the intrinsics it has.
        
        # 8. Extract Abdomen
        abd_verts, abd_faces = extract_abdomen(fitter, skel_verts, skel_faces, img_path)
        
        return skel_verts, skel_faces, abd_verts, abd_faces, fitter
        
    except Exception as e:
        print(f"SKEL Optimization Failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None

def project_and_draw(verts, intrinsics, img_bgr, out_path):
    """
    Project 3D vertices to 2D using intrinsics and draw as point cloud.
    """
    if verts is None or intrinsics is None:
        return

    fx = intrinsics.get('fx')
    fy = intrinsics.get('fy')
    cx = intrinsics.get('cx')
    cy = intrinsics.get('cy')
    
    if None in [fx, fy, cx, cy]:
        print("Missing intrinsics for projection.")
        return

    # Downsample for visualization speed/clarity (every 5th point)
    verts_sub = verts[::5]
    
    vis_img = img_bgr.copy()
    
    # Vectorized Projection
    X = verts_sub[:, 0]
    Y = verts_sub[:, 1]
    Z = verts_sub[:, 2]
    
    # Mask valid Z
    valid = Z > 0.1
    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]
    
    u = (fx * X / Z) + cx
    v = (fy * Y / Z) + cy
    
    h, w = vis_img.shape[:2]
    
    for i in range(len(u)):
        x_p = int(u[i])
        y_p = int(v[i])
        if 0 <= x_p < w and 0 <= y_p < h:
            # Draw yellow dot
            cv2.circle(vis_img, (x_p, y_p), 1, (0, 255, 255), -1)
            
    cv2.imwrite(out_path, vis_img)
    print(f"Projected SKEL visualization saved to: {out_path}")

def extract_abdomen(fitter, vertices, faces, img_path):
    try:
        bone_names = fitter.skel.bone_names
        target_names = ['pelvis', 'lumbar_body', 'thorax']
        TARGET_BONES = [bone_names.index(name) for name in target_names if name in bone_names]
        
        if not TARGET_BONES:
            TARGET_BONES = [0, 11, 12] # Fallback
            
        weights = fitter.skel.skel_weights.to_dense().cpu().numpy()
        vertex_bone_ids = np.argmax(weights, axis=1)
        mask = np.isin(vertex_bone_ids, TARGET_BONES)
        valid_faces = mask[faces].all(axis=1)
        face_indices = np.where(valid_faces)[0]
        
        if len(face_indices) > 0:
            import trimesh
            full_mesh = trimesh.Trimesh(vertices, faces, process=False)
            abdomen_mesh = full_mesh.submesh([face_indices], append=True)
            out_obj = img_path.replace(".jpg", "_skeleton_abdomen.obj").replace(".png", "_skeleton_abdomen.obj")
            abdomen_mesh.export(out_obj)
            print(f"Abdomen Mesh saved to: {out_obj}")
            return abdomen_mesh.vertices, abdomen_mesh.faces
            
    except Exception as e:
        print(f"Abdomen extraction failed: {e}")
    return None, None

def extract_specific_bone_mesh(fitter, vertices, faces, target_bone_name, img_path, threshold=0.1):
    """
    Extracts mesh for a specific bone.
    Uses a weight threshold to include shared vertices at joints, ensuring complete coverage.
    """
    try:
        bone_names = fitter.skel.bone_names
        if target_bone_name in bone_names:
            bone_idx = bone_names.index(target_bone_name)
        else:
            print(f"Bone '{target_bone_name}' not found.")
            return None, None

        # Get skinning weights: (V, J)
        weights = fitter.skel.skel_weights.to_dense().cpu().numpy()
        
        # New Logic: Select vertices where weight for this bone > threshold
        # This captures the overlapping "joint" areas properly.
        mask = weights[:, bone_idx] > threshold
        
        # Debug info
        # print(f"Bone {target_bone_name} (ID {bone_idx}): {mask.sum()} vertices selected with thresh {threshold}")
        
        import trimesh
        full_mesh = trimesh.Trimesh(vertices, faces, process=False)
        
        # Keep faces where ALL vertices are in the mask (strict)
        # OR where ANY vertex is in the mask? 
        # Usually strict is better to avoid flying triangles, but might lose edge faces.
        # Let's try: keep faces where at least 2 vertices are in mask? 
        # Or stick to all vertices. With a low threshold (0.1), 'all' should be fine.
        valid_faces = mask[faces].all(axis=1)
        face_indices = np.where(valid_faces)[0]
        
        if len(face_indices) == 0:
            print(f"No faces found for bone {target_bone_name}.")
            return None, None
            
        bone_mesh = full_mesh.submesh([face_indices], append=True)
        
        out_obj = img_path.replace(".jpg", f"_{target_bone_name}_only.obj").replace(".png", f"_{target_bone_name}_only.obj")
        bone_mesh.export(out_obj)
        # print(f"{target_bone_name.capitalize()} Mesh saved to: {out_obj}")
        
        return bone_mesh.vertices, bone_mesh.faces
        
    except Exception as e:
        print(f"Failed to extract {target_bone_name}: {e}")
        return None, None

def extract_combined_mesh(fitter, vertices, faces, target_bone_names, img_path, suffix="_combined"):
    """
    Extracts a mesh combining multiple bones (e.g. Thorax + Lumbar) to ensure continuous edges.
    """
    try:
        bone_names = fitter.skel.bone_names
        target_indices = []
        for name in target_bone_names:
            if name in bone_names:
                target_indices.append(bone_names.index(name))
            else:
                print(f"Warning: Bone '{name}' not found.")
        
        if not target_indices:
            return None, None

        # Get skinning weights
        weights = fitter.skel.skel_weights.to_dense().cpu().numpy()
        vertex_bone_ids = np.argmax(weights, axis=1)
        
        # Select vertices that belong to ANY of the target bones
        mask = np.isin(vertex_bone_ids, target_indices)
        
        import trimesh
        full_mesh = trimesh.Trimesh(vertices, faces, process=False)
        
        # Keep faces where ALL vertices are in the mask
        # Since we included neighbor bones (e.g. Thorax + Lumbar), the shared faces are preserved!
        valid_faces = mask[faces].all(axis=1)
        face_indices = np.where(valid_faces)[0]
        
        if len(face_indices) == 0:
            print(f"No faces found for combined bones.")
            return None, None
            
        combined_mesh = full_mesh.submesh([face_indices], append=True)
        
        out_obj = img_path.replace(".jpg", f"{suffix}.obj").replace(".png", f"{suffix}.obj")
        combined_mesh.export(out_obj)
        print(f"Combined Mesh ({target_bone_names}) saved to: {out_obj}")
        
        return combined_mesh.vertices, combined_mesh.faces
        
    except Exception as e:
        print(f"Failed to extract combined mesh: {e}")
        return None, None

def _compute_visibility_mask(vertices: np.ndarray, intrinsics: dict, grid_size: int = 20, z_tolerance: float = 0.05) -> np.ndarray:
    """
    粗粒度 z-buffer 可见性过滤：仅保留“从相机看得到”的表面点。
    返回 bool mask (V,).
    """
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    if intrinsics is None or vertices.shape[0] == 0:
        return np.zeros((vertices.shape[0],), dtype=bool)

    fx, fy = float(intrinsics["fx"]), float(intrinsics.get("fy", intrinsics["fx"]))
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    w = int(intrinsics.get("width", 0))
    h = int(intrinsics.get("height", 0))
    if w <= 0 or h <= 0:
        # fallback: allow all
        return np.ones((vertices.shape[0],), dtype=bool)

    z = vertices[:, 2].copy()
    valid = z > 0.1
    z_safe = z.copy()
    z_safe[z_safe < 0.1] = 0.1

    u = (vertices[:, 0] * fx / z_safe) + cx
    v = (vertices[:, 1] * fy / z_safe) + cy

    grid_w = int(w / grid_size) + 1
    grid_h = int(h / grid_size) + 1
    zbuf = np.full((grid_h, grid_w), np.inf, dtype=np.float32)

    ui = np.clip((u / grid_size).astype(int), 0, grid_w - 1)
    vi = np.clip((v / grid_size).astype(int), 0, grid_h - 1)

    # paint near to far
    sort_idx = np.argsort(z_safe)
    for i in sort_idx:
        if not valid[i]:
            continue
        r, c = vi[i], ui[i]
        if z_safe[i] < zbuf[r, c]:
            zbuf[r, c] = z_safe[i]

    min_z = zbuf[vi, ui]
    return (z_safe <= (min_z + float(z_tolerance))) & valid


def _bone_center_xy(weights: np.ndarray, vertices: np.ndarray, bone_names: list, bone_name: str, thr: float = 0.05) -> Optional[np.ndarray]:
    if bone_name not in bone_names:
        return None
    j = bone_names.index(bone_name)
    m = weights[:, j] > float(thr)
    if not np.any(m):
        return None
    return vertices[m][:, :2].mean(axis=0).astype(np.float64)


def _infer_lr_and_inferior_dirs(weights: np.ndarray, vertices: np.ndarray, bone_names: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回 (center_ref_xy, lr_dir_xy, inferior_dir_xy)，都在相机 XY 平面。
    - lr_dir：人体左右轴（优先用左右肱骨中心差）
    - inferior_dir：从胸到骨盆方向（thorax -> pelvis）
    """
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    center_ref = _bone_center_xy(weights, v, bone_names, "thorax")  # 参考中心
    if center_ref is None:
        center_ref = v[:, :2].mean(axis=0).astype(np.float64)

    thorax_c = _bone_center_xy(weights, v, bone_names, "thorax")
    if thorax_c is None:
        thorax_c = center_ref
    pelvis_c = _bone_center_xy(weights, v, bone_names, "pelvis")
    if pelvis_c is None:
        pelvis_c = v[:, :2].mean(axis=0).astype(np.float64)

    inferior_dir = (pelvis_c - thorax_c).astype(np.float64)
    n = float(np.linalg.norm(inferior_dir))
    if n < 1e-6:
        inferior_dir = np.array([0.0, 1.0], dtype=np.float64)
    else:
        inferior_dir = inferior_dir / n

    # left-right from arms if possible
    right_candidates = ["r_humerus", "right_humerus", "humerus_r", "r_upperarm", "right_upperarm"]
    left_candidates = ["l_humerus", "left_humerus", "humerus_l", "l_upperarm", "left_upperarm"]
    r_c = None
    l_c = None
    for nm in right_candidates:
        r_c = _bone_center_xy(weights, v, bone_names, nm)
        if r_c is not None:
            break
    for nm in left_candidates:
        l_c = _bone_center_xy(weights, v, bone_names, nm)
        if l_c is not None:
            break

    if r_c is not None and l_c is not None:
        lr_dir = (r_c - l_c).astype(np.float64)
        nn = float(np.linalg.norm(lr_dir))
        if nn < 1e-6:
            lr_dir = np.array([1.0, 0.0], dtype=np.float64)
        else:
            lr_dir = lr_dir / nn
    else:
        # fallback: image x as lr
        lr_dir = np.array([1.0, 0.0], dtype=np.float64)

    # orthonormalize inferior to lr
    inferior_dir = inferior_dir - float(np.dot(inferior_dir, lr_dir)) * lr_dir
    nn2 = float(np.linalg.norm(inferior_dir))
    if nn2 < 1e-6:
        inferior_dir = np.array([0.0, 1.0], dtype=np.float64)
    else:
        inferior_dir = inferior_dir / nn2

    return center_ref.astype(np.float64), lr_dir.astype(np.float64), inferior_dir.astype(np.float64)

def _project_xyz_to_uv_single(p3: np.ndarray, intr: dict) -> Optional[np.ndarray]:
    """
    Project a single 3D point (x,y,z) in camera frame to uv.
    Returns float32 (2,) or None.
    """
    if p3 is None or intr is None:
        return None
    p3 = np.asarray(p3, dtype=np.float64).reshape(3)
    z = float(p3[2])
    if not np.isfinite(z) or z <= 1e-6:
        return None
    fx = float(intr["fx"])
    fy = float(intr.get("fy", intr["fx"]))
    cx = float(intr["cx"])
    cy = float(intr["cy"])
    u = fx * (float(p3[0]) / z) + cx
    v = fy * (float(p3[1]) / z) + cy
    return np.array([u, v], dtype=np.float32)

def _draw_spine_line_overlay(img_bgr: np.ndarray, intr: dict, thorax_xy: np.ndarray, pelvis_xy: np.ndarray, z_ref: float,
                             color=(255, 0, 0), thickness: int = 4):
    """
    Draw a spine line (thorax->pelvis) onto an image.
    color default: blue in BGR.
    """
    if img_bgr is None:
        return None
    out = img_bgr.copy()
    try:
        p0 = _project_xyz_to_uv_single([float(thorax_xy[0]), float(thorax_xy[1]), float(z_ref)], intr)
        p1 = _project_xyz_to_uv_single([float(pelvis_xy[0]), float(pelvis_xy[1]), float(z_ref)], intr)
        if p0 is None or p1 is None:
            return out
        a = tuple(np.round(p0).astype(np.int32).tolist())
        b = tuple(np.round(p1).astype(np.int32).tolist())
        cv2.line(out, a, b, color, int(thickness))
        cv2.circle(out, a, 6, color, -1)
        cv2.circle(out, b, 6, color, -1)
    except Exception:
        return out
    return out

def _moving_average_xyz(xyz: np.ndarray, win: int = 3) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if xyz.shape[0] < 3 or int(win) <= 1:
        return xyz.astype(np.float32)
    w = int(win)
    w = max(3, w if (w % 2 == 1) else (w + 1))
    r = w // 2
    out = xyz.copy()
    for i in range(xyz.shape[0]):
        j0 = max(0, i - r)
        j1 = min(xyz.shape[0], i + r + 1)
        out[i] = xyz[j0:j1].mean(axis=0)
    return out.astype(np.float32)

def extract_spine_centerline_curve_xyz(
    fitter,
    vertices: np.ndarray,
    faces: np.ndarray,
    intrinsics: dict,
    num_pts: int = 30,
    mid_abs_lr_percentile: float = 8.0,
    smooth_win: int = 5,
    # limit the spine curve to a more reasonable region (avoid "too early" points)
    region: str = "lumbar_to_pelvis",
    inf_margin_m: float = 0.03,
) -> Optional[np.ndarray]:
    """
    通过 mesh 的“中线”估计一条轻微弯曲的 spine curve（相机坐标系 3D 点序列）。
    思路：在 abdomen/torso 点中，沿 inferior_dir 分 bin，每个 bin 取 |coord_lr| 最小的一小撮点的均值作为“脊柱中心点”。
    """
    try:
        v = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        if v.shape[0] == 0:
            return None
        bone_names = list(getattr(fitter.skel, "bone_names", []))
        weights = fitter.skel.skel_weights.to_dense().detach().cpu().numpy()
        vertex_bone_ids = np.argmax(weights, axis=1)
        torso_bones = []
        for nm in ["pelvis", "lumbar_body", "thorax"]:
            if nm in bone_names:
                torso_bones.append(bone_names.index(nm))
        mask_torso = np.isin(vertex_bone_ids, torso_bones) if torso_bones else np.ones((v.shape[0],), dtype=bool)
        visible = _compute_visibility_mask(v, intrinsics, grid_size=20, z_tolerance=0.05)
        cand = mask_torso & visible
        if np.count_nonzero(cand) < 50:
            cand = mask_torso

        center_ref, lr_dir, inferior_dir = _infer_lr_and_inferior_dirs(weights, v, bone_names)
        xy = v[:, :2].astype(np.float64) - center_ref.reshape(1, 2)
        coord_lr = (xy @ lr_dir.reshape(2, 1))[:, 0]
        coord_inf = (xy @ inferior_dir.reshape(2, 1))[:, 0]

        inf_vals = coord_inf[cand]
        if inf_vals.size < 10:
            return None

        # Default range via percentiles
        lo = float(np.percentile(inf_vals, 10.0))
        hi = float(np.percentile(inf_vals, 90.0))

        # Refine range using bone centers (helps keep curve near actual "spine region")
        reg = str(region or "").strip().lower()
        margin = float(inf_margin_m)
        margin = max(0.0, min(0.15, margin))
        lumbar_xy = _bone_center_xy(weights, v.astype(np.float64), bone_names, "lumbar_body")
        pelvis_xy = _bone_center_xy(weights, v.astype(np.float64), bone_names, "pelvis")
        thorax_xy = _bone_center_xy(weights, v.astype(np.float64), bone_names, "thorax")
        def _inf_of_xy(pxy):
            if pxy is None:
                return None
            dxy = (np.asarray(pxy, dtype=np.float64).reshape(2) - center_ref.reshape(2))
            return float(dxy @ inferior_dir.reshape(2))

        inf_lumbar = _inf_of_xy(lumbar_xy)
        inf_pelvis = _inf_of_xy(pelvis_xy)
        inf_thorax = _inf_of_xy(thorax_xy)

        if reg in ("lumbar_to_pelvis", "lumbar-pelvis", "lumbar2pelvis"):
            if (inf_lumbar is not None) and (inf_pelvis is not None):
                lo = min(inf_lumbar, inf_pelvis) - margin
                hi = max(inf_lumbar, inf_pelvis) + margin
        elif reg in ("thorax_to_pelvis", "thorax-pelvis", "thorax2pelvis"):
            if (inf_thorax is not None) and (inf_pelvis is not None):
                lo = min(inf_thorax, inf_pelvis) - margin
                hi = max(inf_thorax, inf_pelvis) + margin

        if hi <= lo + 1e-6:
            lo, hi = float(np.min(inf_vals)), float(np.max(inf_vals))

        n = max(10, int(num_pts))
        edges = np.linspace(lo, hi, n + 1, dtype=np.float64)
        pts = []
        for i in range(n):
            a = edges[i]
            b = edges[i + 1]
            m = cand & (coord_inf >= a) & (coord_inf < b)
            if np.count_nonzero(m) < 10:
                continue
            
            # Robust centerline finding: use Median of LR in this bin
            # This handles asymmetry or side-shifts better than just taking min(abs(lr)).
            bin_lr = coord_lr[m]
            med_lr = float(np.median(bin_lr))
            
            # Filter points near the median LR within a narrow band (e.g. +/- 2cm)
            # This ensures we pick the center of the point cloud mass for this slice.
            band_width = 0.02
            mm = m & (np.abs(coord_lr - med_lr) <= band_width)
            if np.count_nonzero(mm) < 5:
                # fallback: if narrow band is empty, use the median-filtered points directly
                # (actually mm won't be empty if band_width > 0 and we center on median, 
                # but just in case of float precision issues)
                mm = m
            
            p = v[mm].astype(np.float64)
            # use mean xyz; z is surface-ish but ok for projection/visualization
            pts.append(p.mean(axis=0))

        if len(pts) < 5:
            return None
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        # sort by inferior coord
        xy2 = pts[:, :2] - center_ref.reshape(1, 2)
        inf2 = (xy2 @ inferior_dir.reshape(2, 1))[:, 0]
        pts = pts[np.argsort(inf2)]
        pts = _moving_average_xyz(pts, win=smooth_win)
        return pts.astype(np.float32)
    except Exception as e:
        print(f"Spine curve extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def _curve_normals_xy(curve_xy: np.ndarray, lr_dir: np.ndarray) -> np.ndarray:
    """
    给曲线每个点估计一个“右侧法向量”（在相机 XY 平面）。
    """
    c = np.asarray(curve_xy, dtype=np.float64).reshape(-1, 2)
    if c.shape[0] < 2:
        return np.zeros_like(c)
    lr_dir = np.asarray(lr_dir, dtype=np.float64).reshape(2)
    lr_dir = lr_dir / (float(np.linalg.norm(lr_dir)) + 1e-9)
    n_all = np.zeros((c.shape[0], 2), dtype=np.float64)
    for i in range(c.shape[0]):
        if i == 0:
            t = c[i + 1] - c[i]
        elif i == c.shape[0] - 1:
            t = c[i] - c[i - 1]
        else:
            t = c[i + 1] - c[i - 1]
        nt = float(np.linalg.norm(t))
        if nt < 1e-9:
            t = np.array([0.0, 1.0], dtype=np.float64)
            nt = 1.0
        t = t / nt
        n = np.array([-t[1], t[0]], dtype=np.float64)
        nn = float(np.linalg.norm(n)) + 1e-9
        n = n / nn
        # ensure pointing to "right" (dot with lr_dir positive)
        if float(np.dot(n, lr_dir)) < 0.0:
            n = -n
        n_all[i] = n
    return n_all

def extract_kidney_traj_parallel_spine_curve(
    fitter,
    vertices: np.ndarray,
    faces: np.ndarray,
    intrinsics: dict,
    spine_curve_xyz: np.ndarray,
    side: str = "right",
    kidney_length_m: float = 0.22,
    num_samples: int = 40,
    offset_mode: str = "lr_dir",
    start_point_xyz: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    从“lateral band 的 most-inferior 起点”出发，生成一条与 spine curve 平行的肾脏扫描轨迹：
    - spine_curve_xyz: Nx3
    - 输出轨迹点来自 mesh（贴在人体表面），但目标几何是“spine curve 的平行偏移”
    """
    try:
        v = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        if v.shape[0] == 0 or spine_curve_xyz is None or len(spine_curve_xyz) < 5:
            return None

        bone_names = list(getattr(fitter.skel, "bone_names", []))
        weights = fitter.skel.skel_weights.to_dense().detach().cpu().numpy()
        vertex_bone_ids = np.argmax(weights, axis=1)
        torso_bones = []
        for nm in ["pelvis", "lumbar_body", "thorax"]:
            if nm in bone_names:
                torso_bones.append(bone_names.index(nm))
        mask_torso = np.isin(vertex_bone_ids, torso_bones) if torso_bones else np.ones((v.shape[0],), dtype=bool)
        visible = _compute_visibility_mask(v, intrinsics, grid_size=20, z_tolerance=0.05)
        cand = mask_torso & visible
        if np.count_nonzero(cand) < 50:
            cand = mask_torso

        center_ref, lr_dir, inferior_dir = _infer_lr_and_inferior_dirs(weights, v, bone_names)
        xy = v[:, :2].astype(np.float64) - center_ref.reshape(1, 2)
        coord_lr = (xy @ lr_dir.reshape(2, 1))[:, 0]
        coord_inf = (xy @ inferior_dir.reshape(2, 1))[:, 0]

        side_lc = str(side or "right").strip().lower()
        want_right = side_lc in ("right", "r", "rhs")
        side_mask = (coord_lr > 0.0) if want_right else (coord_lr < 0.0)
        cand = cand & side_mask
        if np.count_nonzero(cand) < 50:
            cand = mask_torso & visible  # relax side in worst case

        inf_vals = coord_inf[cand]
        if inf_vals.size < 10:
            return None
        lo = float(np.percentile(inf_vals, 20.0))
        hi = float(np.max(inf_vals))

        # start point:
        # - preferred: use ribline end point (same as previous pipeline end_pt) if provided
        # - fallback: lateral band (85%) then most inferior
        if start_point_xyz is not None:
            sp = np.asarray(start_point_xyz, dtype=np.float64).reshape(3)
            start_xy = (sp[:2].astype(np.float64) - center_ref).reshape(2)
            start_inf = float(((start_xy.reshape(1, 2)) @ inferior_dir.reshape(2, 1))[0, 0])
        else:
            in_range = cand & (coord_inf >= lo)
            lr_vals_in = coord_lr[in_range]
            if lr_vals_in.size < 10:
                in_range = cand
                lr_vals_in = coord_lr[in_range]
            if want_right:
                lat_thr = float(np.percentile(lr_vals_in, 85.0))
                lat_band = in_range & (coord_lr >= lat_thr)
            else:
                lat_thr = float(np.percentile(lr_vals_in, 15.0))
                lat_band = in_range & (coord_lr <= lat_thr)
            if np.count_nonzero(lat_band) == 0:
                lat_band = in_range
            start_idx = int(np.argmax(np.where(lat_band, coord_inf, -1e18)))
            start_xy = (v[start_idx, :2].astype(np.float64) - center_ref).reshape(2)
            start_inf = float(coord_inf[start_idx])

        spine = np.asarray(spine_curve_xyz, dtype=np.float64).reshape(-1, 3)
        spine_xy = spine[:, :2].astype(np.float64) - center_ref.reshape(1, 2)
        spine_inf = (spine_xy @ inferior_dir.reshape(2, 1))[:, 0]
        order = np.argsort(spine_inf)
        spine = spine[order]
        spine_xy = spine_xy[order]
        spine_inf = spine_inf[order]

        # find spine index nearest start_inf
        i0 = int(np.argmin(np.abs(spine_inf - float(start_inf))))
        # Determine "parallel offset direction"
        # - previous version used local curve normal (changes direction along curve)
        # - new default uses global lr_dir so the offset direction is stable and matches user expectation
        mode = str(offset_mode or "lr_dir").strip().lower()
        lr_u = np.asarray(lr_dir, dtype=np.float64).reshape(2)
        lr_u = lr_u / (float(np.linalg.norm(lr_u)) + 1e-9)
        if mode in ("lr", "lr_dir", "lrd", "global_lr"):
            off = float(np.dot(start_xy - spine_xy[i0], lr_u))
            # enforce side sign
            if want_right and off < 0:
                off = -off
            if (not want_right) and off > 0:
                off = -off
            target_xy = spine_xy + lr_u.reshape(1, 2) * off
        else:
            normals = _curve_normals_xy(spine_xy, lr_dir=lr_dir)
            if not want_right:
                normals = -normals
            off = float(np.dot(start_xy - spine_xy[i0], normals[i0]))
            if not np.isfinite(off):
                off = float(np.linalg.norm(start_xy - spine_xy[i0]))
            target_xy = spine_xy + normals * off

        # choose length along curve starting from i0
        L = float(kidney_length_m)
        L = max(0.02, min(0.5, L))
        # cumulative length along target curve
        cum = [0.0]
        for i in range(1, target_xy.shape[0]):
            d = float(np.linalg.norm(target_xy[i] - target_xy[i - 1]))
            cum.append(cum[-1] + d)
        cum = np.asarray(cum, dtype=np.float64)
        # keep only indices with inf >= start_inf (inferior direction)
        keep = np.where(spine_inf >= float(start_inf) - 1e-6)[0]
        if keep.size < 2:
            keep = np.arange(i0, spine_inf.shape[0])
        if keep.size < 2:
            keep = np.arange(0, spine_inf.shape[0])
        # apply length cap from start index
        base_c = cum[i0] if i0 < len(cum) else 0.0
        keep2 = [k for k in keep if (cum[k] - base_c) <= L]
        if len(keep2) < 2:
            keep2 = keep.tolist()
        keep2 = np.asarray(keep2, dtype=np.int32)

        # sample approximately num_samples points along keep2 (keep at least 5)
        ns = max(5, int(num_samples))
        if keep2.size > ns:
            idxs = np.linspace(0, keep2.size - 1, ns, dtype=np.int32)
            keep2 = keep2[idxs]

        # snap to mesh per target point: pick closest in XY within a local inf window
        traj = []
        z = v[:, 2].astype(np.float64)
        for k in keep2:
            ti = float(spine_inf[k])
            # window width based on neighbor spacing
            w = 0.03
            m = cand & (coord_inf >= (ti - w)) & (coord_inf <= (ti + w))
            if np.count_nonzero(m) < 10:
                m = cand
            dx = (xy[:, 0] - float(target_xy[k, 0]))
            dy = (xy[:, 1] - float(target_xy[k, 1]))
            d2 = dx * dx + dy * dy
            # score: xy distance + small z penalty (prefer front)
            score = np.where(m, d2 + 0.02 * (z - np.nanmin(z)), np.inf)
            idx = int(np.argmin(score))
            if np.isfinite(score[idx]):
                traj.append(v[idx])

        if len(traj) < 2:
            return None
        traj = np.asarray(traj, dtype=np.float32).reshape(-1, 3)
        # ensure start point is first
        if start_point_xyz is not None:
            traj = np.concatenate([np.asarray(start_point_xyz, dtype=np.float32).reshape(1, 3), traj], axis=0)
        else:
            traj = np.concatenate([v[start_idx].reshape(1, 3), traj], axis=0)
        return traj.astype(np.float32)
    except Exception as e:
        print(f"Kidney parallel-to-spine-curve extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def _polyline_arclen(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if xy.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    s = np.zeros((xy.shape[0],), dtype=np.float64)
    for i in range(1, xy.shape[0]):
        s[i] = s[i - 1] + float(np.linalg.norm(xy[i] - xy[i - 1]))
    return s

def _pick_point_on_polyline_by_frac(xyz: np.ndarray, frac: float) -> Optional[np.ndarray]:
    """
    Pick a point on a polyline by arc-length fraction (0..1).
    Returns a 3D point (float32) via linear interpolation.
    """
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 2:
        return pts[0].astype(np.float32) if pts.shape[0] == 1 else None
    frac = float(frac)
    frac = max(0.0, min(1.0, frac))
    s = _polyline_arclen(pts[:, :2])
    total = float(s[-1])
    if total <= 1e-9:
        return pts[0].astype(np.float32)
    target = total * frac
    j = int(np.searchsorted(s, target, side="left"))
    j = max(1, min(len(s) - 1, j))
    s0, s1 = float(s[j - 1]), float(s[j])
    t = 0.0 if (s1 - s0) <= 1e-9 else (target - s0) / (s1 - s0)
    p = pts[j - 1] * (1.0 - t) + pts[j] * t
    return p.astype(np.float32)

def estimate_t10_t11_from_spine_curve(
    fitter,
    skel_verts: np.ndarray,
    spine_curve_xyz: np.ndarray,
    intrinsics: dict,
    t10_frac: float = 0.80,
    t11_frac: float = 0.90,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    NOTE: SKEL does not provide explicit T10/T11 joints.
    We approximate T10/T11 along the spine centerline curve segment between thorax and lumbar_body.
    Fractions are along arc-length from thorax (0) to lumbar_body (1).
    """
    try:
        if spine_curve_xyz is None or len(spine_curve_xyz) < 5:
            return None, None
        v = np.asarray(skel_verts, dtype=np.float64).reshape(-1, 3)
        bone_names = list(getattr(fitter.skel, "bone_names", []))
        weights = fitter.skel.skel_weights.to_dense().detach().cpu().numpy()
        center_ref, lr_dir, inferior_dir = _infer_lr_and_inferior_dirs(weights, v, bone_names)
        thorax_xy = _bone_center_xy(weights, v, bone_names, "thorax")
        lumbar_xy = _bone_center_xy(weights, v, bone_names, "lumbar_body")
        if thorax_xy is None or lumbar_xy is None:
            return None, None
        thorax_inf = float(((thorax_xy - center_ref) @ inferior_dir.reshape(2)))
        lumbar_inf = float(((lumbar_xy - center_ref) @ inferior_dir.reshape(2)))
        lo = min(thorax_inf, lumbar_inf)
        hi = max(thorax_inf, lumbar_inf)

        sc = np.asarray(spine_curve_xyz, dtype=np.float64).reshape(-1, 3)
        sc_xy = sc[:, :2] - center_ref.reshape(1, 2)
        sc_inf = (sc_xy @ inferior_dir.reshape(2, 1))[:, 0]
        # keep only within thorax<->lumbar range (with a small margin)
        m = (sc_inf >= (lo - 1e-3)) & (sc_inf <= (hi + 1e-3))
        seg = sc[m]
        if seg.shape[0] < 5:
            seg = sc  # fallback
        # ensure ordering from thorax->lumbar
        seg_xy = seg[:, :2] - center_ref.reshape(1, 2)
        seg_inf = (seg_xy @ inferior_dir.reshape(2, 1))[:, 0]
        if thorax_inf <= lumbar_inf:
            seg = seg[np.argsort(seg_inf)]
        else:
            seg = seg[np.argsort(-seg_inf)]

        t10 = _pick_point_on_polyline_by_frac(seg, float(t10_frac))
        t11 = _pick_point_on_polyline_by_frac(seg, float(t11_frac))
        return t10, t11
    except Exception:
        return None, None

# ------------------------------
# Paper-style kidney scan path (CVA-based)
# Ref: "Point cloud-guided ultrasound robotic scanning path planning for the kidney based on anatomical positioning" (2025)
# Steps (Fig.1 / Sec 3.4):
# ① vertebral line + lowest lumbar line
# ② intersection -> vertebral boundary of CVA
# ③ rotate spinal edge by CVA angle to get rib boundary direction
# ④ start at CVA apex and scan longitudinally along rib boundary
# ------------------------------

def _fit_line_pca_xy(xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a 2D line by PCA.
    Returns (p0, d_unit) where p0 is mean point, d_unit is direction unit vector.
    """
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 2:
        p0 = pts[0] if pts.shape[0] == 1 else np.zeros((2,), dtype=np.float64)
        return p0.astype(np.float64), np.array([1.0, 0.0], dtype=np.float64)
    mu = pts.mean(axis=0)
    x = pts - mu
    cov = (x.T @ x) / max(1, len(x))
    w, v = np.linalg.eigh(cov)
    d = v[:, int(np.argmax(w))].astype(np.float64)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        d = np.array([1.0, 0.0], dtype=np.float64)
    else:
        d = d / n
    return mu.astype(np.float64), d.astype(np.float64)

def _line_intersection_2d(p0: np.ndarray, d0: np.ndarray, p1: np.ndarray, d1: np.ndarray) -> Optional[np.ndarray]:
    """
    Intersection of two infinite 2D lines:
      L0: p0 + t*d0
      L1: p1 + s*d1
    Returns point or None if parallel.
    """
    p0 = np.asarray(p0, dtype=np.float64).reshape(2)
    d0 = np.asarray(d0, dtype=np.float64).reshape(2)
    p1 = np.asarray(p1, dtype=np.float64).reshape(2)
    d1 = np.asarray(d1, dtype=np.float64).reshape(2)
    A = np.stack([d0, -d1], axis=1)  # 2x2
    b = (p1 - p0).reshape(2)
    det = float(np.linalg.det(A))
    if abs(det) < 1e-9:
        return None
    t_s = np.linalg.solve(A, b)
    t = float(t_s[0])
    return (p0 + d0 * t).astype(np.float64)

def _rotate2d(v: np.ndarray, deg: float) -> np.ndarray:
    th = float(deg) * np.pi / 180.0
    c = float(np.cos(th))
    s = float(np.sin(th))
    x, y = [float(x) for x in np.asarray(v, dtype=np.float64).reshape(2)]
    return np.array([c * x - s * y, s * x + c * y], dtype=np.float64)

def extract_lowest_lumbar_line_xy(
    fitter,
    vertices: np.ndarray,
    intrinsics: dict,
    spine_center_ref_xy: np.ndarray,
    lr_dir: np.ndarray,
    inferior_dir: np.ndarray,
    band_half_width_m: float = 0.10,
    num_lr_bins: int = 15,
) -> Optional[np.ndarray]:
    """
    Approximate the "lowest lumbar line" R as a set of points near the spinal midline band,
    by taking the most-inferior point in each lr-bin within a band.
    Returns Nx2 points in camera XY coordinates (not centered).
    """
    v = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    if v.shape[0] == 0:
        return None
    visible = _compute_visibility_mask(v, intrinsics, grid_size=20, z_tolerance=0.05)
    bone_names = list(getattr(fitter.skel, "bone_names", []))
    weights = fitter.skel.skel_weights.to_dense().detach().cpu().numpy()
    vertex_bone_ids = np.argmax(weights, axis=1)
    torso_bones = []
    for nm in ["pelvis", "lumbar_body", "thorax"]:
        if nm in bone_names:
            torso_bones.append(bone_names.index(nm))
    mask_torso = np.isin(vertex_bone_ids, torso_bones) if torso_bones else np.ones((v.shape[0],), dtype=bool)
    cand = mask_torso & visible
    if np.count_nonzero(cand) < 200:
        cand = mask_torso

    xy = v[:, :2].astype(np.float64)
    ref = np.asarray(spine_center_ref_xy, dtype=np.float64).reshape(2)
    dxy = xy - ref.reshape(1, 2)
    lr = (dxy @ np.asarray(lr_dir, dtype=np.float64).reshape(2, 1))[:, 0]
    inf = (dxy @ np.asarray(inferior_dir, dtype=np.float64).reshape(2, 1))[:, 0]

    bw = float(band_half_width_m)
    bw = max(0.02, min(0.30, bw))
    band = cand & (np.abs(lr) <= bw)
    if np.count_nonzero(band) < 50:
        band = cand

    lo = float(np.percentile(lr[band], 1.0))
    hi = float(np.percentile(lr[band], 99.0))
    if hi <= lo + 1e-6:
        lo, hi = -bw, bw
    nb = max(5, int(num_lr_bins))
    edges = np.linspace(lo, hi, nb + 1, dtype=np.float64)

    pts = []
    for i in range(nb):
        a = edges[i]
        b = edges[i + 1]
        m = band & (lr >= a) & (lr < b)
        if np.count_nonzero(m) < 10:
            continue
        idx = int(np.argmax(np.where(m, inf, -1e18)))
        pts.append(xy[idx])
    if len(pts) < 5:
        return None
    return np.asarray(pts, dtype=np.float64).reshape(-1, 2)

def plan_kidney_scan_path_paper(
    fitter,
    vertices: np.ndarray,
    intrinsics: dict,
    spine_curve_xyz: np.ndarray,
    cva_deg: float = 45.0,
    w1_m: float = 0.10,
    path_len_m: float = 0.12,
    step_m: float = 0.01,
    band_half_width_m: float = 0.10,
    num_lr_bins: int = 15,
    snap_window_m: float = 0.03,
) -> Optional[np.ndarray]:
    """
    Implements a practical version of Fig.1 / Sec 3.4 for our SKEL mesh:
    - vertebral line: spine_curve_xyz
    - lowest lumbar line: extracted from torso band
    - CVA vertex: intersection of fitted vertebral line (local) and fitted lowest lumbar line
    - rib boundary direction: rotate vertebral direction by cva_deg in the XY plane
    - scanning path: sample along rotated direction from apex and snap to surface
    Returns Nx3 (camera frame) snapped points.
    """
    v = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    if v.shape[0] == 0 or spine_curve_xyz is None or len(spine_curve_xyz) < 8:
        return None

    bone_names = list(getattr(fitter.skel, "bone_names", []))
    weights = fitter.skel.skel_weights.to_dense().detach().cpu().numpy()
    center_ref, lr_dir, inferior_dir = _infer_lr_and_inferior_dirs(weights, v, bone_names)

    sc = np.asarray(spine_curve_xyz, dtype=np.float64).reshape(-1, 3)
    sc_xy = sc[:, :2].astype(np.float64)
    # choose an L3-ish base point: most inferior on spine curve within lumbar_to_pelvis region
    dxy_sc = sc_xy - center_ref.reshape(1, 2)
    sc_inf = (dxy_sc @ inferior_dir.reshape(2, 1))[:, 0]
    i_l3 = int(np.argmax(sc_inf))
    l3_xy = sc_xy[i_l3]

    # spinal edge point cloud S: take w1_m arc-length on the spine curve starting at i_l3 towards superior
    # (we only need direction; use a local window)
    win_pts = sc_xy[max(0, i_l3 - 8):min(len(sc_xy), i_l3 + 8)]
    p_sp, d_sp = _fit_line_pca_xy(win_pts)

    # lowest lumbar line R from torso band near spine
    R_xy = extract_lowest_lumbar_line_xy(
        fitter=fitter,
        vertices=v,
        intrinsics=intrinsics,
        spine_center_ref_xy=center_ref,
        lr_dir=lr_dir,
        inferior_dir=inferior_dir,
        band_half_width_m=band_half_width_m,
        num_lr_bins=num_lr_bins,
    )
    if R_xy is None or len(R_xy) < 2:
        return None
    p_r, d_r = _fit_line_pca_xy(R_xy)

    # CVA vertex v0: intersection of vertebral line and lowest lumbar line
    v0 = _line_intersection_2d(p_sp, d_sp, p_r, d_r)
    if v0 is None:
        v0 = l3_xy.astype(np.float64)

    # rib boundary direction: rotate vertebral direction by CVA angle
    d_rib = _rotate2d(d_sp, float(cva_deg))
    dn = float(np.linalg.norm(d_rib))
    if dn < 1e-9:
        return None
    d_rib = d_rib / dn

    # sample along rib boundary direction from apex
    L = max(0.03, min(0.25, float(path_len_m)))
    ds = max(0.005, min(0.03, float(step_m)))
    n_steps = int(np.floor(L / ds)) + 1

    # snap each target XY to surface near that "projection plane": here simplified to XY-nearest with inferior window
    visible = _compute_visibility_mask(v, intrinsics, grid_size=20, z_tolerance=0.05)
    cand = visible
    xy = v[:, :2].astype(np.float64)
    dxy = xy - center_ref.reshape(1, 2)
    inf = (dxy @ inferior_dir.reshape(2, 1))[:, 0]
    z = v[:, 2].astype(np.float64)

    out = []
    for i in range(n_steps):
        t = float(i) * ds
        tgt_xy = v0 + d_rib * t
        # restrict by an inferior window around the apex to avoid jumping to unrelated parts
        v0_inf = float(((v0 - center_ref.reshape(2)) @ inferior_dir.reshape(2)))
        m = cand & (np.abs(inf - v0_inf) <= max(0.10, snap_window_m + 0.02 + t))
        if np.count_nonzero(m) < 100:
            m = cand
        dx = xy[:, 0] - float(tgt_xy[0])
        dy = xy[:, 1] - float(tgt_xy[1])
        d2 = dx * dx + dy * dy
        score = np.where(m, d2 + 0.02 * (z - np.nanmin(z)), np.inf)
        idx = int(np.argmin(score))
        if np.isfinite(score[idx]):
            out.append(v[idx])
    if len(out) < 2:
        return None
    return np.asarray(out, dtype=np.float32).reshape(-1, 3)

def plan_kidney_scan_path_costal_margin_based(
    fitter,
    vertices: np.ndarray,
    faces: np.ndarray,
    img_path: str,
    intrinsics: dict,
    spine_curve_xyz: np.ndarray,
    side: str = "right",
    path_len_m: float = 0.12,
    step_m: float = 0.01,
    snap_window_m: float = 0.03,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[np.ndarray]]:
    """
    User-requested variant:
    - angle comes from the costal margin direction (ribline), not a fixed deg
    - start point is the intersection of spine line and costal margin line (in XY), then snapped to surface
    Returns: (path_xyz, angle_deg_between_spine_and_cm, start_xyz)
    """
    v = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    if v.shape[0] == 0 or spine_curve_xyz is None or len(spine_curve_xyz) < 8:
        return None, None, None

    bone_names = list(getattr(fitter.skel, "bone_names", []))
    weights = fitter.skel.skel_weights.to_dense().detach().cpu().numpy()
    center_ref, lr_dir, inferior_dir = _infer_lr_and_inferior_dirs(weights, v, bone_names)

    # 1) costal margin (ribline) points
    rib_line_3d = extract_right_costal_margin(fitter, v, faces, img_path, intrinsics)
    if rib_line_3d is None or len(rib_line_3d) < 5:
        return None, None, None
    rib_xy = np.asarray(rib_line_3d, dtype=np.float64).reshape(-1, 3)[:, :2]
    p_cm, d_cm = _fit_line_pca_xy(rib_xy)

    # 2) spine line direction from local window around lumbar region
    sc = np.asarray(spine_curve_xyz, dtype=np.float64).reshape(-1, 3)
    sc_xy = sc[:, :2]
    dxy_sc = sc_xy - center_ref.reshape(1, 2)
    sc_inf = (dxy_sc @ inferior_dir.reshape(2, 1))[:, 0]
    i_l = int(np.argmax(sc_inf))
    win = sc_xy[max(0, i_l - 8):min(len(sc_xy), i_l + 8)]
    p_sp, d_sp = _fit_line_pca_xy(win)

    # 3) intersection in XY as start point (apex proxy)
    v0_xy = _line_intersection_2d(p_sp, d_sp, p_cm, d_cm)
    if v0_xy is None:
        # fallback: closest spine point to costal margin line
        # distance from sc_xy to cm line
        n_cm = np.array([-d_cm[1], d_cm[0]], dtype=np.float64)
        n_cm = n_cm / (float(np.linalg.norm(n_cm)) + 1e-9)
        dist = np.abs(((sc_xy - p_cm.reshape(1, 2)) @ n_cm.reshape(2, 1))[:, 0])
        v0_xy = sc_xy[int(np.argmin(dist))]

    # 4) angle derived from costal margin vs spine
    dot = float(np.clip(np.dot(d_sp, d_cm), -1.0, 1.0))
    ang = float(np.degrees(np.arccos(abs(dot))))

    # 5) scan direction: use costal margin direction, but force "down" along inferior_dir
    scan_dir = d_cm.copy()
    if float(np.dot(scan_dir, inferior_dir.reshape(2))) < 0.0:
        scan_dir = -scan_dir
    scan_dir = scan_dir / (float(np.linalg.norm(scan_dir)) + 1e-9)

    # 6) snap start point to surface
    visible = _compute_visibility_mask(v, intrinsics, grid_size=20, z_tolerance=0.05)
    cand = visible
    xy = v[:, :2].astype(np.float64)
    dxy = xy - center_ref.reshape(1, 2)
    inf = (dxy @ inferior_dir.reshape(2, 1))[:, 0]
    z = v[:, 2].astype(np.float64)
    v0_inf = float(((v0_xy - center_ref.reshape(2)) @ inferior_dir.reshape(2)))

    dx0 = xy[:, 0] - float(v0_xy[0])
    dy0 = xy[:, 1] - float(v0_xy[1])
    d20 = dx0 * dx0 + dy0 * dy0
    m0 = cand & (np.abs(inf - v0_inf) <= 0.12)
    if np.count_nonzero(m0) < 100:
        m0 = cand
    score0 = np.where(m0, d20 + 0.02 * (z - np.nanmin(z)), np.inf)
    i0 = int(np.argmin(score0))
    start_xyz = v[i0].astype(np.float32)

    # 7) sample along scan_dir and snap each point
    L = max(0.03, min(0.25, float(path_len_m)))
    ds = max(0.005, min(0.03, float(step_m)))
    n_steps = int(np.floor(L / ds)) + 1

    out = []
    for i in range(n_steps):
        t = float(i) * ds
        tgt_xy = v0_xy + scan_dir * t
        # local window grows slightly with t to reduce snapping jumps
        w = max(0.10, float(snap_window_m) + 0.02 + t)
        m = cand & (np.abs(inf - v0_inf) <= w)
        if np.count_nonzero(m) < 100:
            m = cand
        dx = xy[:, 0] - float(tgt_xy[0])
        dy = xy[:, 1] - float(tgt_xy[1])
        d2 = dx * dx + dy * dy
        score = np.where(m, d2 + 0.02 * (z - np.nanmin(z)), np.inf)
        idx = int(np.argmin(score))
        if np.isfinite(score[idx]):
            out.append(v[idx])
    if len(out) < 2:
        return None, ang, start_xyz
    return np.asarray(out, dtype=np.float32).reshape(-1, 3), ang, start_xyz

def plan_kidney_scan_path_angle_line_offset(
    fitter,
    vertices: np.ndarray,
    faces: np.ndarray,
    img_path: str,
    intrinsics: dict,
    spine_curve_xyz: np.ndarray,
    side: str = "right",
    offset_m: float = 0.03,
    step_m: float = 0.01,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    User-requested:
    - Define an "angle line" as a STRAIGHT line from start point (spine ∩ costal margin)
      to the END of the rib (use lateral end of costal margin).
    - Then generate a PARALLEL OFFSET line (like drawing two rib lines) and use that as scan path.

    Returns (offset_path_xyz, start_xyz, rib_end_xyz)
    """
    v = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    if v.shape[0] == 0 or spine_curve_xyz is None or len(spine_curve_xyz) < 8:
        return None, None, None

    bone_names = list(getattr(fitter.skel, "bone_names", []))
    weights = fitter.skel.skel_weights.to_dense().detach().cpu().numpy()
    center_ref, lr_dir, inferior_dir = _infer_lr_and_inferior_dirs(weights, v, bone_names)

    # costal margin (ribline) points
    rib_line_3d = extract_right_costal_margin(fitter, v, faces, img_path, intrinsics)
    if rib_line_3d is None or len(rib_line_3d) < 5:
        return None, None, None
    rib_xyz = np.asarray(rib_line_3d, dtype=np.float64).reshape(-1, 3)
    rib_xy = rib_xyz[:, :2]
    p_cm, d_cm = _fit_line_pca_xy(rib_xy)

    # spine local line near lumbar region
    sc = np.asarray(spine_curve_xyz, dtype=np.float64).reshape(-1, 3)
    sc_xy = sc[:, :2]
    dxy_sc = sc_xy - center_ref.reshape(1, 2)
    sc_inf = (dxy_sc @ inferior_dir.reshape(2, 1))[:, 0]
    i_l = int(np.argmax(sc_inf))
    win = sc_xy[max(0, i_l - 8):min(len(sc_xy), i_l + 8)]
    p_sp, d_sp = _fit_line_pca_xy(win)

    # start point = intersection(spine line, costal margin line) in XY
    start_xy = _line_intersection_2d(p_sp, d_sp, p_cm, d_cm)
    if start_xy is None:
        # fallback: closest spine point to costal margin line
        n_cm = np.array([-d_cm[1], d_cm[0]], dtype=np.float64)
        n_cm = n_cm / (float(np.linalg.norm(n_cm)) + 1e-9)
        dist = np.abs(((sc_xy - p_cm.reshape(1, 2)) @ n_cm.reshape(2, 1))[:, 0])
        start_xy = sc_xy[int(np.argmin(dist))]

    # rib end = lateral-most point on the ribline (not the "inferior end")
    dxy_r = rib_xy - center_ref.reshape(1, 2)
    rib_lr = (dxy_r @ lr_dir.reshape(2, 1))[:, 0]
    want_right = str(side or "right").strip().lower() in ("right", "r", "rhs")
    rib_end_i = int(np.argmax(rib_lr) if want_right else np.argmin(rib_lr))
    rib_end_xyz = rib_xyz[rib_end_i].astype(np.float32)
    rib_end_xy = rib_xy[rib_end_i]

    # angle line direction: start -> rib_end
    d = (rib_end_xy - start_xy).astype(np.float64)
    dn = float(np.linalg.norm(d))
    if dn < 1e-6:
        return None, None, rib_end_xyz
    d = d / dn

    # offset direction: perpendicular to d, choose sign to shift "down" (inferior) like a parallel band
    n = np.array([-d[1], d[0]], dtype=np.float64)
    nn = float(np.linalg.norm(n))
    if nn < 1e-9:
        return None, None, rib_end_xyz
    n = n / nn
    if float(np.dot(n, inferior_dir.reshape(2))) < 0.0:
        n = -n

    off = float(offset_m)
    off = max(-0.20, min(0.20, off))
    # offset line endpoints (parallel to angle line)
    p0 = start_xy + n * off
    p1 = rib_end_xy + n * off

    # snap start point to surface near p0 (for exporting start_xyz)
    visible = _compute_visibility_mask(v, intrinsics, grid_size=20, z_tolerance=0.05)
    cand = visible
    xy = v[:, :2].astype(np.float64)
    z = v[:, 2].astype(np.float64)
    dx0 = xy[:, 0] - float(p0[0])
    dy0 = xy[:, 1] - float(p0[1])
    d20 = dx0 * dx0 + dy0 * dy0
    score0 = np.where(cand, d20 + 0.02 * (z - np.nanmin(z)), np.inf)
    i0 = int(np.argmin(score0))
    start_xyz = v[i0].astype(np.float32)

    # sample along offset line and snap each point
    L = float(np.linalg.norm(p1 - p0))
    ds = max(0.005, min(0.03, float(step_m)))
    n_steps = max(2, int(np.floor(L / ds)) + 1)
    out = []
    for i in range(n_steps):
        t = (float(i) / float(n_steps - 1))
        tgt = p0 * (1.0 - t) + p1 * t
        dx = xy[:, 0] - float(tgt[0])
        dy = xy[:, 1] - float(tgt[1])
        d2 = dx * dx + dy * dy
        score = np.where(cand, d2 + 0.02 * (z - np.nanmin(z)), np.inf)
        idx = int(np.argmin(score))
        if np.isfinite(score[idx]):
            out.append(v[idx])
    if len(out) < 2:
        return None, start_xyz, rib_end_xyz
    return np.asarray(out, dtype=np.float32).reshape(-1, 3), start_xyz, rib_end_xyz

def plan_kidney_scan_path_spine_next_to_rib_other_side_offset(
    fitter,
    vertices: np.ndarray,
    faces: np.ndarray,
    img_path: str,
    intrinsics: dict,
    spine_curve_xyz: np.ndarray,
    side: str = "right",
    # endpoint rules (user requested)
    rib_end_side: str = "left",
    spine_next_k: int = 1,
    # offset like upper_only_skel_rib_both (two parallel lines)
    offset_m: float = 0.03,
    step_m: float = 0.01,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    User clarified:
    - start point: the NEXT middle point of spine piece => pick a point on spine curve near CVA intersection,
      then move "next" along the curve (spine_next_k).
    - end point: the OTHER side of the rib (e.g., want LEFT side even if current was right).
    - connect start/end with a STRAIGHT line, then create a parallel OFFSET line like upper_only_skel_rib_both.

    Returns:
      (offset_path_xyz, start_spine_xyz, rib_end_xyz, base_angle_line_xyz_2pts)
    """
    v = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    if v.shape[0] == 0 or spine_curve_xyz is None or len(spine_curve_xyz) < 8:
        return None, None, None, None

    bone_names = list(getattr(fitter.skel, "bone_names", []))
    weights = fitter.skel.skel_weights.to_dense().detach().cpu().numpy()
    center_ref, lr_dir, inferior_dir = _infer_lr_and_inferior_dirs(weights, v, bone_names)

    # ribline points (costal margin)
    rib_line_3d = extract_right_costal_margin(fitter, v, faces, img_path, intrinsics)
    if rib_line_3d is None or len(rib_line_3d) < 5:
        return None, None, None, None
    rib_xyz = np.asarray(rib_line_3d, dtype=np.float64).reshape(-1, 3)
    rib_xy = rib_xyz[:, :2]
    p_cm, d_cm = _fit_line_pca_xy(rib_xy)

    # spine local line (for intersection reference)
    sc = np.asarray(spine_curve_xyz, dtype=np.float64).reshape(-1, 3)
    sc_xy = sc[:, :2]
    dxy_sc = sc_xy - center_ref.reshape(1, 2)
    sc_inf = (dxy_sc @ inferior_dir.reshape(2, 1))[:, 0]
    i_l = int(np.argmax(sc_inf))
    win = sc_xy[max(0, i_l - 8):min(len(sc_xy), i_l + 8)]
    p_sp, d_sp = _fit_line_pca_xy(win)

    # intersection(spine line, costal margin line) to anchor where we pick "next spine point"
    inter_xy = _line_intersection_2d(p_sp, d_sp, p_cm, d_cm)
    if inter_xy is None:
        inter_xy = sc_xy[int(np.argmin(np.linalg.norm(sc_xy - p_cm.reshape(1, 2), axis=1)))]

    # pick nearest spine curve point to intersection, then move "next" along inferior direction
    dist_sp = np.linalg.norm(sc_xy - inter_xy.reshape(1, 2), axis=1)
    i0 = int(np.argmin(dist_sp))
    k = int(spine_next_k)
    k = max(0, min(10, k))
    # ensure ordering along inferior direction
    order = np.argsort(sc_inf)  # low->high (inferior increases per our convention)
    pos = int(np.where(order == i0)[0][0])
    pos2 = min(len(order) - 1, pos + k)
    i_start = int(order[pos2])
    start_spine_xyz = sc[i_start].astype(np.float32)
    start_xy = sc_xy[i_start]

    # rib end on OTHER side
    dxy_r = rib_xy - center_ref.reshape(1, 2)
    rib_lr = (dxy_r @ lr_dir.reshape(2, 1))[:, 0]
    rib_end_side_lc = str(rib_end_side or "left").strip().lower()
    if rib_end_side_lc in ("left", "l", "lhs"):
        rib_end_i = int(np.argmin(rib_lr))
    elif rib_end_side_lc in ("right", "r", "rhs"):
        rib_end_i = int(np.argmax(rib_lr))
    else:
        # auto: opposite of the scanning side
        want_right = str(side or "right").strip().lower() in ("right", "r", "rhs")
        rib_end_i = int(np.argmin(rib_lr) if want_right else np.argmax(rib_lr))
    rib_end_xyz = rib_xyz[rib_end_i].astype(np.float32)
    rib_end_xy = rib_xy[rib_end_i]

    # base angle line: start -> rib_end
    d = (rib_end_xy - start_xy).astype(np.float64)
    dn = float(np.linalg.norm(d))
    if dn < 1e-6:
        return None, start_spine_xyz, rib_end_xyz, None
    d = d / dn

    # offset normal: perpendicular, choose sign to point "inferior" (like rib_both's down normal)
    n = np.array([-d[1], d[0]], dtype=np.float64)
    n = n / (float(np.linalg.norm(n)) + 1e-9)
    if float(np.dot(n, inferior_dir.reshape(2))) < 0.0:
        n = -n
    off = float(offset_m)
    off = max(-0.20, min(0.20, off))
    p0 = start_xy + n * off
    p1 = rib_end_xy + n * off

    # snap: use visible surface
    visible = _compute_visibility_mask(v, intrinsics, grid_size=20, z_tolerance=0.05)
    cand = visible
    xy = v[:, :2].astype(np.float64)
    z = v[:, 2].astype(np.float64)

    # sample along offset line and snap each point
    L = float(np.linalg.norm(p1 - p0))
    ds = max(0.005, min(0.03, float(step_m)))
    n_steps = max(2, int(np.floor(L / ds)) + 1)
    out = []
    for i in range(n_steps):
        t = (float(i) / float(n_steps - 1))
        tgt = p0 * (1.0 - t) + p1 * t
        dx = xy[:, 0] - float(tgt[0])
        dy = xy[:, 1] - float(tgt[1])
        d2 = dx * dx + dy * dy
        score = np.where(cand, d2 + 0.02 * (z - np.nanmin(z)), np.inf)
        idx = int(np.argmin(score))
        if np.isfinite(score[idx]):
            out.append(v[idx])
    if len(out) < 2:
        return None, start_spine_xyz, rib_end_xyz, None

    base_line = np.stack(
        [np.array([start_xy[0], start_xy[1], float(np.median(v[:, 2]))], dtype=np.float32),
         np.array([rib_end_xy[0], rib_end_xy[1], float(np.median(v[:, 2]))], dtype=np.float32)],
        axis=0,
    )
    return np.asarray(out, dtype=np.float32).reshape(-1, 3), start_spine_xyz, rib_end_xyz, base_line

def find_spine_costal_margin_cross_point_on_spine_curve(
    fitter,
    skel_verts: np.ndarray,
    skel_faces: np.ndarray,
    img_path: str,
    intrinsics: dict,
    spine_curve_xyz: np.ndarray,
    rib_trim_head_ratio: float = 0.0,
    side: str = "right",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Step-1 (per user request):
    Find the "costal margin ↔ spine cross point", but spine is the BLUE spine curve (polyline),
    not a fitted straight spine line.

    Returns:
      (cross_xyz_on_spine_curve, rib_line_3d)
    """
    try:
        if spine_curve_xyz is None or len(spine_curve_xyz) < 5:
            return None, None
        rib_line_3d = extract_right_costal_margin(
            fitter, skel_verts, skel_faces, img_path, intrinsics, side=str(side or "right")
        )
        if rib_line_3d is None or len(rib_line_3d) < 5:
            return None, None

        v = np.asarray(skel_verts, dtype=np.float64).reshape(-1, 3)
        bone_names = list(getattr(fitter.skel, "bone_names", []))
        weights = fitter.skel.skel_weights.to_dense().detach().cpu().numpy()
        center_ref, lr_dir, inferior_dir = _infer_lr_and_inferior_dirs(weights, v, bone_names)

        rib_xyz = np.asarray(rib_line_3d, dtype=np.float64).reshape(-1, 3)
        # 与主流程一致：裁掉肋缘点序前 rib_trim_head_ratio(剑突/上段)，
        # 用“裁切后的下外侧肋缘”做 PCA 直线，再延长去交脊柱曲线 → 交点更贴下肋弓。
        rib_xyz_for_pca = _trim_head_points(rib_xyz, rib_trim_head_ratio) if rib_trim_head_ratio > 0 else rib_xyz
        if len(rib_xyz_for_pca) < 3:
            rib_xyz_for_pca = rib_xyz
        rib_xy = rib_xyz_for_pca[:, :2]
        p_cm, d_cm = _fit_line_pca_xy(rib_xy)

        spine = np.asarray(spine_curve_xyz, dtype=np.float64).reshape(-1, 3)
        spine_xy = spine[:, :2]

        # target inferior coordinate: use rib median as reference
        rib_inf = ((rib_xy - center_ref.reshape(1, 2)) @ inferior_dir.reshape(2, 1))[:, 0]
        rib_inf_med = float(np.median(rib_inf))

        # Try true intersection between (infinite) costal-margin line and spine-curve segments
        best = None  # (score, i, t, q_xy)
        for i in range(len(spine_xy) - 1):
            a = spine_xy[i]
            b = spine_xy[i + 1]
            ds = (b - a).astype(np.float64)
            den = float(np.dot(ds, ds))
            if den < 1e-12:
                continue
            q = _line_intersection_2d(a, ds, p_cm, d_cm)
            if q is None:
                continue
            t = float(np.dot((q - a), ds) / den)
            if t < 0.0 or t > 1.0:
                continue
            q_inf = float(((q - center_ref.reshape(2)) @ inferior_dir.reshape(2)))
            score = abs(q_inf - rib_inf_med)
            if (best is None) or (score < best[0]):
                best = (score, i, t, q)

        if best is not None:
            _, i, t, q = best
            # interpolate XYZ on spine segment using same t (XY space is consistent with XYZ)
            p = spine[i] * (1.0 - t) + spine[i + 1] * t
            return p.astype(np.float32), rib_xyz.astype(np.float32)

        # Fallback: nearest spine point to the costal-margin line in XY
        n_cm = np.array([-d_cm[1], d_cm[0]], dtype=np.float64)
        n_cm = n_cm / (float(np.linalg.norm(n_cm)) + 1e-9)
        dist = np.abs(((spine_xy - p_cm.reshape(1, 2)) @ n_cm.reshape(2, 1))[:, 0])
        j = int(np.argmin(dist))
        return spine[j].astype(np.float32), rib_xyz.astype(np.float32)
    except Exception:
        return None, None

def _advance_point_along_spine_curve_by_arclen(
    fitter,
    skel_verts: np.ndarray,
    spine_curve_xyz: np.ndarray,
    from_xyz: np.ndarray,
    advance_m: float,
    direction: str = "inferior",
) -> Optional[np.ndarray]:
    """
    Move along the spine curve by arc-length starting from the closest point to `from_xyz`.
    direction:
      - "inferior": towards pelvis (down in body)
      - "superior": towards thorax (up in body)
    """
    try:
        curve = np.asarray(spine_curve_xyz, dtype=np.float64).reshape(-1, 3)
        if curve.shape[0] < 2:
            return curve[0].astype(np.float32) if curve.shape[0] == 1 else None
        from_xyz = np.asarray(from_xyz, dtype=np.float64).reshape(3)
        v = np.asarray(skel_verts, dtype=np.float64).reshape(-1, 3)
        bone_names = list(getattr(fitter.skel, "bone_names", []))
        weights = fitter.skel.skel_weights.to_dense().detach().cpu().numpy()
        center_ref, lr_dir, inferior_dir = _infer_lr_and_inferior_dirs(weights, v, bone_names)

        # order curve along inferior coordinate
        xy = curve[:, :2] - center_ref.reshape(1, 2)
        inf = (xy @ inferior_dir.reshape(2, 1))[:, 0]
        order = np.argsort(inf)  # increasing inferior
        curve = curve[order]

        # cumulative arc-length in 3D
        ds = np.linalg.norm(curve[1:] - curve[:-1], axis=1)
        s = np.concatenate([[0.0], np.cumsum(ds)], axis=0)

        # closest vertex index to from_xyz
        d = np.linalg.norm(curve - from_xyz.reshape(1, 3), axis=1)
        i0 = int(np.argmin(d))
        s0 = float(s[i0])

        L = float(advance_m)
        L = max(0.0, min(0.50, L))
        dir_lc = str(direction or "inferior").strip().lower()
        if dir_lc in ("superior", "up", "thorax"):
            target = s0 - L
        else:
            target = s0 + L
        target = max(0.0, min(float(s[-1]), target))

        j = int(np.searchsorted(s, target, side="left"))
        j = max(1, min(len(s) - 1, j))
        s0j, s1j = float(s[j - 1]), float(s[j])
        t = 0.0 if (s1j - s0j) <= 1e-9 else (target - s0j) / (s1j - s0j)
        p = curve[j - 1] * (1.0 - t) + curve[j] * t
        return p.astype(np.float32)
    except Exception:
        return None

def _advance_point_along_polyline_by_arclen(
    xyz: np.ndarray,
    start_idx: int,
    delta_m: float,
    direction: str = "backward",
) -> Optional[np.ndarray]:
    """
    Move along a 3D polyline by arc-length.
    - direction="forward": towards increasing index
    - direction="backward": towards decreasing index
    Returns interpolated 3D point.
    """
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 2:
        return pts[0].astype(np.float32) if pts.shape[0] == 1 else None
    i0 = int(start_idx)
    i0 = max(0, min(len(pts) - 1, i0))
    ds = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(ds)], axis=0)
    L = float(delta_m)
    L = max(0.0, min(0.50, L))
    dir_lc = str(direction or "backward").strip().lower()
    if dir_lc in ("forward", "fwd", "inc", "+"):
        target = float(s[i0] + L)
    else:
        target = float(s[i0] - L)
    target = max(0.0, min(float(s[-1]), target))
    j = int(np.searchsorted(s, target, side="left"))
    j = max(1, min(len(s) - 1, j))
    s0j, s1j = float(s[j - 1]), float(s[j])
    t = 0.0 if (s1j - s0j) <= 1e-9 else (target - s0j) / (s1j - s0j)
    p = pts[j - 1] * (1.0 - t) + pts[j] * t
    return p.astype(np.float32)

def _normal_down_from_dir_uv(d: np.ndarray) -> np.ndarray:
    """
    Same idea as cliff_skel_trajectory._normal_down_from_dir:
    Given a 2D direction d in image UV, return a perpendicular normal whose sign points to image-down (v+).
    """
    d = np.asarray(d, dtype=np.float64).reshape(2)
    n = np.array([-d[1], d[0]], dtype=np.float64)
    nn = float(np.linalg.norm(n))
    if nn < 1e-9:
        n = np.array([0.0, 1.0], dtype=np.float64)
        nn = 1.0
    n = n / nn
    # choose sign so it points down (positive v)
    if float(n[1]) < 0.0:
        n = -n
    return n.astype(np.float64)

def _intersect_infinite_line_with_polyline_uv(
    line_p: np.ndarray,
    line_dir: np.ndarray,
    poly_uv: np.ndarray,
    prefer_forward: bool = True,
) -> Optional[Tuple[np.ndarray, float]]:
    """
    Intersect an infinite 2D line (line_p + t*line_dir) with a polyline (piecewise segments).
    Returns (intersection_uv, t_along_line). If multiple intersections, choose the closest forward t>=0
    (or closest |t| if no forward intersection).
    """
    p = np.asarray(line_p, dtype=np.float64).reshape(2)
    d = np.asarray(line_dir, dtype=np.float64).reshape(2)
    dn = float(np.linalg.norm(d))
    if dn < 1e-9:
        return None
    d = d / dn
    poly = np.asarray(poly_uv, dtype=np.float64).reshape(-1, 2)
    if poly.shape[0] < 2:
        return None

    best_fwd = None  # (t, pt)
    best_any = None  # (abs_t, t, pt)

    for i in range(poly.shape[0] - 1):
        a = poly[i]
        b = poly[i + 1]
        e = (b - a).astype(np.float64)
        # solve p + t d = a + s e
        A = np.stack([d, -e], axis=1)  # 2x2
        det = float(np.linalg.det(A))
        if abs(det) < 1e-9:
            continue
        rhs = (a - p).reshape(2)
        t_s = np.linalg.solve(A, rhs)
        t = float(t_s[0])
        s = float(t_s[1])
        if s < -1e-6 or s > 1.0 + 1e-6:
            continue
        pt = p + d * t
        if prefer_forward and t >= 0.0:
            if best_fwd is None or t < best_fwd[0]:
                best_fwd = (t, pt)
        at = abs(t)
        if best_any is None or at < best_any[0]:
            best_any = (at, t, pt)

    if best_fwd is not None:
        return best_fwd[1].astype(np.float32), float(best_fwd[0])
    if best_any is not None:
        return best_any[2].astype(np.float32), float(best_any[1])
    return None

def find_costal_margin_end_point_leftest_lowest(
    fitter,
    skel_verts: np.ndarray,
    rib_line_3d: np.ndarray,
    lr_band_percentile: float = 3.0,
) -> Optional[np.ndarray]:
    """
    Mirror of the previous "rightest + lowest" endpoint logic, but for:
      - leftest (most negative LR) within a lateral band
      - then pick the most inferior point inside that band

    Returns a 3D point in camera coordinates.
    """
    try:
        v = np.asarray(skel_verts, dtype=np.float64).reshape(-1, 3)
        bone_names = list(getattr(fitter.skel, "bone_names", []))
        weights = fitter.skel.skel_weights.to_dense().detach().cpu().numpy()
        center_ref, lr_dir, inferior_dir = _infer_lr_and_inferior_dirs(weights, v, bone_names)

        rib_xyz = np.asarray(rib_line_3d, dtype=np.float64).reshape(-1, 3)
        if rib_xyz.shape[0] < 3:
            return rib_xyz[-1].astype(np.float32) if rib_xyz.shape[0] > 0 else None

        xy = rib_xyz[:, :2] - center_ref.reshape(1, 2)
        coord_lr = (xy @ lr_dir.reshape(2, 1))[:, 0]
        coord_inf = (xy @ inferior_dir.reshape(2, 1))[:, 0]

        p = float(lr_band_percentile)
        p = max(0.0, min(50.0, p))
        thr = float(np.percentile(coord_lr, p))
        band = coord_lr <= thr
        if np.count_nonzero(band) == 0:
            band = np.ones((rib_xyz.shape[0],), dtype=bool)

        idx = int(np.argmax(np.where(band, coord_inf, -1e18)))
        return rib_xyz[idx].astype(np.float32)
    except Exception:
        return None


def find_costal_margin_end_point_rightest_lowest(
    fitter,
    skel_verts: np.ndarray,
    rib_line_3d: np.ndarray,
    lr_band_percentile: float = 3.0,
) -> Optional[np.ndarray]:
    """Right kidney: rightest (most positive LR) within lateral band, then most inferior."""
    try:
        v = np.asarray(skel_verts, dtype=np.float64).reshape(-1, 3)
        bone_names = list(getattr(fitter.skel, "bone_names", []))
        weights = fitter.skel.skel_weights.to_dense().detach().cpu().numpy()
        center_ref, lr_dir, inferior_dir = _infer_lr_and_inferior_dirs(weights, v, bone_names)

        rib_xyz = np.asarray(rib_line_3d, dtype=np.float64).reshape(-1, 3)
        if rib_xyz.shape[0] < 3:
            return rib_xyz[-1].astype(np.float32) if rib_xyz.shape[0] > 0 else None

        xy = rib_xyz[:, :2] - center_ref.reshape(1, 2)
        coord_lr = (xy @ lr_dir.reshape(2, 1))[:, 0]
        coord_inf = (xy @ inferior_dir.reshape(2, 1))[:, 0]

        p = float(lr_band_percentile)
        p = max(0.0, min(50.0, p))
        thr = float(np.percentile(coord_lr, 100.0 - p))
        band = coord_lr >= thr
        if np.count_nonzero(band) == 0:
            band = np.ones((rib_xyz.shape[0],), dtype=bool)

        idx = int(np.argmax(np.where(band, coord_inf, -1e18)))
        return rib_xyz[idx].astype(np.float32)
    except Exception:
        return None

def find_skeleton_end_point_left_low_deepest(
    fitter,
    skel_verts: np.ndarray,
    skel_faces: np.ndarray,
    lr_band_percentile: float = 15.0,
    z_depth_percentile: float = 85.0,
    prefer_bones: Optional[list] = None,
) -> Optional[np.ndarray]:
    """
    User request: end point should be on the *skeleton* and be:
      - leftest (most negative LR) within a left lateral band
      - depthest (furthest from camera => largest Z) within a depth band
      - lowest (most inferior) within that intersection

    NOTE: we do NOT apply visibility/front-normal filtering here, because we explicitly want the BACK side.
    """
    try:
        v = np.asarray(skel_verts, dtype=np.float64).reshape(-1, 3)
        if v.shape[0] == 0:
            return None
        bone_names = list(getattr(fitter.skel, "bone_names", []))
        weights = fitter.skel.skel_weights.to_dense().detach().cpu().numpy()
        center_ref, lr_dir, inferior_dir = _infer_lr_and_inferior_dirs(weights, v, bone_names)

        vertex_bone_ids = np.argmax(weights, axis=1)
        # abdomen region bones:
        # For endpoint selection we use a *tighter* abdomen proxy to avoid drifting into upper thorax.
        if prefer_bones is None:
            prefer_bones = ["pelvis", "lumbar_body"]
        bone_ids = [bone_names.index(nm) for nm in prefer_bones if nm in bone_names]
        mask_bones = np.isin(vertex_bone_ids, bone_ids) if bone_ids else np.ones((v.shape[0],), dtype=bool)
        # additional abdomen constraint: also require vertex to belong to the abdomen submesh mask (faces all-in-mask)
        # this removes shoulders/upper thorax stray points when bone assignment is noisy.
        try:
            f = np.asarray(skel_faces, dtype=np.int64).reshape(-1, 3)
            valid_faces = mask_bones[f].all(axis=1)
            if np.any(valid_faces):
                abd_vert_ids = np.unique(f[valid_faces].reshape(-1))
                mask_abd = np.zeros((v.shape[0],), dtype=bool)
                mask_abd[abd_vert_ids] = True
            else:
                mask_abd = mask_bones
        except Exception:
            mask_abd = mask_bones

        xy = v[:, :2].astype(np.float64) - center_ref.reshape(1, 2)
        coord_lr = (xy @ lr_dir.reshape(2, 1))[:, 0]
        coord_inf = (xy @ inferior_dir.reshape(2, 1))[:, 0]
        z = v[:, 2].astype(np.float64)

        m = mask_abd & np.isfinite(coord_lr) & np.isfinite(coord_inf) & np.isfinite(z) & (z > 0.1)
        if np.count_nonzero(m) < 100:
            m = np.isfinite(coord_lr) & np.isfinite(coord_inf) & np.isfinite(z) & (z > 0.1)
        if np.count_nonzero(m) < 10:
            return None

        # left lateral band
        p_lr = float(lr_band_percentile)
        p_lr = max(0.0, min(50.0, p_lr))
        thr_lr = float(np.percentile(coord_lr[m], p_lr))
        left_band = m & (coord_lr <= thr_lr)
        if np.count_nonzero(left_band) < 10:
            left_band = m

        # depth band (back side): largest Z
        pz = float(z_depth_percentile)
        pz = max(50.0, min(99.5, pz))
        thr_z = float(np.percentile(z[left_band], pz))
        deep_band = left_band & (z >= thr_z)
        if np.count_nonzero(deep_band) < 10:
            deep_band = left_band

        idx = int(np.argmax(np.where(deep_band, coord_inf, -1e18)))
        return v[idx].astype(np.float32)
    except Exception:
        return None


def extract_kidney_side_trajectory(
    fitter,
    vertices: np.ndarray,
    faces: np.ndarray,
    intrinsics: dict,
    side: str = "right",
    inf_percentile_lo: float = 20.0,
    inf_percentile_hi: float = 85.0,
    lr_band_percentile: float = 97.0,
    num_bins: int = 40,
) -> Optional[np.ndarray]:
    """
    “肾脏扫描”用的简化轨迹（相机坐标系 3D 点）：
    - 使用 abdomen/torso 区域（pelvis/lumbar_body/thorax）
    - 仅取可见前表面点
    - 在人体“右侧/左侧”上，沿 inferior_dir（从胸到骨盆方向）采样
    - 每个 inf bin 选“最靠右(或最靠左)”且尽量前表面的点
    输出：Nx3（camera frame）。
    说明：起点将改为 ribline 的 end_pt 同款逻辑（lateral band 内最 inferior 的点），更稳定。
    """
    try:
        import trimesh

        v = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        if v.shape[0] == 0:
            return None

        mesh = trimesh.Trimesh(v, faces, process=False)
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32).reshape(-1, 3)

        bone_names = list(getattr(fitter.skel, "bone_names", []))
        weights = fitter.skel.skel_weights.to_dense().detach().cpu().numpy()  # (V,J)
        vertex_bone_ids = np.argmax(weights, axis=1)

        abd_bones = []
        for nm in ["pelvis", "lumbar_body", "thorax"]:
            if nm in bone_names:
                abd_bones.append(bone_names.index(nm))
        if not abd_bones:
            # fallback: use all vertices
            mask_abd = np.ones((v.shape[0],), dtype=bool)
        else:
            mask_abd = np.isin(vertex_bone_ids, abd_bones)

        visible = _compute_visibility_mask(v, intrinsics, grid_size=20, z_tolerance=0.05)
        front_normal = normals[:, 2] < -0.1

        center_ref, lr_dir, inferior_dir = _infer_lr_and_inferior_dirs(weights, v, bone_names)

        # NOTE (per user requirement): "down" means anatomical inferior (thorax -> pelvis),
        # not image v+. So we do NOT flip inferior_dir based on image projection.
        xy = v[:, :2].astype(np.float64) - center_ref.reshape(1, 2)
        coord_lr = (xy @ lr_dir.reshape(2, 1))[:, 0]
        coord_inf = (xy @ inferior_dir.reshape(2, 1))[:, 0]

        side_lc = str(side or "right").strip().lower()
        want_right = side_lc in ("right", "r", "rhs")
        side_mask = (coord_lr > 0.0) if want_right else (coord_lr < 0.0)

        cand = mask_abd & visible & front_normal & side_mask
        if np.count_nonzero(cand) < 200:
            # relax normal constraint
            cand = mask_abd & visible & side_mask
        if np.count_nonzero(cand) < 50:
            # fallback: ignore side if we really have too few points
            cand = mask_abd & visible

        inf_vals = coord_inf[cand]
        if inf_vals.size < 10:
            return None
        lo = float(np.percentile(inf_vals, float(inf_percentile_lo)))
        hi = float(np.percentile(inf_vals, float(inf_percentile_hi)))
        if hi <= lo + 1e-6:
            lo, hi = float(np.min(inf_vals)), float(np.max(inf_vals))

        # start point (same idea as ribline end_pt):
        # pick lateral band, then choose the most-inferior point in that band.
        in_range = cand & (coord_inf >= lo) & (coord_inf <= hi)
        if np.count_nonzero(in_range) == 0:
            in_range = cand

        lr_vals_in = coord_lr[in_range]
        if lr_vals_in.size < 10:
            lr_vals_in = coord_lr[cand]
            in_range = cand

        if want_right:
            lat_thr = float(np.percentile(lr_vals_in, 85.0))
            lat_band = in_range & (coord_lr >= lat_thr)
        else:
            lat_thr = float(np.percentile(lr_vals_in, 15.0))
            lat_band = in_range & (coord_lr <= lat_thr)
        if np.count_nonzero(lat_band) == 0:
            lat_band = in_range

        start_idx = int(np.argmax(np.where(lat_band, coord_inf, -1e18)))  # most inferior
        start_inf = float(coord_inf[start_idx])

        # Build a trajectory PARALLEL to the spine line:
        # - spine direction in camera XY plane is inferior_dir (thorax -> pelvis)
        # - define the target line as the line passing through the start point with direction inferior_dir
        # - for each axial bin along inferior_dir, pick the point closest to this line (and front-most)
        start_xy = v[start_idx, :2].astype(np.float64)
        spine_dir = inferior_dir.astype(np.float64)
        spine_dir = spine_dir / (float(np.linalg.norm(spine_dir)) + 1e-9)
        # normal to spine line in XY
        n = np.array([-spine_dir[1], spine_dir[0]], dtype=np.float64)
        n = n / (float(np.linalg.norm(n)) + 1e-9)
        # signed distance to target line: n·(xy - start_xy)
        dist_line = np.abs((xy - start_xy.reshape(1, 2)) @ n.reshape(2, 1))[:, 0]

        # Extend inferior along spine_dir: s coordinate is coord_inf (already along inferior_dir)
        max_inf = float(np.max(coord_inf[cand]))
        bin_lo = float(start_inf)
        bin_hi = max(float(hi), max_inf)
        if bin_hi <= bin_lo + 1e-6:
            p0 = v[start_idx].reshape(1, 3).astype(np.float32)
            return p0

        nb = max(5, int(num_bins))
        # ascending edges: start -> more inferior
        edges = np.linspace(bin_lo, bin_hi, nb + 1, dtype=np.float64)

        pts = []
        for i in range(nb):
            a = edges[i]
            b = edges[i + 1]
            m = cand & (coord_inf >= a) & (coord_inf < b)
            if np.count_nonzero(m) == 0:
                continue
            # restrict to a lateral band near the target line to avoid hugging pelvis boundary
            # we pick points with smallest distance to the line, then choose the front-most (smallest z)
            d = dist_line
            # take the closest X% in this bin
            dvals = d[m]
            thr_d = float(np.percentile(dvals, 20.0))  # closest 20%
            mm = m & (d <= thr_d)
            if np.count_nonzero(mm) == 0:
                mm = m
            z = v[:, 2]  # prefer front-most
            idx = int(np.argmin(np.where(mm, z, np.inf)))
            if np.isfinite(z[idx]):
                pts.append(v[idx])

        if not pts:
            return None

        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        inf_p = (pts[:, :2].astype(np.float64) - center_ref.reshape(1, 2)) @ inferior_dir.reshape(2, 1)
        inf_p = inf_p[:, 0]
        # keep only points that are >= start_inf (extend inferior only)
        keep_mask = inf_p >= (float(start_inf) - 1e-6)
        if np.count_nonzero(keep_mask) > 0:
            pts = pts[keep_mask]
            inf_p = inf_p[keep_mask]
        # sort by inf increasing: start -> more inferior
        order = np.argsort(inf_p)
        pts = pts[order]

        # ensure start point is the first element (and only keep points >= start_inf, so monotonic)
        p0 = v[start_idx].reshape(1, 3).astype(np.float32)
        pts = np.concatenate([p0, pts], axis=0)
        try:
            inf_all = (pts[:, :2].astype(np.float64) - center_ref.reshape(1, 2)) @ inferior_dir.reshape(2, 1)
            inf_all = inf_all[:, 0]
            print(f"[kidney_traj] start_inf={float(inf_all[0]):.4f}, end_inf={float(inf_all[-1]):.4f}, N={len(pts)}")
        except Exception:
            pass
        # de-duplicate close points
        keep = [0]
        for k in range(1, len(pts)):
            if float(np.linalg.norm(pts[k] - pts[keep[-1]])) > 1e-4:
                keep.append(k)
        return pts[keep]
    except Exception as e:
        print(f"Kidney trajectory extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_right_costal_margin(fitter, vertices, faces, img_path, intrinsics, side: str = "right"):
    """
    Extracts the right costal margin (rib line) using geometric heuristics + binning + NORMAL + VISIBILITY filtering.
    Returns: (N, 3) array of 3D points forming the line.
    """
    try:
        # --- side="left": 背面视角下直接提取左侧肋缘会被"正面/可见性"筛选搞退化，
        #     改为复用稳定的右侧肋缘，再沿矢状面(coord_lr=0)镜像到左侧。
        #     镜像法向取 LR 轴且 z=0：只翻左右、不动深度(背面视角左右肋深度相近)，
        #     反射保持点序(中线→外侧)，因此返回结果 [0]近脊柱、[-1]外侧端，与右侧一致。---
        if str(side or "right").strip().lower() in ("left", "l", "lhs"):
            R = extract_right_costal_margin(fitter, vertices, faces, img_path, intrinsics, side="right")
            if R is not None and len(np.asarray(R).reshape(-1, 3)) >= 2:
                try:
                    v_all = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
                    _bn = list(fitter.skel.bone_names)
                    _w = fitter.skel.skel_weights.to_dense().cpu().numpy()
                    c_xy, lr_xy, _inf = _infer_lr_and_inferior_dirs(_w, v_all, _bn)
                    n3 = np.array([float(lr_xy[0]), float(lr_xy[1]), 0.0], dtype=np.float64)
                    nn = float(np.linalg.norm(n3))
                    if nn > 1e-9:
                        n3 = n3 / nn
                        C3 = np.array([float(c_xy[0]), float(c_xy[1]), 0.0], dtype=np.float64)
                        Rp = np.asarray(R, dtype=np.float64).reshape(-1, 3)
                        s = (Rp - C3.reshape(1, 3)) @ n3.reshape(3, 1)  # (N,1) 沿 LR 的有符号距离
                        L = Rp - 2.0 * s * n3.reshape(1, 3)             # 关于矢状面镜像 → 翻到左侧
                        return L.astype(np.float32)
                except Exception as _e:
                    print(f"[costal_margin] left-mirror failed, fallback to direct-left logic: {_e}")
            # 右侧提取失败 → 落回下面的原始 direct-left 逻辑

        # 0. Compute Vertex Normals for the whole mesh (Robust Front/Back check)
        import trimesh
        full_mesh = trimesh.Trimesh(vertices, faces, process=False)
        vertex_normals = full_mesh.vertex_normals # (V, 3)
        
        # --- NEW: Visibility Filtering using Z-Buffer ---
        # Project all vertices to 2D
        # intrinsics keys: fx, fy, cx, cy
        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']
        
        # Simple projection: u = fx * x / z + cx, v = fy * y / z + cy
        valid_proj_mask = vertices[:, 2] > 0.1
        z_safe = vertices[:, 2].copy()
        z_safe[z_safe < 0.1] = 0.1
        
        u = (vertices[:, 0] * fx / z_safe) + cx
        v = (vertices[:, 1] * fy / z_safe) + cy
        
        # Create a coarse Z-buffer grid (e.g. 20x20 pixels)
        grid_size = 20
        grid_w = int(intrinsics['width'] / grid_size) + 1
        grid_h = int(intrinsics['height'] / grid_size) + 1
        z_buffer = np.full((grid_h, grid_w), np.inf)
        
        ui = np.clip((u / grid_size).astype(int), 0, grid_w - 1)
        vi = np.clip((v / grid_size).astype(int), 0, grid_h - 1)
        
        sort_idx = np.argsort(vertices[:, 2])
        for i in sort_idx:
            if not valid_proj_mask[i]: continue
            r, c = vi[i], ui[i]
            if vertices[i, 2] < z_buffer[r, c]:
                z_buffer[r, c] = vertices[i, 2]
                
        z_tolerance = 0.05
        min_z_vals = z_buffer[vi, ui]
        is_visible = (vertices[:, 2] <= min_z_vals + z_tolerance) & valid_proj_mask
        
        # 1. Identify Thorax + Lumbar vertices (for rib margin), and Pelvis/arms (for body axes)
        bone_names = fitter.skel.bone_names
        weights = fitter.skel.skel_weights.to_dense().cpu().numpy()  # (V, J)

        # Rib margin search bones
        rib_bones = []
        for name in ["thorax", "lumbar_body"]:
            if name in bone_names:
                rib_bones.append(bone_names.index(name))
        if not rib_bones:
            print("Warning: No relevant thorax/lumbar bones found.")
            return None

        vertex_bone_ids = np.argmax(weights, axis=1)
        mask_rib = np.isin(vertex_bone_ids, rib_bones)
        if not np.any(mask_rib):
            print("No rib-related vertices found.")
            return None

        # 2. Build body axes in the IMAGE plane (XY of camera coordinates)
        def _bone_center_2d(bone_name: str, thr: float = 0.05) -> Optional[np.ndarray]:
            if bone_name not in bone_names:
                return None
            j = bone_names.index(bone_name)
            m = weights[:, j] > thr
            if not np.any(m):
                return None
            return vertices[m][:, :2].mean(axis=0).astype(np.float64)

        thorax_c = _bone_center_2d("thorax")
        pelvis_c = _bone_center_2d("pelvis")
        if thorax_c is None:
            thorax_c = vertices[mask_rib][:, :2].mean(axis=0).astype(np.float64)
        if pelvis_c is None:
            pelvis_c = vertices[:, :2].mean(axis=0).astype(np.float64)

        # Inferior direction (thorax -> pelvis) in XY
        inferior_dir = (pelvis_c - thorax_c).astype(np.float64)
        n_inf = np.linalg.norm(inferior_dir)
        if n_inf < 1e-6:
            inferior_dir = np.array([0.0, 1.0], dtype=np.float64)
        else:
            inferior_dir = inferior_dir / n_inf

        # Left-right direction: try to use arm bones if available; otherwise fallback to image X.
        lr_dir = None
        # common naming guesses (SKEL may vary)
        right_candidates = ["r_humerus", "right_humerus", "humerus_r", "r_upperarm", "right_upperarm"]
        left_candidates = ["l_humerus", "left_humerus", "humerus_l", "l_upperarm", "left_upperarm"]
        r_c = None
        l_c = None
        for nm in right_candidates:
            r_c = _bone_center_2d(nm)
            if r_c is not None:
                break
        for nm in left_candidates:
            l_c = _bone_center_2d(nm)
            if l_c is not None:
                break
        if r_c is not None and l_c is not None:
            v_lr = (r_c - l_c).astype(np.float64)
            n_lr = np.linalg.norm(v_lr)
            if n_lr > 1e-6:
                lr_dir = v_lr / n_lr

        if lr_dir is None:
            # fallback: use image-x as left-right
            lr_dir = np.array([1.0, 0.0], dtype=np.float64)

        # Orthonormalize: remove lr component from inferior (so axes are stable)
        inferior_dir = inferior_dir - float(np.dot(inferior_dir, lr_dir)) * lr_dir
        n_inf2 = np.linalg.norm(inferior_dir)
        if n_inf2 < 1e-6:
            inferior_dir = np.array([0.0, 1.0], dtype=np.float64)
        else:
            inferior_dir = inferior_dir / n_inf2

        # Reference center (mid-sternum approx)
        center_ref = thorax_c.astype(np.float64)

        # 3. Filtering: front-facing & visible
        # Keep previous visibility / normals logic, but define "right side" in body LR coordinates.
        xy = vertices[:, :2].astype(np.float64) - center_ref.reshape(1, 2)
        coord_lr = (xy @ lr_dir.reshape(2, 1))[:, 0]
        coord_inf = (xy @ inferior_dir.reshape(2, 1))[:, 0]

        # choose side in body LR coordinates
        side_lc = str(side or "right").strip().lower()
        want_right = side_lc in ("right", "r", "rhs")
        # right side: positive lr half (body-right); left side: negative lr half
        is_right = coord_lr > 0.0 if want_right else coord_lr < 0.0

        relevant_z = vertices[mask_rib & is_right, 2]
        if len(relevant_z) == 0:
            # if arm naming failed and lr_dir fallback wrong, try the opposite side
            is_right = coord_lr < 0.0 if want_right else coord_lr > 0.0
            relevant_z = vertices[mask_rib & is_right, 2]
            if len(relevant_z) == 0:
                return None
        z_min = relevant_z.min()
        z_max = relevant_z.max()
        z_threshold = z_min + (z_max - z_min) * 0.5
        is_front_depth = vertices[:, 2] < z_threshold
        is_front_normal = vertex_normals[:, 2] < -0.1

        final_mask = mask_rib & is_right & is_front_depth & is_front_normal & is_visible
        front_right_verts = vertices[final_mask]
        fr_lr = coord_lr[final_mask]
        fr_inf = coord_inf[final_mask]

        if len(front_right_verts) < 50:
            print("Filtering too strict, falling back to Visibility+Bone only.")
            final_mask = mask_rib & is_right & is_visible
            front_right_verts = vertices[final_mask]
            fr_lr = coord_lr[final_mask]
            fr_inf = coord_inf[final_mask]

        if len(front_right_verts) == 0:
            return None

        # 4. Xiphoid (start): midline region = smallest |lr|, then most inferior (max coord_inf)
        abs_lr = np.abs(fr_lr)
        mid_thr = float(np.percentile(abs_lr, 15.0))
        mid_mask = abs_lr <= mid_thr
        midline_verts = front_right_verts[mid_mask]
        mid_inf = fr_inf[mid_mask]
        if len(midline_verts) < 10:
            midline_verts = front_right_verts
            mid_inf = fr_inf
        xiphoid_pt = midline_verts[int(np.argmax(mid_inf))]

        # 5. End point: lateral extreme on chosen side + most inferior in that lateral band
        # right: high percentile; left: low percentile
        if want_right:
            lat_thr = float(np.percentile(fr_lr, 85.0))
            lat_mask = fr_lr >= lat_thr
        else:
            lat_thr = float(np.percentile(fr_lr, 15.0))
            lat_mask = fr_lr <= lat_thr
        lateral_band = front_right_verts[lat_mask]
        lateral_inf = fr_inf[lat_mask]
        if len(lateral_band) < 10:
            lateral_band = front_right_verts
            lateral_inf = fr_inf
        end_pt = lateral_band[int(np.argmax(lateral_inf))]

        # 6. Scan outward: bin along lr (from midline towards lateral), pick most inferior per bin
        xip_xy = (xiphoid_pt[:2].astype(np.float64) - center_ref)
        end_xy = (end_pt[:2].astype(np.float64) - center_ref)
        lr_start = float(xip_xy @ lr_dir)
        lr_end = float(end_xy @ lr_dir)
        if want_right:
            if lr_end <= lr_start:
                lr_end = float(np.max(fr_lr))
        else:
            if lr_end >= lr_start:
                lr_end = float(np.min(fr_lr))

        num_bins = 40
        lr_step = (lr_end - lr_start) / float(num_bins)
        lr_step = max(lr_step, 1e-6)

        costal_margin_points = [xiphoid_pt]
        current_lr0 = lr_start
        for _ in range(num_bins):
            current_lr1 = current_lr0 + lr_step
            lo = min(current_lr0, current_lr1)
            hi = max(current_lr0, current_lr1)
            in_bin = (fr_lr >= lo) & (fr_lr < hi)
            bin_points = front_right_verts[in_bin]
            if len(bin_points) > 0:
                bp_xy = bin_points[:, :2].astype(np.float64) - center_ref.reshape(1, 2)
                bp_inf = (bp_xy @ inferior_dir.reshape(2, 1))[:, 0]
                target_inf = float(np.percentile(bp_inf, 98))
                idx = int(np.abs(bp_inf - target_inf).argmin())
                costal_margin_points.append(bin_points[idx])
            current_lr0 = current_lr1

        # Force add the determined end point (helps stabilize endpoint)
        if len(costal_margin_points) == 0 or np.linalg.norm(costal_margin_points[-1] - end_pt) > 1e-6:
            costal_margin_points.append(end_pt)

        # Debug overlay: show start/end and body axes in image (helps verify standing vs lying)
        try:
            dbg = cv2.imread(img_path)
            if dbg is not None and intrinsics is not None:
                fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
                def _proj(pt3):
                    X, Y, Z = float(pt3[0]), float(pt3[1]), float(pt3[2])
                    Z = max(Z, 1e-6)
                    uu = int(round(fx * X / Z + cx))
                    vv = int(round(fy * Y / Z + cy))
                    return uu, vv
                p_start = _proj(xiphoid_pt)
                p_end = _proj(end_pt)
                cv2.circle(dbg, p_start, 6, (0, 0, 255), -1)  # red start
                cv2.circle(dbg, p_end, 6, (0, 255, 0), -1)    # green end
                # draw axes at thorax center (use Z from a near vertex)
                # pick a representative Z: use mean Z of rib mask
                z_ref = float(np.median(vertices[mask_rib, 2]))
                origin3 = np.array([center_ref[0], center_ref[1], z_ref], dtype=np.float64)
                # scale in world xy (meters-ish)
                s = 0.15
                lr3 = np.array([lr_dir[0], lr_dir[1], 0.0], dtype=np.float64) * s
                inf3 = np.array([inferior_dir[0], inferior_dir[1], 0.0], dtype=np.float64) * s
                o2 = _proj(origin3)
                lr2 = _proj(origin3 + lr3)
                inf2 = _proj(origin3 + inf3)
                cv2.arrowedLine(dbg, o2, lr2, (255, 255, 0), 2, tipLength=0.2)   # cyan: right
                cv2.arrowedLine(dbg, o2, inf2, (0, 255, 255), 2, tipLength=0.2)  # yellow: inferior
                out_dbg = img_path.replace(".jpg", "_ribline_axes_dbg.png").replace(".png", "_ribline_axes_dbg.png")
                cv2.imwrite(out_dbg, dbg)
        except Exception:
            pass
            
        return np.array(costal_margin_points)

    except Exception as e:
        print(f"Costal margin extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def project_and_draw_lines(points_3d, intrinsics, img_bgr, out_path, color=(255, 0, 0), thickness=3, trim_tail_ratio: float = 0.0):
    """
    Project 3D points and connect them with a line.
    """
    if points_3d is None or len(points_3d) < 2 or intrinsics is None:
        return

    # Optionally drop the tail part of the trajectory (helps when endpoint is unstable)
    if trim_tail_ratio is not None and trim_tail_ratio > 0:
        n = len(points_3d)
        keep_n = int(round(n * (1.0 - float(trim_tail_ratio))))
        keep_n = max(2, min(n, keep_n))
        points_3d = points_3d[:keep_n]

    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']
    
    vis_img = img_bgr.copy()
    
    points_2d = []
    for v in points_3d:
        X, Y, Z = v
        if Z <= 0.1: continue
        u = int(fx * X / Z + cx)
        v = int(fy * Y / Z + cy)
        points_2d.append((u, v))
        
    # Draw lines connecting points
    for i in range(len(points_2d) - 1):
        pt1 = points_2d[i]
        pt2 = points_2d[i+1]
        cv2.line(vis_img, pt1, pt2, color, thickness)
        # Draw keypoints
        cv2.circle(vis_img, pt1, 4, (0, 255, 255), -1)
        
    cv2.imwrite(out_path, vis_img)
    print(f"Rib line visualization saved to: {out_path}")


def _out_path(img_path: str, suffix: str, ext: str = ".png") -> str:
    """
    Build output path consistently with desired extension.
    Example: _out_path(".../ak_burst_03.jpg", "_demo_result") -> ".../ak_burst_03_demo_result.png"
    """
    base, _ = os.path.splitext(img_path)
    return f"{base}{suffix}{ext}"

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}")

    # Resolve ckpt path relative to this file (avoid cwd-dependent FileNotFoundError)
    ckpt_path = args.ckpt
    if ckpt_path and (not os.path.isabs(ckpt_path)) and (not os.path.exists(ckpt_path)):
        cand = os.path.join(os.path.dirname(os.path.abspath(__file__)), ckpt_path)
        if os.path.exists(cand):
            ckpt_path = cand
    print(f"CLIFF ckpt: {ckpt_path}")

    # 1. Load Image
    if not os.path.exists(args.input_path):
        print(f"Error: {args.input_path} not found.")
        return
    
    img_path = args.input_path
    orig_img_bgr = cv2.imread(img_path)
    if orig_img_bgr is None:
        print("Error reading image.")
        return
        
    # 2. Load Intrinsics
    intrinsics = load_intrinsics(img_path)
    # Helpful debug: show which intrinsics file we expect
    try:
        dir_name = os.path.dirname(img_path)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        expected_intr = os.path.join(dir_name, f"{base_name}_intrinsics.json")
        print(f"Expected intrinsics json: {expected_intr} (exists={os.path.exists(expected_intr)})")
    except Exception:
        pass
    if intrinsics:
        print(f"Loaded Real Intrinsics: {intrinsics}")
        fx = intrinsics.get('fx', estimate_focal_length(orig_img_bgr.shape[0], orig_img_bgr.shape[1]))
        fy = intrinsics.get('fy', fx)
        # Use average for focal length parameter
        focal_length_val = (fx + fy) / 2.0
    else:
        print("Using Estimated Focal Length")
        focal_length_val = estimate_focal_length(orig_img_bgr.shape[0], orig_img_bgr.shape[1])
    
    print(f"Focal Length Used: {focal_length_val}")

    # 3. Detection (YOLOv3)
    print("--- Detection ---")
    human_detector = HumanDetector()
    # Mock dataset for single image
    detection_dataset = DetectionDataset([orig_img_bgr], human_detector.in_dim)
    detection_data_loader = DataLoader(detection_dataset, batch_size=1, num_workers=0)
    
    batch = next(iter(detection_data_loader))
    norm_img = batch["norm_img"].to(device).float()
    dim = batch["dim"].to(device).float()
    
    detection_result = human_detector.detect_batch(norm_img, dim)
    detection_result = detection_result.cpu().numpy()
    
    if len(detection_result) == 0:
        print("No person detected.")
        return
    
    # Pick best person (largest area) if multiple? 
    # Demo logic just processes all. We will take the first one or largest.
    # detection_result: [batch_id, min_x, min_y, max_x, max_y, conf, cls_conf, cls_pred]
    # Filter for batch 0
    mask = detection_result[:, 0] == 0
    dets = detection_result[mask]
    
    if len(dets) == 0:
        print("No detections for image.")
        return
        
    # Optional: Pick largest
    # ... demo.py processes all, let's process all but focus on 0 for skel?
    # MocapDataset handles multiple detections.
    
    print(f"Detected {len(dets)} person(s).")

    # 4. CLIFF Inference
    print("--- CLIFF Inference ---")
    cliff = eval("cliff_" + args.backbone)
    cliff_model = cliff(constants.SMPL_MEAN_PARAMS).to(device)
    state_dict = torch.load(ckpt_path)['model']
    state_dict = strip_prefix_if_present(state_dict, prefix="module.")
    cliff_model.load_state_dict(state_dict, strict=True)
    cliff_model.eval()
    
    smpl_model = smplx.create(constants.SMPL_MODEL_DIR, "smpl").to(device)
    
    # Prepare Mocap Dataset
    mocap_db = MocapDataset([orig_img_bgr], detection_result)
    mocap_data_loader = DataLoader(mocap_db, batch_size=1, num_workers=0)
    
    pred_vert_arr = []
    
    for batch in tqdm(mocap_data_loader):
        norm_img = batch["norm_img"].to(device).float()
        center = batch["center"].to(device).float()
        scale = batch["scale"].to(device).float()
        img_h = batch["img_h"].to(device).float()
        img_w = batch["img_w"].to(device).float()
        
        # KEY CHANGE: OVERWRITE FOCAL LENGTH FROM INTRINSICS
        # MocapDataset estimates it. We override it.
        focal_length = torch.tensor([focal_length_val], device=device).float()
        
        cx, cy, b = center[:, 0], center[:, 1], scale * 200
        bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
        bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8
        bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)

        with torch.no_grad():
            pred_rotmat, pred_betas, pred_cam_crop = cliff_model(norm_img, bbox_info)
            
        # Convert to full (Global Translation)
        full_img_shape = torch.stack((img_h, img_w), dim=-1)
        pred_cam_full = cam_crop2full(pred_cam_crop, center, scale, full_img_shape, focal_length)
        
        # SMPL Forward
        pred_output = smpl_model(betas=pred_betas,
                                 body_pose=pred_rotmat[:, 1:],
                                 global_orient=pred_rotmat[:, [0]],
                                 pose2rot=False,
                                 transl=pred_cam_full)
        
        pred_vertices = pred_output.vertices
        pred_vert_arr.extend(pred_vertices.cpu().numpy())
        
        # --- 5. Run SKEL on FIRST detection only (for now) ---
        # Assuming single person of interest
        print("--- Running SKEL on current detection ---")
        skel_verts, skel_faces, abd_verts, abd_faces, fitter = run_skel_optimization(device, pred_rotmat[[0]], pred_betas[[0]], pred_cam_full[[0]], img_path)
        
        # 兜底：如果没有真实相机内参，也允许继续生成 overlay/ribline（使用估算 intrinsics）
        intrinsics_use = intrinsics
        if intrinsics_use is None:
            try:
                h0, w0 = orig_img_bgr.shape[:2]
                intrinsics_use = {
                    "fx": float(focal_length_val),
                    "fy": float(focal_length_val),
                    "cx": float(w0) / 2.0,
                    "cy": float(h0) / 2.0,
                    "width": int(w0),
                    "height": int(h0),
                }
                print("[warn] intrinsics missing; use estimated intrinsics for visualization/export.")
            except Exception:
                intrinsics_use = None
        
        if skel_verts is not None and intrinsics_use:
             # 1. Project Points
             print("--- Projecting SKEL to Image ---")
             out_proj_path = _out_path(img_path, "_demo_projected", ".png")
             project_and_draw(skel_verts, intrinsics_use, orig_img_bgr, out_proj_path)
             
             # 2. Render Full Skeleton Mesh Overlay
             print("--- Rendering Full Skeleton Mesh ---")
             out_skel_path = _out_path(img_path, "_demo_skel_overlay", ".png")
             skel_overlay_bgr = None
             try:
                 renderer_skel = Renderer(
                     focal_length=focal_length_val,
                     img_w=orig_img_bgr.shape[1],
                     img_h=orig_img_bgr.shape[0],
                     faces=skel_faces,
                     same_mesh_color=True,
                 )
                 # render_front_view expects list of verts [verts_person1, verts_person2...]
                 skel_overlay = renderer_skel.render_front_view(
                     [skel_verts],
                     bg_img_rgb=orig_img_bgr[:, :, ::-1].copy(),
                 )
                 skel_overlay_bgr = skel_overlay[:, :, ::-1]
                 cv2.imwrite(out_skel_path, skel_overlay_bgr)
                 print(f"Skeleton Overlay saved to: {out_skel_path}")
                 renderer_skel.delete()
             except Exception as e:
                 print(f"[warn] Skeleton overlay render failed (skip): {e}")
                 skel_overlay_bgr = None

             # 3. Render Abdomen Mesh Overlay
             if abd_verts is not None and abd_faces is not None:
                 print("--- Rendering Abdomen Mesh ---")
                 renderer_abd = Renderer(focal_length=focal_length_val, img_w=orig_img_bgr.shape[1], img_h=orig_img_bgr.shape[0],
                                          faces=abd_faces, same_mesh_color=True)
                 abd_overlay = renderer_abd.render_front_view([abd_verts], bg_img_rgb=orig_img_bgr[:, :, ::-1].copy())
                 out_abd_path = _out_path(img_path, "_demo_abdomen_overlay", ".png")
                 cv2.imwrite(out_abd_path, abd_overlay[:, :, ::-1])
                 print(f"Abdomen Overlay saved to: {out_abd_path}")
                 renderer_abd.delete()

             # 4. Extract and Render Thorax (ID 12) AND Lumbar (ID 11)
             print("--- Rendering Thorax (ID 12) & Lumbar (ID 11) ---")
             
             # Extract Thorax (relaxed threshold)
             tx_verts, tx_faces = extract_specific_bone_mesh(fitter, skel_verts, skel_faces, 'thorax', img_path, threshold=0.1)
             
             # Extract Lumbar
             lb_verts, lb_faces = extract_specific_bone_mesh(fitter, skel_verts, skel_faces, 'lumbar_body', img_path, threshold=0.1)

             if tx_verts is not None:
                 renderer_tx = Renderer(focal_length=focal_length_val, img_w=orig_img_bgr.shape[1], img_h=orig_img_bgr.shape[0],
                                          faces=tx_faces, same_mesh_color=True)
                 tx_overlay = renderer_tx.render_front_view([tx_verts], bg_img_rgb=orig_img_bgr[:, :, ::-1].copy())
                 out_tx_path = _out_path(img_path, "_demo_thorax_only", ".png")
                 cv2.imwrite(out_tx_path, tx_overlay[:, :, ::-1])
                 print(f"Thorax Only Overlay (Improved) saved to: {out_tx_path}")
                 renderer_tx.delete()
                 
             if lb_verts is not None:
                 renderer_lb = Renderer(focal_length=focal_length_val, img_w=orig_img_bgr.shape[1], img_h=orig_img_bgr.shape[0],
                                          faces=lb_faces, same_mesh_color=True)
                 lb_overlay = renderer_lb.render_front_view([lb_verts], bg_img_rgb=orig_img_bgr[:, :, ::-1].copy())
                 out_lb_path = _out_path(img_path, "_demo_lumbar_only", ".png")
                 cv2.imwrite(out_lb_path, lb_overlay[:, :, ::-1])
                 print(f"Lumbar Only Overlay saved to: {out_lb_path}")
                 renderer_lb.delete()

             # 4b. Extract and Render Thorax + Lumbar Combined (Best Visual Completeness)
             print("--- Rendering Thorax + Lumbar Combined ---")
             tl_verts, tl_faces = extract_combined_mesh(fitter, skel_verts, skel_faces, ['thorax', 'lumbar_body'], img_path, suffix="_demo_thorax_lumbar_combined")
             if tl_verts is not None:
                 renderer_tl = Renderer(focal_length=focal_length_val, img_w=orig_img_bgr.shape[1], img_h=orig_img_bgr.shape[0],
                                          faces=tl_faces, same_mesh_color=True)
                 tl_overlay = renderer_tl.render_front_view([tl_verts], bg_img_rgb=orig_img_bgr[:, :, ::-1].copy())
                 out_tl_path = _out_path(img_path, "_demo_thorax_lumbar_combined", ".png")
                 cv2.imwrite(out_tl_path, tl_overlay[:, :, ::-1])
                 print(f"Thorax+Lumbar Combined Overlay saved to: {out_tl_path}")
                 renderer_tl.delete()

             # 5. Kidney trajectory (no rib)
             print("--- Extracting Kidney Trajectory (parallel to spine curve) ---")
             kidney_side = str(getattr(args, "kidney_side", "right"))
             spine_curve_xyz = extract_spine_centerline_curve_xyz(
                 fitter=fitter,
                 vertices=skel_verts,
                 faces=skel_faces,
                 intrinsics=intrinsics_use,
                 num_pts=int(getattr(args, "spine_num_pts", 30)),
                 mid_abs_lr_percentile=float(getattr(args, "spine_mid_abs_lr_percentile", 8.0)),
                 smooth_win=int(getattr(args, "spine_smooth_win", 5)),
                 region=str(getattr(args, "spine_region", "lumbar_to_pelvis")),
                 inf_margin_m=float(getattr(args, "spine_inf_margin_m", 0.03)),
             )
             if spine_curve_xyz is not None and len(spine_curve_xyz) >= 5:
                 # export spine curve for debug / downstream
                 try:
                     out_sp_xyz = os.path.join(os.path.dirname(img_path), "spine_curve_xyz_cam.npy")
                     np.save(out_sp_xyz, np.asarray(spine_curve_xyz, dtype=np.float32))
                     print(f"[export] Saved: {out_sp_xyz}")
                 except Exception:
                     pass

                 # export spine curve in UV for downstream scan planning (same image coordinate system as overlays)
                 try:
                     spine_uv = project_points_to_uv(
                         np.asarray(spine_curve_xyz, dtype=np.float32),
                         intrinsics_use,
                     ).astype(np.float32)
                     out_sp_uv = os.path.join(os.path.dirname(img_path), "spine_curve_uv.npy")
                     np.save(out_sp_uv, spine_uv)
                     print(f"[export] Saved: {out_sp_uv}")
                 except Exception:
                     pass
                 # approximate T10/T11 along spine curve (no explicit T10/T11 in SKEL)
                 try:
                     t10, t11 = estimate_t10_t11_from_spine_curve(
                         fitter=fitter,
                         skel_verts=skel_verts,
                         spine_curve_xyz=spine_curve_xyz,
                         intrinsics=intrinsics_use,
                         t10_frac=float(getattr(args, "t10_frac", 0.80)),
                         t11_frac=float(getattr(args, "t11_frac", 0.90)),
                     )
                     if t10 is not None:
                         np.save(os.path.join(os.path.dirname(img_path), "t10_xyz_cam.npy"), np.asarray(t10, dtype=np.float32))
                         print("[export] Saved: t10_xyz_cam.npy")
                     if t11 is not None:
                         np.save(os.path.join(os.path.dirname(img_path), "t11_xyz_cam.npy"), np.asarray(t11, dtype=np.float32))
                         print("[export] Saved: t11_xyz_cam.npy")
                 except Exception:
                     t10, t11 = None, None

                 # Step-1: spine(costal margin) cross point ON THE SPINE CURVE (blue line)
                 try:
                     cross_xyz, _rib_dbg = find_spine_costal_margin_cross_point_on_spine_curve(
                         fitter=fitter,
                         skel_verts=skel_verts,
                         skel_faces=skel_faces,
                         img_path=img_path,
                         intrinsics=intrinsics_use,
                         spine_curve_xyz=spine_curve_xyz,
                     )
                     if cross_xyz is not None:
                         out_cross = os.path.join(os.path.dirname(img_path), "spine_costal_margin_cross_xyz_cam.npy")
                         np.save(out_cross, np.asarray(cross_xyz, dtype=np.float32))
                         print(f"[export] Saved: {out_cross}")
                         # Step-2: extend along the BLUE spine curve to define kidney start point
                         try:
                             adv_m = float(getattr(args, "cross_to_start_len_m", 0.03))
                             adv_dir = str(getattr(args, "cross_to_start_dir", "inferior"))
                             start_xyz = _advance_point_along_spine_curve_by_arclen(
                                 fitter=fitter,
                                 skel_verts=skel_verts,
                                 spine_curve_xyz=spine_curve_xyz,
                                 from_xyz=cross_xyz,
                                 advance_m=adv_m,
                                 direction=adv_dir,
                             )
                             if start_xyz is not None:
                                 out_start = os.path.join(os.path.dirname(img_path), "spine_costal_margin_start_xyz_cam.npy")
                                 np.save(out_start, np.asarray(start_xyz, dtype=np.float32))
                                 print(f"[export] Saved: {out_start}")
                         except Exception:
                             start_xyz = None
                         # Step-3 (reverted per user): end point = leftest + lowest on the LEFT costal margin (ribline)
                         try:
                             rib_left = extract_right_costal_margin(
                                 fitter, skel_verts, skel_faces, img_path, intrinsics_use, side="left"
                             )
                             end_xyz = None
                             if rib_left is not None and len(rib_left) >= 2:
                                 # extraction stabilizes endpoint as the last point (end_pt forced appended)
                                 rib_left = np.asarray(rib_left, dtype=np.float32).reshape(-1, 3)
                                 end_xyz_raw = rib_left[-1].astype(np.float32)
                                 # user tweak: move "up" along the rib by a small distance (default 0.06m)
                                 up_m = float(getattr(args, "end_up_along_rib_m", 0.0))
                                 if up_m > 1e-6:
                                     end_xyz = _advance_point_along_polyline_by_arclen(
                                         rib_left,
                                         start_idx=len(rib_left) - 1,
                                         delta_m=up_m,
                                         direction="backward",
                                     )
                                 else:
                                     end_xyz = end_xyz_raw
                             if end_xyz is not None:
                                 out_end = os.path.join(os.path.dirname(img_path), "costal_margin_end_left_lowest_xyz_cam.npy")
                                 np.save(out_end, np.asarray(end_xyz, dtype=np.float32))
                                 print(f"[export] Saved: {out_end}")
                         except Exception:
                             end_xyz = None
                         # visualize: ONLY keep the moved point (green) on top of the spine-curve overlay
                         try:
                             base_path = _out_path(img_path, "_demo_skel_overlay_spine_curve", ".png")
                             base = cv2.imread(base_path) if os.path.exists(base_path) else None
                             if base is not None:
                                 # draw start/end with tiny markers (user requested)
                                 if ('start_xyz' in locals()) and (start_xyz is not None):
                                     uv2 = _project_xyz_to_uv_single(start_xyz, intrinsics_use)
                                     if uv2 is not None:
                                         cv2.circle(base, tuple(np.round(uv2).astype(np.int32).tolist()), 1, (0, 255, 0), -1)
                                 # draw end (magenta)
                                 if ('end_xyz' in locals()) and (end_xyz is not None):
                                     uv3 = _project_xyz_to_uv_single(end_xyz, intrinsics_use)
                                     if uv3 is not None:
                                         cv2.circle(base, tuple(np.round(uv3).astype(np.int32).tolist()), 1, (255, 0, 255), -1)
                                 # connect start->end and build offset line (like rgbpair_latest rib_both)
                                 if ('start_xyz' in locals()) and (start_xyz is not None) and ('end_xyz' in locals()) and (end_xyz is not None):
                                     try:
                                         uv_s = _project_xyz_to_uv_single(start_xyz, intrinsics_use)
                                         uv_e = _project_xyz_to_uv_single(end_xyz, intrinsics_use)
                                         if uv_s is not None and uv_e is not None:
                                             uv_s = np.asarray(uv_s, dtype=np.float64).reshape(2)
                                             uv_e = np.asarray(uv_e, dtype=np.float64).reshape(2)
                                             d = uv_e - uv_s
                                             dn = float(np.linalg.norm(d))
                                             if dn > 1e-6:
                                                 d = d / dn
                                                 n_down = _normal_down_from_dir_uv(d)
                                                 shift_px = float(getattr(args, "kidney_line_shift_px", 12.0))
                                                 uv_line = np.stack([uv_s, uv_e], axis=0).astype(np.float32)
                                                 uv_line_shift = (uv_line + n_down.reshape(1, 2) * float(shift_px)).astype(np.float32)
                                                 # Slide the offset line along its own direction until its START hits the spine curve (blue)
                                                 # This matches: "first draw offset line, then move along direction until start crosses spine line".
                                                 if spine_curve_xyz is not None and len(spine_curve_xyz) >= 5:
                                                     spine_uv = project_points_to_uv(
                                                         np.asarray(spine_curve_xyz, dtype=np.float32), intrinsics_use
                                                     ).astype(np.float32)
                                                     d_uv = (uv_line[1] - uv_line[0]).astype(np.float64)
                                                     d_uvn = float(np.linalg.norm(d_uv))
                                                     if d_uvn > 1e-6:
                                                         d_uv = d_uv / d_uvn
                                                         hit = _intersect_infinite_line_with_polyline_uv(
                                                             line_p=uv_line_shift[0],
                                                             line_dir=d_uv,
                                                             poly_uv=spine_uv,
                                                             prefer_forward=True,
                                                         )
                                                         if hit is not None:
                                                             hit_uv, _t = hit
                                                             L = float(np.linalg.norm(uv_line_shift[1] - uv_line_shift[0]))
                                                             uv_line_shift = np.stack(
                                                                 [hit_uv, (hit_uv.astype(np.float64) + d_uv * L).astype(np.float32)],
                                                                 axis=0,
                                                             ).astype(np.float32)
                                                 # export for downstream
                                                 out_uv = os.path.join(os.path.dirname(img_path), "kidney_line_uv.npy")
                                                 out_uv_shift = os.path.join(os.path.dirname(img_path), "kidney_line_uv_shifted.npy")
                                                 np.save(out_uv, uv_line.astype(np.float32))
                                                 np.save(out_uv_shift, uv_line_shift.astype(np.float32))
                                                 # draw base line (yellow) + shifted line (red)
                                                 pts0 = np.round(uv_line).astype(np.int32).reshape((-1, 1, 2))
                                                 pts1 = np.round(uv_line_shift).astype(np.int32).reshape((-1, 1, 2))
                                                 cv2.polylines(base, [pts0], isClosed=False, color=(0, 255, 255), thickness=2)
                                                 cv2.polylines(base, [pts1], isClosed=False, color=(0, 0, 255), thickness=2)

                                                 # --- export separated overlays (user request) ---
                                                 try:
                                                     skel_path = _out_path(img_path, "_demo_skel_overlay", ".png")
                                                     skel_base = cv2.imread(skel_path) if os.path.exists(skel_path) else None
                                                     if skel_base is not None:
                                                         # 1) spine curve only (blue)
                                                         spine_only = skel_base.copy()
                                                         if spine_curve_xyz is not None and len(spine_curve_xyz) >= 5:
                                                             spine_uv2 = project_points_to_uv(
                                                                 np.asarray(spine_curve_xyz, dtype=np.float32), intrinsics_use
                                                             ).astype(np.float32)
                                                             spine_only = _draw_uv_polyline(spine_only, spine_uv2, color=(255, 0, 0), thickness=4)
                                                         out_sp_only = _out_path(img_path, "_demo_skel_overlay_spine_curve_only", ".png")
                                                         cv2.imwrite(out_sp_only, spine_only)

                                                         # 2) offset kidney line only (red)
                                                         kidney_only = skel_base.copy()
                                                         cv2.polylines(kidney_only, [pts1], isClosed=False, color=(0, 0, 255), thickness=2)
                                                         out_k_only = _out_path(img_path, "_demo_skel_overlay_kidney_offset_only", ".png")
                                                         cv2.imwrite(out_k_only, kidney_only)
                                                 except Exception:
                                                     pass
                                     except Exception:
                                         pass
                                 outp = _out_path(img_path, "_demo_skel_overlay_spine_curve_start_end", ".png")
                                 cv2.imwrite(outp, base)
                                 print(f"[export] Saved: {outp}")
                         except Exception:
                             pass
                 except Exception as e:
                     print(f"[warn] spine-costal cross export failed (skip): {e}")

             traj_xyz = extract_kidney_traj_parallel_spine_curve(
                 fitter=fitter,
                 vertices=skel_verts,
                 faces=skel_faces,
                 intrinsics=intrinsics_use,
                 spine_curve_xyz=spine_curve_xyz,
                 side=kidney_side,
                 kidney_length_m=float(getattr(args, "kidney_length_m", 0.22)),
                 num_samples=int(getattr(args, "kidney_num_bins", 40)),
                 offset_mode=str(getattr(args, "kidney_offset_mode", "lr_dir")),
                 start_point_xyz=None,
             )
             # Override kidney start point with ribline end point (same as previous rib pipeline end_pt)
             try:
                 rib_line_3d = extract_right_costal_margin(fitter, skel_verts, skel_faces, img_path, intrinsics_use)
                 if rib_line_3d is not None and len(rib_line_3d) >= 2:
                     rib_end_xyz = np.asarray(rib_line_3d[-1], dtype=np.float32).reshape(3)
                     traj_xyz = extract_kidney_traj_parallel_spine_curve(
                         fitter=fitter,
                         vertices=skel_verts,
                         faces=skel_faces,
                         intrinsics=intrinsics_use,
                         spine_curve_xyz=spine_curve_xyz,
                         side=kidney_side,
                         kidney_length_m=float(getattr(args, "kidney_length_m", 0.22)),
                         num_samples=int(getattr(args, "kidney_num_bins", 40)),
                         offset_mode=str(getattr(args, "kidney_offset_mode", "lr_dir")),
                         start_point_xyz=rib_end_xyz,
                     )
                     print("[kidney] start point overridden by ribline end point.")
                 else:
                     print("[warn] ribline end point not available; using fallback kidney start.")
             except Exception as e:
                 print(f"[warn] failed to compute ribline end point for kidney start (fallback): {e}")
             if traj_xyz is None:
                 print("[warn] kidney trajectory extraction failed (traj_xyz is None). Try increasing --kidney_length_m or relaxing --spine_mid_abs_lr_percentile.")
             if traj_xyz is not None and len(traj_xyz) >= 2:
                 out_xyz = os.path.join(os.path.dirname(img_path), f"kidney_traj_{kidney_side.lower()}_xyz_cam.npy")
                 np.save(out_xyz, np.asarray(traj_xyz, dtype=np.float32))
                 print(f"[export] Saved: {out_xyz}")

                 traj_uv = project_points_to_uv(np.asarray(traj_xyz, dtype=np.float32), intrinsics_use)
                 out_uv = os.path.join(os.path.dirname(img_path), f"kidney_traj_{kidney_side.lower()}_uv.npy")
                 np.save(out_uv, traj_uv.astype(np.float32))
                 print(f"[export] Saved: {out_uv}")

                 # quick overlay on RGB
                 try:
                     overlay = orig_img_bgr.copy()
                     pts_i = np.round(traj_uv).astype(np.int32).reshape((-1, 1, 2))
                     # magenta trajectory for visibility
                     cv2.polylines(overlay, [pts_i], isClosed=False, color=(255, 0, 255), thickness=6)
                     # draw dotted points to make it obvious even if line overlaps mesh texture
                     pts_flat = pts_i.reshape(-1, 2)
                     for j in range(0, len(pts_flat), 3):
                         cv2.circle(overlay, tuple(pts_flat[j].tolist()), 3, (255, 0, 255), -1)
                     # mark start/end points
                     p0 = tuple(np.round(traj_uv[0]).astype(np.int32).tolist())
                     p1 = tuple(np.round(traj_uv[-1]).astype(np.int32).tolist())
                     cv2.circle(overlay, p0, 9, (0, 255, 255), -1)  # yellow start
                     cv2.circle(overlay, p1, 9, (0, 255, 0), -1)    # green end
                     out_vis = _out_path(img_path, f"_demo_kidney_traj_{kidney_side.lower()}", ".png")
                     cv2.imwrite(out_vis, overlay)
                     print(f"[export] Saved: {out_vis}")
                 except Exception as e:
                     print(f"[warn] Kidney overlay export failed (skip): {e}")

                 # overlay on skeleton mesh (original body + skeleton + trajectory + spine curve)
                 try:
                     skel_bg = None
                     # prefer in-memory render result if available
                     if 'skel_overlay_bgr' in locals() and skel_overlay_bgr is not None:
                         skel_bg = skel_overlay_bgr.copy()
                     else:
                         # fallback to file on disk
                         skel_path = _out_path(img_path, "_demo_skel_overlay", ".png")
                         if os.path.exists(skel_path):
                             tmp = cv2.imread(skel_path)
                             if tmp is not None:
                                 skel_bg = tmp
                     if skel_bg is not None:
                        # draw spine curve if available (blue)
                        try:
                            if spine_curve_xyz is not None and len(spine_curve_xyz) >= 5:
                                spine_uv = project_points_to_uv(np.asarray(spine_curve_xyz, dtype=np.float32), intrinsics_use)
                                skel_bg = _draw_uv_polyline(skel_bg, spine_uv, color=(255, 0, 0), thickness=4)
                        except Exception:
                            pass
                        # draw T10/T11 markers if available
                        try:
                            if 't10' in locals() and t10 is not None:
                                uv = _project_xyz_to_uv_single(t10, intrinsics_use)
                                if uv is not None:
                                    cv2.circle(skel_bg, tuple(np.round(uv).astype(np.int32).tolist()), 8, (0, 128, 255), -1)  # orange
                            if 't11' in locals() and t11 is not None:
                                uv = _project_xyz_to_uv_single(t11, intrinsics_use)
                                if uv is not None:
                                    cv2.circle(skel_bg, tuple(np.round(uv).astype(np.int32).tolist()), 8, (0, 0, 255), -1)    # red
                        except Exception:
                            pass

                        pts_i = np.round(traj_uv).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(skel_bg, [pts_i], isClosed=False, color=(255, 0, 255), thickness=6)
                        pts_flat = pts_i.reshape(-1, 2)
                        for j in range(0, len(pts_flat), 3):
                            cv2.circle(skel_bg, tuple(pts_flat[j].tolist()), 3, (255, 0, 255), -1)
                        p0 = tuple(np.round(traj_uv[0]).astype(np.int32).tolist())
                        p1 = tuple(np.round(traj_uv[-1]).astype(np.int32).tolist())
                        cv2.circle(skel_bg, p0, 9, (0, 255, 255), -1)  # start
                        cv2.circle(skel_bg, p1, 9, (0, 255, 0), -1)    # end
                        out_combo = _out_path(img_path, f"_demo_skel_overlay_kidney_traj_{kidney_side.lower()}", ".png")
                        cv2.imwrite(out_combo, skel_bg)
                        print(f"[export] Saved: {out_combo}")
                     else:
                         print("[warn] skeleton overlay not available; skip skel+traj composite export.")
                 except Exception as e:
                     print(f"[warn] Skeleton+traj overlay export failed (skip): {e}")

                 # spine curve overlay only (original body + skeleton + spine curve)
                 try:
                     skel_path = _out_path(img_path, "_demo_skel_overlay", ".png")
                     base = cv2.imread(skel_path) if os.path.exists(skel_path) else None
                     if base is not None:
                         if spine_curve_xyz is not None and len(spine_curve_xyz) >= 5:
                             spine_uv = project_points_to_uv(np.asarray(spine_curve_xyz, dtype=np.float32), intrinsics_use)
                             out_sp = _draw_uv_polyline(base, spine_uv, color=(255, 0, 0), thickness=4)
                             out_sp_path = _out_path(img_path, "_demo_skel_overlay_spine_curve", ".png")
                             cv2.imwrite(out_sp_path, out_sp)
                             print(f"[export] Saved: {out_sp_path}")
                 except Exception as e:
                     print(f"[warn] spine curve overlay export failed (skip): {e}")

                 # Paper-style CVA scan path overlay
                 try:
                     paper_path = plan_kidney_scan_path_paper(
                         fitter=fitter,
                         vertices=skel_verts,
                         intrinsics=intrinsics_use,
                         spine_curve_xyz=spine_curve_xyz,
                         cva_deg=float(getattr(args, "cva_deg", 45.0)),
                         w1_m=float(getattr(args, "cva_w1_m", 0.10)),
                         path_len_m=float(getattr(args, "cva_path_len_m", 0.12)),
                         step_m=float(getattr(args, "cva_step_m", 0.01)),
                         band_half_width_m=float(getattr(args, "cva_band_half_width_m", 0.10)),
                         num_lr_bins=int(getattr(args, "cva_num_lr_bins", 15)),
                         snap_window_m=float(getattr(args, "cva_snap_window_m", 0.03)),
                     )
                     if paper_path is not None and len(paper_path) >= 2:
                         np.save(os.path.join(os.path.dirname(img_path), "kidney_scan_paper_path_xyz_cam.npy"), np.asarray(paper_path, dtype=np.float32))
                         paper_uv = project_points_to_uv(np.asarray(paper_path, dtype=np.float32), intrinsics_use)
                         np.save(os.path.join(os.path.dirname(img_path), "kidney_scan_paper_path_uv.npy"), paper_uv.astype(np.float32))
                         # draw on skeleton overlay
                         skel_path = _out_path(img_path, "_demo_skel_overlay", ".png")
                         base = cv2.imread(skel_path) if os.path.exists(skel_path) else None
                         if base is not None:
                             base2 = base.copy()
                             # spine curve (blue) for context
                             if spine_curve_xyz is not None and len(spine_curve_xyz) >= 5:
                                 spine_uv = project_points_to_uv(np.asarray(spine_curve_xyz, dtype=np.float32), intrinsics_use)
                                 base2 = _draw_uv_polyline(base2, spine_uv, color=(255, 0, 0), thickness=3)
                             # paper path (green)
                             pts_i = np.round(paper_uv).astype(np.int32).reshape((-1, 1, 2))
                             cv2.polylines(base2, [pts_i], isClosed=False, color=(0, 255, 0), thickness=5)
                             p0 = tuple(np.round(paper_uv[0]).astype(np.int32).tolist())
                             cv2.circle(base2, p0, 9, (0, 255, 255), -1)  # start
                             outp = _out_path(img_path, "_demo_skel_overlay_kidney_scan_paper_path", ".png")
                             cv2.imwrite(outp, base2)
                             print(f"[export] Saved: {outp}")
                     else:
                         print("[warn] paper-style scan path not generated (insufficient points).")
                 except Exception as e:
                     print(f"[warn] paper-style scan path export failed (skip): {e}")

                 # Costal-margin-angle based path (user requested)
                 try:
                     cm_path, cm_ang_deg, cm_start_xyz = plan_kidney_scan_path_costal_margin_based(
                         fitter=fitter,
                         vertices=skel_verts,
                         faces=skel_faces,
                         img_path=img_path,
                         intrinsics=intrinsics_use,
                         spine_curve_xyz=spine_curve_xyz,
                         side=str(getattr(args, "kidney_side", "right")),
                         path_len_m=float(getattr(args, "cm_path_len_m", 0.12)),
                         step_m=float(getattr(args, "cm_step_m", 0.01)),
                         snap_window_m=float(getattr(args, "cm_snap_window_m", 0.03)),
                     )
                     if cm_ang_deg is not None:
                         print(f"[cm-path] angle(spine vs costal_margin) = {cm_ang_deg:.1f} deg")
                     if cm_path is not None and len(cm_path) >= 2:
                         np.save(os.path.join(os.path.dirname(img_path), "kidney_scan_costal_margin_path_xyz_cam.npy"),
                                 np.asarray(cm_path, dtype=np.float32))
                         cm_uv = project_points_to_uv(np.asarray(cm_path, dtype=np.float32), intrinsics_use)
                         np.save(os.path.join(os.path.dirname(img_path), "kidney_scan_costal_margin_path_uv.npy"),
                                 cm_uv.astype(np.float32))
                         if cm_start_xyz is not None:
                             np.save(os.path.join(os.path.dirname(img_path), "kidney_scan_costal_margin_start_xyz_cam.npy"),
                                     np.asarray(cm_start_xyz, dtype=np.float32))
                         # draw on skeleton overlay
                         skel_path = _out_path(img_path, "_demo_skel_overlay", ".png")
                         base = cv2.imread(skel_path) if os.path.exists(skel_path) else None
                         if base is not None:
                             base2 = base.copy()
                             # spine curve (blue)
                             if spine_curve_xyz is not None and len(spine_curve_xyz) >= 5:
                                 spine_uv = project_points_to_uv(np.asarray(spine_curve_xyz, dtype=np.float32), intrinsics_use)
                                 base2 = _draw_uv_polyline(base2, spine_uv, color=(255, 0, 0), thickness=3)
                             # cm path (green)
                             pts_i = np.round(cm_uv).astype(np.int32).reshape((-1, 1, 2))
                             cv2.polylines(base2, [pts_i], isClosed=False, color=(0, 255, 0), thickness=6)
                             # mark start from spine∩costal_margin (yellow)
                             p0 = tuple(np.round(cm_uv[0]).astype(np.int32).tolist())
                             cv2.circle(base2, p0, 10, (0, 255, 255), -1)
                             outp = _out_path(img_path, "_demo_skel_overlay_kidney_scan_costal_margin_path", ".png")
                             cv2.imwrite(outp, base2)
                             print(f"[export] Saved: {outp}")
                     else:
                         print("[warn] costal-margin-based scan path not generated (insufficient points).")
                 except Exception as e:
                     print(f"[warn] costal-margin-based scan path export failed (skip): {e}")

                 # Angle line (start->rib_end) + parallel offset path (user requested)
                 try:
                     angle_path, angle_start_xyz, angle_rib_end_xyz = plan_kidney_scan_path_angle_line_offset(
                         fitter=fitter,
                         vertices=skel_verts,
                         faces=skel_faces,
                         img_path=img_path,
                         intrinsics=intrinsics_use,
                         spine_curve_xyz=spine_curve_xyz,
                         side=str(getattr(args, "kidney_side", "right")),
                         offset_m=float(getattr(args, "angle_offset_m", 0.03)),
                         step_m=float(getattr(args, "angle_step_m", 0.01)),
                     )
                     if angle_path is not None and len(angle_path) >= 2:
                         np.save(os.path.join(os.path.dirname(img_path), "kidney_scan_angle_offset_path_xyz_cam.npy"),
                                 np.asarray(angle_path, dtype=np.float32))
                         angle_uv = project_points_to_uv(np.asarray(angle_path, dtype=np.float32), intrinsics_use)
                         np.save(os.path.join(os.path.dirname(img_path), "kidney_scan_angle_offset_path_uv.npy"),
                                 angle_uv.astype(np.float32))
                         if angle_start_xyz is not None:
                             np.save(os.path.join(os.path.dirname(img_path), "kidney_scan_angle_offset_start_xyz_cam.npy"),
                                     np.asarray(angle_start_xyz, dtype=np.float32))
                         if angle_rib_end_xyz is not None:
                             np.save(os.path.join(os.path.dirname(img_path), "kidney_scan_angle_offset_rib_end_xyz_cam.npy"),
                                     np.asarray(angle_rib_end_xyz, dtype=np.float32))
                         # draw on skeleton overlay: angle line + offset path
                         skel_path = _out_path(img_path, "_demo_skel_overlay", ".png")
                         base = cv2.imread(skel_path) if os.path.exists(skel_path) else None
                         if base is not None:
                             base2 = base.copy()
                             # offset path (magenta)
                             pts_i = np.round(angle_uv).astype(np.int32).reshape((-1, 1, 2))
                             cv2.polylines(base2, [pts_i], isClosed=False, color=(255, 0, 255), thickness=6)
                             p0 = tuple(np.round(angle_uv[0]).astype(np.int32).tolist())
                             p1 = tuple(np.round(angle_uv[-1]).astype(np.int32).tolist())
                             cv2.circle(base2, p0, 10, (0, 255, 255), -1)  # start
                             cv2.circle(base2, p1, 9, (0, 255, 0), -1)     # end
                             outp = _out_path(img_path, "_demo_skel_overlay_kidney_scan_angle_offset_path", ".png")
                         cv2.imwrite(outp, base2)
                         print(f"[export] Saved: {outp}")
                     else:
                         print("[warn] angle+offset path not generated (insufficient points).")
                 except Exception as e:
                     print(f"[warn] angle+offset path export failed (skip): {e}")

                 # (disabled) legacy angle-offset2 debug export (kept out to avoid clutter/indent issues)

    # 6. Visualization
    print("--- Visualization ---")
    pred_vert_arr = np.array(pred_vert_arr)
    # We only have 1 image (img_idx 0)
    img_idx = 0
    chosen_mask = detection_result[:, 0] == img_idx
    chosen_vert_arr = pred_vert_arr[chosen_mask]
    
    if len(chosen_vert_arr) > 0:
        renderer = Renderer(focal_length=focal_length_val, img_w=orig_img_bgr.shape[1], img_h=orig_img_bgr.shape[0],
                            faces=smpl_model.faces, same_mesh_color=False)
        front_view = renderer.render_front_view(chosen_vert_arr, bg_img_rgb=orig_img_bgr[:, :, ::-1].copy())
        
        out_img_path = _out_path(img_path, "_demo_result", ".png")
        cv2.imwrite(out_img_path, front_view[:, :, ::-1])
        print(f"Visualization saved to: {out_img_path}")
        
        # BBox
        bbox_img = orig_img_bgr.copy()
        for min_x, min_y, max_x, max_y, conf in dets[:, 1:6]:
             cv2.rectangle(bbox_img, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2)
        cv2.imwrite(_out_path(img_path, "_demo_bbox", ".png"), bbox_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True, help='path to the input image')
    parser.add_argument('--ckpt', default="data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt")
    parser.add_argument("--backbone", default="hr48", choices=['res50', 'hr48'])
    # Kidney trajectory controls
    parser.add_argument("--kidney_side", default="left", choices=["right", "left"],
                        help="要扫描哪侧肾（患者体侧）：right=右肾/右侧，left=左肾/左侧")
    parser.add_argument("--kidney_inf_percentile_lo", type=float, default=20.0,
                        help="沿 inferior_dir 的取样范围下界（百分位），避免过胸/过骨盆")
    parser.add_argument("--kidney_inf_percentile_hi", type=float, default=85.0,
                        help="沿 inferior_dir 的取样范围上界（百分位）")
    parser.add_argument("--kidney_lr_band_percentile", type=float, default=97.0,
                        help="每个 inf bin 内取“最靠右/最靠左”的 lr 百分位阈值（越大越贴边）")
    parser.add_argument("--kidney_num_bins", type=int, default=40,
                        help="沿 inferior_dir 的采样 bin 数（点越多轨迹越密）")
    parser.add_argument("--kidney_length_m", type=float, default=0.22,
                        help="肾脏轨迹长度（沿 spine curve 的弧长近似，单位米）。会从起点沿 inferior 方向截断。")
    parser.add_argument("--spine_num_pts", type=int, default=30,
                        help="spine curve 采样点数（越大越平滑/越贴合）")
    parser.add_argument("--spine_mid_abs_lr_percentile", type=float, default=8.0,
                        help="每个 inferior bin 里取 |lr| 最小的百分位作为‘脊柱中线’候选（越小越靠中线）")
    parser.add_argument("--spine_smooth_win", type=int, default=5,
                        help="spine curve 平滑窗口（奇数更合适，默认 5）")
    parser.add_argument("--spine_region", default="lumbar_to_pelvis",
                        choices=["lumbar_to_pelvis", "thorax_to_pelvis"],
                        help="限制 spine curve 的范围，避免起点太早/跑偏：默认 lumbar_to_pelvis")
    parser.add_argument("--spine_inf_margin_m", type=float, default=0.03,
                        help="spine curve 区间的前后扩展 margin（米），默认 0.03")
    parser.add_argument("--kidney_offset_mode", default="lr_dir",
                        choices=["lr_dir", "curve_normal"],
                        help="kidney 平行偏移方向：lr_dir=沿人体左右轴固定偏移(更稳定)；curve_normal=沿曲线法向偏移(旧行为)")
    parser.add_argument("--t10_frac", type=float, default=0.80,
                        help="T10 近似位置：在 thorax->lumbar_body spine curve 段上的弧长比例（0~1）。")
    parser.add_argument("--t11_frac", type=float, default=0.90,
                        help="T11 近似位置：在 thorax->lumbar_body spine curve 段上的弧长比例（0~1）。")
    parser.add_argument("--cva_deg", type=float, default=45.0,
                        help="论文中的 costovertebral angle θ（度）。论文建议 45° 起步，可逐步减小。")
    parser.add_argument("--cva_w1_m", type=float, default=0.10,
                        help="论文中的 capture window w1（米），沿脊柱边界取一段用于构造 CVA rib boundary。")
    parser.add_argument("--cva_path_len_m", type=float, default=0.12,
                        help="论文式扫描路径长度（米），默认 0.12m（≈成人肾长 10–12cm）。")
    parser.add_argument("--cva_step_m", type=float, default=0.01,
                        help="论文式路径点间距（米）。")
    parser.add_argument("--cva_band_half_width_m", type=float, default=0.10,
                        help="提取 lowest lumbar line 时的中线带宽（半宽，米）。")
    parser.add_argument("--cva_num_lr_bins", type=int, default=15,
                        help="提取 lowest lumbar line 的 lr 分箱数。")
    parser.add_argument("--cva_snap_window_m", type=float, default=0.03,
                        help="把论文式直线/平面投影回表面时的局部窗口（米）。")
    parser.add_argument("--cm_path_len_m", type=float, default=0.12,
                        help="基于 costal margin 方向的扫描路径长度（米）。默认 0.12m。")
    parser.add_argument("--cm_step_m", type=float, default=0.01,
                        help="基于 costal margin 的路径点间距（米）。")
    parser.add_argument("--cm_snap_window_m", type=float, default=0.03,
                        help="costal margin 模式下吸附到表面的局部窗口（米）。")
    parser.add_argument("--angle_offset_m", type=float, default=0.03,
                        help="按“start->rib_end”的角度直线生成平行偏移线：偏移距离（米）。正值默认朝 inferior 方向偏移。")
    parser.add_argument("--angle_step_m", type=float, default=0.01,
                        help="角度直线平行偏移模式：采样点间距（米）。")
    parser.add_argument("--angle_rib_end_side", default="right", choices=["left", "right", "auto"],
                        help="角度直线的 rib end 取哪一侧。你最新需求是 left。auto=与 kidney_side 相反。")
    parser.add_argument("--angle_spine_next_k", type=int, default=1,
                        help="起点取 spine curve 上最接近(脊柱∩肋缘)的点，然后沿 inferior 方向取第 k 个“下一段中点”。默认 1。")
    parser.add_argument("--cross_to_start_len_m", type=float, default=0.03,
                        help="Step2: 从 costal margin×spine_curve 交点沿 spine_curve 延伸的长度（米）。")
    parser.add_argument("--cross_to_start_dir", default="inferior", choices=["inferior", "superior"],
                        help="Step2: 延伸方向：inferior=向骨盆(身体下)，superior=向胸(身体上)。")
    parser.add_argument("--end_lr_band_percentile", type=float, default=15.0,
                        help="Step3: end point 取 leftest band 的 LR 百分位阈值（越小越靠左侧极值）。默认 15。")
    parser.add_argument("--end_z_depth_percentile", type=float, default=85.0,
                        help="Step3: end point 取 depthest(最大Z) 的百分位阈值（越大越靠后）。默认 85。")
    parser.add_argument("--end_up_along_rib_m", type=float, default=0.0,
                        help="在 rib-based end point 的基础上，沿肋缘曲线向上(回到 xiphoid 方向)移动的弧长距离（米）。例如 0.06。")
    parser.add_argument("--kidney_line_shift_px", type=float, default=12.0,
                        help="连接 start->end 的直线，沿其法向量(指向图像下)偏移的像素距离。默认 12（同 rgbpair_latest rib_shift_px）。")

    # optional: invert stitch transforms back to upper image size
    parser.add_argument("--export_upper_only", action="store_true",
                        help="把 stitched_upright 的 overlay 逆变换回 upper 图尺寸（逆 letterbox→裁下半身→逆旋转）")
    parser.add_argument("--upper_img", default="", help="原始上半身图路径（用于重建拼接前尺寸/旋转）")
    parser.add_argument("--lower_img", default="", help="原始下半身图路径（用于重建拼接前尺寸/旋转）")
    parser.add_argument("--stitch_rotate", default="cw", choices=["none", "cw", "ccw", "180"],
                        help="拼接前对 upper/lower 施加的旋转（需与 stitch_two_rgb.py 参数一致）")
    parser.add_argument("--stitch_mode", default="vertical", choices=["vertical", "horizontal"],
                        help="拼接方向（需与 stitch_two_rgb.py 参数一致）")
    parser.add_argument("--stitch_overlap_ratio", type=float, default=0.0,
                        help="拼接时对第二张裁剪比例（需与 stitch_two_rgb.py --overlap_ratio 一致）")
    parser.add_argument("--stitch_letterbox_w", type=int, default=720, help="拼接时 letterbox 输出宽（需一致）")
    parser.add_argument("--stitch_letterbox_h", type=int, default=1280, help="拼接时 letterbox 输出高（需一致）")
    args = parser.parse_args()
    
    main(args)
def try_build_kidney_offset_line_uv_shifted(
    fitter,
    vertices,
    faces,
    img_path,
    intrinsics,
    spine_curve_uv,
    spine_curve_xyz,
    cross_to_start_len_m=0.03,
    cross_to_start_dir="inferior",
    end_up_along_rib_m=0.0,
    kidney_line_shift_px=12.0,
    rib_trim_head_ratio=0.0,
    kidney_side="left",
    end_lr_band_percentile=15.0,
):
    """
    尝试构建肾脏扫描的偏移直线（Offset Line）：
    1. 起点：裁切后的肋缘 PCA 直线延长 ∩ 脊柱曲线，再沿脊柱曲线向下( inferior )移动一小段。
    2. 终点：裁切后肋缘上「最左 + 最下」点（可沿肋缘回调一段距离）。
    3. 连接起点与终点形成直线，沿法向(图像下)偏移 kidney_line_shift_px。
    4. 将偏移后直线起点吸附回脊柱曲线（保持长度不变）。

    仅 left kidney 默认 rib_trim_head_ratio=0.40；胆囊等 traj 不走此函数。
    """
    import numpy as np
    import os

    side_lc = str(kidney_side or "left").strip().lower()
    if side_lc not in ("left", "right"):
        side_lc = "left"
    rib_side = "left" if side_lc == "left" else "right"

    trim_r = float(rib_trim_head_ratio or 0.0)
    if side_lc == "left" and trim_r <= 0.0:
        trim_r = 0.40
    
    # 1. 找到脊柱曲线与（裁切后）肋缘延长线的交点
    try:
        cross_xyz, rib_line_3d = find_spine_costal_margin_cross_point_on_spine_curve(
            fitter=fitter,
            skel_verts=vertices,
            skel_faces=faces,
            img_path=img_path,
            intrinsics=intrinsics,
            spine_curve_xyz=spine_curve_xyz,
            rib_trim_head_ratio=trim_r,
            side=rib_side,
        )
        if cross_xyz is None:
            return None
            
        # 2. 确定起点：从交点沿脊柱曲线延伸
        start_xyz = _advance_point_along_spine_curve_by_arclen(
            fitter=fitter,
            skel_verts=vertices,
            spine_curve_xyz=spine_curve_xyz,
            from_xyz=cross_xyz,
            advance_m=cross_to_start_len_m,
            direction=cross_to_start_dir,
        )
        if start_xyz is None:
            return None
            
    except Exception as e:
        print(f"[try_build_kidney] Start point failed: {e}")
        return None

    # 3. 确定终点：裁切后肋缘的 leftest+lowest（与起点同一侧肋缘）
    try:
        if rib_line_3d is None or len(rib_line_3d) < 2:
            rib_line_3d = extract_right_costal_margin(
                fitter, vertices, faces, img_path, intrinsics, side=rib_side
            )
        if rib_line_3d is None or len(rib_line_3d) < 2:
            return None

        rib_xyz = np.asarray(rib_line_3d, dtype=np.float32).reshape(-1, 3)
        rib_trimmed = _trim_head_points(rib_xyz, trim_r) if trim_r > 0 else rib_xyz
        if len(rib_trimmed) < 2:
            rib_trimmed = rib_xyz

        if rib_side == "left":
            end_xyz_raw = find_costal_margin_end_point_leftest_lowest(
                fitter,
                vertices,
                rib_trimmed,
                lr_band_percentile=float(end_lr_band_percentile),
            )
        else:
            end_xyz_raw = find_costal_margin_end_point_rightest_lowest(
                fitter,
                vertices,
                rib_trimmed,
                lr_band_percentile=float(end_lr_band_percentile),
            )
        if end_xyz_raw is None:
            end_xyz_raw = rib_trimmed[-1].astype(np.float32)
        
        # 可选：沿肋缘回调（向脊柱/剑突方向）
        if end_up_along_rib_m > 1e-6:
            d_end = np.linalg.norm(rib_trimmed.reshape(-1, 3) - end_xyz_raw.reshape(1, 3), axis=1)
            end_idx = int(np.argmin(d_end)) if d_end.size > 0 else (len(rib_trimmed) - 1)
            end_xyz = _advance_point_along_polyline_by_arclen(
                rib_trimmed,
                start_idx=end_idx,
                delta_m=end_up_along_rib_m,
                direction="backward",
            )
        else:
            end_xyz = end_xyz_raw
            
        if end_xyz is None:
            return None
            
    except Exception as e:
        print(f"[try_build_kidney] End point failed: {e}")
        return None

    # 4. 构建 UV 直线并偏移
    try:
        uv_s = _project_xyz_to_uv_single(start_xyz, intrinsics)
        uv_e = _project_xyz_to_uv_single(end_xyz, intrinsics)
        
        if uv_s is None or uv_e is None:
            return None
            
        uv_s = np.asarray(uv_s, dtype=np.float64).reshape(2)
        uv_e = np.asarray(uv_e, dtype=np.float64).reshape(2)
        
        d = uv_e - uv_s
        dn = float(np.linalg.norm(d))
        if dn < 1e-6:
            return None
            
        d = d / dn
        n_down = _normal_down_from_dir_uv(d)
        
        uv_line = np.stack([uv_s, uv_e], axis=0).astype(np.float32)
        # 偏移
        uv_line_shift = (uv_line + n_down.reshape(1, 2) * float(kidney_line_shift_px)).astype(np.float32)
        
        # 5. 将偏移后直线的起点吸附回 Spine Curve (Slide start point back to spine)
        if spine_curve_xyz is not None and len(spine_curve_xyz) >= 5:
            d_uv = (uv_line[1] - uv_line[0]).astype(np.float64)
            d_uvn = float(np.linalg.norm(d_uv))
            if d_uvn > 1e-6:
                d_uv = d_uv / d_uvn
                # 使用偏移线的起点，沿直线方向寻找与脊柱曲线的交点
                # 注意：这里我们用 spine_curve_uv (如果传入了)
                if spine_curve_uv is not None:
                    hit = _intersect_infinite_line_with_polyline_uv(
                        line_p=uv_line_shift[0],
                        line_dir=d_uv,
                        poly_uv=spine_curve_uv,
                        prefer_forward=True,
                    )
                    if hit is not None:
                        hit_uv, _t = hit
                        # 保持原有长度
                        L = float(np.linalg.norm(uv_line_shift[1] - uv_line_shift[0]))
                        uv_line_shift = np.stack(
                            [hit_uv, (hit_uv.astype(np.float64) + d_uv * L).astype(np.float32)],
                            axis=0,
                        ).astype(np.float32)
        
        return uv_line_shift

    except Exception as e:
        print(f"[try_build_kidney] UV construction failed: {e}")
        return None
