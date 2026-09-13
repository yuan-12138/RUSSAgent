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
import math

# For kidney/spine "post-SKEL only" extraction. We deliberately reuse the same CLiFF+SKEL
# pipeline in this file for all tasks; only the trajectory extraction differs.
try:
    import kidney_spine_geometry as _kidney_post
except Exception:
    _kidney_post = None

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

# ------------------------------
# Output controls (env flags)
# ------------------------------
def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, None)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

# 默认：只保留核心产物（轨迹用的 uv npy + upper_only），不生成一堆 debug PNG / mesh obj / 多余 npy
SKEL_SAVE_DEBUG_PNGS = _env_flag("SKEL_SAVE_DEBUG_PNGS", False)
SKEL_SAVE_MESHES = _env_flag("SKEL_SAVE_MESHES", False)
SKEL_SAVE_EXTRA_NPY = _env_flag("SKEL_SAVE_EXTRA_NPY", False)

# ------------------------------
# SKEL fitter singleton + speed knobs
# ------------------------------
_SKEL_FITTER = None
_SKEL_FITTER_DEVICE_STR = None
_SKEL_FITTER_TUNED = False

def _tune_skel_cfg_half_iterations(fitter, iter_scale: float = 0.5) -> None:
    """
    将 SKEL fitting 的迭代相关超参数整体缩放（默认砍半）。
    主要影响 cfg.optim_steps[*].max_iter (LBFGS 内迭代) 和 num_steps (外层循环次数)。
    """
    if fitter is None:
        return
    try:
        scale = float(iter_scale)
    except Exception:
        scale = 0.5
    scale = max(0.05, min(1.0, scale))

    try:
        for step in getattr(fitter.cfg, "optim_steps", []):
            if hasattr(step, "max_iter"):
                old = int(step.max_iter)
                step.max_iter = max(1, int(math.ceil(old * scale)))
            if hasattr(step, "num_steps"):
                old = int(step.num_steps)
                step.num_steps = max(1, int(math.ceil(old * scale)))
    except Exception:
        # 如果 cfg 结构变化，不影响主流程
        pass

def _get_skel_fitter(device: torch.device, gender: str = "male"):
    """
    只初始化一次 SkelFitter（避免每张图重复加载/构建），并在首次初始化时做“减半迭代 + 日志降频”。
    """
    global _SKEL_FITTER, _SKEL_FITTER_DEVICE_STR, _SKEL_FITTER_TUNED
    if "SkelFitter" not in globals():
        return None

    dev_str = str(device)
    if _SKEL_FITTER is not None and _SKEL_FITTER_DEVICE_STR == dev_str:
        return _SKEL_FITTER

    _SKEL_FITTER = SkelFitter(gender=gender, device=device)
    _SKEL_FITTER_DEVICE_STR = dev_str
    _SKEL_FITTER_TUNED = False
    return _SKEL_FITTER

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


def spine_uv_visible_half(uv: np.ndarray) -> np.ndarray:
    """
    Match *_demo_skel_overlay_spine_curve_only.png:
    hide the first half of the spine polyline (start index = len // 2).
    """
    pts = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 4:
        return pts
    half_idx = len(pts) // 2
    visible = pts[half_idx:]
    return visible if len(visible) >= 2 else pts


def spine_uv_scan_line_endpoints(uv: np.ndarray) -> Optional[np.ndarray]:
    """Visible lumbar half -> 2 UV endpoints for straight spine scan (kidney-style)."""
    visible = spine_uv_visible_half(uv)
    if len(visible) < 2:
        return None
    return np.stack([visible[0], visible[-1]], axis=0).astype(np.float32)


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
        # 日志：默认静默 + 每 N 次 closure 打一次（在 SKEL aligner.py 里读取）
        os.environ.setdefault("SKEL_QUIET", "1")
        os.environ.setdefault("SKEL_PRINT_EVERY", "20")
        
        # 2. Init Fitter (singleton)
        fitter = _get_skel_fitter(device=device, gender='male')
        if fitter is None:
            raise RuntimeError("SkelFitter is not available (import failed)")
        global _SKEL_FITTER_TUNED
        if not _SKEL_FITTER_TUNED:
            # 强制：fitting 迭代次数砍半（最卡的地方）
            # 允许通过环境变量传参（由 rgbpair_skel_pipeline.py 负责设置默认值）
            try:
                iter_scale = float(os.environ.get("SKEL_ITER_SCALE", "0.45"))
            except Exception:
                iter_scale = 0.4
            _tune_skel_cfg_half_iterations(fitter, iter_scale=iter_scale)
            _SKEL_FITTER_TUNED = True
        
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
        # 禁止优化 betas：对单张图，CLIFF 的 pred_betas 足够，避免 betas 抖动/不稳定
        try:
            if isinstance(res, dict):
                res["betas"] = betas_in_np
        except Exception:
            pass
        
        # 5. Forward to get Mesh
        final_poses = torch.from_numpy(res['poses']).to(device)
        final_betas = torch.from_numpy(betas_in_np).to(device)
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
        
        # 6. Save meshes (optional)
        if SKEL_SAVE_MESHES:
            import trimesh
            out_obj = img_path.replace(".jpg", "_skeleton.obj").replace(".png", "_skeleton.obj")
            mesh = trimesh.Trimesh(skel_verts, skel_faces, process=False)
            mesh.export(out_obj)
            print(f"SKEL Mesh saved to: {out_obj}")

        # 7. Project and Draw (New Feature)
        # We need intrinsics here. We can assume intrinsics is available in scope or pass it.
        # Let's return skel_verts to main so main can handle drawing with the intrinsics it has.
        
        # 8. Extract Abdomen (only needed for debug visualization / export)
        abd_verts, abd_faces = None, None
        if SKEL_SAVE_DEBUG_PNGS or SKEL_SAVE_MESHES:
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


def project_lines_on_image(points_3d, intrinsics, img_bgr, color=(255, 0, 0), thickness=5, trim_tail_ratio=0.4):
    """
    把 3D 点投影并画折线到 img_bgr 上（不写文件，返回新图）。
    用于 upper_only_skel_rib_both.png 的最小输出模式。
    """
    if points_3d is None or intrinsics is None or img_bgr is None:
        return None
    pts = np.asarray(points_3d, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] < 2:
        return img_bgr.copy()
    # optional tail trim (keep head part)
    r = float(trim_tail_ratio or 0.0)
    r = max(0.0, min(0.95, r))
    k = int(round(len(pts) * (1.0 - r)))
    k = max(2, min(len(pts), k))
    pts = pts[:k]
    fx = float(intrinsics.get("fx"))
    fy = float(intrinsics.get("fy", fx))
    cx = float(intrinsics.get("cx"))
    cy = float(intrinsics.get("cy"))
    out = img_bgr.copy()
    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
    valid = Z > 0.1
    X, Y, Z = X[valid], Y[valid], Z[valid]
    if len(Z) < 2:
        return out
    u = (fx * X / Z) + cx
    v = (fy * Y / Z) + cy
    h, w = out.shape[:2]
    pts_2d = []
    for i in range(len(u)):
        x_p = int(u[i])
        y_p = int(v[i])
        if 0 <= x_p < w and 0 <= y_p < h:
            pts_2d.append((x_p, y_p))
    for i in range(len(pts_2d) - 1):
        cv2.line(out, pts_2d[i], pts_2d[i + 1], color, int(thickness))
    return out

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

def extract_right_costal_margin(fitter, vertices, faces, img_path, intrinsics):
    """
    Extracts the right costal margin (rib line) using geometric heuristics + binning + NORMAL + VISIBILITY filtering.
    Returns: (N, 3) array of 3D points forming the line.
    """
    try:
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

        # right side: take positive lr half (body-right)
        is_right = coord_lr > 0.0

        relevant_z = vertices[mask_rib & is_right, 2]
        if len(relevant_z) == 0:
            # if arm naming failed and lr_dir fallback wrong, try the opposite side
            is_right = coord_lr < 0.0
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

        # 5. End point: lateral extreme on right + most inferior in that lateral band
        lat_thr = float(np.percentile(fr_lr, 85.0))
        lat_mask = fr_lr >= lat_thr
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
        if lr_end <= lr_start:
            lr_end = float(np.max(fr_lr))

        num_bins = 40
        lr_step = (lr_end - lr_start) / float(num_bins)
        lr_step = max(lr_step, 1e-6)

        costal_margin_points = [xiphoid_pt]
        current_lr0 = lr_start
        for _ in range(num_bins):
            current_lr1 = current_lr0 + lr_step
            in_bin = (fr_lr >= current_lr0) & (fr_lr < current_lr1)
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
        if SKEL_SAVE_DEBUG_PNGS:
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
             # ---- Minimal outputs by default (skip many debug png/mesh exports) ----
             save_debug_pngs = bool(SKEL_SAVE_DEBUG_PNGS)
             save_meshes = bool(SKEL_SAVE_MESHES)
             save_extra_npy = bool(SKEL_SAVE_EXTRA_NPY)

             # 1) Projected point cloud PNG (debug only)
             if save_debug_pngs:
                 print("--- Projecting SKEL to Image ---")
                 out_proj_path = _out_path(img_path, "_demo_projected", ".png")
                 project_and_draw(skel_verts, intrinsics_use, orig_img_bgr, out_proj_path)

             # 2) Skeleton overlay (always render in-memory if needed for upper_only; write file only if debug)
             skel_overlay_bgr = None
             try:
                 renderer_skel = Renderer(
                     focal_length=focal_length_val,
                     img_w=orig_img_bgr.shape[1],
                     img_h=orig_img_bgr.shape[0],
                     faces=skel_faces,
                     same_mesh_color=True,
                 )
                 skel_overlay = renderer_skel.render_front_view(
                     [skel_verts],
                     bg_img_rgb=orig_img_bgr[:, :, ::-1].copy(),
                 )
                 skel_overlay_bgr = skel_overlay[:, :, ::-1]
                 if save_debug_pngs:
                     out_skel_path = _out_path(img_path, "_demo_skel_overlay", ".png")
                     cv2.imwrite(out_skel_path, skel_overlay_bgr)
                     print(f"Skeleton Overlay saved to: {out_skel_path}")
                 renderer_skel.delete()
             except Exception as e:
                 print(f"[warn] Skeleton overlay render failed (skip): {e}")
                 skel_overlay_bgr = None

             # 3) Abdomen / Thorax / Lumbar overlays (debug only)
             if save_debug_pngs:
                 if abd_verts is not None and abd_faces is not None:
                     try:
                         print("--- Rendering Abdomen Mesh ---")
                         renderer_abd = Renderer(
                             focal_length=focal_length_val,
                             img_w=orig_img_bgr.shape[1],
                             img_h=orig_img_bgr.shape[0],
                             faces=abd_faces,
                             same_mesh_color=True,
                         )
                         abd_overlay = renderer_abd.render_front_view([abd_verts], bg_img_rgb=orig_img_bgr[:, :, ::-1].copy())
                         out_abd_path = _out_path(img_path, "_demo_abdomen_overlay", ".png")
                         cv2.imwrite(out_abd_path, abd_overlay[:, :, ::-1])
                         print(f"Abdomen Overlay saved to: {out_abd_path}")
                         renderer_abd.delete()
                     except Exception as e:
                         print(f"[warn] Abdomen overlay render failed (skip): {e}")

                 try:
                     print("--- Rendering Thorax (ID 12) & Lumbar (ID 11) ---")
                     tx_verts, tx_faces = extract_specific_bone_mesh(fitter, skel_verts, skel_faces, 'thorax', img_path, threshold=0.1)
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

                     print("--- Rendering Thorax + Lumbar Combined ---")
                     # export combined mesh/overlay only in debug or mesh export mode
                     if save_debug_pngs or save_meshes:
                         tl_verts, tl_faces = extract_combined_mesh(fitter, skel_verts, skel_faces, ['thorax', 'lumbar_body'], img_path, suffix="_demo_thorax_lumbar_combined")
                         if tl_verts is not None:
                             if save_debug_pngs:
                                 renderer_tl = Renderer(focal_length=focal_length_val, img_w=orig_img_bgr.shape[1], img_h=orig_img_bgr.shape[0],
                                                        faces=tl_faces, same_mesh_color=True)
                                 tl_overlay = renderer_tl.render_front_view([tl_verts], bg_img_rgb=orig_img_bgr[:, :, ::-1].copy())
                                 out_tl_path = _out_path(img_path, "_demo_thorax_lumbar_combined", ".png")
                                 cv2.imwrite(out_tl_path, tl_overlay[:, :, ::-1])
                                 print(f"Thorax+Lumbar Combined Overlay saved to: {out_tl_path}")
                                 renderer_tl.delete()
                 except Exception as e:
                     print(f"[warn] Thorax/Lumbar debug render failed (skip): {e}")

             # 3.5) Post-SKEL trajectory extraction for kidney/spine (does NOT change CLiFF+SKEL)
             traj_kind = str(getattr(args, "traj_kind", "gallbladder") or "gallbladder").strip().lower()
             kidney_side = str(getattr(args, "kidney_side", "right") or "right").strip().lower()
             if kidney_side not in ("right", "left"):
                 kidney_side = "right"

             spine_uv_stitched = None
             kidney_line_uv_shifted_stitched = None

             if traj_kind in ("spine", "kidney"):
                 if _kidney_post is None:
                     print("[warn] kidney/spine requested but kidney_spine_geometry.py is not importable; skip.")
                 else:
                     # ---- spine curve (for spine-only + kidney start reference) ----
                     try:
                         spine_curve_xyz = _kidney_post.extract_spine_centerline_curve_xyz(
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
                             spine_uv_stitched = project_points_to_uv(
                                 np.asarray(spine_curve_xyz, dtype=np.float32),
                                 intrinsics_use,
                             ).astype(np.float32)
                     except Exception as e:
                         print(f"[warn] spine curve extraction failed (skip): {e}")

                         # save spine-only overlay (stitched coords)
                     try:
                         if spine_uv_stitched is not None and skel_overlay_bgr is not None:
                             spine_img = skel_overlay_bgr.copy()
                             spine_uv_visible = spine_uv_visible_half(spine_uv_stitched)
                             if len(spine_uv_visible) >= 2:
                                 spine_img = _draw_uv_polyline(spine_img, spine_uv_visible, color=(255, 0, 0), thickness=3)
                             out_sp = _out_path(img_path, "_demo_skel_overlay_spine_curve_only", ".png")
                             cv2.imwrite(out_sp, spine_img)
                             print(f"[export] Saved: {out_sp}")
                     except Exception as e:
                         print(f"[warn] spine-only overlay failed (skip): {e}")

                     # save spine straight scan line overlay (visible half endpoints, traj_kind=spine)
                     try:
                         if traj_kind == "spine" and spine_uv_stitched is not None and skel_overlay_bgr is not None:
                             line_uv = spine_uv_scan_line_endpoints(spine_uv_stitched)
                             if line_uv is not None:
                                 scan_img = skel_overlay_bgr.copy()
                                 scan_img = _draw_uv_polyline(
                                     scan_img, line_uv, color=(0, 255, 255), thickness=4
                                 )
                                 out_straight = _out_path(
                                     img_path, "_demo_skel_overlay_spine_scan_line_straight", ".png"
                                 )
                                 cv2.imwrite(out_straight, scan_img)
                                 print(f"[export] Saved: {out_straight}")
                     except Exception as e:
                         print(f"[warn] spine straight scan line overlay failed (skip): {e}")

                     # ---- kidney offset-only line ----
                     if traj_kind == "kidney":
                         # 与肋缘可视化一致：左肾默认用裁掉前 40% 的肋缘做交点(肋缘线延长∩脊柱→下移作起点)
                         kidney_rib_trim = float(getattr(args, "rib_trim_head_ratio", 0.0))
                         if kidney_side == "left" and kidney_rib_trim <= 0.0:
                             kidney_rib_trim = 0.40
                         kidney_start_mode = str(
                             getattr(args, "kidney_start_mode", "thorax_spine_lowest") or "thorax_spine_lowest"
                         ).strip().lower()
                         try:
                             from kidney_spine_trajectory import build_kidney_offset_line_uv_shifted_by_mode

                             kidney_line_uv_shifted_stitched, kidney_start_mode = (
                                 build_kidney_offset_line_uv_shifted_by_mode(
                                     fitter=fitter,
                                     skel_verts=skel_verts,
                                     skel_faces=skel_faces,
                                     img_path=img_path,
                                     intrinsics=intrinsics_use,
                                     spine_curve_uv=spine_uv_stitched,
                                     spine_curve_xyz=spine_curve_xyz if "spine_curve_xyz" in locals() else None,
                                     kidney_start_mode=kidney_start_mode,
                                     cross_to_start_len_m=float(getattr(args, "cross_to_start_len_m", 0.03)),
                                     cross_to_start_dir=str(getattr(args, "cross_to_start_dir", "inferior")),
                                     end_up_along_rib_m=float(getattr(args, "end_up_along_rib_m", 0.0)),
                                     kidney_line_shift_px=float(getattr(args, "kidney_line_shift_px", 8.0)),
                                     rib_trim_head_ratio=kidney_rib_trim,
                                     kidney_side=kidney_side,
                                     end_lr_band_percentile=float(getattr(args, "end_lr_band_percentile", 15.0)),
                                 )
                             )
                             print(
                                 f"[kidney] start_mode={kidney_start_mode} side={kidney_side}"
                             )
                             if kidney_line_uv_shifted_stitched is not None:
                                 out_k_st = os.path.join(os.path.dirname(img_path), "kidney_line_uv_shifted.npy")
                                 np.save(out_k_st, kidney_line_uv_shifted_stitched.astype(np.float32))
                                 print(f"[export] Saved: {out_k_st}")
                         except Exception as e:
                             print(f"[warn] kidney offset-only line build failed (skip): {e}")

                         # save kidney offset-only overlay (stitched coords)
                         try:
                             if kidney_line_uv_shifted_stitched is not None and skel_overlay_bgr is not None:
                                 kid_img = skel_overlay_bgr.copy()
                                 kid_img = _draw_uv_polyline(kid_img, kidney_line_uv_shifted_stitched, color=(0, 0, 255), thickness=4)
                                 out_k = _out_path(img_path, "_demo_skel_overlay_kidney_offset_only", ".png")
                                 cv2.imwrite(out_k, kid_img)
                                 print(f"[export] Saved: {out_k}")
                         except Exception as e:
                             print(f"[warn] kidney offset-only overlay failed (skip): {e}")

                         # ===== DIAGNOSTIC: 把脊柱线 + 肋缘线 完整画到 2D overlay，看是哪条线的问题 =====
                         # 默认关闭(会重复提取肋缘/交点，浪费算力)；需要时 KIDNEY_DIAG=1 开启。
                         try:
                             if os.environ.get("KIDNEY_DIAG", "") and skel_overlay_bgr is not None:
                                 diag = skel_overlay_bgr.copy()
                                 sx = spine_curve_xyz if "spine_curve_xyz" in locals() else None

                                 cross_uv = None
                                 rib_def_uv = None
                                 try:
                                     cross_xyz, rib_def_3d = _kidney_post.find_spine_costal_margin_cross_point_on_spine_curve(
                                         fitter=fitter, skel_verts=skel_verts, skel_faces=skel_faces,
                                         img_path=img_path, intrinsics=intrinsics_use, spine_curve_xyz=sx,
                                     )
                                     if rib_def_3d is not None and len(rib_def_3d) >= 2:
                                         rib_def_uv = project_points_to_uv(np.asarray(rib_def_3d, dtype=np.float32), intrinsics_use).astype(np.float32)
                                     if cross_xyz is not None:
                                         cross_uv = project_points_to_uv(np.asarray(cross_xyz, dtype=np.float32).reshape(1, 3), intrinsics_use).astype(np.float32)[0]
                                 except Exception as e:
                                     print(f"[diag] cross/rib_default failed: {e}")

                                 rib_left_uv = None
                                 try:
                                     rib_left_3d = _kidney_post.extract_right_costal_margin(
                                         fitter, skel_verts, skel_faces, img_path, intrinsics_use, side="left")
                                     if rib_left_3d is not None and len(rib_left_3d) >= 2:
                                         rib_left_uv = project_points_to_uv(np.asarray(rib_left_3d, dtype=np.float32), intrinsics_use).astype(np.float32)
                                 except Exception as e:
                                     print(f"[diag] rib_left failed: {e}")

                                 if spine_uv_stitched is not None:
                                     diag = _draw_uv_polyline(diag, spine_uv_stitched, color=(255, 0, 0), thickness=2)
                                     for p in spine_uv_stitched.astype(int):
                                         cv2.circle(diag, (int(p[0]), int(p[1])), 3, (255, 0, 0), -1)
                                 if rib_def_uv is not None:
                                     diag = _draw_uv_polyline(diag, rib_def_uv, color=(0, 255, 0), thickness=2)
                                     for p in rib_def_uv.astype(int):
                                         cv2.circle(diag, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)
                                 if rib_left_uv is not None:
                                     for p in rib_left_uv.astype(int):
                                         cv2.circle(diag, (int(p[0]), int(p[1])), 5, (255, 0, 255), -1)
                                 if cross_uv is not None:
                                     cv2.circle(diag, (int(cross_uv[0]), int(cross_uv[1])), 9, (0, 255, 255), -1)
                                 if kidney_line_uv_shifted_stitched is not None:
                                     diag = _draw_uv_polyline(diag, kidney_line_uv_shifted_stitched, color=(0, 0, 255), thickness=3)

                                 out_diag = _out_path(img_path, "_demo_kidney_DIAG", ".png")
                                 cv2.imwrite(out_diag, diag)
                                 print(f"[export] Saved: {out_diag}")
                                 print(f"[diag] spine_pts={0 if spine_uv_stitched is None else len(spine_uv_stitched)} "
                                       f"rib_default_pts={0 if rib_def_uv is None else len(rib_def_uv)} "
                                       f"rib_left_pts={0 if rib_left_uv is None else len(rib_left_uv)} "
                                       f"cross_uv={None if cross_uv is None else cross_uv.tolist()}")
                         except Exception as e:
                             print(f"[warn] kidney DIAG overlay failed (skip): {e}")

             # 3.6) Thorax spine lowest overlay (always saved alongside trajectory exports)
             try:
                 from kidney_spine_trajectory import save_thorax_spine_lowest_overlay

                 _spine_for_lowest = (
                     spine_curve_xyz
                     if "spine_curve_xyz" in locals() and spine_curve_xyz is not None
                     else None
                 )
                 out_thorax_low = save_thorax_spine_lowest_overlay(
                     fitter=fitter,
                     skel_verts=skel_verts,
                     skel_faces=skel_faces,
                     img_path=img_path,
                     orig_img_bgr=orig_img_bgr,
                     focal_length_val=focal_length_val,
                     intrinsics=intrinsics_use,
                     spine_curve_xyz=_spine_for_lowest,
                     out_path=_out_path(img_path, "_demo_thorax_spine_lowest", ".png"),
                     save_xyz_npy=True,
                 )
                 if out_thorax_low:
                     print(f"[export] Saved: {out_thorax_low}")
             except Exception as e:
                 print(f"[warn] thorax spine lowest overlay failed (skip): {e}")

             # 3.7) Lumbar overlay with highest lumbar spine point (L1 superior on mesh)
             try:
                 from kidney_spine_trajectory import save_lumbar_spine_highest_overlay

                 out_lumbar_high = save_lumbar_spine_highest_overlay(
                     fitter=fitter,
                     skel_verts=skel_verts,
                     skel_faces=skel_faces,
                     img_path=img_path,
                     orig_img_bgr=orig_img_bgr,
                     focal_length_val=focal_length_val,
                     out_path=_out_path(img_path, "_demo_lumbar_spine_highest", ".png"),
                     save_xyz_npy=True,
                 )
                 if out_lumbar_high:
                     print(f"[export] Saved: {out_lumbar_high}")
             except Exception as e:
                 print(f"[warn] lumbar spine highest overlay failed (skip): {e}")

             # 3.8) Thorax mesh left-lowest point (direct on thorax mesh; auxiliary)
             try:
                 from kidney_spine_trajectory import save_thorax_mesh_left_lowest_overlay

                 out_thorax_ll = save_thorax_mesh_left_lowest_overlay(
                     fitter=fitter,
                     skel_verts=skel_verts,
                     skel_faces=skel_faces,
                     img_path=img_path,
                     orig_img_bgr=orig_img_bgr,
                     focal_length_val=focal_length_val,
                     intrinsics=intrinsics_use,
                     out_path=_out_path(img_path, "_demo_thorax_mesh_left_lowest", ".png"),
                     save_xyz_npy=True,
                 )
                 if out_thorax_ll:
                     print(f"[export] Saved: {out_thorax_ll}")
             except Exception as e:
                 print(f"[warn] thorax mesh left-lowest overlay failed (skip): {e}")

             # 4) Extract ribline (needed for uv + upper_only)
             print("--- Extracting Right Costal Margin ---")
             rib_line_3d = extract_right_costal_margin(fitter, skel_verts, skel_faces, img_path, intrinsics_use)
             if rib_line_3d is not None:
                 # Do NOT generate these debug pngs by default:
                 # *_demo_rib_line.png, *_demo_skel_overlay.png, *_demo_thorax_lumbar_combined.png,
                 # *_ribline_axes_dbg.png, rgb_ribline_uv.png, rgb_ribline_uv_both.png
                 if save_debug_pngs:
                     out_rib_path = _out_path(img_path, "_demo_rib_line", ".png")
                     project_and_draw_lines(rib_line_3d, intrinsics_use, orig_img_bgr, out_rib_path, color=(255, 0, 0), thickness=5, trim_tail_ratio=0.4)

                 # ===== Export for downstream "回投到点云/深度 -> 目标位姿" =====
                 # Save 3D ribline in camera coordinates (optional)
                 if save_extra_npy:
                     rib_xyz_path = os.path.join(os.path.dirname(img_path), "ribline_xyz_cam.npy")
                     np.save(rib_xyz_path, np.asarray(rib_line_3d, dtype=np.float32))
                     print(f"[export] Saved: {rib_xyz_path}")

                 # Save 2D pixel ribline (optional)
                 rib_uv = project_points_to_uv(np.asarray(rib_line_3d, dtype=np.float32), intrinsics_use)
                 if save_extra_npy:
                     rib_uv_path = os.path.join(os.path.dirname(img_path), "ribline_uv.npy")
                     np.save(rib_uv_path, rib_uv)
                     print(f"[export] Saved: {rib_uv_path}")

                 # --- NEW: 2D-only postprocess per your diagram ---
                 # 黑线：对“原始肋缘折线”做 PCA 得到的直线（主方向）
                 # 绿线：黑线的法向量（互相垂直），并选择符号使其指向图像向下 (v+)
                 # 红线：沿绿线方向把“原始肋缘折线”整体平移 shift_px 得到
         # NOTE: we do NOT trim near the xiphoid region here (unless rib_trim_head_ratio>0 is explicitly set)
                 trim_head_ratio = float(getattr(args, "rib_trim_head_ratio", 0.0))
                 # 仅左肾扫描：默认裁掉肋缘点序前 40%(剑突/上段)，只保留下外侧肋缘；胆囊等保持原逻辑
                 if traj_kind == "kidney" and kidney_side == "left" and trim_head_ratio <= 0.0:
                     trim_head_ratio = 0.40
                 rib_uv_draw = _trim_head_uv(rib_uv, trim_head_ratio=trim_head_ratio) if trim_head_ratio > 0 else rib_uv

                 pca_dir, pca_line = _pca_dir_and_line_uv(rib_uv_draw)
                 n_down = _normal_down_from_dir(pca_dir)  # 绿法向量

                 # NEW: 单向曲线拟合（2D），用于绘制黄色线，避免“走一半又怪上去”
                 rib_uv_fit = _fit_monotonic_curve_uv(rib_uv_draw, pca_dir, n_down)
                 # ribline_uv_fit.npy：仅在需要调试/或没有 shifted 输出时保存
                 if save_extra_npy or float(getattr(args, "rib_shift_px", 0.0)) == 0.0:
                     try:
                         rib_uv_fit_path = os.path.join(os.path.dirname(img_path), "ribline_uv_fit.npy")
                         np.save(rib_uv_fit_path, rib_uv_fit.astype(np.float32))
                         print(f"[export] Saved: {rib_uv_fit_path}")
                     except Exception:
                         pass

                 # 仅导出“向下法向量”（用于你验证偏移方向）；不再导出/绘制黑色 PCA 线
                 if save_extra_npy:
                     try:
                         rib_uv_normal_path = os.path.join(os.path.dirname(img_path), "ribline_uv_normal_down.npy")
                         np.save(rib_uv_normal_path, n_down.astype(np.float32))
                         print(f"[export] Saved: {rib_uv_normal_path}")
                     except Exception:
                         pass

                 shift_px = float(getattr(args, "rib_shift_px", 0.0))
                 rib_uv_shift = None
                 rib_uv_shift_draw = None
                 if shift_px != 0.0:
                     # 关键：沿“绿法向量”平移“原始肋缘折线”
                     rib_uv_shift = rib_uv + n_down.reshape(1, 2) * float(shift_px)
                     # 绘制/upper_only 使用“拟合后的单向曲线”再偏移（更稳定）
                     rib_uv_shift_draw = rib_uv_fit + n_down.reshape(1, 2) * float(shift_px)

                     # 轨迹生成会用到：默认仍保存
                     rib_uv_shift_path = os.path.join(os.path.dirname(img_path), "ribline_uv_shifted.npy")
                     np.save(rib_uv_shift_path, rib_uv_shift.astype(np.float32))
                     print(f"[export] Saved: {rib_uv_shift_path}")
                     try:
                         rib_uv_shift_fit_path = os.path.join(os.path.dirname(img_path), "ribline_uv_shifted_fit.npy")
                         np.save(rib_uv_shift_fit_path, rib_uv_shift_draw.astype(np.float32))
                         print(f"[export] Saved: {rib_uv_shift_fit_path}")
                     except Exception:
                         pass

                 # 画图时黑线/绿线应是直线（你的要求），所以单独用 pca_line 与 arrow 表示

                 # 只在内存里构建“skeleton + rib 两条线”叠图，供 upper_only 使用；不落盘（除非 debug）
                 skel_both_img = None
                 try:
                     if skel_overlay_bgr is not None:
                         skel_both_img = skel_overlay_bgr.copy()
                         skel_both_img = _draw_uv_polyline(skel_both_img, rib_uv_fit, color=(0, 255, 255), thickness=3)
                         if rib_uv_shift_draw is not None:
                             skel_both_img = _draw_uv_polyline(skel_both_img, rib_uv_shift_draw, color=(0, 0, 255), thickness=3)
                         # 可选：debug 时才保存
                         if save_debug_pngs:
                             out_skel_both = _out_path(img_path, "_demo_skel_overlay_rib_axes", ".png")
                             cv2.imwrite(out_skel_both, skel_both_img)
                             print(f"[export] Saved: {out_skel_both}")
                 except Exception as e:
                     if save_debug_pngs:
                         print(f"[warn] build skel_both overlay failed (skip): {e}")

                 # rgb_ribline_uv.png / rgb_ribline_uv_both.png 都属于 debug 可视化：默认不生成
                 if save_debug_pngs:
                     try:
                         uv_vis = orig_img_bgr.copy()
                         pts = rib_uv.astype(np.int32).reshape((-1, 1, 2))
                         cv2.polylines(uv_vis, [pts], isClosed=False, color=(255, 0, 0), thickness=3)
                         uv_vis_path = os.path.join(os.path.dirname(img_path), "rgb_ribline_uv.png")
                         cv2.imwrite(uv_vis_path, uv_vis)
                         print(f"[export] Saved: {uv_vis_path}")
                     except Exception as e:
                         print(f"[export] Failed to save rgb_ribline_uv.png: {e}")

                 # --- NEW: export "upper-only" overlays by undoing stitch transforms (optional) ---
                 if getattr(args, "export_upper_only", False):
                     try:
                         upper_path = args.upper_img
                         lower_path = args.lower_img
                         rot = args.stitch_rotate
                         mode = args.stitch_mode
                         overlap_ratio = float(args.stitch_overlap_ratio)
                         lb_w = int(args.stitch_letterbox_w)
                         lb_h = int(args.stitch_letterbox_h)

                         if not (upper_path and os.path.exists(upper_path)):
                             raise RuntimeError("upper_img not found")
                         if not (lower_path and os.path.exists(lower_path)):
                             raise RuntimeError("lower_img not found")

                         upper0 = cv2.imread(upper_path)
                         lower0 = cv2.imread(lower_path)
                         if upper0 is None or lower0 is None:
                             raise RuntimeError("failed to read upper/lower images")

                         upper_r = _rotate_img(upper0, rot)
                         lower_r = _rotate_img(lower0, rot)

                         # reconstruct stitch content size before letterbox (must match stitch_two_rgb.py)
                         if mode == "vertical":
                             # match width: resize lower to upper width
                             if lower_r.shape[1] != upper_r.shape[1]:
                                 new_h = int(round(lower_r.shape[0] * (float(upper_r.shape[1]) / float(lower_r.shape[1]))))
                                 lower_r = cv2.resize(lower_r, (upper_r.shape[1], new_h), interpolation=cv2.INTER_AREA)
                             ov = int(round(lower_r.shape[0] * overlap_ratio))
                             ov = max(0, min(lower_r.shape[0] - 1, ov))
                             lower_crop = lower_r[ov:, :, :]
                             content_w = upper_r.shape[1]
                             content_h = upper_r.shape[0] + lower_crop.shape[0]
                             upper_h = upper_r.shape[0]
                         else:
                             # horizontal: match height, crop left of lower by ratio
                             if lower_r.shape[0] != upper_r.shape[0]:
                                 new_w = int(round(lower_r.shape[1] * (float(upper_r.shape[0]) / float(lower_r.shape[0]))))
                                 lower_r = cv2.resize(lower_r, (new_w, upper_r.shape[0]), interpolation=cv2.INTER_AREA)
                             ov = int(round(lower_r.shape[1] * overlap_ratio))
                             ov = max(0, min(lower_r.shape[1] - 1, ov))
                             lower_crop = lower_r[:, ov:, :]
                             content_w = upper_r.shape[1] + lower_crop.shape[1]
                             content_h = upper_r.shape[0]
                             upper_h = upper_r.shape[0]

                         # 最小输出模式：直接用内存里的 overlay（不依赖磁盘中的 *_demo_*.png）
                         src_overlay = None
                         if 'skel_both_img' in locals() and skel_both_img is not None:
                             src_overlay = skel_both_img
                         elif skel_overlay_bgr is not None:
                             # 退化：至少保证能生成 upper_only skeleton
                             src_overlay = skel_overlay_bgr
                         else:
                             # 退化：最后使用原图（仍可走 inverse-transform，便于调试）
                             src_overlay = orig_img_bgr

                         unlb = _undo_letterbox(src_overlay, content_w, content_h, lb_w, lb_h)

                         # crop to upper part (remove stitched lower body)
                         if mode == "vertical":
                             upper_only_r = unlb[:upper_h, :, :]
                         else:
                             upper_only_r = unlb[:, :upper_r.shape[1], :]

                         # inverse rotate back to match original upper image size
                         upper_only = _inv_rotate_img(upper_only_r, rot)
                         # ensure exact size equals upper0
                         upper_only = cv2.resize(upper_only, (upper0.shape[1], upper0.shape[0]), interpolation=cv2.INTER_AREA)

                         out_upper = os.path.join(os.path.dirname(img_path), "upper_only_skel_rib_both.png")
                         cv2.imwrite(out_upper, upper_only)
                         print(f"[export] Saved: {out_upper}")

                        # --- ALSO export UV trajectories in *upper0* image coordinate system ---
                         # 注意：rib_uv_fit / rib_uv_shift_draw 是在 stitched_upright(letterbox 输出)坐标系里的点
                         try:
                            # 1) gallbladder ribline (existing behavior)
                            if "rib_uv_fit" in locals() and rib_uv_fit is not None and len(rib_uv_fit) > 0:
                                uv_fit_content = _undo_letterbox_points(rib_uv_fit, content_w, content_h, lb_w, lb_h)
                                if mode == "vertical":
                                    mask = (uv_fit_content[:, 1] >= 0) & (uv_fit_content[:, 1] < float(upper_h))
                                    uv_fit_upper_r = uv_fit_content[mask]
                                else:
                                    mask = (uv_fit_content[:, 0] >= 0) & (uv_fit_content[:, 0] < float(upper_r.shape[1]))
                                    uv_fit_upper_r = uv_fit_content[mask]
                                uv_fit_upper0 = _inv_rotate_points_to_original(uv_fit_upper_r, rot, upper0.shape[0], upper0.shape[1])
                                uv_fit_upper0_path = os.path.join(os.path.dirname(img_path), "ribline_uv_upper_fit.npy")
                                np.save(uv_fit_upper0_path, uv_fit_upper0.astype(np.float32))
                                print(f"[export] Saved: {uv_fit_upper0_path}")

                            if "rib_uv_shift_draw" in locals() and rib_uv_shift_draw is not None and len(rib_uv_shift_draw) > 0:
                                uv_shift_content = _undo_letterbox_points(rib_uv_shift_draw, content_w, content_h, lb_w, lb_h)
                                if mode == "vertical":
                                    mask = (uv_shift_content[:, 1] >= 0) & (uv_shift_content[:, 1] < float(upper_h))
                                    uv_shift_upper_r = uv_shift_content[mask]
                                else:
                                    mask = (uv_shift_content[:, 0] >= 0) & (uv_shift_content[:, 0] < float(upper_r.shape[1]))
                                    uv_shift_upper_r = uv_shift_content[mask]
                                uv_shift_upper0 = _inv_rotate_points_to_original(uv_shift_upper_r, rot, upper0.shape[0], upper0.shape[1])
                                uv_shift_upper0_path = os.path.join(os.path.dirname(img_path), "ribline_uv_upper_shifted_fit.npy")
                                np.save(uv_shift_upper0_path, uv_shift_upper0.astype(np.float32))
                                print(f"[export] Saved: {uv_shift_upper0_path}")

                            # 2) spine-only uv in upper coords (scan segment = visible half, same as overlay PNG)
                            if "spine_uv_stitched" in locals() and spine_uv_stitched is not None and len(spine_uv_stitched) > 0:
                                spine_uv_export = (
                                    spine_uv_visible_half(spine_uv_stitched)
                                    if str(traj_kind).strip().lower() == "spine"
                                    else spine_uv_stitched
                                )
                                if str(traj_kind).strip().lower() == "spine":
                                    n_vis = len(spine_uv_export)
                                    endpoints = spine_uv_scan_line_endpoints(spine_uv_stitched)
                                    if endpoints is not None:
                                        spine_uv_export = endpoints
                                    print(
                                        f"[export] spine_curve_uv_upper: visible half "
                                        f"[{len(spine_uv_stitched) // 2}:] ({n_vis}/{len(spine_uv_stitched)} pts) "
                                        f"-> 2 endpoints (kidney-style); densify in path_stroke_preview_uv"
                                    )
                                uv_sp_content = _undo_letterbox_points(spine_uv_export, content_w, content_h, lb_w, lb_h)
                                if mode == "vertical":
                                    mask = (uv_sp_content[:, 1] >= 0) & (uv_sp_content[:, 1] < float(upper_h))
                                    uv_sp_upper_r = uv_sp_content[mask]
                                else:
                                    mask = (uv_sp_content[:, 0] >= 0) & (uv_sp_content[:, 0] < float(upper_r.shape[1]))
                                    uv_sp_upper_r = uv_sp_content[mask]
                                uv_sp_upper0 = _inv_rotate_points_to_original(uv_sp_upper_r, rot, upper0.shape[0], upper0.shape[1])
                                out_sp_uv = os.path.join(os.path.dirname(img_path), "spine_curve_uv_upper.npy")
                                np.save(out_sp_uv, uv_sp_upper0.astype(np.float32))
                                print(f"[export] Saved: {out_sp_uv}")

                            # 3) kidney offset-only uv in upper coords
                            if "kidney_line_uv_shifted_stitched" in locals() and kidney_line_uv_shifted_stitched is not None and len(kidney_line_uv_shifted_stitched) > 0:
                                uv_k_content = _undo_letterbox_points(kidney_line_uv_shifted_stitched, content_w, content_h, lb_w, lb_h)
                                if mode == "vertical":
                                    mask = (uv_k_content[:, 1] >= 0) & (uv_k_content[:, 1] < float(upper_h))
                                    uv_k_upper_r = uv_k_content[mask]
                                else:
                                    mask = (uv_k_content[:, 0] >= 0) & (uv_k_content[:, 0] < float(upper_r.shape[1]))
                                    uv_k_upper_r = uv_k_content[mask]
                                uv_k_upper0 = _inv_rotate_points_to_original(uv_k_upper_r, rot, upper0.shape[0], upper0.shape[1])
                                out_k_uv = os.path.join(os.path.dirname(img_path), "kidney_line_uv_upper_shifted.npy")
                                np.save(out_k_uv, uv_k_upper0.astype(np.float32))
                                print(f"[export] Saved: {out_k_uv}")

                         except Exception as e:
                            print(f"[export] upper_only uv export failed: {e}")
                     except Exception as e:
                         print(f"[export] upper_only export failed: {e}")

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
    parser.add_argument("--rib_shift_px", type=float, default=0.0,
                        help="沿肋缘线切向方向向下平移的像素距离（在 uv 上操作）。0=不生成偏移线")
    parser.add_argument("--rib_trim_head_ratio", type=float, default=0.0,
                        help="可选：隐藏剑突附近的拐角：丢弃肋缘线开头的比例（0~1）。默认 0（不隐藏）")
    parser.add_argument("--rib_normal_arrow_len_px", type=float, default=120.0,
                        help="绿法向量箭头长度（像素）")

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

    # Post-SKEL trajectory selection (CLiFF+SKEL is identical; only extraction differs)
    parser.add_argument("--traj_kind", default="gallbladder", choices=["gallbladder", "kidney", "spine"])
    parser.add_argument("--kidney_side", default="left", choices=["right", "left"])
    parser.add_argument(
        "--kidney_start_mode",
        default="thorax_spine_lowest",
        choices=["legacy", "thorax_spine_lowest"],
        help="Kidney trajectory start: legacy (rib-cross+spine advance) or thorax_spine_lowest (left kidney only).",
    )
    parser.add_argument("--kidney_line_shift_px", type=float, default=8.0)
    parser.add_argument("--cross_to_start_len_m", type=float, default=0.02)
    parser.add_argument("--cross_to_start_dir", default="inferior", choices=["inferior", "superior"])
    parser.add_argument("--end_up_along_rib_m", type=float, default=0.0)

    parser.add_argument("--spine_num_pts", type=int, default=30)
    parser.add_argument("--spine_mid_abs_lr_percentile", type=float, default=8.0)
    parser.add_argument("--spine_smooth_win", type=int, default=5)
    parser.add_argument("--spine_region", default="lumbar_to_pelvis", choices=["lumbar_to_pelvis", "thorax_to_pelvis"])
    parser.add_argument("--spine_inf_margin_m", type=float, default=0.03)

    args = parser.parse_args()
    
    main(args)