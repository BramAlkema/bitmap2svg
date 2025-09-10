from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import cv2
from PIL import Image
import io
from cairosvg import svg2png  # pip install cairosvg

@dataclass
class Metrics:
    ssim: float
    edge_iou: float
    bytes: int

def _render_svg(svg_text: str, size: tuple[int,int], scale: int) -> np.ndarray:
    W,H = size
    png_bytes = svg2png(bytestring=svg_text.encode("utf-8"), output_width=W*scale, output_height=H*scale)
    arr = np.array(Image.open(io.BytesIO(png_bytes)).convert("L"))
    return arr

def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.astype(np.float32) / 255.0
    b64 = b.astype(np.float32) / 255.0
    mu_a, mu_b = a64.mean(), b64.mean()
    va, vb = a64.var(), b64.var()
    cov = ((a64 - mu_a)*(b64 - mu_b)).mean()
    C1, C2 = 0.01**2, 0.03**2
    return float(((2*mu_a*mu_b + C1)*(2*cov + C2))/((mu_a**2 + mu_b**2 + C1)*(va + vb + C2) + 1e-8))

def _edges(u8: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(u8, cv2.CV_32F, 0, 1, ksize=3)
    m = np.hypot(gx, gy)
    m = (m > (0.2*m.max())).astype(np.uint8)
    return m

def _edge_iou(a_edges: np.ndarray, b_edges: np.ndarray) -> float:
    inter = np.logical_and(a_edges>0, b_edges>0).sum()
    union = np.logical_or(a_edges>0, b_edges>0).sum()
    return float(inter/union) if union else 1.0

def evaluate(svg_text: str, img, cfg) -> Dict[str, Any]:
    scale = getattr(cfg, "ssim_scale", 4)
    tgt = cv2.resize(img.gray, (img.size[0]*scale, img.size[1]*scale), interpolation=cv2.INTER_NEAREST)
    ren = _render_svg(svg_text, img.size, scale=scale)
    ssim = _ssim(ren, tgt)
    iou = _edge_iou(_edges(ren), _edges(tgt))
    return {"ssim": ssim, "edge_iou": iou, "bytes": len(svg_text.encode("utf-8"))}