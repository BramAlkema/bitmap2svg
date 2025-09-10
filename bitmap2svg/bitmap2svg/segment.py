from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple
import numpy as np
import cv2

@dataclass
class Layer:
    mask: np.ndarray          # HxW uint8 (0/255)
    color: Tuple[int,int,int,int]  # RGBA

def _kmeans_palette(rgba: np.ndarray, k: int) -> np.ndarray:
    H, W, _ = rgba.shape
    X = rgba.reshape(-1, 4).astype(np.float32)
    # Ignore transparent pixels in clustering
    opaque = X[:,3] > 10
    Xo = X[opaque][:,:3]  # RGB only
    if len(Xo) == 0:
        # fallback single layer: black
        return np.array([[0,0,0,255]], dtype=np.uint8)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    k_eff = min(k, max(1, len(Xo)//200))  # guard against tiny images
    if k_eff < 1: k_eff = 1
    _ret, labels, centers = cv2.kmeans(Xo, k_eff, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = np.clip(np.round(centers), 0, 255).astype(np.uint8)
    # Add alpha=255
    centers_rgba = np.concatenate([centers, 255*np.ones((centers.shape[0],1), dtype=np.uint8)], axis=1)
    return centers_rgba

def to_layers(img, cfg) -> Iterable[Layer]:
    """Segment into k flat-colour layers using k-means in RGB space (logos are flat)."""
    rgba = img.rgba
    H, W, _ = rgba.shape
    palette = _kmeans_palette(rgba, cfg.k_colors)

    # Assign each pixel to nearest palette colour (RGB only, ignore transparent)
    rgb = rgba[:,:,:3].astype(np.int16)
    a = rgba[:,:,3]
    layers: List[Layer] = []
    for c in palette:
        crgb = c[:3].astype(np.int16)
        # L2 distance mask
        dist = np.sum((rgb - crgb)**2, axis=2)
        # For this simple approach, pick pixels closest to this center among all centers
        # Compute winner-takes-all once:
    # Precompute full assignment for all pixels
    centers = palette[:,:3].astype(np.int16)
    diff = rgb[:, :, None, :] - centers[None, None, :, :]
    dists = np.sum(diff*diff, axis=3)  # HxWxK
    assign = np.argmin(dists, axis=2)  # HxW

    for idx, c in enumerate(palette):
        mask = (assign == idx) & (a > 10)
        mask_u8 = np.zeros((H,W), dtype=np.uint8)
        mask_u8[mask] = 255
        # Small cleanup: open/close to kill speckles
        kernel = np.ones((3,3), np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
        if mask_u8.sum() == 0:
            continue
        layers.append(Layer(mask=mask_u8, color=tuple(int(x) for x in c)))
    # Sort big → small to draw background first
    layers.sort(key=lambda L: int(L.mask.sum()), reverse=True)
    return layers

def mask_to_bw(img, layer: Layer) -> np.ndarray:
    """Return a binary (0/255) image for Potrace."""
    # Ensure outer background is 0, shape is 255
    return layer.mask