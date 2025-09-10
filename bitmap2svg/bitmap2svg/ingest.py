from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image
import cv2

@dataclass
class LoadedImage:
    pil: Image.Image
    rgba: np.ndarray          # HxWx4 uint8
    gray: np.ndarray          # HxW uint8
    edges: np.ndarray         # HxW float32 in [0,1]
    size: Tuple[int, int]     # (W,H)

def _sobel_edges(arr_u8: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(arr_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(arr_u8, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    m = mag.max()
    return mag / m if m > 0 else mag

def load(path: str | Path) -> LoadedImage:
    pil = Image.open(path).convert("RGBA")
    rgba = np.array(pil)
    gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
    edges = _sobel_edges(gray)
    H, W = gray.shape
    return LoadedImage(
        pil=pil,
        rgba=rgba,
        gray=gray,
        edges=edges,
        size=(W, H),
    )