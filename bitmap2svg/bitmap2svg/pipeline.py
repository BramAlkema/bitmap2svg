"""Core vectorisation pipeline with caching and batching utilities."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Tuple

import numpy as np

from .bwtrace_cv2 import trace_bitmap

from .bezier import fit as bezier_fit
from .config import Settings
from .ingest import LoadedImage
from .qa import evaluate
from .segment import mask_to_bw, to_layers
from .simplify import rdp_all
from .svg_io import compose
from .vector_critic import snap, SnapCfg


@lru_cache(maxsize=128)
def _trace_cached(bw_bytes: bytes, width: int, height: int) -> List[List[Tuple[float, float]]]:
    """Trace binary image data with OpenCV and cache the result."""
    bw_u8 = np.frombuffer(bw_bytes, dtype=np.uint8).reshape((height, width))
    return trace_bitmap(bw_u8)


def _trace(bw_u8: np.ndarray) -> List[List[Tuple[float, float]]]:
    """Wrapper around the cached trace call using array data."""
    h, w = bw_u8.shape
    return _trace_cached(bw_u8.tobytes(), w, h)


def vectorise(img: LoadedImage, cfg: Settings):
    """Vectorise a single loaded image into an SVG result."""
    layers = to_layers(img, cfg)
    composed = []
    for layer in layers:
        bw = mask_to_bw(img, layer)
        seeds = _trace(bw)
        polys = rdp_all(seeds, epsilon=cfg.rdp_epsilon)
        snapped = snap(polys, cfg.snap)
        items = [(t, p) for (t, p) in snapped if t in ("circle", "rect")]
        poly_left = [p for (t, p) in snapped if t == "poly"]
        bez = bezier_fit(poly_left, cfg.bezier)
        items.extend(bez)
        composed.append((items, layer.color))
    svg = compose(composed, img.size, cfg.svg).minified
    metrics = evaluate(svg, img, cfg.qa)
    return type("SVGResult", (), {"svg_min": svg, "svg_pretty": svg, "metrics": metrics})


def vectorise_batch(images: Iterable[LoadedImage], cfg: Settings):
    """Vectorise a batch of images, leveraging cached traces."""
    return [vectorise(img, cfg) for img in images]

