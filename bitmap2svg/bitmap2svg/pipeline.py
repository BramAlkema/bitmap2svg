"""Core vectorisation pipeline with caching and batching utilities."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Tuple

import numpy as np
import potrace

from .bezier import fit as bezier_fit
from .config import Settings
from .ingest import LoadedImage
from .qa import evaluate
from .segment import mask_to_bw, to_layers
from .simplify import rdp_all
from .svg_io import compose
from .vector_critic import snap, SnapCfg


@lru_cache(maxsize=128)
def _potrace_trace_cached(bw_bytes: bytes, width: int, height: int) -> List[List[Tuple[float, float]]]:
    """Trace binary image data with Potrace and cache the result."""
    bw_u8 = np.frombuffer(bw_bytes, dtype=np.uint8).reshape((height, width))
    bmp = potrace.Bitmap(bw_u8 > 0)
    seeds: List[List[Tuple[float, float]]] = []
    for curve in bmp.trace().curves:
        pts: List[Tuple[float, float]] = []
        for seg in curve.segments:
            pts.append((float(seg.c.x), float(seg.c.y)))
        if len(pts) >= 3 and pts[0] != pts[-1]:
            pts.append(pts[0])
        if len(pts) >= 3:
            seeds.append(pts)
    return seeds


def _potrace_trace(bw_u8: np.ndarray) -> List[List[Tuple[float, float]]]:
    """Wrapper around the cached Potrace call using array data."""
    h, w = bw_u8.shape
    return _potrace_trace_cached(bw_u8.tobytes(), w, h)


def vectorise(img: LoadedImage, cfg: Settings):
    """Vectorise a single loaded image into an SVG result."""
    layers = to_layers(img, cfg)
    composed = []
    for layer in layers:
        bw = mask_to_bw(img, layer)
        seeds = _potrace_trace(bw)
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
    """Vectorise a batch of images, leveraging cached Potrace traces."""
    return [vectorise(img, cfg) for img in images]

