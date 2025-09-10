from __future__ import annotations
from typing import Iterable
from .config import Settings
from .ingest import LoadedImage
from .segment import to_layers, mask_to_bw
from .simplify import rdp_all
from .vector_critic import snap, SnapCfg
from .bezier import fit as bezier_fit
from .svg_io import compose
from .qa import evaluate
import potrace

def _potrace_trace(bw_u8):
    bmp = potrace.Bitmap(bw_u8 > 0)
    seeds = []
    for curve in bmp.trace().curves:
        pts = []
        for seg in curve.segments:
            pts.append((float(seg.c.x), float(seg.c.y)))
        if len(pts) >= 3 and pts[0] != pts[-1]:
            pts.append(pts[0])
        if len(pts) >= 3:
            seeds.append(pts)
    return seeds

def vectorise(img: LoadedImage, cfg: Settings):
    layers = to_layers(img, cfg)
    composed = []
    for layer in layers:
        bw = mask_to_bw(img, layer)
        seeds = _potrace_trace(bw)
        polys = rdp_all(seeds, epsilon=cfg.rdp_epsilon)
        snapped = snap(polys, cfg.snap)
        beziers = bezier_fit([p for t, p in snapped if t == "poly"], cfg.bezier)
        items = [(t, p) for (t, p) in snapped if t in ("circle", "rect")]
        poly_left = [p for (t, p) in snapped if t == "poly"]
        bez = fit(poly_left, cfg.bezier)
        items.extend(bez)
        composed.append((items, layer.color))
    svg = compose(composed, img.size, cfg.svg).minified
    metrics = evaluate(svg, img, cfg.qa)
    return type("SVGResult", (), {"svg_min": svg, "svg_pretty": svg, "metrics": metrics})