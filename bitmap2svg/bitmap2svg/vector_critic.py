from __future__ import annotations
from typing import List, Tuple, Sequence
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

PathLike = List[Tuple[float, float]]

class SnapCfg:
    circle_tol: float = 1.3
    rect_iou: float = 0.95

def _fit_circle(points: Sequence[Tuple[float,float]], tol: float = 1.5):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 6:
        return None
    x, y = pts[:,0], pts[:,1]
    x_m, y_m = x.mean(), y.mean()
    u, v = x - x_m, y - y_m
    Suu, Svv, Suv = np.dot(u,u), np.dot(v,v), np.dot(u,v)
    Suuu, Svvv = np.dot(u, u*u), np.dot(v, v*v)
    Suvv, Svuu = np.dot(u, v*v), np.dot(v, u*u)
    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = 0.5 * np.array([Suuu + Suvv, Svvv + Svuu])
    try:
        cx, cy = np.linalg.solve(A, B) + np.array([x_m, y_m])
    except np.linalg.LinAlgError:
        return None
    r = np.mean(np.sqrt((x - cx)**2 + (y - cy)**2))
    res = np.std(np.sqrt((x - cx)**2 + (y - cy)**2))
    if res < tol:
        return (float(cx), float(cy), float(r))
    return None

def _fit_axis_rect(points: Sequence[Tuple[float,float]], iou_thresh: float=0.95):
    poly = Polygon(points).buffer(0)
    if not poly.is_valid:
        return None
    minx, miny, maxx, maxy = poly.bounds
    rect = Polygon([(minx,miny),(maxx,miny),(maxx,maxy),(minx,maxy)])
    denom = rect.union(poly).area
    if denom == 0:
        return None
    iou = rect.intersection(poly).area / denom
    if iou >= iou_thresh:
        return (float(minx), float(miny), float(maxx-minx), float(maxy-miny))
    return None

def snap(polylines: List[PathLike], cfg: SnapCfg):
    out = []
    for pts in polylines:
        if len(pts) < 4:
            continue
        circ = _fit_circle(pts, tol=cfg.circle_tol)
        if circ:
            out.append(("circle", circ))
            continue
        rect = _fit_axis_rect(pts, iou_thresh=cfg.rect_iou)
        if rect:
            out.append(("rect", rect))
            continue
        out.append(("poly", pts))
    return out