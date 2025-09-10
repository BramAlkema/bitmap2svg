from __future__ import annotations
from pydantic import BaseModel

class SwarmCfg(BaseModel):
    iters: int = 80
    step: float = 0.8

class SnapCfg(BaseModel):
    circle_tol: float = 1.3
    rect_iou: float = 0.95

class BezierCfg(BaseModel):
    max_err_px: float = 1.5
    max_segments: int = 256

class QACfg(BaseModel):
    ssim_scale: int = 4
    edge_iou_thresh: float = 0.97
    ssim_thresh: float = 0.97

class SVGCfg(BaseModel):
    decimals: int = 3
    readable: bool = True

class Settings(BaseModel):
    k_colors: int = 4
    rdp_epsilon: float = 1.2
    swarm: SwarmCfg = SwarmCfg()
    snap: SnapCfg = SnapCfg()
    bezier: BezierCfg = BezierCfg()
    qa: QACfg = QACfg()
    svg: SVGCfg = SVGCfg()
    use_llm: bool = False