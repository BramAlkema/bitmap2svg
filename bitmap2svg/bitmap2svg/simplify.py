from __future__ import annotations
from typing import List, Tuple
import numpy as np
from rdp import rdp

def rdp_all(seeds: list[list[tuple[float,float]]], epsilon: float) -> list[list[tuple[float,float]]]:
    out: list[list[tuple[float,float]]] = []
    for chain in seeds:
        arr = np.asarray(chain, dtype=float)
        simp = rdp(arr, epsilon=epsilon)
        pts = [ (float(x), float(y)) for x,y in simp ]
        # Ensure closed
        if len(pts) >= 3 and pts[0] != pts[-1]:
            pts.append(pts[0])
        out.append(pts)
    return out