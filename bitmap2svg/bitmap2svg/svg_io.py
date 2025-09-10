from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple
import svgwrite

@dataclass
class SVGOut:
    minified: str
    pretty: str

def _path_from_poly(dwg, pts):
    d = f"M {pts[0][0]:.3f} {pts[0][1]:.3f} " + " ".join(f"L {x:.3f} {y:.3f}" for x,y in pts[1:]) + " Z"
    return dwg.path(d=d)

def compose(paths_with_color: Iterable[tuple[list[tuple[str, list]], tuple[int,int,int,int]]],
            size: tuple[int,int], cfg) -> SVGOut:
    W, H = size
    dwg = svgwrite.Drawing(size=(W, H), viewBox=f"0 0 {W} {H}")
    root = dwg.g(id="logo")
    for items, color in paths_with_color:
        rgba = f"rgba({color[0]},{color[1]},{color[2]},{color[3]/255:.3f})"
        for typ, payload in items:
            if typ == "circle":
                cx, cy, r = payload
                el = dwg.circle(center=(round(cx,cfg.decimals), round(cy,cfg.decimals)),
                                r=round(r, cfg.decimals), fill=rgba)
            elif typ == "rect":
                x,y,w,h = payload
                el = dwg.rect(insert=(round(x,cfg.decimals), round(y,cfg.decimals)),
                              size=(round(w,cfg.decimals), round(h,cfg.decimals)),
                              fill=rgba)
            elif typ == "poly":
                el = _path_from_poly(dwg, payload)
                el.update(fill=rgba)
            elif typ == "bezier":
                if not payload:
                    continue
                p0 = payload[0][0]
                d = f"M {p0[0]:.{cfg.decimals}f} {p0[1]:.{cfg.decimals}f} "
                for (s0, c1, c2, s1) in payload:
                    d += f"C {c1[0]:.{cfg.decimals}f} {c1[1]:.{cfg.decimals}f} " \
                         f"{c2[0]:.{cfg.decimals}f} {c2[1]:.{cfg.decimals}f} " \
                         f"{s1[0]:.{cfg.decimals}f} {s1[1]:.{cfg.decimals}f} "
                el = dwg.path(d=d + "Z", fill=rgba)
            else:
                continue
            root.add(el)
    dwg.add(root)
    pretty = dwg.tostring()
    minified = " ".join(pretty.split())
    return SVGOut(minified=minified, pretty=pretty)