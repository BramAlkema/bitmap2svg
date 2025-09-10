from __future__ import annotations
from typing import List, Tuple
import numpy as np

Point = Tuple[float, float]
CurveSeg = Tuple[Point, Point, Point, Point]

class BezierCfg:
    max_err_px: float = 1.5
    max_segments: int = 256

def fit(polylines: List[List[Point]], cfg: BezierCfg) -> List[Tuple[str, List[CurveSeg]]]:
    out: List[Tuple[str, List[CurveSeg]]] = []
    for pts in polylines:
        if len(pts) < 3:
            seg = [ (pts[0], pts[0], pts[-1], pts[-1]) ]
            out.append(("bezier", seg))
            continue

        closed = _is_closed(pts)
        P = _prepare_points(pts, closed)

        left_tan  = _unit(_vsub(P[1], P[0]))
        right_tan = _unit(_vsub(P[-2], P[-1])) * -1.0

        segments: List[CurveSeg] = []
        _fit_subcurve(P, left_tan, right_tan, cfg.max_err_px, cfg.max_segments, segments)
        out.append(("bezier", segments))
    return out

def _fit_subcurve(P: np.ndarray, left_tan: np.ndarray, right_tan: np.ndarray,
                  err: float, seg_budget: int, out: List[CurveSeg]):
    if seg_budget <= 0:
        c = _generate_bezier(P, _chord_params(P), left_tan, right_tan)
        out.append(c)
        return

    U = _chord_params(P)
    C = _generate_bezier(P, U, left_tan, right_tan)

    split_i, max_err = _find_max_error(P, C, U)
    if max_err <= err**2:
        out.append(C)
        return

    ok = False
    for _ in range(3):
        U = _reparameterize(P, C, U)
        C = _generate_bezier(P, U, left_tan, right_tan)
        split_i, max_err = _find_max_error(P, C, U)
        if max_err <= err**2:
            ok = True
            break
    if ok:
        out.append(C)
        return

    center = split_i
    tan_l = _unit(_vsub(P[center+1], P[center-1]))
    tan_r = -tan_l

    _fit_subcurve(P[:center+1], left_tan, tan_l, err, seg_budget-1, out)
    _fit_subcurve(P[center:], tan_r, right_tan, err, seg_budget-1, out)

def _generate_bezier(P: np.ndarray, U: np.ndarray,
                     left_tan: np.ndarray, right_tan: np.ndarray) -> CurveSeg:
    n = len(P) - 1
    A = np.zeros((n+1, 2, 2), dtype=np.float64)
    for i in range(n+1):
        t = U[i]; b = _basis(t)
        A[i,0,:] = left_tan * b[1]
        A[i,1,:] = right_tan * b[2]

    C = np.zeros((2,2), dtype=np.float64)
    X = np.zeros(2, dtype=np.float64)
    for i in range(n+1):
        t = U[i]; b = _basis(t)
        tmp = P[i] - (P[0]*b[0] + P[-1]*b[3])
        C[0,0] += np.dot(A[i,0], A[i,0])
        C[0,1] += np.dot(A[i,0], A[i,1])
        C[1,0] += np.dot(A[i,1], A[i,0])
        C[1,1] += np.dot(A[i,1], A[i,1])
        X[0]   += np.dot(A[i,0], tmp)
        X[1]   += np.dot(A[i,1], tmp)

    detC0C1 = C[0,0]*C[1,1] - C[1,0]*C[0,1]
    alpha_l = alpha_r = 0.0
    if abs(detC0C1) > 1e-12:
        invC = np.linalg.inv(C)
        alpha_l, alpha_r = invC @ X
    else:
        chord = np.linalg.norm(P[-1] - P[0])
        alpha_l = alpha_r = chord/3.0

    seg_len = np.linalg.norm(P[-1] - P[0])
    eps = 1e-6
    if alpha_l < eps or alpha_r < eps:
        alpha_l = alpha_r = seg_len/3.0

    C1 = P[0]   + left_tan  * alpha_l
    C2 = P[-1]  + right_tan * alpha_r
    return (tuple(P[0]), tuple(C1), tuple(C2), tuple(P[-1]))

def _find_max_error(P: np.ndarray, C: CurveSeg, U: np.ndarray):
    p0, c1, c2, p3 = [np.asarray(x, dtype=np.float64) for x in C]
    max_err = -1.0; split_i = len(P)//2
    for i in range(1, len(P)-1):
        t = U[i]
        Q = _bezier_point(p0, c1, c2, p3, t)
        v = Q - P[i]
        e = v[0]*v[0] + v[1]*v[1]
        if e > max_err:
            max_err = e
            split_i = i
    return split_i, max_err

def _reparameterize(P: np.ndarray, C: CurveSeg, U: np.ndarray) -> np.ndarray:
    p0, c1, c2, p3 = [np.asarray(x, dtype=np.float64) for x in C]
    out = np.empty_like(U)
    for i, (Pi, ui) in enumerate(zip(P, U)):
        out[i] = _find_root(p0, c1, c2, p3, Pi, ui)
    out = np.clip(out, 0.0, 1.0)
    for i in range(1, len(out)):
        if out[i] <= out[i-1]:
            out[i] = min(1.0, out[i-1] + 1e-4)
    return out

def _is_closed(pts: List[Point]) -> bool:
    return np.linalg.norm(np.asarray(pts[0]) - np.asarray(pts[-1])) < 1e-6

def _prepare_points(pts: List[Point], closed: bool) -> np.ndarray:
    P = np.asarray(pts, dtype=np.float64)
    if closed:
        turns = []
        for i in range(1, len(P)-1):
            v1 = _unit(P[i] - P[i-1]); v2 = _unit(P[i+1] - P[i])
            ang = 1.0 - float(np.dot(v1, v2))
            turns.append((ang, i))
        turns.sort(reverse=True)
        j = turns[0][1] if turns else 0
        P = np.vstack([P[j:], P[1:j+1]])
        if np.linalg.norm(P[0] - P[-1]) > 1e-6:
            P = np.vstack([P, P[0]])
    else:
        if np.linalg.norm(P[0] - P[1]) < 1e-6:
            P = P[1:]
        if np.linalg.norm(P[-1] - P[-2]) < 1e-6:
            P = P[:-1]
    return P

def _chord_params(P: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(P[1:] - P[:-1], axis=1)
    total = d.sum()
    if total <= 1e-12:
        return np.linspace(0.0, 1.0, len(P))
    u = np.zeros(len(P), dtype=np.float64)
    u[1:] = np.cumsum(d) / total
    return u

def _basis(t: float) -> np.ndarray:
    mt = 1.0 - t
    return np.array([mt*mt*mt, 3*mt*mt*t, 3*mt*t*t, t*t*t], dtype=np.float64)

def _bezier_point(p0, p1, p2, p3, t: float) -> np.ndarray:
    b = _basis(t)
    return b[0]*p0 + b[1]*p1 + b[2]*p2 + b[3]*p3

def _bezier_first_derivative(p0, p1, p2, p3, t: float) -> np.ndarray:
    mt = 1.0 - t
    return 3.0 * ( (p1 - p0)*mt*mt + 2.0*(p2 - p1)*mt*t + (p3 - p2)*t*t )

def _bezier_second_derivative(p0, p1, p2, p3, t: float) -> np.ndarray:
    return 6.0 * ( (p2 - 2.0*p1 + p0)*(1.0 - t) + (p3 - 2.0*p2 + p1)*t )

def _find_root(p0, p1, p2, p3, P, u: float) -> float:
    Q  = _bezier_point(p0, p1, p2, p3, u)
    Q1 = _bezier_first_derivative(p0, p1, p2, p3, u)
    Q2 = _bezier_second_derivative(p0, p1, p2, p3, u)
    num = (Q - P) @ Q1
    den = (Q1 @ Q1) + (Q - P) @ Q2
    if abs(den) < 1e-12:
        return u
    return u - num/den

def _vsub(a, b): return np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v