from __future__ import annotations
import numpy as np
import pytest
from bitmap2svg.vector_critic import snap, SnapCfg

def test_snap_circle():
    points = [(0, 0), (1, 1), (0, 2), (2, 0)]
    cfg = SnapCfg(circle_tol=1.5)
    result = snap([points], cfg)
    assert len(result) == 1
    assert result[0][0] == "circle"

def test_snap_rectangle():
    points = [(0, 0), (0, 2), (2, 2), (2, 0)]
    cfg = SnapCfg(rect_iou=0.95)
    result = snap([points], cfg)
    assert len(result) == 1
    assert result[0][0] == "rect"

def test_snap_polygon():
    points = [(0, 0), (1, 1), (1, 0), (0, 1)]
    cfg = SnapCfg(circle_tol=1.5)
    result = snap([points], cfg)
    assert len(result) == 1
    assert result[0][0] == "poly"

def test_snap_empty():
    points = []
    cfg = SnapCfg(circle_tol=1.5)
    result = snap([points], cfg)
    assert len(result) == 0

def test_snap_invalid():
    points = [(0, 0), (0, 0)]
    cfg = SnapCfg(circle_tol=1.5)
    result = snap([points], cfg)
    assert len(result) == 0