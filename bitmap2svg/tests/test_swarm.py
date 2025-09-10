from __future__ import annotations
import numpy as np
import pytest
from bitmap2svg.swarm import refine

@pytest.fixture
def sample_polylines():
    return [
        [(0, 0), (1, 2), (2, 1)],
        [(3, 3), (4, 5), (5, 4)]
    ]

@pytest.fixture
def sample_edges():
    return np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32)

def test_refine(sample_polylines, sample_edges):
    refined = refine(sample_polylines, sample_edges, {"iters": 50, "step": 0.5})
    assert isinstance(refined, list)
    assert len(refined) == len(sample_polylines)  # Ensure the number of polylines is unchanged
    # Additional assertions can be added to check the properties of the refined paths