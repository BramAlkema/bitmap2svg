"""Tests for caching and batching behaviour in pipeline."""

import base64

import pytest

from bitmap2svg.ingest import load
from bitmap2svg.config import Settings
from bitmap2svg.pipeline import (
    vectorise,
    vectorise_batch,
    _potrace_trace_cached,
)


@pytest.fixture
def sample_image(tmp_path):
    png_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/woAAgMBg8UoAA=="
    )
    img_path = tmp_path / "sample.png"
    img_path.write_bytes(base64.b64decode(png_base64))
    return load(img_path)


def test_caching(sample_image):
    cfg = Settings()
    _potrace_trace_cached.cache_clear()
    vectorise(sample_image, cfg)
    first_hits = _potrace_trace_cached.cache_info().hits
    vectorise(sample_image, cfg)
    second_hits = _potrace_trace_cached.cache_info().hits
    assert second_hits > first_hits


def test_vectorise_batch(sample_image):
    cfg = Settings()
    _potrace_trace_cached.cache_clear()
    images = [sample_image, sample_image]
    results = vectorise_batch(images, cfg)
    assert len(results) == 2
    assert _potrace_trace_cached.cache_info().hits > 0

