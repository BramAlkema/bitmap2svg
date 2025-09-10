from bitmap2svg.ingest import load
from bitmap2svg.pipeline import vectorise
from bitmap2svg.config import Settings
import pytest
import os

@pytest.fixture
def sample_image():
    # Provide a path to a sample image for testing
    return "tests/assets/sample_logo.png"

def test_vectorise(sample_image):
    cfg = Settings()
    img = load(sample_image)
    result = vectorise(img, cfg)
    
    assert result.svg_min is not None
    assert isinstance(result.svg_min, str)
    assert len(result.svg_min) > 0

def test_metrics(sample_image):
    cfg = Settings()
    img = load(sample_image)
    result = vectorise(img, cfg)
    
    assert "ssim" in result.metrics
    assert "edge_iou" in result.metrics
    assert "bytes" in result.metrics

def test_invalid_image():
    with pytest.raises(FileNotFoundError):
        load("tests/assets/non_existent_image.png")