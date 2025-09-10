from bitmap2svg.ingest import load
from bitmap2svg.pipeline import vectorise
from bitmap2svg.config import Settings
import pytest
import base64


@pytest.fixture
def sample_image(tmp_path):
    png_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/woAAgMBg8UoAA=="
    )
    img_path = tmp_path / "sample.png"
    img_path.write_bytes(base64.b64decode(png_base64))
    return img_path


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


def test_invalid_image(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "non_existent_image.png")

