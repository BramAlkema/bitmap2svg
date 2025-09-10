import base64
from pathlib import Path

from typer.testing import CliRunner

from bitmap2svg.cli import app


PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/woAAgMBg8UoAA=="


def _write_sample(path: Path) -> None:
    path.write_bytes(base64.b64decode(PNG_B64))


def test_batch_parallel(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _write_sample(src / "a.png")
    _write_sample(src / "b.png")

    out = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(app, ["batch", str(src), str(out), "--jobs", "2", "--quiet"])

    assert result.exit_code == 0
    assert (out / "a.svg").exists()
    assert (out / "b.svg").exists()

