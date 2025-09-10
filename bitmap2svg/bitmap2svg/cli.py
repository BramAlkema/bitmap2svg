from __future__ import annotations

"""Command line interface for bitmap2svg.

This module exposes two commands:

``vectorise``
    Convert a single image to SVG.

``batch``
    Convert all images under a directory to SVG, optionally using multiple
    threads for faster processing. Progress can be disabled with ``--quiet``.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import typer
from rich.progress import track

from .config import Settings
from .ingest import load
from .pipeline import vectorise

app = typer.Typer(add_completion=False)


@app.command("vectorise")
def vectorise_cmd(input: str, out: str = "out.svg", cfg: str | None = None) -> None:
    """Vectorise a single image ``input`` and write the SVG to ``out``."""
    settings = Settings.model_validate_json(Path(cfg).read_text()) if cfg else Settings()
    img = load(input)
    res = vectorise(img, settings)
    Path(out).write_text(res.svg_min, encoding="utf-8")
    typer.echo(json.dumps(res.metrics, indent=2))


def _iter_images(src: Path) -> list[Path]:
    """Return all image files under ``src`` that we know how to handle."""
    return [
        p
        for p in src.glob("**/*.*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]


def _process(p: Path, dst: Path, settings: Settings):
    """Process ``p`` returning a tuple of (ok, path, metrics_or_error)."""
    try:
        img = load(p)
        res = vectorise(img, settings)
        Path(dst, p.with_suffix(".svg").name).write_text(res.svg_min, encoding="utf-8")
        return True, p, res.metrics
    except Exception as e:  # pragma: no cover - exception path
        return False, p, e


@app.command("batch")
def batch_cmd(
    src: str,
    dst: str = "out",
    cfg: str | None = None,
    jobs: int = 1,
    quiet: bool = False,
) -> None:
    """Vectorise all images in ``src`` placing results in ``dst``.

    ``jobs`` controls the number of worker threads. When set to 1 the images are
    processed sequentially. Set ``quiet`` to ``True`` to disable the progress
    display which is useful for automated testing.
    """

    settings = Settings.model_validate_json(Path(cfg).read_text()) if cfg else Settings()
    src_p = Path(src)
    dst_p = Path(dst)
    dst_p.mkdir(parents=True, exist_ok=True)
    paths = _iter_images(src_p)

    def iterator() -> Iterable:
        if jobs > 1:
            with ThreadPoolExecutor(max_workers=jobs) as ex:
                yield from ex.map(lambda p: _process(p, dst_p, settings), paths)
        else:
            for p in paths:
                yield _process(p, dst_p, settings)

    it = iterator()
    if not quiet:
        it = track(it, total=len(paths), description="Vectorising")

    for ok, p, data in it:
        if ok:
            typer.echo(f"OK {p.name}  {data}")
        else:
            typer.secho(f"FAIL {p}: {data}", fg="red")


if __name__ == "__main__":  # pragma: no cover
    app()

