from __future__ import annotations
import json
from pathlib import Path
import typer
from .config import Settings
from .ingest import load
from .pipeline import vectorise

app = typer.Typer(add_completion=False)

@app.command("vectorise")
def vectorise_cmd(input: str, out: str = "out.svg", cfg: str | None = None):
    settings = Settings.model_validate_json(Path(cfg).read_text()) if cfg else Settings()
    img = load(input)
    res = vectorise(img, settings)
    Path(out).write_text(res.svg_min, encoding="utf-8")
    typer.echo(json.dumps(res.metrics, indent=2))

@app.command("batch")
def batch_cmd(src: str, dst: str = "out", cfg: str | None = None):
    settings = Settings.model_validate_json(Path(cfg).read_text()) if cfg else Settings()
    Path(dst).mkdir(parents=True, exist_ok=True)
    for p in Path(src).glob("**/*.*"):
        if p.suffix.lower() not in {".png",".jpg",".jpeg",".webp"}:
            continue
        try:
            img = load(p)
            res = vectorise(img, settings)
            Path(dst, p.with_suffix(".svg").name).write_text(res.svg_min, encoding="utf-8")
            typer.echo(f"OK {p.name}  {res.metrics}")
        except Exception as e:
            typer.secho(f"FAIL {p}: {e}", fg="red")

if __name__ == "__main__":
    app()