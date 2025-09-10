"""FastAPI service exposing vectorisation endpoints with caching and batching."""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile
from starlette.responses import JSONResponse

from bitmap2svg.config import Settings
from bitmap2svg.ingest import load
from bitmap2svg.pipeline import vectorise


app = FastAPI()


@lru_cache(maxsize=32)
def _vectorise_cached(data: bytes, cfg_json: str):
    """Cache the vectorisation of raw image bytes with a given config."""
    cfg = Settings.model_validate_json(cfg_json)
    img = load(BytesIO(data))
    return vectorise(img, cfg)


@app.post("/vectorise")
async def vectorise_image(file: UploadFile = File(...), cfg_path: str | None = None):
    try:
        cfg = (
            Settings.model_validate_json(Path(cfg_path).read_text())
            if cfg_path
            else Settings()
        )
        data = await file.read()
        res = _vectorise_cached(data, cfg.model_dump_json())
        return JSONResponse(content={"svg": res.svg_min, "metrics": res.metrics})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.post("/vectorise-batch")
async def vectorise_batch(files: List[UploadFile], cfg_path: str | None = None):
    try:
        cfg = (
            Settings.model_validate_json(Path(cfg_path).read_text())
            if cfg_path
            else Settings()
        )
        results = []
        for f in files:
            data = await f.read()
            res = _vectorise_cached(data, cfg.model_dump_json())
            results.append({"svg": res.svg_min, "metrics": res.metrics})
        return JSONResponse(content={"results": results})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.get("/status")
def status():
    return {"status": "Service is running"}

