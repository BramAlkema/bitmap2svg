from fastapi import FastAPI, UploadFile, File
from starlette.responses import JSONResponse
from bitmap2svg.pipeline import vectorise
from bitmap2svg.ingest import load
from bitmap2svg.config import Settings
import json
from pathlib import Path

app = FastAPI()

@app.post("/vectorise")
async def vectorise_image(file: UploadFile = File(...), cfg_path: str = None):
    try:
        # Load configuration settings
        cfg = Settings.model_validate_json(Path(cfg_path).read_text()) if cfg_path else Settings()
        
        # Load the image
        img = load(file.file)
        
        # Vectorise the image
        res = vectorise(img, cfg)
        
        # Return the SVG result and metrics
        return JSONResponse(content={
            "svg": res.svg_min,
            "metrics": res.metrics
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.get("/status")
def status():
    return {"status": "Service is running"}