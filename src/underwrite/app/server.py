"""Review UI server.

Serves a single-page review interface backed by a **precomputed batch**, so the demo
runs anywhere with no GPU and no model download. Nothing here touches customer data:
the batch is built from the public CORD corpus by ``scripts/build_demo_batch.py``.

Run locally:

    uvicorn underwrite.app.server:app --reload
    # or, without installing the `app` extras:
    python -m underwrite.app.server
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"
DEFAULT_BATCH = Path("reports/demo_batch.json")

app = FastAPI(title="Underwrite review", version="0.1.0")


def _batch_path() -> Path:
    import os

    return Path(os.environ.get("UNDERWRITE_BATCH", DEFAULT_BATCH))


@app.get("/api/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/api/batch")
def batch() -> JSONResponse:
    path = _batch_path()
    if not path.is_file():
        return JSONResponse(
            {"error": f"no demo batch at {path}; run scripts/build_demo_batch.py"},
            status_code=404,
        )
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
