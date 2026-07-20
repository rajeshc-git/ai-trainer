"""FastAPI application entrypoint for the AI Model Fine-Tuner.

Wires together the dataset, training and models routers, exposes a health
check, and configures permissive CORS so the Vite dev server (port 5173) can
talk to the API during local development.
"""

from __future__ import annotations

try:
    # pyrefly: ignore [missing-import]
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from routers import dataset, inference, models, search, training
from schemas import HealthResponse
from utils import detect_gpu, gpu_total_mb

app = FastAPI(
    title="AI Model Fine-Tuner",
    description="Fine-tune Hugging Face models on your own dataset — no code required.",
    version="1.0.0",
)

# CORS — the browser frontend runs on a different origin (5173).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dataset.router)
app.include_router(training.router)
app.include_router(models.router)
app.include_router(inference.router)
app.include_router(search.router)


@app.on_event("startup")
async def _startup() -> None:
    """Log GPU availability once at boot for easy diagnostics in docker logs."""
    has_gpu, gpu_name = detect_gpu()
    if has_gpu:
        print(f"[startup] GPU available: {gpu_name}", flush=True)
    else:
        print("[startup] No GPU detected — training will run on CPU (slow).", flush=True)
    print("[startup] AI Model Fine-Tuner backend is ready.", flush=True)


@app.get("/health", response_model=HealthResponse)
async def health() -> JSONResponse:
    """Liveness + GPU probe used by the Docker healthcheck and the frontend."""
    has_gpu, gpu_name = detect_gpu()
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "gpu": has_gpu,
            "gpu_name": gpu_name,
            "gpu_total_mb": gpu_total_mb() if has_gpu else None,
        },
    )


@app.get("/")
async def root() -> dict[str, str]:
    """Friendly root pointing at the docs."""
    return {"message": "AI Model Fine-Tuner API. See /docs for the OpenAPI UI."}
