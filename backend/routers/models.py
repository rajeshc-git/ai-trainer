"""Saved-model endpoints: list, download (zip + GGUF), GGUF export, and delete."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import uuid
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel

import gguf
from utils import MODELS_DIR, now, save_job

router = APIRouter(prefix="/api/models", tags=["models"])


class GgufExportRequest(BaseModel):
    """Body of ``POST /api/models/{job_id}/export/gguf``."""

    quant: str = gguf.DEFAULT_QUANT


def _dir_size(path: Path) -> int:
    """Return total size in bytes of all files under ``path``."""
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


@router.get("/list")
def list_models() -> JSONResponse:
    """List all locally saved fine-tuned models.

    Reads the ``ft_metadata.json`` sidecar written by the trainer for each
    saved model; directories without metadata are still listed with best-effort
    information.
    """
    models = []
    for child in sorted(MODELS_DIR.iterdir() if MODELS_DIR.exists() else []):
        if not child.is_dir() or child.name.startswith("."):
            continue
        meta_path = child / "ft_metadata.json"
        if not meta_path.exists():
            # Skip directories that are not finished model saves.
            if not any((child / f).exists() for f in ("config.json", "adapter_config.json")):
                continue
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {}
        models.append(
            {
                "job_id": meta.get("job_id", child.name),
                "name": meta.get("name", child.name),
                "base_model": meta.get("base_model"),
                "created_at": meta.get("created_at"),
                "dataset_rows": meta.get("dataset_rows"),
                "final_loss": meta.get("final_loss"),
                "best_eval_loss": meta.get("best_eval_loss"),
                "architecture": meta.get("architecture"),
                "size_bytes": _dir_size(child),
            }
        )
    models.sort(key=lambda m: m.get("created_at") or 0, reverse=True)
    return JSONResponse(status_code=200, content={"models": models})


@router.get("/{job_id}/download")
def download_model(job_id: str) -> FileResponse:
    """Stream a saved model directory as a zip archive.

    The archive is written to a temp file on disk (not held in RAM) and
    streamed from there, then deleted as soon as the response is sent.
    Model weights (.safetensors/.bin) are already incompressible, so we use
    ZIP_STORED — no CPU spent on compression, no size penalty, lossless.
    """
    model_dir = MODELS_DIR / job_id
    if not model_dir.exists() or not model_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail={"error": "Model not found.", "suggestion": "Refresh the models list."},
        )

    fd, tmp_path = tempfile.mkstemp(suffix=".zip", prefix=f"{job_id}-")
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_STORED) as zf:
            for p in model_dir.rglob("*"):
                if p.is_file():
                    zf.write(p, p.relative_to(model_dir))
    except Exception:
        # Don't leak the temp file if zipping fails mid-way.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    def _cleanup() -> None:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # FileResponse streams from disk in chunks and sets Content-Length from the
    # file size, which gives the UI an accurate progress bar.
    return FileResponse(
        tmp_path,
        media_type="application/zip",
        filename=f"{job_id}.zip",
        background=BackgroundTask(_cleanup),
    )


@router.delete("/{job_id}")
def delete_model(job_id: str) -> JSONResponse:
    """Delete a saved model directory from disk."""
    model_dir = MODELS_DIR / job_id
    if not model_dir.exists():
        raise HTTPException(
            status_code=404,
            detail={"error": "Model not found.", "suggestion": "It may already be deleted."},
        )
    shutil.rmtree(model_dir)
    # Drop it from the inference cache so memory is freed immediately.
    try:
        import inference

        inference.unload(job_id)
    except Exception:
        pass
    print(f"[api] Deleted model {job_id}", flush=True)
    return JSONResponse(status_code=200, content={"deleted": job_id})


# ─────────────────────────────────────────────────────────────
# GGUF export (Ollama / llama.cpp)
# ─────────────────────────────────────────────────────────────
@router.post("/{job_id}/export/gguf")
def export_gguf(job_id: str, req: GgufExportRequest) -> JSONResponse:
    """Start a GGUF export of a trained model and return its ``export_id``.

    The heavy merge→convert→quantize pipeline runs in a daemon thread and
    streams progress on the shared log channel, so the client can watch it live
    over the existing ``/ws/train/{export_id}`` WebSocket.
    """
    model_dir = MODELS_DIR / job_id
    if not model_dir.exists() or not model_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail={"error": "Model not found.", "suggestion": "Refresh the models list."},
        )

    quant = (req.quant or gguf.DEFAULT_QUANT).upper()
    if quant not in gguf.ALLOWED_QUANTS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Unsupported quantization '{quant}'.",
                "suggestion": f"Choose one of: {', '.join(gguf.ALLOWED_QUANTS)}.",
            },
        )

    export_id = uuid.uuid4().hex
    save_job(
        export_id,
        {
            "status": "queued",
            "kind": "gguf_export",
            "model_job_id": job_id,
            "quant": quant,
            "percent": 0.0,
            "started_at": now(),
        },
    )
    thread = threading.Thread(
        target=gguf.run_export,
        args=(export_id, job_id, quant),
        daemon=True,
        name=f"gguf-{export_id[:8]}",
    )
    thread.start()
    print(f"[api] Launching GGUF export {export_id} ({quant}) for model {job_id}", flush=True)
    return JSONResponse(
        status_code=200,
        content={"export_id": export_id, "quant": quant, "status": "queued"},
    )


@router.get("/{job_id}/gguf")
def list_gguf_exports(job_id: str) -> JSONResponse:
    """List the GGUF files already exported for a model."""
    return JSONResponse(status_code=200, content={"files": gguf.list_gguf(job_id)})


@router.get("/{job_id}/gguf/{filename}/download")
def download_gguf(job_id: str, filename: str) -> FileResponse:
    """Stream a single exported GGUF file."""
    # Guard against path traversal — only allow a plain .gguf file name.
    safe = Path(filename).name
    if safe != filename or not safe.endswith(".gguf"):
        raise HTTPException(status_code=400, detail={"error": "Invalid file name."})
    path = MODELS_DIR / job_id / "gguf" / safe
    if not path.exists() or not path.is_file():
        raise HTTPException(
            status_code=404,
            detail={"error": "GGUF file not found.", "suggestion": "Export it first."},
        )
    return FileResponse(
        str(path),
        media_type="application/octet-stream",
        filename=safe,
    )


@router.delete("/{job_id}/gguf/{filename}")
def delete_gguf(job_id: str, filename: str) -> JSONResponse:
    """Delete a single exported GGUF file."""
    # Guard against path traversal — only allow a plain .gguf file name.
    safe = Path(filename).name
    if safe != filename or not safe.endswith(".gguf"):
        raise HTTPException(status_code=400, detail={"error": "Invalid file name."})
    path = MODELS_DIR / job_id / "gguf" / safe
    if not path.exists() or not path.is_file():
        raise HTTPException(
            status_code=404,
            detail={"error": "GGUF file not found."},
        )
    path.unlink()
    return JSONResponse(status_code=200, content={"deleted": safe})
