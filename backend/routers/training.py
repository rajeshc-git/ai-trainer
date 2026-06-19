"""Training endpoints: start, status, cancel, and the live-log WebSocket."""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

import trainer
from schemas import JobStatus, JobStatusResponse, TrainRequest, TrainResponse
from utils import (
    DATASETS_DIR,
    get_redis,
    is_cancelled,
    list_jobs,
    load_job,
    log_channel,
    now,
    save_job,
    set_cancel_flag,
    delete_job,
)

router = APIRouter(tags=["training"])


def _launch_job(job_id: str, params: dict) -> None:
    """Run a training job in a daemon thread so the event loop stays free.

    The heavy training work is blocking and CPU/GPU-bound, so a dedicated
    thread (rather than the asyncio loop) keeps the API responsive and lets
    multiple jobs run concurrently.
    """
    thread = threading.Thread(
        target=trainer.run_training,
        args=(job_id, params),
        daemon=True,
        name=f"train-{job_id[:8]}",
    )
    thread.start()


@router.post("/api/train", response_model=TrainResponse)
async def start_training(req: TrainRequest) -> JSONResponse:
    """Start a fine-tuning job and return its ``job_id`` immediately."""
    dataset_path = DATASETS_DIR / f"{req.dataset_id}.csv"
    if not dataset_path.exists():
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Dataset not found.",
                "suggestion": "Upload and validate your CSV again before training.",
            },
        )

    job_id = uuid.uuid4().hex
    params = {
        "model_name": req.model_name,
        "dataset_path": str(dataset_path),
        "epochs": req.epochs,
        "learning_rate": req.learning_rate,
        "batch_size": req.batch_size,
        "max_length": req.max_length,
        "output_dir": req.output_dir,
    }

    save_job(
        job_id,
        {
            "status": JobStatus.QUEUED.value,
            "model_name": req.model_name,
            "total_epochs": req.epochs,
            "percent": 0.0,
            "started_at": now(),
            "params": params,
        },
    )
    print(f"[api] Launching training job {job_id} for model {req.model_name}", flush=True)
    _launch_job(job_id, params)

    return JSONResponse(
        status_code=200,
        content={
            "job_id": job_id,
            "status": JobStatus.QUEUED.value,
            "message": "Training job started.",
        },
    )


@router.get("/api/train/{job_id}/status", response_model=JobStatusResponse)
async def job_status(job_id: str) -> JSONResponse:
    """Return the current status snapshot for a job."""
    job = load_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "Job not found.", "suggestion": "Check the job id."},
        )
    # Normalise into the response schema (fill missing keys with defaults).
    resp = JobStatusResponse(
        job_id=job_id,
        status=job.get("status", "queued"),
        percent=job.get("percent", 0.0),
        current_epoch=job.get("current_epoch", 0),
        total_epochs=job.get("total_epochs", 0),
        current_step=job.get("current_step", 0),
        total_steps=job.get("total_steps", 0),
        loss=job.get("loss"),
        eval_loss=job.get("eval_loss"),
        best_eval_loss=job.get("best_eval_loss"),
        learning_rate=job.get("learning_rate"),
        eta_seconds=job.get("eta_seconds"),
        gpu_memory_mb=job.get("gpu_memory_mb"),
        gpu_total_mb=job.get("gpu_total_mb"),
        gpu_memory_percent=job.get("gpu_memory_percent"),
        model_name=job.get("model_name"),
        started_at=job.get("started_at"),
        finished_at=job.get("finished_at"),
        error=job.get("error"),
        suggestion=job.get("suggestion"),
    )
    return JSONResponse(status_code=200, content=resp.model_dump())


@router.get("/api/train/{job_id}/cancel")
async def cancel_job(job_id: str) -> JSONResponse:
    """Request cancellation of a running job."""
    job = load_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "Job not found.", "suggestion": "Check the job id."},
        )
    if job.get("status") in ("completed", "failed", "cancelled"):
        return JSONResponse(
            status_code=200,
            content={"job_id": job_id, "message": "Job already finished."},
        )
    set_cancel_flag(job_id)
    print(f"[api] Cancel requested for job {job_id}", flush=True)
    return JSONResponse(
        status_code=200,
        content={"job_id": job_id, "message": "Cancellation requested."},
    )


@router.get("/api/train")
async def list_training_jobs() -> JSONResponse:
    """Return all training jobs (used by the dashboard table)."""
    return JSONResponse(status_code=200, content={"jobs": list_jobs()})


@router.delete("/api/train/{job_id}")
async def delete_training_job(job_id: str) -> JSONResponse:
    """Delete a training job's metadata from Redis."""
    job = load_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "Job not found.", "suggestion": "Check the job id."},
        )
    if job.get("status") in ("queued", "running"):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Cannot delete an active training job.",
                "suggestion": "Cancel the job first before removing it from the history.",
            },
        )
    delete_job(job_id)
    return JSONResponse(
        status_code=200,
        content={"job_id": job_id, "message": "Job record deleted successfully."},
    )


@router.websocket("/ws/train/{job_id}")
async def ws_training_logs(websocket: WebSocket, job_id: str) -> None:
    """Stream live training logs for a job over a WebSocket.

    On connect we first replay the buffered backlog (so a client that joins
    late still sees earlier lines), then subscribe to the Redis pub/sub
    channel and forward each message. The loop also emits periodic status
    frames and exits cleanly when the job finishes or the socket drops.
    """
    await websocket.accept()
    r = get_redis()

    # 1) Replay backlog.
    try:
        backlog = r.lrange(f"ftlogbuf:{job_id}", 0, -1)
        for line in backlog:
            await websocket.send_text(line)
    except Exception:
        pass

    # 2) Subscribe to live logs.
    pubsub = r.pubsub()
    pubsub.subscribe(log_channel(job_id))

    try:
        while True:
            message = pubsub.get_message(ignore_subscribe_messages=True, timeout=0.5)
            if message and message.get("type") == "message":
                await websocket.send_text(message["data"])

            # Piggyback a status frame so the client always has fresh metrics.
            job = load_job(job_id)
            if job is not None:
                await websocket.send_text(
                    json.dumps({"type": "status", "status": job})
                )
                if job.get("status") in ("completed", "failed", "cancelled"):
                    await websocket.send_text(
                        json.dumps(
                            {"type": "done", "status": job.get("status")}
                        )
                    )
                    break
            await asyncio.sleep(0.4)
    except WebSocketDisconnect:
        print(f"[ws] client disconnected from job {job_id}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[ws] error on job {job_id}: {exc}", flush=True)
    finally:
        try:
            pubsub.close()
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
