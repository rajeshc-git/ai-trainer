"""Utility helpers: GPU detection, dataset parsing, Redis access, job store.

This module centralises the small pieces of shared logic so the routers
and trainer stay focused on their own concerns.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
# pyrefly: ignore [missing-import]
import redis

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
DATASETS_DIR = Path(os.getenv("DATASETS_DIR", "/datasets"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models"))
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = ["input", "output"]
OPTIONAL_COLUMNS = ["instruction"]

# ─────────────────────────────────────────────────────────────
# Redis (job state + pub/sub for live logs)
# ─────────────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
JOB_KEY_PREFIX = "ftjob:"
LOG_CHANNEL_PREFIX = "ftlogs:"


def get_redis() -> redis.Redis:
    """Return a decoded (str) Redis client.

    A new lightweight client is created per call; ``redis-py`` pools
    connections under the hood, so this is cheap and thread-safe.
    """
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)


def log_channel(job_id: str) -> str:
    """Return the pub/sub channel name for a job's live logs."""
    return f"{LOG_CHANNEL_PREFIX}{job_id}"


def publish_log(job_id: str, level: str, message: str) -> None:
    """Publish one structured log line to a job's Redis channel + backlog buffer.

    Shared by the trainer, the model downloader and the GGUF exporter so they all
    stream into the same live-log channel the ``/ws/train/{id}`` WebSocket relays.
    A bounded backlog (``ftlogbuf:{id}``) lets a late-joining client replay history.

    Args:
        job_id: The job/export id whose channel to publish on.
        level: One of ``INFO``, ``WARNING``, ``ERROR``.
        message: Human-readable log text.
    """
    payload = json.dumps({"ts": now(), "level": level, "message": message})
    r = get_redis()
    r.publish(log_channel(job_id), payload)
    r.rpush(f"ftlogbuf:{job_id}", payload)
    r.ltrim(f"ftlogbuf:{job_id}", -500, -1)
    r.expire(f"ftlogbuf:{job_id}", 86400)
    # Mirror to stdout for `docker compose logs`.
    print(f"[{job_id}] {level}: {message}", flush=True)


# ─────────────────────────────────────────────────────────────
# GPU detection
# ─────────────────────────────────────────────────────────────
def detect_gpu() -> tuple[bool, Optional[str]]:
    """Detect whether a CUDA GPU is available.

    Returns:
        A tuple ``(has_gpu, gpu_name)``. ``gpu_name`` is ``None`` on CPU.
    """
    try:
        # pyrefly: ignore [missing-import]
        import torch

        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover - torch may be missing on dev box
        pass
    return False, None


def gpu_total_mb() -> Optional[float]:
    """Return the total VRAM of GPU 0 in MB, or None on CPU.

    This is the size of the card and never changes — the frontend uses it to
    tell the user how much memory their GPU has (so they can judge which models
    will fit).
    """
    try:
        # pyrefly: ignore [missing-import]
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return round(props.total_memory / (1024 * 1024), 1)
    except Exception:
        pass
    return None


def gpu_memory_stats() -> Optional[dict[str, float]]:
    """Return live, device-wide GPU memory usage in MB, or None on CPU.

    Uses ``torch.cuda.mem_get_info`` so the numbers reflect **everything** on
    the card — model weights, activations, the KV cache and any other process —
    not just what this PyTorch process reserved. That is exactly what a user
    needs to answer "does this model fit on my GPU?".

    Returns a dict with ``used_mb``, ``total_mb``, ``free_mb``, ``percent``
    (device-wide) plus ``reserved_mb``/``allocated_mb`` (this process only).
    """
    try:
        # pyrefly: ignore [missing-import]
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info(0)
        used = total - free
        return {
            "used_mb": round(used / (1024 * 1024), 1),
            "total_mb": round(total / (1024 * 1024), 1),
            "free_mb": round(free / (1024 * 1024), 1),
            "percent": round(100.0 * used / total, 1) if total else 0.0,
            "reserved_mb": round(torch.cuda.memory_reserved(0) / (1024 * 1024), 1),
            "allocated_mb": round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1),
        }
    except Exception:
        return None


def gpu_memory_mb() -> Optional[float]:
    """Return live device-wide *used* GPU memory in MB, or None on CPU.

    Kept for backwards compatibility; prefer :func:`gpu_memory_stats`.
    """
    stats = gpu_memory_stats()
    return stats["used_mb"] if stats else None


def release_gpu_memory() -> None:
    """Return freed VRAM to the driver after model objects are dropped.

    PyTorch's caching allocator keeps memory *reserved* once it has been used,
    so simply dropping the Python references to a model/optimizer leaves the
    card looking full in ``nvidia-smi`` and to ``device_map='auto'`` until the
    process exits. Callers must drop their references first, then call this to
    run a GC pass and hand the now-unreferenced blocks back to the GPU.

    Safe to call on a CPU-only box (it just runs ``gc.collect``).
    """
    import gc

    gc.collect()
    try:
        # pyrefly: ignore [missing-import]
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────
def build_prompt(row: dict[str, Any]) -> str:
    """Convert a dataset row into the canonical instruction prompt.

    Format::

        ### Instruction:
        {instruction}
        ### Input:
        {input}
        ### Response:
        {output}

    The ``instruction`` block is omitted when the column is absent or empty.
    """
    instruction = str(row.get("instruction") or "").strip()
    input_text = str(row.get("input") or "").strip()
    output_text = str(row.get("output") or "").strip()

    parts: list[str] = []
    if instruction:
        parts.append(f"### Instruction:\n{instruction}")
    parts.append(f"### Input:\n{input_text}")
    parts.append(f"### Response:\n{output_text}")
    return "\n".join(parts)


def build_seq2seq_src(row: dict[str, Any]) -> str:
    """Build the encoder input for a seq2seq (T5-style) model.

    Encoder-decoder models take the instruction+input as the source and learn
    to produce the output as the target, so the ``### Response`` cue used for
    causal models is intentionally omitted here. Trainer and inference share
    this helper so the model sees the same format in both phases.
    """
    instruction = str(row.get("instruction") or "").strip()
    input_text = str(row.get("input") or "").strip()
    if instruction:
        return f"{instruction}\n{input_text}".strip()
    return input_text


def validate_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Validate a dataset DataFrame against the required schema.

    Returns a dict describing the validation outcome, including detected
    columns, null counts, a 5-row sample, and a friendly suggestion when
    something is wrong.
    """
    columns = [str(c) for c in df.columns]
    detected = []
    missing = []
    for col in REQUIRED_COLUMNS:
        present = col in df.columns
        null_count = int(df[col].isnull().sum()) if present else 0
        detected.append({"name": col, "present": present, "null_count": null_count})
        if not present:
            missing.append(col)
    for col in OPTIONAL_COLUMNS:
        if col in df.columns:
            detected.append(
                {
                    "name": col,
                    "present": True,
                    "null_count": int(df[col].isnull().sum()),
                }
            )

    valid = True
    error: Optional[str] = None
    suggestion: Optional[str] = None

    if missing:
        valid = False
        error = f"Missing required column(s): {', '.join(missing)}."
        suggestion = (
            "Your CSV must contain an 'input' and an 'output' column. "
            "Download the template to see the exact format."
        )
    elif len(df) == 0:
        valid = False
        error = "The dataset is empty."
        suggestion = "Add at least one row with example input/output pairs."
    else:
        # Null check on required columns
        null_required = [
            c for c in REQUIRED_COLUMNS if int(df[c].isnull().sum()) > 0
        ]
        if null_required:
            valid = False
            error = f"Found empty cells in: {', '.join(null_required)}."
            suggestion = "Remove blank rows or fill in the missing values."

    # Round-trip through pandas' JSON encoder so numpy types (int64/float64)
    # and NaN become JSON-safe before they reach the response serializer.
    sample = json.loads(df.head(5).where(df.head(5).notnull(), "").to_json(orient="records"))

    return {
        "valid": valid,
        "row_count": int(len(df)),
        "columns": columns,
        "detected": detected,
        "sample_rows": sample,
        "error": error,
        "suggestion": suggestion,
    }


def extract_model_id(raw: str) -> str:
    """Extract a bare Hugging Face model id from a URL or plain name.

    Examples:
        >>> extract_model_id("https://huggingface.co/gpt2")
        'gpt2'
        >>> extract_model_id("mistralai/Mistral-7B-v0.1")
        'mistralai/Mistral-7B-v0.1'
    """
    raw = raw.strip()
    m = re.search(r"huggingface\.co/([^/\s]+(?:/[^/\s?#]+)?)", raw)
    if m:
        return m.group(1)
    return raw


# ─────────────────────────────────────────────────────────────
# Job state persistence (Redis hash)
# ─────────────────────────────────────────────────────────────
def save_job(job_id: str, data: dict[str, Any]) -> None:
    """Persist (merge) a job's state into Redis as a JSON blob."""
    r = get_redis()
    existing = load_job(job_id) or {}
    existing.update(data)
    existing["job_id"] = job_id
    r.set(f"{JOB_KEY_PREFIX}{job_id}", json.dumps(existing))
    r.sadd("ftjobs:all", job_id)


def load_job(job_id: str) -> Optional[dict[str, Any]]:
    """Load a job's state from Redis, or None if it does not exist."""
    r = get_redis()
    raw = r.get(f"{JOB_KEY_PREFIX}{job_id}")
    if raw is None:
        return None
    return json.loads(raw)


def list_jobs() -> list[dict[str, Any]]:
    """Return all known jobs, newest first."""
    r = get_redis()
    ids = r.smembers("ftjobs:all")
    jobs = [load_job(jid) for jid in ids]
    jobs = [j for j in jobs if j and j.get("kind") != "gguf_export"]
    jobs.sort(key=lambda j: j.get("started_at") or 0, reverse=True)
    return jobs


def delete_job(job_id: str) -> None:
    """Delete a job's state and log buffers from Redis."""
    r = get_redis()
    r.delete(f"{JOB_KEY_PREFIX}{job_id}")
    r.srem("ftjobs:all", job_id)
    r.delete(f"ftlogbuf:{job_id}")


def set_cancel_flag(job_id: str) -> None:
    """Signal a running job that it should stop at the next checkpoint."""
    get_redis().set(f"ftcancel:{job_id}", "1", ex=3600)


def is_cancelled(job_id: str) -> bool:
    """Return True if a cancel has been requested for this job."""
    return get_redis().get(f"ftcancel:{job_id}") == "1"


def now() -> float:
    """Return the current unix timestamp (wrapped for easy testing)."""
    return time.time()
