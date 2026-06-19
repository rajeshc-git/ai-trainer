"""Pydantic models shared across the API.

These schemas define the request/response contracts for the
fine-tuning service. Keeping them in one module makes the API
surface easy to audit and reuse from the routers.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Lifecycle states of a training job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class HealthResponse(BaseModel):
    """Returned by ``GET /health``."""

    status: str = Field(..., description="Always 'ok' when the service is up.")
    gpu: bool = Field(..., description="True if a CUDA GPU is available.")
    gpu_name: Optional[str] = Field(
        None, description="Human-readable GPU name, or None on CPU."
    )
    gpu_total_mb: Optional[float] = Field(
        None, description="Total VRAM of the GPU in MB, or None on CPU."
    )


class ColumnPreview(BaseModel):
    """Detection result for a single CSV column."""

    name: str
    present: bool
    null_count: int = 0


class DatasetValidationResponse(BaseModel):
    """Returned by ``POST /api/dataset/validate``."""

    valid: bool
    row_count: int
    columns: list[str]
    detected: list[ColumnPreview]
    sample_rows: list[dict[str, Any]]
    file_size_bytes: int
    error: Optional[str] = None
    suggestion: Optional[str] = None
    dataset_id: Optional[str] = Field(
        None, description="Server-side id of the stored dataset for training."
    )


class TrainRequest(BaseModel):
    """Body of ``POST /api/train``."""

    model_name: str = Field(
        ...,
        description="Hugging Face model id, e.g. 'sshleifer/tiny-gpt2'.",
        examples=["sshleifer/tiny-gpt2"],
    )
    dataset_id: str = Field(..., description="Id returned by /api/dataset/validate.")
    epochs: int = Field(3, ge=1, le=10)
    learning_rate: float = Field(2e-4, gt=0, le=1e-2)
    batch_size: int = Field(4, ge=1, le=16)
    max_length: int = Field(512, ge=64, le=2048)
    output_dir: Optional[str] = Field(
        None, description="Override save directory; defaults to /models/{job_id}."
    )

    # pydantic v2: silence the 'model_' protected-namespace warning
    model_config = {"protected_namespaces": ()}


class TrainResponse(BaseModel):
    """Returned by ``POST /api/train``."""

    job_id: str
    status: JobStatus
    message: str


class JobStatusResponse(BaseModel):
    """Returned by ``GET /api/train/{job_id}/status``."""

    job_id: str
    status: JobStatus
    percent: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0
    total_steps: int = 0
    loss: Optional[float] = None
    eval_loss: Optional[float] = None
    best_eval_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    eta_seconds: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    gpu_total_mb: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    model_name: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    suggestion: Optional[str] = None

    model_config = {"protected_namespaces": ()}


class SavedModel(BaseModel):
    """One entry in ``GET /api/models/list``."""

    job_id: str
    name: str
    base_model: Optional[str] = None
    created_at: Optional[float] = None
    dataset_rows: Optional[int] = None
    final_loss: Optional[float] = None
    best_eval_loss: Optional[float] = None
    size_bytes: int = 0

    model_config = {"protected_namespaces": ()}


class ErrorResponse(BaseModel):
    """Uniform error envelope used across the API."""

    error: str
    suggestion: str
