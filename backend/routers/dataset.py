"""Dataset endpoints: template download + CSV validation/upload."""

from __future__ import annotations

import io
import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from schemas import DatasetValidationResponse
from utils import DATASETS_DIR, validate_dataframe

router = APIRouter(prefix="/api/dataset", tags=["dataset"])


# A small, friendly example dataset used as the downloadable template.
_TEMPLATE_ROWS = [
    {
        "instruction": "Translate the English sentence to French.",
        "input": "Hello, how are you?",
        "output": "Bonjour, comment allez-vous ?",
    },
    {
        "instruction": "Translate the English sentence to French.",
        "input": "I love programming.",
        "output": "J'adore la programmation.",
    },
    {
        "instruction": "Translate the English sentence to French.",
        "input": "Where is the train station?",
        "output": "Où est la gare ?",
    },
]


@router.get("/template")
def download_template() -> StreamingResponse:
    """Stream a ready-to-edit CSV template with the correct columns.

    The ``instruction`` column is optional but included to demonstrate
    instruction-tuning datasets.
    """
    df = pd.DataFrame(_TEMPLATE_ROWS, columns=["instruction", "input", "output"])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="dataset_template.csv"'
        },
    )


@router.post("/validate", response_model=DatasetValidationResponse)
async def validate_dataset(file: UploadFile = File(...)) -> JSONResponse:
    """Validate an uploaded CSV and (if valid) persist it for training.

    Returns column detection, a 5-row preview, the row count and file size.
    On success a ``dataset_id`` is returned which the caller passes to
    ``POST /api/train``.
    """
    contents = await file.read()
    size = len(contents)

    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        return JSONResponse(
            status_code=200,
            content={
                "valid": False,
                "row_count": 0,
                "columns": [],
                "detected": [],
                "sample_rows": [],
                "file_size_bytes": size,
                "error": "That file could not be read as a CSV.",
                "suggestion": "Make sure you uploaded a comma-separated .csv file.",
                "dataset_id": None,
            },
        )

    result = validate_dataframe(df)
    result["file_size_bytes"] = size
    result["dataset_id"] = None

    if result["valid"]:
        dataset_id = uuid.uuid4().hex
        dest = DATASETS_DIR / f"{dataset_id}.csv"
        df.to_csv(dest, index=False)
        result["dataset_id"] = dataset_id

    return JSONResponse(status_code=200, content=result)
