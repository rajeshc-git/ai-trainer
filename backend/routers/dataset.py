"""Dataset endpoints: template download + CSV validation/upload."""

from __future__ import annotations

import io
import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from pydantic import BaseModel
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


@router.post("/extract-and-clean")
async def extract_and_clean_file(file: UploadFile = File(...)) -> JSONResponse:
    """CEF Pipeline: Process any raw file (PDF, Excel, CSV, TXT, JSON, MD).

    Extracts instruction-response pairs, cleans noise, deduplicates, computes
    a quality score, saves canonical CSV to DATASETS_DIR, and returns dataset_id.
    """
    contents = await file.read()
    size = len(contents)
    filename = file.filename or "uploaded_data.txt"

    try:
        import cef_engine

        cef_result = cef_engine.process_cef_file(filename, contents)
        pairs = cef_result.get("pairs", [])
        if not pairs:
            return JSONResponse(
                status_code=200,
                content={
                    "valid": False,
                    "row_count": 0,
                    "quality_score": 0.0,
                    "sample_rows": [],
                    "file_size_bytes": size,
                    "error": "No valid training pairs could be extracted from that file.",
                    "suggestion": "Make sure your document contains text paragraphs, headers, or Q&A pairs.",
                    "dataset_id": None,
                },
            )

        df = pd.DataFrame(pairs, columns=["instruction", "input", "output"])
        dataset_id = uuid.uuid4().hex
        dest = DATASETS_DIR / f"{dataset_id}.csv"
        df.to_csv(dest, index=False)

        val_result = validate_dataframe(df)
        val_result.update(
            {
                "valid": True,
                "dataset_id": dataset_id,
                "file_size_bytes": size,
                "quality_score": cef_result.get("quality_score", 100.0),
                "raw_items_found": cef_result.get("raw_items_found", len(pairs)),
                "clean_pairs_count": len(pairs),
                "sample_rows": [p for p in pairs[:10]],
            }
        )
        return JSONResponse(status_code=200, content=val_result)

    except Exception as exc:
        return JSONResponse(
            status_code=200,
            content={
                "valid": False,
                "row_count": 0,
                "file_size_bytes": size,
                "error": f"Failed to extract dataset: {exc}",
                "suggestion": "Try uploading a readable PDF, Excel (.xlsx), TXT, or CSV document.",
                "dataset_id": None,
            },
        )


class UrlExtractRequest(BaseModel):
    url: str


@router.post("/extract-url")
async def extract_url_dataset(body: UrlExtractRequest) -> JSONResponse:
    """CEF Pipeline: Scrape and extract fine-tuning pairs directly from a Web URL."""
    url = body.url.strip()
    if not url.startswith(("http://", "https://")):
        return JSONResponse(
            status_code=400,
            detail={
                "error": "Invalid URL format.",
                "suggestion": "URL must start with http:// or https://",
            },
        )

    try:
        import cef_engine

        extracted_pairs = await cef_engine.extract_from_url(url)
        cleaned_pairs = cef_engine.deduplicate_pairs(extracted_pairs)
        quality_score = cef_engine.calculate_quality_score(cleaned_pairs, len(extracted_pairs))

        if not cleaned_pairs:
            return JSONResponse(
                status_code=200,
                content={
                    "valid": False,
                    "row_count": 0,
                    "quality_score": 0.0,
                    "sample_rows": [],
                    "error": "Could not extract readable article text from that URL.",
                    "suggestion": "Verify the URL is publicly accessible and contains text content.",
                    "dataset_id": None,
                },
            )

        pairs_dict = [p.to_dict() for p in cleaned_pairs]
        df = pd.DataFrame(pairs_dict, columns=["instruction", "input", "output"])
        dataset_id = uuid.uuid4().hex
        dest = DATASETS_DIR / f"{dataset_id}.csv"
        df.to_csv(dest, index=False)

        val_result = validate_dataframe(df)
        val_result.update(
            {
                "valid": True,
                "dataset_id": dataset_id,
                "quality_score": quality_score,
                "raw_items_found": len(extracted_pairs),
                "clean_pairs_count": len(cleaned_pairs),
                "sample_rows": pairs_dict[:10],
            }
        )
        return JSONResponse(status_code=200, content=val_result)

    except Exception as exc:
        return JSONResponse(
            status_code=200,
            content={
                "valid": False,
                "row_count": 0,
                "error": f"Failed to scrape URL: {exc}",
                "suggestion": "Check your internet connection and verify the website URL.",
                "dataset_id": None,
            },
        )
