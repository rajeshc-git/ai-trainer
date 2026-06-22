"""Hugging Face / Unsloth model search + details.

Lets the frontend search the Hub live for openly-licensed *and* gated/private
models (Llama, Gemma, …) to fine-tune. "Unsloth" is the ``unsloth/`` Hub org,
whose pre-quantized 4-bit checkpoints are great QLoRA starting points — so a
single Hub search with an optional ``author=unsloth`` filter covers both portals.

* ``GET /api/hf/search`` — cheap list (id, downloads, likes, gated). Params are
  *estimated* from the id client-side for an instant VRAM-fit hint.
* ``GET /api/hf/info``   — exact parameter count + download size for one model,
  used when the user selects it so the fit check is precise.

The HF token (for gated/private models) is read from the environment, exactly as
``transformers``/``huggingface_hub`` do, so existing ``.env`` config just works.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/hf", tags=["search"])


def _token() -> Optional[str]:
    """Return the configured HF token (gated/private access), or None."""
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or None


def _gated_flag(value: Any) -> bool:
    """Normalise the Hub's gated field (False | 'auto' | 'manual') to a bool."""
    return bool(value) and value not in (False, "false", "False")


@router.get("/search")
def search_models(
    q: str = Query("", description="Free-text query, e.g. 'llama' or 'qwen instruct'."),
    source: str = Query("all", description="'all' or 'unsloth' (the unsloth/ org)."),
    limit: int = Query(24, ge=1, le=50),
) -> JSONResponse:
    """Search the Hugging Face Hub for text-generation models to fine-tune.

    An empty query returns the most-downloaded models (a sensible "trending"
    default). ``source=unsloth`` restricts results to the ``unsloth/`` org.
    """
    try:
        from huggingface_hub import HfApi
    except Exception as exc:  # pragma: no cover - hub missing on dev box
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model search is unavailable.",
                "suggestion": f"huggingface_hub failed to import ({exc}).",
            },
        ) from exc

    api = HfApi(token=_token())
    kwargs: dict[str, Any] = {
        "search": q or None,
        "pipeline_tag": "text-generation",
        "sort": "downloads",
        "limit": limit,
    }
    if source == "unsloth":
        kwargs["author"] = "unsloth"

    try:
        results = list(api.list_models(**kwargs))
    except TypeError:
        # Older/newer signature differences — retry with fallback parameters
        fallback = {
            "search": q or None,
            "sort": "downloads",
            "limit": limit,
        }
        if source == "unsloth":
            fallback["author"] = "unsloth"
        results = list(api.list_models(**fallback))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail={
                "error": "Could not reach Hugging Face.",
                "suggestion": f"Check the backend's internet connection. ({exc})",
            },
        ) from exc

    models = []
    for m in results:
        mid = getattr(m, "id", None) or getattr(m, "modelId", None)
        if not mid:
            continue
        models.append(
            {
                "id": mid,
                "downloads": getattr(m, "downloads", None),
                "likes": getattr(m, "likes", None),
                "pipeline_tag": getattr(m, "pipeline_tag", None),
                "gated": _gated_flag(getattr(m, "gated", None)),
                "private": bool(getattr(m, "private", False)),
            }
        )
    return JSONResponse(status_code=200, content={"models": models})


@router.get("/info")
def model_info(model_id: str = Query(..., description="HF repo id.")) -> JSONResponse:
    """Return the exact parameter count + total download size for one model.

    Used on selection so the VRAM-fit check and "download size" are precise
    rather than estimated from the model name.
    """
    try:
        from huggingface_hub import HfApi
    except Exception as exc:  # pragma: no cover
        raise HTTPException(
            status_code=503,
            detail={"error": "Model info is unavailable.", "suggestion": str(exc)},
        ) from exc

    api = HfApi(token=_token())

    # Call 1 — base details + per-file sizes (``files_metadata`` populates
    # ``siblings[].size``). This also carries gated/downloads/likes/pipeline_tag.
    try:
        info = api.model_info(model_id, files_metadata=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Could not load model details.",
                "suggestion": (
                    "Check the model id. Gated/private models need an HF_TOKEN "
                    f"in your .env. ({exc})"
                ),
            },
        ) from exc

    # Total download size = sum of repo file sizes (best-effort).
    size_bytes = 0
    for sib in getattr(info, "siblings", None) or []:
        sz = getattr(sib, "size", None)
        if isinstance(sz, (int, float)):
            size_bytes += int(sz)

    # Call 2 — exact parameter count from the safetensors index. ``expand`` is
    # mutually exclusive with ``files_metadata``, so this must be a separate call.
    params: Optional[int] = None
    try:
        st_info = api.model_info(model_id, expand=["safetensors"])
        st = getattr(st_info, "safetensors", None)
        total = getattr(st, "total", None) if st is not None else None
        if isinstance(total, (int, float)) and total > 0:
            params = int(total)
    except Exception:
        params = None  # not all repos publish a safetensors index; estimate by name

    return JSONResponse(
        status_code=200,
        content={
            "id": model_id,
            "params": params,
            "gated": _gated_flag(getattr(info, "gated", None)),
            "private": bool(getattr(info, "private", False)),
            "downloads": getattr(info, "downloads", None),
            "likes": getattr(info, "likes", None),
            "pipeline_tag": getattr(info, "pipeline_tag", None),
            "size_bytes": size_bytes or None,
        },
    )
