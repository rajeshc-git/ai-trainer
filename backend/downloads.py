"""Model download with live, streamed progress.

Hugging Face models are normally fetched silently the first time
``from_pretrained`` is called. To give the user a real-time "downloading 45%"
read-out in the training log UI, we pre-fetch the model with
``huggingface_hub.snapshot_download`` using a custom ``tqdm`` class that
publishes aggregate progress to the job's Redis log channel. The snapshot lands
in the shared HF cache (``HF_HOME``), so the subsequent ``from_pretrained`` is a
fast cache hit with no second download.

Heavy imports are done lazily (matching ``trainer.py``) so importing this module
is cheap even where the ML wheels are unavailable.
"""

from __future__ import annotations

import os
import time
from typing import Optional

from utils import publish_log


def _fmt_gb(n_bytes: float) -> str:
    """Format a byte count as a compact GB string."""
    return f"{n_bytes / (1024 ** 3):.2f} GB"


def ensure_model_downloaded(
    job_id: str, model_id: str, token: Optional[str] = None
) -> Optional[str]:
    """Pre-download ``model_id`` into the HF cache, streaming progress to logs.

    Args:
        job_id: Job/export id whose log channel receives progress lines.
        model_id: Hugging Face repo id (e.g. ``Qwen/Qwen2.5-7B-Instruct``).
        token: Optional HF token for gated/private models. Falls back to the
            ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` environment variables.

    Returns:
        The local snapshot directory path, or ``None`` if the download could not
        be performed via the hub API (caller should fall back to a plain load).
    """
    token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or None

    try:
        from huggingface_hub import snapshot_download
        from tqdm.auto import tqdm as _base_tqdm
    except Exception as exc:  # pragma: no cover - hub/tqdm missing on dev box
        publish_log(job_id, "WARNING", f"Could not import download tools ({exc}).")
        return None

    state = {"last": 0.0}

    class _LogTqdm(_base_tqdm):
        """A tqdm subclass that streams throttled *byte* download progress to logs.

        ``huggingface_hub`` creates many tqdm bars: per-file byte bars (``unit ==
        'B'``) and an overall "Fetching N files" count bar. We only report the
        byte bars — that's the real "downloading X%" the user wants — throttled to
        ~1 line/sec so the log stays readable on big multi-shard models.
        """

        def update(self, n: float = 1):  # noqa: ANN201
            res = super().update(n)
            try:
                # Only byte-progress bars carry a real download percentage.
                if getattr(self, "unit", "") != "B":
                    return res
                total = float(self.total or 0)
                if total <= 0:
                    return res
                now_t = time.time()
                frac = min(1.0, float(self.n) / total)
                # Emit on each ~1s tick and at completion.
                if now_t - state["last"] >= 1.0 or frac >= 1.0:
                    state["last"] = now_t
                    desc = (getattr(self, "desc", "") or "model").strip().rstrip(":")
                    publish_log(
                        job_id,
                        "INFO",
                        f"Downloading {desc}: {frac * 100:.0f}% "
                        f"({_fmt_gb(self.n)}/{_fmt_gb(total)})",
                    )
            except Exception:
                pass
            return res

    # The per-file byte bars in huggingface_hub are created inside
    # ``file_download`` with its module-level ``tqdm`` (NOT the ``tqdm_class`` we
    # pass to ``snapshot_download`` — that only styles the outer files bar). So we
    # temporarily swap that symbol to capture real byte progress, then restore it.
    fd_mod = None
    orig_tqdm = None
    try:
        import huggingface_hub.file_download as fd_mod  # type: ignore

        if hasattr(fd_mod, "tqdm"):
            orig_tqdm = fd_mod.tqdm
            fd_mod.tqdm = _LogTqdm  # type: ignore[attr-defined]
    except Exception:
        fd_mod = None

    publish_log(job_id, "INFO", f"Fetching '{model_id}' from Hugging Face…")
    try:
        local_dir = snapshot_download(
            model_id,
            token=token,
            tqdm_class=_LogTqdm,
        )
        publish_log(job_id, "INFO", f"Download complete: '{model_id}'.")
        return local_dir
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, then re-fall-back
        # Don't fail the whole job here — let the normal loader try (it may hit
        # cache, or raise a clearer error that _humanize_error can translate).
        publish_log(
            job_id,
            "WARNING",
            f"Streamed download unavailable ({exc}); falling back to a direct load.",
        )
        return None
    finally:
        if fd_mod is not None and orig_tqdm is not None:
            try:
                fd_mod.tqdm = orig_tqdm  # type: ignore[attr-defined]
            except Exception:
                pass
