"""GGUF export engine.

Converts a trained LoRA/QLoRA adapter into a single quantized **GGUF** file that
Ollama, llama.cpp, LM Studio and friends can run directly. This is the official
recommended pipeline:

    1. Merge the LoRA adapter into the base weights (fp16) — the converter needs
       *full* weights, not a delta.
    2. ``convert_hf_to_gguf.py`` → an F16 GGUF.
    3. ``llama-quantize`` → the requested quant (Q4_K_M / Q5_K_M / Q8_0).

Every heavy stage (merge, convert, quantize) runs in its own short-lived
subprocess, so the RAM/VRAM it uses — including PyTorch's CUDA context — is
returned to the OS the moment that step finishes, instead of lingering in the
long-lived API server until the container restarts. The merge uses the GPU when
there's enough free VRAM (falling back to CPU, which needs ~16 GB system RAM for
a 7–8B model). Progress for every stage is streamed to the job's Redis log channel —
the same one the training WebSocket relays — keyed by a fresh ``export_id`` so the
frontend can watch it live with the existing ``/ws/train/{id}`` socket.

Heavy ML imports are lazy, matching ``trainer.py`` / ``inference.py``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

from utils import (
    MODELS_DIR,
    detect_gpu,
    gpu_memory_stats,
    now,
    publish_log,
    release_gpu_memory,
    save_job,
)

# Quant levels offered to the user. Q5_K_M is the default sweet spot (high
# quality, ~⅓ the F16 size); Q8_0 is near-lossless; Q4_K_M is the smallest.
ALLOWED_QUANTS = ("Q4_K_M", "Q5_K_M", "Q8_0")
DEFAULT_QUANT = "Q5_K_M"

LLAMA_CPP_DIR = Path(os.getenv("LLAMA_CPP_DIR", "/opt/llama.cpp"))

# Standalone worker that performs the LoRA→base merge in its own process, so all
# of its RAM/VRAM (including the CUDA context) is reclaimed by the OS on exit.
MERGE_SCRIPT = Path(__file__).resolve().parent / "merge_lora.py"


def _log(export_id: str, msg: str) -> None:
    publish_log(export_id, "INFO", msg)


def _warn(export_id: str, msg: str) -> None:
    publish_log(export_id, "WARNING", msg)


def _err(export_id: str, msg: str) -> None:
    publish_log(export_id, "ERROR", msg)


def gguf_dir(job_id: str) -> Path:
    """Return (and create) the directory holding a model's GGUF exports."""
    d = MODELS_DIR / job_id / "gguf"
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_gguf(job_id: str) -> list[dict[str, Any]]:
    """List exported GGUF files for a model, newest first."""
    d = MODELS_DIR / job_id / "gguf"
    if not d.exists():
        return []
    out = []
    for p in d.glob("*.gguf"):
        quant = next((q for q in ALLOWED_QUANTS if q.lower() in p.name.lower()), None)
        st = p.stat()
        out.append(
            {
                "filename": p.name,
                "quant": quant,
                "size_bytes": st.st_size,
                "created": st.st_mtime,
            }
        )
    out.sort(key=lambda x: x["created"], reverse=True)
    return out


def _convert_script() -> Optional[Path]:
    """Locate llama.cpp's HF→GGUF converter (name has changed over time)."""
    for name in ("convert_hf_to_gguf.py", "convert-hf-to-gguf.py"):
        p = LLAMA_CPP_DIR / name
        if p.exists():
            return p
    return None


def _quantize_bin() -> Optional[Path]:
    """Locate the built ``llama-quantize`` binary (name/path varies by build)."""
    candidates = [
        LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize",
        LLAMA_CPP_DIR / "build" / "bin" / "quantize",
        LLAMA_CPP_DIR / "llama-quantize",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# Matches llama.cpp's per-tensor progress counter, e.g. "[  96/ 339]".
_PROGRESS_RE = re.compile(r"\[\s*(\d+)\s*/\s*(\d+)\s*\]")


def _run_streaming(
    export_id: str,
    cmd: list[str],
    label: str,
    cwd: Optional[str] = None,
    progress: Optional[tuple[float, float]] = None,
) -> None:
    """Run a subprocess, streaming each output line to the job log.

    Output is line-buffered (``stdbuf`` when available, plus ``PYTHONUNBUFFERED``)
    so progress shows up the instant the child prints it, instead of being held
    in libc's 4 KB block buffer until the pipe fills — which made long stages
    look frozen. When ``progress=(lo, hi)`` is given and the child emits
    llama.cpp-style ``[ n/ m]`` counters, the job percent is interpolated across
    that range so the UI advances during the stage.
    """
    # Force line buffering on the child so its output streams live. stdbuf wraps
    # any binary (incl. the C++ llama-quantize); PYTHONUNBUFFERED covers the
    # Python converter/merge workers.
    run_cmd = list(cmd)
    stdbuf = shutil.which("stdbuf")
    if stdbuf:
        run_cmd = [stdbuf, "-oL", "-eL", *run_cmd]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    _log(export_id, f"$ {label}")
    proc = subprocess.Popen(
        run_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=cwd or str(LLAMA_CPP_DIR),
        env=env,
    )
    assert proc.stdout is not None
    last_pct = -1.0
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        publish_log(export_id, "INFO", line)
        if progress:
            m = _PROGRESS_RE.search(line)
            if m:
                n, total = int(m.group(1)), int(m.group(2))
                if total > 0:
                    lo, hi = progress
                    pct = lo + (hi - lo) * (n / total)
                    if pct - last_pct >= 1.0:  # throttle Redis writes
                        last_pct = pct
                        save_job(export_id, {"percent": round(pct, 1)})
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed (exit code {proc.returncode}).")


def _sanitize(name: str) -> str:
    """Make a model name safe for a filename."""
    keep = "-_.()"
    return "".join(c if (c.isalnum() or c in keep) else "-" for c in name).strip("-") or "model"


def _estimate_params_b(model_id: str) -> Optional[float]:
    """Best-effort parameter count (in billions) parsed from a model id.

    e.g. 'Mistral-7B' → 7, 'Phi-3-mini... 3.8b' → 3.8, 'pythia-160m' → 0.16,
    'Mixtral-8x7B' → ~56. Returns None when nothing parseable is found.
    """
    s = (model_id or "").lower()
    moe = re.search(r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*b", s)
    if moe:
        return float(moe.group(1)) * float(moe.group(2))
    b = re.search(r"(\d+(?:\.\d+)?)\s*b(?:[^a-z]|$)", s)
    if b:
        return float(b.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*m(?:[^a-z]|$)", s)
    if m:
        return float(m.group(1)) / 1000
    return None


def _pick_merge_device(base_model_name: str) -> tuple[str, str]:
    """Choose ``'cuda'`` or ``'cpu'`` for the fp16 merge based on live free VRAM.

    The merge needs the base model in fp16 (~2 bytes/param). We use the GPU (fast)
    only when the card has comfortably more *free* VRAM than that needs (~1.3×) and
    isn't already busy with another job; otherwise the CPU (safe — uses system RAM
    and never disturbs a training/inference run on the GPU). Returns
    ``(device, human_reason)``.
    """
    has_gpu, gpu_name = detect_gpu()
    if not has_gpu:
        return "cpu", "no CUDA GPU detected"
    stats = gpu_memory_stats()
    if not stats:
        return "cpu", "GPU memory stats unavailable"

    free = stats["free_mb"]
    total = stats["total_mb"] or 0.0
    params_b = _estimate_params_b(base_model_name)
    fp16_mb = params_b * 2 * 1024 if params_b else None  # ~2 bytes per parameter
    needed_mb = fp16_mb * 1.3 if fp16_mb else None        # +30% working headroom

    # If a large chunk of VRAM is already in use, a job is likely running — leave
    # the GPU alone and merge on CPU.
    if total and (free / total) < 0.6:
        return "cpu", f"GPU busy ({free:.0f}/{total:.0f} MB free) — not disturbing it"
    if needed_mb is not None:
        if free >= needed_mb:
            return "cuda", f"{free:.0f} MB free ≥ ~{needed_mb:.0f} MB needed ({gpu_name})"
        return "cpu", f"only {free:.0f} MB free, ~{needed_mb:.0f} MB needed for fp16"
    # Unknown size — only risk the GPU with a generous floor of free VRAM.
    if free >= 14000:
        return "cuda", f"{free:.0f} MB free, size unknown ({gpu_name})"
    return "cpu", f"size unknown and only {free:.0f} MB free"


def run_export(export_id: str, job_id: str, quant: str) -> None:
    """Top-level export entry point. Blocking; run in a daemon thread."""
    try:
        _export_impl(export_id, job_id, quant)
    except Exception as exc:  # noqa: BLE001 - report everything to the user
        _err(export_id, f"GGUF export failed: {exc}")
        _err(export_id, traceback.format_exc())
        save_job(
            export_id,
            {
                "status": "failed",
                "error": "GGUF export failed.",
                "suggestion": str(exc),
                "finished_at": now(),
            },
        )


def _export_impl(export_id: str, job_id: str, quant: str) -> None:
    import json

    quant = (quant or DEFAULT_QUANT).upper()
    if quant not in ALLOWED_QUANTS:
        raise ValueError(
            f"Unsupported quantization '{quant}'. Choose one of {', '.join(ALLOWED_QUANTS)}."
        )

    model_dir = MODELS_DIR / job_id
    if not model_dir.exists() or not (model_dir / "adapter_config.json").exists():
        raise FileNotFoundError(
            "That trained model could not be found (no adapter on disk)."
        )

    # Metadata / architecture gate — GGUF + llama.cpp target causal LMs.
    meta = {}
    meta_path = model_dir / "ft_metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
    if meta.get("architecture") == "seq2seq":
        raise ValueError(
            "GGUF export supports causal (GPT-style) models only — this is a "
            "seq2seq (T5-style) model. Use the .zip download for it instead."
        )

    convert = _convert_script()
    qbin = _quantize_bin()
    if convert is None or qbin is None:
        raise RuntimeError(
            "llama.cpp tooling not found in the image. Rebuild with "
            "`docker compose build` so the converter + llama-quantize are present."
        )

    base_name = meta.get("name") or model_dir.name
    out_name = f"{_sanitize(base_name)}-{quant}.gguf"
    out_dir = gguf_dir(job_id)
    out_path = out_dir / out_name

    save_job(
        export_id,
        {
            "status": "running",
            "kind": "gguf_export",
            "model_job_id": job_id,
            "quant": quant,
            "percent": 0.0,
            "started_at": now(),
        },
    )
    _log(export_id, f"Starting GGUF export ({quant}) for '{base_name}'.")

    tmp_dir = out_dir / f".tmp-{export_id[:8]}"
    merged_dir = tmp_dir / "merged"
    f16_path = tmp_dir / "model-f16.gguf"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── 1) Merge LoRA into fp16 base weights (isolated worker) ──
        # The merge loads the FULL base model (~2 bytes/param) into RAM/VRAM. We
        # run it in a short-lived subprocess so that when it exits the OS reclaims
        # ALL of that memory — including PyTorch's CUDA context, which an
        # in-process empty_cache() can never release until the server restarts.
        adapter_cfg = json.loads((model_dir / "adapter_config.json").read_text())
        base_model_name = adapter_cfg.get("base_model_name_or_path") or model_dir.name
        _log(export_id, f"Base model: {base_model_name}")

        # GPU when there's comfortably enough free VRAM (fast), else CPU (safe;
        # uses system RAM, won't disturb the GPU). The worker re-checks at load
        # time and falls back to CPU itself if the GPU OOMs.
        device, reason = _pick_merge_device(base_model_name)
        _log(export_id, f"Merge device: {device.upper()} — {reason}")
        _log(
            export_id,
            "Merging LoRA adapter into the base model (fp16) in an isolated worker"
            " — this can take a few minutes"
            + (", ~16 GB system RAM for a 7–8B model." if device == "cpu" else ".")
            + " Its RAM/VRAM is returned to the OS the moment it finishes.",
        )
        save_job(export_id, {"percent": 10.0})

        merged_dir.mkdir(parents=True, exist_ok=True)
        _run_streaming(
            export_id,
            [
                sys.executable,
                str(MERGE_SCRIPT),
                "--adapter-dir", str(model_dir),
                "--out-dir", str(merged_dir),
                "--device", device,
            ],
            label="merge_lora.py → merged fp16 model",
            cwd=str(MERGE_SCRIPT.parent),
        )
        if not (merged_dir / "config.json").exists():
            raise RuntimeError("The merge worker did not produce a merged model.")
        save_job(export_id, {"percent": 40.0})

        # ── 2) Convert merged HF model → F16 GGUF ───────────────
        _log(export_id, "Converting merged model to F16 GGUF…")
        _run_streaming(
            export_id,
            [
                sys.executable,
                str(convert),
                str(merged_dir),
                "--outfile",
                str(f16_path),
                "--outtype",
                "f16",
            ],
            label=f"convert_hf_to_gguf.py → {f16_path.name}",
        )
        if not f16_path.exists():
            raise RuntimeError("Conversion did not produce an F16 GGUF file.")
        save_job(export_id, {"percent": 70.0})

        # ── 3) Quantize F16 GGUF → requested quant ──────────────
        # Quantization is CPU-bound and embarrassingly parallel; tell
        # llama-quantize to use every core (its positional `nthreads` arg).
        # Thread count affects only speed, never the output bytes.
        nthreads = str(os.cpu_count() or 1)
        _log(export_id, f"Quantizing to {quant} using {nthreads} threads…")
        _run_streaming(
            export_id,
            [str(qbin), str(f16_path), str(out_path), quant, nthreads],
            label=f"llama-quantize → {out_name}",
            progress=(70.0, 99.0),
        )
        if not out_path.exists():
            raise RuntimeError("Quantization did not produce a GGUF file.")

        size_mb = out_path.stat().st_size / (1024 * 1024)
        save_job(
            export_id,
            {
                "status": "completed",
                "percent": 100.0,
                "gguf_file": out_name,
                "gguf_size_bytes": out_path.stat().st_size,
                "finished_at": now(),
            },
        )
        _log(
            export_id,
            f"Done! Exported {out_name} ({size_mb:.0f} MB). "
            f"Run it with Ollama or llama.cpp.",
        )
    finally:
        # The merge / convert / quantize steps each ran in their own subprocess
        # and freed their RAM/VRAM by exiting; here we just drop the big on-disk
        # intermediates (merged dir + f16 GGUF) and any transient parent state.
        release_gpu_memory()
        shutil.rmtree(tmp_dir, ignore_errors=True)
