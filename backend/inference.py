"""Inference engine for trained models.

Loads a fine-tuned LoRA adapter on top of its base model and generates text.
A small LRU-style cache keeps recently-used models resident in (GPU) memory so
repeated chat turns are fast; the least-recently-used model is evicted when the
cache is full. A background reaper also auto-unloads any model left idle past
``INFERENCE_IDLE_TTL_SECONDS`` (default 600s), so VRAM/RAM is returned to the OS
on its own without the user clicking "Free memory".

Like ``trainer.py``, the heavy ML imports are done lazily so the API process
starts quickly and stays usable even where those wheels can't be imported.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

from utils import (
    MODELS_DIR,
    build_prompt,
    build_seq2seq_src,
    detect_gpu,
    release_gpu_memory,
)

# job_id -> (model, tokenizer, metadata)
_CACHE: "OrderedDict[str, tuple[Any, Any, dict]]" = OrderedDict()
_CACHE_LOCK = threading.Lock()
_MAX_CACHED = 2

# Auto-evict models that haven't been used in a while so their VRAM/RAM is
# returned to the OS without the user having to click "Free memory". Set
# INFERENCE_IDLE_TTL_SECONDS=0 to disable. The reaper thread wakes periodically
# and unloads anything idle longer than the TTL.
_IDLE_TTL = float(os.getenv("INFERENCE_IDLE_TTL_SECONDS", "600"))
_REAPER_INTERVAL = 60.0
_LAST_USED: dict[str, float] = {}  # job_id -> monotonic timestamp of last use
_reaper_started = False


def _touch(job_id: str) -> None:
    """Mark a model as just-used (call while holding ``_CACHE_LOCK``)."""
    _LAST_USED[job_id] = time.monotonic()


def _reap_idle_once() -> list[str]:
    """Evict every model idle longer than the TTL. Returns the job ids dropped."""
    if _IDLE_TTL <= 0:
        return []
    now_t = time.monotonic()
    expired: list[str] = []
    with _CACHE_LOCK:
        for jid in list(_CACHE.keys()):
            if now_t - _LAST_USED.get(jid, now_t) >= _IDLE_TTL:
                _CACHE.pop(jid, None)
                _LAST_USED.pop(jid, None)
                expired.append(jid)
    # Hand the freed blocks back to the GPU outside the lock.
    if expired:
        release_gpu_memory()
    return expired


def _reaper_loop() -> None:
    while True:
        time.sleep(_REAPER_INTERVAL)
        try:
            dropped = _reap_idle_once()
            if dropped:
                print(
                    f"[inference] Auto-unloaded {len(dropped)} idle model(s) "
                    f"after {_IDLE_TTL:.0f}s: {', '.join(dropped)}",
                    flush=True,
                )
        except Exception:  # noqa: BLE001 - a reaper hiccup must never crash the app
            pass


def _ensure_reaper() -> None:
    """Start the idle-eviction thread once, lazily, on first model load."""
    global _reaper_started
    if _reaper_started or _IDLE_TTL <= 0:
        return
    with _CACHE_LOCK:
        if _reaper_started:
            return
        _reaper_started = True
        threading.Thread(target=_reaper_loop, name="inference-reaper", daemon=True).start()


class InferenceError(Exception):
    """Raised with a friendly (message, suggestion) for the API layer."""

    def __init__(self, message: str, suggestion: str) -> None:
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion


def _model_dir(job_id: str) -> Path:
    """Return the on-disk directory for a saved model, validating existence."""
    path = MODELS_DIR / job_id
    if not path.exists() or not path.is_dir():
        raise InferenceError(
            "That model could not be found.",
            "It may have been deleted. Refresh the My Models page.",
        )
    return path


def _load_metadata(model_dir: Path) -> dict:
    """Read the ft_metadata.json sidecar (best-effort)."""
    meta_path = model_dir / "ft_metadata.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except Exception:
            return {}
    return {}


def _load_model(job_id: str) -> tuple[Any, Any, dict]:
    """Load (or fetch from cache) the model+tokenizer for a job.

    Returns:
        A tuple ``(model, tokenizer, metadata)``.
    """
    _ensure_reaper()
    with _CACHE_LOCK:
        if job_id in _CACHE:
            _CACHE.move_to_end(job_id)
            _touch(job_id)
            return _CACHE[job_id]

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
    )

    model_dir = _model_dir(job_id)
    meta = _load_metadata(model_dir)
    is_seq2seq = meta.get("architecture") == "seq2seq"
    has_gpu, _ = detect_gpu()

    try:
        # The saved directory is a PEFT adapter; discover its base model.
        peft_config = PeftConfig.from_pretrained(str(model_dir))
        base_model_name = peft_config.base_model_name_or_path

        model_cls = AutoModelForSeq2SeqLM if is_seq2seq else AutoModelForCausalLM
        load_kwargs: dict[str, Any] = {}
        if has_gpu:
            # Load the base model in 4-bit, exactly as it was trained (QLoRA).
            # Loading it in full fp16 instead would need ~2x the VRAM, overflow a
            # 16GB card, and make device_map="auto" offload layers to system RAM
            # — which is what makes a quantized model answer at CPU speed (a "hi"
            # taking minutes). Matching the 4-bit load keeps it entirely on GPU.
            use_bf16 = torch.cuda.is_bf16_supported()
            compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                )
            except Exception:
                # bitsandbytes unavailable — fall back to fp16 (may not fit big models).
                load_kwargs["torch_dtype"] = compute_dtype
            load_kwargs["device_map"] = "auto"

        base = model_cls.from_pretrained(base_model_name, **load_kwargs)
        model = PeftModel.from_pretrained(base, str(model_dir))
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    except InferenceError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise InferenceError(
            "The model could not be loaded for inference.",
            f"Make sure the base model is still available on Hugging Face. ({exc})",
        ) from exc

    entry = (model, tokenizer, {**meta, "is_seq2seq": is_seq2seq})
    evicted = []
    with _CACHE_LOCK:
        _CACHE[job_id] = entry
        _CACHE.move_to_end(job_id)
        _touch(job_id)
        while len(_CACHE) > _MAX_CACHED:
            old_id, old_entry = _CACHE.popitem(last=False)
            _LAST_USED.pop(old_id, None)
            evicted.append(old_entry)
    # Free the evicted models' VRAM outside the lock. Dropping the dict entry
    # alone is not enough — the caching allocator keeps the memory reserved
    # until we drop the refs and empty the cache.
    if evicted:
        del evicted
        release_gpu_memory()
    return entry


def generate(
    job_id: str,
    message: str,
    *,
    instruction: Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> str:
    """Generate a single response from a fine-tuned model.

    The user message is wrapped in the same instruction prompt format used
    during training so the model sees familiar inputs.

    Args:
        job_id: Which saved model to use.
        message: The user's input text.
        instruction: Optional system/instruction string.
        max_new_tokens: Cap on generated tokens.
        temperature: Sampling temperature (0 = greedy).
        top_p: Nucleus sampling cutoff.

    Returns:
        The model's generated response, with the prompt stripped.

    Raises:
        InferenceError: With a friendly message on any failure.
    """
    import torch

    model, tokenizer, meta = _load_model(job_id)
    is_seq2seq = bool(meta.get("is_seq2seq"))

    row = {"instruction": instruction or "", "input": message, "output": ""}
    if is_seq2seq:
        # Match the encoder-input format used during seq2seq training.
        prompt = build_seq2seq_src(row)
    else:
        # Causal models continue from the "### Response:" cue.
        prompt = build_prompt(row).rstrip()

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": temperature > 0,
            "temperature": max(temperature, 1e-4),
            "top_p": top_p,
            "pad_token_id": tokenizer.pad_token_id,
        }

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        if is_seq2seq:
            text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return text.strip()

        # Causal: strip the prompt tokens, decode only the new tokens.
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][prompt_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        # Cut at the next instruction marker if the model rambles on.
        for marker in ("### Instruction", "### Input", "### Response"):
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx]
        return text.strip()
    except Exception as exc:  # noqa: BLE001
        text = str(exc).lower()
        if "out of memory" in text:
            raise InferenceError(
                "Not enough GPU memory to run this model.",
                "Try a smaller base model, or reduce max new tokens.",
            ) from exc
        raise InferenceError(
            "The model failed to generate a response.",
            "Try again, or pick a different model.",
        ) from exc


def unload(job_id: str) -> bool:
    """Evict one model from the cache and free its VRAM/RAM.

    Called when a model is deleted, or on demand from the "Stop / Unload"
    button so the user can reclaim memory like ejecting a model in LM Studio.

    Returns:
        True if a resident model was unloaded, False if it wasn't loaded.
    """
    with _CACHE_LOCK:
        entry = _CACHE.pop(job_id, None)
        _LAST_USED.pop(job_id, None)
    if entry is None:
        return False
    # Drop our reference, then hand the freed blocks back to the GPU.
    del entry
    release_gpu_memory()
    return True


def unload_all() -> int:
    """Evict every cached model and free all the memory they held.

    Returns:
        The number of models that were unloaded.
    """
    with _CACHE_LOCK:
        count = len(_CACHE)
        _CACHE.clear()
        _LAST_USED.clear()
    if count:
        release_gpu_memory()
    return count


def loaded_models() -> list[str]:
    """Return the job ids of the models currently resident in memory."""
    with _CACHE_LOCK:
        return list(_CACHE.keys())
