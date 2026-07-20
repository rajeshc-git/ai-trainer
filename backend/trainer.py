"""Fine-tuning engine.

Runs a LoRA + 4-bit (QLoRA-style) supervised fine-tune of a Hugging Face
model on a user-supplied CSV dataset. Supports both causal LM (GPT-style)
and seq2seq (T5-style) architectures, auto-detected from the model config.

Live progress (per-step loss, lr, eta, GPU memory) is streamed to the
frontend by publishing JSON log lines to a Redis channel which the
WebSocket endpoint relays to the browser.

The heavy ML imports (torch, transformers, ...) are performed lazily inside
``run_training`` so the FastAPI process can start fast and the rest of the
API works even on a box where those wheels are not importable.
"""

from __future__ import annotations

import json
import shutil
import traceback
from pathlib import Path
from typing import Any, Optional

# ─────────────────────────────────────────────────────────────
# Enterprise training defaults
# ─────────────────────────────────────────────────────────────
# A fixed seed makes a run reproducible (same data + config → same weights),
# which matters for audited/regulated workloads.
TRAIN_SEED = 42
# Below this many rows a held-out split is too small to give a meaningful
# eval signal, so we skip it (and tell the user) rather than mislead.
MIN_ROWS_FOR_EVAL = 16
# Never spend more than this many rows on validation, even on larger sets.
MAX_EVAL_ROWS = 200
# Stop early after this many epochs with no eval_loss improvement (overfit guard).
EARLY_STOPPING_PATIENCE = 3

from utils import (
    MODELS_DIR,
    build_prompt,
    build_seq2seq_src,
    detect_gpu,
    extract_model_id,
    gpu_memory_stats,
    gpu_total_mb,
    is_cancelled,
    load_job,
    load_tokenizer,
    publish_log,
    get_redis,
    now,
    save_job,
)


# ─────────────────────────────────────────────────────────────
# Log streaming
# ─────────────────────────────────────────────────────────────
def _publish(job_id: str, level: str, message: str) -> None:
    """Publish a single structured log line to the job's Redis channel.

    Thin wrapper around :func:`utils.publish_log` (shared with the downloader
    and GGUF exporter) so every subsystem streams into the same live-log channel.
    """
    publish_log(job_id, level, message)


def log_info(job_id: str, msg: str) -> None:
    """Stream an INFO-level line."""
    _publish(job_id, "INFO", msg)


def log_warn(job_id: str, msg: str) -> None:
    """Stream a WARNING-level line."""
    _publish(job_id, "WARNING", msg)


def log_error(job_id: str, msg: str) -> None:
    """Stream an ERROR-level line."""
    _publish(job_id, "ERROR", msg)


# ─────────────────────────────────────────────────────────────
# GPU cleanup
# ─────────────────────────────────────────────────────────────
def _free_gpu_memory(*objs: Any) -> None:
    """Release GPU/host memory held by training objects.

    PyTorch's caching allocator keeps VRAM *reserved* after Python frees the
    tensors, so without an explicit ``empty_cache`` the previous run's weights
    and optimizer state stay on the card until the whole process exits. That
    leftover reservation is what makes a *second* training job fail with
    "Some modules are dispatched on the CPU or the disk" — ``device_map='auto'``
    sees the card as nearly full and offloads layers.

    Pass any references that must die first (model, trainer, optimizer); they
    are deleted, then a GC pass + ``torch.cuda.empty_cache()`` returns the
    freed blocks to the driver so ``nvidia-smi`` drops back to baseline.
    """
    # Best-effort: explicitly drop the optimizer's GPU state if we were handed
    # a Trainer, since that (paged_adamw_8bit) is often the largest consumer.
    for obj in objs:
        try:
            opt = getattr(obj, "optimizer", None)
            if opt is not None:
                obj.optimizer = None
                del opt
        except Exception:
            pass
    del objs

    from utils import release_gpu_memory

    release_gpu_memory()


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────
def run_training(job_id: str, params: dict[str, Any]) -> None:
    """Execute a full fine-tuning job. Blocking; run in a background thread.

    Args:
        job_id: Unique id for this job.
        params: A dict with keys ``model_name``, ``dataset_path``,
            ``dataset_rows``, ``epochs``, ``learning_rate``, ``batch_size``,
            ``max_length`` and optionally ``output_dir``.
    """
    try:
        _train_impl(job_id, params)
    except Exception as exc:  # noqa: BLE001 - top-level guard, report everything
        error, suggestion = _humanize_error(exc, params.get("model_name", ""))
        log_error(job_id, f"{error} — {suggestion}")
        log_error(job_id, traceback.format_exc())
        save_job(
            job_id,
            {
                "status": "failed",
                "error": error,
                "suggestion": suggestion,
                "finished_at": now(),
            },
        )


def _train_impl(job_id: str, params: dict[str, Any]) -> None:
    """The real training routine (kept separate for clean error wrapping).

    The body is wrapped so that whichever way it exits — completed, cancelled
    or raising — the model and trainer are torn down and the GPU cache is
    emptied. This is what lets a subsequent job reuse the card; see
    :func:`_free_gpu_memory`.
    """
    model = None
    trainer = None
    try:
        model, trainer = _train_body(job_id, params)
    finally:
        _free_gpu_memory(trainer, model)
        log_info(job_id, "Released GPU memory.")


def _train_body(job_id: str, params: dict[str, Any]) -> tuple[Any, Any]:
    """Run the fine-tune and return ``(model, trainer)`` for cleanup."""
    # pyrefly: ignore [missing-import]
    import torch
    from datasets import Dataset
    # pyrefly: ignore [missing-import]
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        TrainerCallback,
        set_seed,
    )
    import pandas as pd

    # Reproducibility: seed everything (Python/NumPy/Torch) before any model
    # init or data shuffling so the run can be reproduced for audit.
    set_seed(TRAIN_SEED)

    model_name = extract_model_id(params["model_name"])
    dataset_path = params["dataset_path"]
    epochs = int(params["epochs"])
    learning_rate = float(params["learning_rate"])
    batch_size = int(params["batch_size"])
    max_length = int(params["max_length"])
    output_dir = params.get("output_dir") or str(MODELS_DIR / job_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    has_gpu, gpu_name = detect_gpu()
    # Prefer bf16 on hardware that supports it (Ampere+): it has the same range
    # as fp32, so it avoids the loss spikes/NaNs fp16 can hit on long 4-bit runs.
    use_bf16 = bool(has_gpu and torch.cuda.is_bf16_supported())
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    save_job(
        job_id,
        {
            "status": "running",
            "model_name": model_name,
            "total_epochs": epochs,
            "started_at": now(),
            "percent": 0.0,
            "gpu_total_mb": gpu_total_mb() if has_gpu else None,
        },
    )

    log_info(job_id, f"Starting fine-tune of '{model_name}'")
    if has_gpu:
        log_info(job_id, f"GPU detected: {gpu_name}")
        log_info(job_id, f"Mixed precision: {'bf16' if use_bf16 else 'fp16'}")
    else:
        log_warn(
            job_id,
            "No GPU detected — falling back to CPU. Training will be VERY slow. "
            "4-bit quantization is disabled on CPU.",
        )

    # ── Load dataset ────────────────────────────────────────
    log_info(job_id, f"Loading dataset from {dataset_path}")
    df = pd.read_csv(dataset_path).fillna("")
    df["__text__"] = df.apply(lambda r: build_prompt(r.to_dict()), axis=1)
    log_info(job_id, f"Loaded {len(df)} training examples")

    # ── Pre-fetch the base model with live progress ─────────
    # Stream a real-time download %, then load from the populated HF cache.
    import os as _os

    from downloads import ensure_model_downloaded

    hf_token = _os.getenv("HF_TOKEN") or _os.getenv("HUGGING_FACE_HUB_TOKEN") or None

    ensure_model_downloaded(
        job_id,
        model_name,
        token=hf_token,
    )

    # ── Detect architecture ─────────────────────────────────
    log_info(job_id, "Inspecting model architecture...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    is_seq2seq = bool(getattr(config, "is_encoder_decoder", False))
    arch = "seq2seq (T5-style)" if is_seq2seq else "causal LM (GPT-style)"
    log_info(job_id, f"Detected architecture: {arch}")

    # ── Tokenizer ───────────────────────────────────────────
    # load_tokenizer() falls back to the slow tokenizer if the fast (Rust) one
    # can't parse this model's tokenizer.json, so any downloadable model loads.
    tokenizer = load_tokenizer(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    # ── Quantization config (GPU only) ──────────────────────
    quant_config = None
    if has_gpu:
        try:
            # pyrefly: ignore [missing-import]
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            log_info(job_id, "4-bit quantization enabled (saves VRAM)")
        except Exception as exc:  # pragma: no cover
            log_warn(job_id, f"Could not enable 4-bit quantization: {exc}")
            quant_config = None

    # ── Load model ──────────────────────────────────────────
    log_info(job_id, "Downloading / loading base model (this can take a while)...")
    model_cls = AutoModelForSeq2SeqLM if is_seq2seq else AutoModelForCausalLM
    model_kwargs: dict[str, Any] = {}
    if quant_config is not None:
        # bitsandbytes requires an explicit device map.
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"
    elif has_gpu:
        # No device_map here: let the Trainer move the model to the GPU itself.
        # Loading with device_map="auto" can make Trainer.train() refuse to run.
        model_kwargs["torch_dtype"] = compute_dtype

    model_kwargs["trust_remote_code"] = True
    if hf_token:
        model_kwargs["token"] = hf_token
    model = model_cls.from_pretrained(model_name, **model_kwargs)
    log_info(job_id, "Base model loaded")

    # ── LoRA via PEFT ───────────────────────────────────────
    # pyrefly: ignore [missing-import]
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if quant_config is not None:
        model = prepare_model_for_kbit_training(model)

    target_modules = _auto_target_modules(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules,
        task_type="SEQ_2_SEQ_LM" if is_seq2seq else "CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log_info(
        job_id,
        f"LoRA applied (r=16, alpha=32). Trainable params: "
        f"{trainable:,} / {total:,} ({100 * trainable / max(total, 1):.2f}%)",
    )

    # ── Truncation check (surface silent data loss) ─────────
    _warn_if_truncating(job_id, tokenizer, df, is_seq2seq, max_length)

    # ── Build HF dataset ────────────────────────────────────
    if is_seq2seq:
        # Encoder gets instruction+input; decoder learns the output.
        def _split(r: dict[str, Any]) -> dict[str, str]:
            return {
                "src": build_seq2seq_src(r),
                "tgt": str(r.get("output") or ""),
            }

        records = df.apply(lambda r: _split(r.to_dict()), axis=1).tolist()
        hf_ds = Dataset.from_list(records)
        pad_id = tokenizer.pad_token_id

        def _tok(batch: dict[str, list]) -> dict[str, Any]:
            model_inputs = tokenizer(
                batch["src"], max_length=max_length, truncation=True, padding="max_length"
            )
            labels = tokenizer(
                batch["tgt"], max_length=max_length, truncation=True, padding="max_length"
            )
            # Mask pad tokens in the labels so they don't contribute to the loss.
            model_inputs["labels"] = [
                [(tok if tok != pad_id else -100) for tok in seq]
                for seq in labels["input_ids"]
            ]
            return model_inputs

        train_ds = hf_ds.map(_tok, batched=True, remove_columns=hf_ds.column_names)
    else:
        # Causal: hand SFTTrainer the raw "text" column and let it tokenize
        # (using dataset_text_field + max_seq_length). Pre-tokenizing here
        # would conflict with that and is unnecessary.
        train_ds = Dataset.from_dict({"text": df["__text__"].tolist()})

    # ── Held-out validation split (overfit detection) ───────
    # Small datasets overfit fast; without a held-out set the falling training
    # loss is misleading. We carve off a validation slice so eval_loss + early
    # stopping can catch overfitting and pick the best epoch's weights.
    eval_ds = None
    n_total = len(train_ds)
    if n_total >= MIN_ROWS_FOR_EVAL:
        frac = 0.1 if n_total >= 200 else 0.15
        eval_n = max(2, min(round(n_total * frac), MAX_EVAL_ROWS))
        split = train_ds.train_test_split(test_size=eval_n, seed=TRAIN_SEED, shuffle=True)
        train_ds, eval_ds = split["train"], split["test"]
        log_info(
            job_id,
            f"Validation split: {len(train_ds)} train / {len(eval_ds)} eval "
            f"examples (early stopping on eval_loss, patience {EARLY_STOPPING_PATIENCE}).",
        )
    else:
        log_warn(
            job_id,
            f"Only {n_total} examples — too few for a reliable validation split, "
            f"so training runs without held-out eval / early stopping. Add more "
            f"rows (≥{MIN_ROWS_FOR_EVAL}) to enable overfit detection.",
        )

    steps_per_epoch = max(1, len(train_ds) // batch_size)
    total_steps = steps_per_epoch * epochs
    save_job(job_id, {"total_steps": total_steps})

    # ── Live-logging callback ───────────────────────────────
    class _LiveCallback(TrainerCallback):
        """Streams loss/lr/eta/GPU-mem per step and enforces cancellation."""

        def __init__(self) -> None:
            self._t0 = now()

        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
            logs = logs or {}
            step = int(state.global_step)
            loss = logs.get("loss")
            lr = logs.get("learning_rate")
            elapsed = now() - self._t0
            done = max(step, 1)
            eta = (elapsed / done) * (total_steps - step) if step else None
            percent = min(100.0, 100.0 * step / max(total_steps, 1))
            mem = gpu_memory_stats()
            epoch = int(state.epoch) + 1 if state.epoch is not None else 0

            update: dict[str, Any] = {
                "percent": round(percent, 1),
                "current_step": step,
                "current_epoch": min(epoch, epochs),
                "gpu_memory_mb": mem["used_mb"] if mem else None,
                "gpu_total_mb": mem["total_mb"] if mem else None,
                "gpu_memory_percent": mem["percent"] if mem else None,
            }
            if loss is not None:
                update["loss"] = round(float(loss), 4)
            if lr is not None:
                update["learning_rate"] = float(lr)
            if eta is not None:
                update["eta_seconds"] = round(eta, 1)
            # Eval logs arrive on their own step (no train "loss" key).
            eval_loss = logs.get("eval_loss")
            if eval_loss is not None:
                update["eval_loss"] = round(float(eval_loss), 4)
                log_info(job_id, f"validation | eval_loss {float(eval_loss):.4f}")
            save_job(job_id, update)

            if loss is not None:
                gpu_txt = (
                    f"{mem['used_mb']:.0f}/{mem['total_mb']:.0f}MB ({mem['percent']:.0f}%)"
                    if mem
                    else "CPU"
                )
                # lr/eta may be absent on some logged steps; format defensively.
                lr_txt = f"{lr:.2e}" if lr is not None else "n/a"
                eta_txt = f"{eta:.0f}s" if eta is not None else "n/a"
                log_info(
                    job_id,
                    f"step {step}/{total_steps} | epoch {min(epoch, epochs)}/{epochs} "
                    f"| loss {float(loss):.4f} | lr {lr_txt} "
                    f"| eta {eta_txt} | gpu {gpu_txt}",
                )

        def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
            if is_cancelled(job_id):
                log_warn(job_id, "Cancellation requested — stopping training.")
                control.should_training_stop = True
            return control

        def on_epoch_end(self, args, state, control, **kwargs):  # noqa: ANN001
            log_info(job_id, f"Epoch {int(state.epoch)} complete.")
            return control

    # ── Trainer ─────────────────────────────────────────────
    log_info(job_id, "Configuring trainer (TRL SFTTrainer)...")
    final_loss, trainer = _run_trainer(
        job_id=job_id,
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        eval_ds=eval_ds,
        is_seq2seq=is_seq2seq,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_length=max_length,
        output_dir=output_dir,
        has_gpu=has_gpu,
        use_bf16=use_bf16,
        callback=_LiveCallback(),
    )

    if is_cancelled(job_id):
        save_job(job_id, {"status": "cancelled", "finished_at": now()})
        log_warn(job_id, "Training cancelled by user.")
        get_redis().delete(f"ftcancel:{job_id}")
        return model, trainer

    # With early stopping + load_best_model_at_end, ``model`` now holds the
    # best epoch's weights (lowest eval_loss), not necessarily the last.
    best_eval_loss = None
    try:
        bm = getattr(trainer.state, "best_metric", None)
        if bm is not None:
            best_eval_loss = round(float(bm), 4)
    except Exception:  # pragma: no cover
        pass

    # ── Save ────────────────────────────────────────────────
    log_info(job_id, f"Saving fine-tuned model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Drop the Trainer's per-epoch checkpoints: the best weights are already
    # saved above, and leaving checkpoint-* dirs bloats the model download zip.
    for ckpt in Path(output_dir).glob("checkpoint-*"):
        if ckpt.is_dir():
            shutil.rmtree(ckpt, ignore_errors=True)

    # Write a small metadata sidecar for the "My Models" page.
    meta = {
        "job_id": job_id,
        "name": f"{model_name.split('/')[-1]}-ft-{job_id[:8]}",
        "base_model": model_name,
        "created_at": now(),
        "dataset_rows": int(len(df)),
        "final_loss": final_loss,
        "best_eval_loss": best_eval_loss,
        "architecture": "seq2seq" if is_seq2seq else "causal",
        "seed": TRAIN_SEED,
        "precision": "bf16" if use_bf16 else ("fp16" if has_gpu else "fp32"),
    }
    (Path(output_dir) / "ft_metadata.json").write_text(json.dumps(meta, indent=2))

    save_job(
        job_id,
        {
            "status": "completed",
            "percent": 100.0,
            "final_loss": final_loss,
            "best_eval_loss": best_eval_loss,
            "dataset_rows": int(len(df)),
            "finished_at": now(),
        },
    )
    log_info(job_id, f"Done! Final loss: {final_loss}. Model saved to {output_dir}")
    return model, trainer


def _run_trainer(
    *,
    job_id: str,
    model: Any,
    tokenizer: Any,
    train_ds: Any,
    eval_ds: Any,
    is_seq2seq: bool,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    max_length: int,
    output_dir: str,
    has_gpu: bool,
    use_bf16: bool,
    callback: Any,
) -> tuple[Optional[float], Any]:
    """Build and run the appropriate Trainer; return ``(final_loss, trainer)``.

    Uses TRL's ``SFTTrainer`` for causal models and the standard
    ``Seq2SeqTrainer`` for encoder-decoder models. The trainer is returned so
    the caller can tear it down (it owns the optimizer's GPU state).

    When ``eval_ds`` is provided, evaluation runs each epoch, the best epoch's
    weights are restored at the end (``load_best_model_at_end``) and training
    stops early once ``eval_loss`` stops improving.
    """
    # pyrefly: ignore [missing-import]
    from transformers import (
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainingArguments,
    )

    common = dict(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=1,
        report_to=[],
        bf16=use_bf16,
        fp16=has_gpu and not use_bf16,
        optim="paged_adamw_8bit" if has_gpu else "adamw_torch",
        gradient_accumulation_steps=1,
        warmup_ratio=0.03,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=TRAIN_SEED,
        data_seed=TRAIN_SEED,
        disable_tqdm=True,
    )

    callbacks = [callback]
    if eval_ds is not None:
        # Evaluate + checkpoint each epoch, keep only the best, and stop early
        # once eval_loss plateaus — the core overfit guard for small datasets.
        common.update(
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)
        )
    else:
        common.update(save_strategy="no")

    if is_seq2seq:
        args = Seq2SeqTrainingArguments(**common)
        collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            callbacks=callbacks,
        )
    else:
        # pyrefly: ignore [missing-import]
        from trl import SFTConfig, SFTTrainer

        # TRL >= 0.9 takes its own config object that subclasses TrainingArguments.
        try:
            sft_args = SFTConfig(
                max_seq_length=max_length,
                packing=False,
                dataset_text_field="text",
                **common,
            )
        except TypeError:
            sft_args = SFTConfig(
                max_length=max_length,
                packing=False,
                dataset_text_field="text",
                **common,
            )
        try:
            trainer = SFTTrainer(
                model=model,
                args=sft_args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                tokenizer=tokenizer,
                callbacks=callbacks,
            )
        except TypeError:
            # Fallback for TRL signature differences across versions.
            trainer = SFTTrainer(
                model=model,
                args=TrainingArguments(**common),
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                callbacks=callbacks,
            )

    result = trainer.train()
    final_loss = None
    try:
        final_loss = round(float(result.training_loss), 4)
    except Exception:  # pragma: no cover
        pass
    return final_loss, trainer


def _warn_if_truncating(
    job_id: str, tokenizer: Any, df: Any, is_seq2seq: bool, max_length: int
) -> None:
    """Warn when rows exceed ``max_length`` and will be silently truncated.

    Truncation drops the *tail* of an over-long example. For domains like
    medical/insurance records that can quietly discard clinically relevant
    context, so we surface exactly how many rows are affected and the longest
    one — letting the user raise ``max_length`` instead of losing data unawares.
    """
    if is_seq2seq:
        texts = [build_seq2seq_src(r.to_dict()) for _, r in df.iterrows()]
        texts += [str(r.get("output") or "") for _, r in df.iterrows()]
    else:
        texts = df["__text__"].tolist()

    over = 0
    longest = 0
    for t in texts:
        n = len(tokenizer(t, truncation=False)["input_ids"])
        longest = max(longest, n)
        if n > max_length:
            over += 1

    if over:
        pct = 100.0 * over / max(len(texts), 1)
        log_warn(
            job_id,
            f"{over} example(s) ({pct:.0f}%) exceed max_length={max_length} tokens "
            f"and will be TRUNCATED — their tail content is dropped. Longest is "
            f"{longest} tokens. Raise 'Max Length' to keep the full text if that "
            f"content matters.",
        )
    else:
        log_info(
            job_id,
            f"All examples fit within max_length={max_length} (longest {longest} tokens).",
        )


def _auto_target_modules(model: Any) -> list[str]:
    """Heuristically pick LoRA target modules ('auto') for the given model.

    Scans the module names for the common attention/projection linear layers
    used by the major architectures (GPT-2, LLaMA/Mistral, T5, GPT-NeoX, ...).
    """
    candidates = {
        # llama / mistral / qwen2 / gemma (attention + MLP projections)
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        # phi-3 / fused-qkv families
        "qkv_proj", "gate_up_proj", "Wqkv", "out_proj",
        # gpt-neox / falcon
        "query_key_value", "dense_h_to_4h", "dense_4h_to_h",
        # gpt-2
        "c_attn", "c_proj", "c_fc",
        # t5
        "q", "k", "v", "o", "wi", "wo", "wi_0", "wi_1",
        # bert-like
        "query", "key", "value", "dense",
    }
    found: list[str] = []
    for name, _ in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in candidates and leaf not in found:
            found.append(leaf)
    # Sensible default if nothing matched.
    return found or ["q_proj", "v_proj"]


def _humanize_error(exc: Exception, model_name: str = "") -> tuple[str, str]:
    """Map a raw exception to a friendly (error, suggestion) pair."""
    text = str(exc).lower()
    if "fine-grained" in text or "canreadgatedrepos" in text or "gated repositories in your" in text:
        return (
            f"HF_TOKEN permission issue for gated model '{model_name or 'Hugging Face'}'.",
            "Your HF_TOKEN is a Fine-Grained token with 'Access to public gated repos' turned OFF. Go to https://huggingface.co/settings/tokens, create a 'Classic (Read)' token or enable 'Read access to public gated repos', update your .env file, then try again.",
        )
    if any(
        s in text
        for s in ("gated repo", "403 client error", "restricted", "authorized list", "access to model")
    ):
        url = f"https://huggingface.co/{model_name}" if model_name else "Hugging Face"
        return (
            f"Access restricted for gated model '{model_name or 'Hugging Face'}' (403 Forbidden).",
            f"You must accept the model license terms first. Visit {url} in your browser to grant access, ensure your HF_TOKEN is in .env, then try again.",
        )
    if (
        "untagged enum" in text
        or "modelwrapper" in text
        or "data did not match any variant" in text
        or "tokenizer format is newer" in text
    ):
        return (
            "This model's tokenizer is too new for the installed libraries.",
            "Update the backend's transformers/tokenizers and rebuild the image, "
            "then try again.",
        )
    if "out of memory" in text or "cuda oom" in text or "outofmemory" in text:
        return (
            "Not enough GPU memory.",
            "Try reducing batch size or max_length, then start again.",
        )
    if any(
        s in text
        for s in ("not a valid model", "404", "repository not found", "couldn't find", "is not the path")
    ):
        return (
            "Could not load model from Hugging Face.",
            "Check the model name and try again. Make sure it is public.",
        )
    if "connection" in text or "timed out" in text or "max retries" in text:
        return (
            "Network error while contacting Hugging Face.",
            "Check your internet connection and try again.",
        )
    return (f"Training failed: {exc}", "See the live logs above for details.")
