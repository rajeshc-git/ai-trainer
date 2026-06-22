"""Standalone LoRA-merge worker.

Run as a subprocess by ``gguf.py`` so that **all** the RAM/VRAM used to load and
merge the base model is returned to the OS the instant this process exits —
something an in-process merge can't guarantee. PyTorch's CUDA context and large
CPU allocations outlive ``torch.cuda.empty_cache()`` for the life of the host
process, so doing the merge inside the long-lived API server left the card/RAM
looking full until the container was restarted. A short-lived child process
sidesteps that entirely: when it dies, the kernel reclaims everything.

Usage:
    python merge_lora.py --adapter-dir DIR --out-dir DIR [--device cuda|cpu]

All progress is printed to stdout (line-buffered) so the parent can relay it to
the job's live log.
"""

from __future__ import annotations

import argparse
import os
import sys


def _log(msg: str) -> None:
    print(msg, flush=True)


def merge(adapter_dir: str, out_dir: str, device: str) -> None:
    # pyrefly: ignore [missing-import]
    import torch
    # pyrefly: ignore [missing-import]
    from peft import PeftConfig, PeftModel
    # pyrefly: ignore [missing-import]
    from transformers import AutoModelForCausalLM

    from utils import load_tokenizer

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or None
    peft_config = PeftConfig.from_pretrained(adapter_dir)
    base_model_name = peft_config.base_model_name_or_path
    _log(f"Base model: {base_model_name}")

    def _load_base(dev: str):
        kwargs = dict(torch_dtype=torch.float16, low_cpu_mem_usage=True, token=token, trust_remote_code=True)
        # Whole model on one device — never "auto" (which could offload and
        # silently run at CPU speed).
        kwargs["device_map"] = {"": 0} if dev == "cuda" else "cpu"
        return AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs)

    try:
        base = _load_base(device)
    except Exception as exc:  # noqa: BLE001 - GPU OOM/placement → retry on CPU
        msg = str(exc).lower()
        if device == "cuda" and ("out of memory" in msg or "cuda" in msg):
            _log(f"GPU merge failed ({exc}); falling back to CPU.")
            device = "cpu"
            base = _load_base("cpu")
        else:
            raise

    _log(f"Merging adapter on {device.upper()}…")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.merge_and_unload()
    _log("Adapter merged. Saving full fp16 model…")

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    load_tokenizer(adapter_dir).save_pretrained(out_dir)
    _log("Merge complete.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model.")
    ap.add_argument("--adapter-dir", required=True, help="Directory holding the trained adapter.")
    ap.add_argument("--out-dir", required=True, help="Where to write the merged fp16 model.")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()
    merge(args.adapter_dir, args.out_dir, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
