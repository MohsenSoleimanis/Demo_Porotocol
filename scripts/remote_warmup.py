"""Pre-download Qwen-VL and warm up Docling on the Lightning AI machine.

Run this ONCE after installing requirements-remote.txt so the first /parse/*
call doesn't pay the multi-gigabyte download tax. Designed to be called from
inside the remote venv on the Lightning AI Studio (not from a local laptop).

Usage on the remote box:
    source .venv/bin/activate
    python scripts/remote_warmup.py
"""

from __future__ import annotations

import os
import sys
import time


def warm_qwen():
    model_id = os.getenv("FMLS_QWEN_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
    print(f"[warmup] downloading Qwen model: {model_id}")
    t0 = time.perf_counter()
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    import torch

    AutoProcessor.from_pretrained(model_id)
    Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    print(f"[warmup] qwen ready ({time.perf_counter() - t0:.1f}s)")


def warm_docling():
    print("[warmup] initializing Docling (this triggers its layout model download)")
    t0 = time.perf_counter()
    from docling.document_converter import DocumentConverter

    DocumentConverter()
    print(f"[warmup] docling ready ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    try:
        warm_docling()
    except Exception as e:
        print(f"[warmup] docling failed: {e}", file=sys.stderr)
    try:
        warm_qwen()
    except Exception as e:
        print(f"[warmup] qwen failed: {e}", file=sys.stderr)
    print("[warmup] done")
