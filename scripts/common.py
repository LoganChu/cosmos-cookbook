"""Shared helpers: model aliases, example assets, memory modes and VRAM/latency reporting."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"

MODELS = {
    "edge": "nvidia/Cosmos3-Edge",
    "nano": "nvidia/Cosmos3-Nano",
    "edge-policy": "nvidia/Cosmos3-Edge-Policy-DROID",
    "nano-policy": "nvidia/Cosmos3-Nano-Policy-DROID",
}

# Offload modes for the Diffusers generator, cheapest in VRAM last.
OFFLOAD_MODES = ("none", "model", "sequential")


# On a shared GPU, set COSMOS_VRAM_CAP_GIB so this process fails with its own
# out-of-memory error instead of starving another job.
if os.environ.get("COSMOS_VRAM_CAP_GIB") and torch.cuda.is_available():
    _cap = float(os.environ["COSMOS_VRAM_CAP_GIB"]) * 1024**3
    torch.cuda.set_per_process_memory_fraction(min(1.0, _cap / torch.cuda.mem_get_info()[1]))


def resolve_model(name: str) -> str:
    """Accept an alias (edge, nano, ...), a Hugging Face repo ID or a local path."""
    return MODELS.get(name, name)


def asset(repo: str, filename: str) -> Path:
    """Fetch one example file from a model repo's assets/ folder (cached by huggingface_hub)."""
    return Path(hf_hub_download(resolve_model(repo), filename))


def gib(num_bytes: float) -> float:
    return num_bytes / 1024**3


def free_vram_gib() -> float:
    free, _total = torch.cuda.mem_get_info()
    return gib(free)


def apply_offload(pipe, mode: str):
    """Place a Diffusers pipeline on the GPU under one of three memory modes.

    none        everything resident on the GPU; fastest, needs the most VRAM
    model       whole components swap between CPU RAM and GPU as they are needed
    sequential  submodules stream through the GPU one at a time; slowest, least VRAM
    Both offload modes keep the weights in system RAM, so RAM becomes the limit.
    """
    if mode == "none":
        pipe.to("cuda")
    elif mode == "model":
        pipe.enable_model_cpu_offload()
    elif mode == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        raise ValueError(f"offload must be one of {OFFLOAD_MODES}, got {mode!r}")
    return pipe


@contextmanager
def measure(label: str):
    """Print wall time and peak VRAM allocated by this process for the enclosed block."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    stats: dict[str, float] = {}
    try:
        yield stats
    finally:
        torch.cuda.synchronize()
        stats["seconds"] = time.perf_counter() - start
        stats["peak_gib"] = gib(torch.cuda.max_memory_allocated())
        print(f"[{label}] {stats['seconds']:.2f} s, peak VRAM {stats['peak_gib']:.1f} GiB")


def output_path(name: str) -> Path:
    OUTPUTS.mkdir(exist_ok=True)
    return OUTPUTS / name
