"""Report whether this machine is ready to run Cosmos 3, and which models fit right now."""

from __future__ import annotations

import shutil

import psutil
import torch
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

from common import gib

# NVIDIA's published per-device VRAM and host RAM floors (GiB), from the Cosmos3 NIM
# support matrix. See docs/vram-fit.md for the source and the full table.
FLOORS = [
    ("Edge reasoner (BF16)", 12, 8),
    ("Edge DROID policy (BF16)", 15, 16),
    ("Nano reasoner (BF16)", 23.1, 24),
    ("Nano reasoner (FP8)", 23.1, 17),
    ("Nano generator (FP8, layer offload)", 32, 36),
    ("Nano generator (BF16, layer offload, guardrails on CPU)", 31, 64),
]


def main() -> None:
    import diffusers
    import transformers

    print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), "
          f"diffusers {diffusers.__version__}, transformers {transformers.__version__}")

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device visible. Check the driver and TORCH_BACKEND in scripts/setup.sh.")

    free, total = (gib(x) for x in torch.cuda.mem_get_info())
    ram = psutil.virtual_memory()
    ram_total, ram_free = gib(ram.total), gib(ram.available)
    disk_free = gib(shutil.disk_usage(".").free)
    print(f"GPU   {torch.cuda.get_device_name(0)}: {free:.1f} GiB free of {total:.1f} GiB")
    print(f"RAM   {ram_free:.1f} GiB available of {ram_total:.1f} GiB")
    print(f"Disk  {disk_free:.0f} GiB free")

    print("\nFit against NVIDIA's floors (VRAM / host RAM, GiB):")
    for name, vram, host in FLOORS:
        if total < vram or ram_total < host:
            verdict = "does not fit this machine"
        elif free < vram or ram_free < host:
            verdict = "fits, but not with what is running now"
        else:
            verdict = "fits now"
        print(f"  {name:<58} {vram:>5} / {host:<3} {verdict}")

    print("\nHugging Face access:")
    api = HfApi()
    for repo in ("nvidia/Cosmos3-Edge", "nvidia/Cosmos3-Nano", "nvidia/Cosmos-1.0-Guardrail"):
        try:
            api.auth_check(repo)
            print(f"  ok       {repo}")
        except HfHubHTTPError as err:
            print(f"  blocked  {repo} ({err.response.status_code}); accept its terms on huggingface.co "
                  "and run `uvx hf@latest auth login`")


if __name__ == "__main__":
    main()
