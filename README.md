# cosmos-cookbook

A local playground for NVIDIA Cosmos 3 world models on a single 32 GB GPU: scripts that
run every capability of Cosmos3-Edge, a fit guide for what else runs on this card, and an
explanation of how the models work and where they sit in a Physical AI stack.

- [docs/how-cosmos-works.md](docs/how-cosmos-works.md): architecture, the workflows it enables, and what "real time" means
- [docs/vram-fit.md](docs/vram-fit.md): every Cosmos model against a 32 GB card, quantized and not

## Setup

```bash
make setup          # .venv with PyTorch (CUDA 13), Diffusers and Transformers from main
make check          # GPU/RAM headroom, which models fit right now, Hugging Face access
make download-edge  # Cosmos3-Edge and Cosmos3-Edge-Policy-DROID, about 9 GB each
make framework      # NVIDIA's cosmos-framework: sample robot episode, FP8 export, policy server
```

Needs `uv`, an NVIDIA driver with CUDA 13 support (580 or newer; use
`TORCH_BACKEND=cu128 make setup` for older drivers), and a Hugging Face login
(`uvx hf@latest auth login`).

## What you can run

| Command | What it shows | Measured on the RTX 5090 |
|---|---|---|
| `make reason` | **Understand.** Image or video in, text out: scene description, task plans, hazards | 4.6 GiB, about 107 tokens/s |
| `make watch` | **Understand, live.** The same question asked of each 2 s window of a video | 0.5 s per window: 4x faster than real time |
| `make generate` | **Predict.** One image and a prompt become 5 s of video | 256p: 8 s, 8.3 GiB. 480p: 62 s, 9.7 GiB with `--tile-vae` |
| `make forward` | **Simulate.** A frame plus robot actions become the video the robot would see | 1.4 s per 16-action chunk, 8.4 GiB |
| `make inverse` | **Label.** A driving video becomes the 60-step trajectory behind it | 21 s per clip, 9.2 GiB with `--tile-vae` |
| `make policy` | **Act.** Three camera views and an instruction become 16 robot actions | see below |

Every script takes `--help`, and `--model nano` where Nano supports the task. Outputs land
in `outputs/`.

All timings were taken while another job held about 15 GB and 90% of the same GPU, so
treat them as lower bounds on speed. Re-run on an idle card for real numbers.

### The real-time result

The policy must deliver 16 actions before the previous 16 finish executing: 1.07 s at
DROID's 15 Hz control rate.

| `scripts/policy.py` flags | Time per plan | Keeps up at 15 Hz? |
|---|---|---|
| default (30 denoising steps, decode the imagined video) | 5.4 s | No, 5x over |
| `--steps 4` | 3.5 s | No, 3x over |
| `--steps 4 --actions-only` | **0.81 s** | **Yes** |

Two changes get there: fewer denoising steps, and not decoding video nobody on the robot
will watch. The second one mattered more. Not checked here: how much action quality the
4-step setting gives up. NVIDIA's own real-time benchmark also uses 4 steps, but with
guidance 3.0 where this script uses 1.0.

## What fits on this card

| | Edge (4B) | Nano (16B) | Super (64B) |
|---|---|---|---|
| Reasoner | Yes, 5 GiB | Yes, 16.4 GiB, about 65 tok/s (needs the card mostly idle) | No |
| Generator | Yes, 8 to 10 GiB | Not on this machine: fits in VRAM only with layer offload, which needs 64 GiB of system RAM (30 here) | No |
| Robot policy | Yes, 9 GiB, real time | No | No |

Details, quantization options and the older Cosmos models are in
[docs/vram-fit.md](docs/vram-fit.md).

## Things to know

- **Guardrails are off by default.** NVIDIA's safety checker lives in the gated repo
  `nvidia/Cosmos-1.0-Guardrail` (35 GB). Accept its terms on Hugging Face, then pass
  `--guardrails`.
- **Serving (`make serve-reasoner`, `make serve-generator`) is untested here.** The targets
  use NVIDIA's documented vLLM containers, but Docker on this machine has no NVIDIA
  container toolkit, so containers cannot see the GPU until that is installed.
- **Shared GPU.** `COSMOS_VRAM_CAP_GIB=12 make generate` makes a run fail at 12 GiB rather
  than crowd out another job.
- With thinking disabled (`--no-thinking`), Edge appends bracketed spans such as
  `[0, 499]` to plan steps. That is the model's output, not a parsing bug here.
- Scripts are thin wrappers over `Cosmos3OmniPipeline` (Diffusers) and the Cosmos 3
  classes in Transformers. Both currently require installs from `main`.

## Layout

```
scripts/        common.py (aliases, memory modes, timing) and one script per capability
docs/           the two guides above
third_party/    cosmos-framework checkout (gitignored)
outputs/        generated video and action files (gitignored)
```

Models and code from NVIDIA are under the OpenMDW-1.1 license.
