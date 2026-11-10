# What fits on a 32 GB GPU

An RTX 5090 is sold as 32 GB and reports 31.3 GiB usable. A desktop session takes 1 to
3 GiB of that. This machine also has 30 GiB of system RAM, which matters as much as VRAM
once offloading is involved.

Two kinds of numbers appear below. **Measured** means run from this repo on this 5090.
**NVIDIA floor** means the minimum NVIDIA publishes for its NIM container, which includes
guardrail models and serving overhead, so it is higher than a bare PyTorch run.

## Cosmos 3

| Model and task | Precision | NVIDIA floor: VRAM / host RAM | Measured peak here | Verdict on this machine |
|---|---|---|---|---|
| **Edge** reasoner | BF16 | 12 / 8 GiB | 4.6 GiB (image), 5.5 GiB (6 s video) | Fits with room to spare |
| **Edge** image-to-video, 256p, 49 frames | BF16 | not published | 8.3 GiB | Fits |
| **Edge** image-to-video, 480p, 121 frames | BF16 | not published | 9.7 GiB with `--tile-vae`; over 11.5 GiB without | Fits |
| **Edge** forward dynamics, 256p | BF16 | not published | 8.4 GiB | Fits |
| **Edge** inverse dynamics, 480p, 61 frames | BF16 | not published | 9.2 GiB with `--tile-vae` | Fits |
| **Edge-Policy-DROID** | BF16 | 15 / 16 GiB | 8.8 to 9.4 GiB | Fits |
| **Nano** reasoner | BF16 | 23.1 / 24 GiB | 16.4 GiB (image and 6 s video) | Fits with the GPU otherwise idle |
| **Nano** reasoner | FP8 or NVFP4 | 23.1 / 17 GiB | not yet run | Should fit; NVFP4 needs a Blackwell GPU, which this is |
| **Nano** generator, all on GPU | BF16 | 58 / 40 GiB | | Does not fit |
| **Nano** generator, all on GPU | FP8 | 44 / 35 GiB | | Does not fit |
| **Nano** generator, layer offload | FP8 | 32 / 36 GiB | | Just over on both VRAM and RAM |
| **Nano** generator, layer offload, guardrails on CPU | BF16 or FP8 | 31 / 64 GiB | | VRAM fits; **needs 64 GiB RAM, this machine has 30** |
| **Nano-Policy-DROID** | BF16 | 58 / 40 GiB | | Does not fit |
| **Nano-Sim-\*** (Auto, Bimanual, Transfer) | BF16 | tested on H100 / B200 | | Does not fit |
| **Super**, any task | FP8, layer offload | 38 / 80 GiB at best | | Does not fit |

The short version: **Edge does everything on this card. Nano's Reasoner fits. Nano's
Generator is blocked by system RAM, not VRAM.** NVIDIA's own notes say the 31 GiB
layer-offload configuration exists specifically for 32 GB cards like the 5090, but it
parks the weights in host memory and needs 64 GiB of it. More RAM is the upgrade that
unlocks Nano video generation here.

`scripts/generate.py --model nano --offload sequential` will attempt it through Diffusers,
which has no fixed floor. With 30 GiB of RAM and a 35 GB checkpoint it will swap heavily;
treat it as an experiment.

### About quantization

- **FP8 for Nano** is an official NVIDIA artifact: the `fp8` branch of `nvidia/Cosmos3-Nano`
  (21 GB instead of 35 GB). `make download-nano-fp8` fetches it and `make fp8-deps`
  installs the loader. NVIDIA's docs link a separate `Cosmos3-Nano-FP8` repository; that
  link returned 404 on 2026-11-09, and the branch is where the weights are.
- **FP8 saves less than the file size suggests.** Only weights shrink. Activations, the
  VAE and the vision encoder stay at full size, and the exported checkpoint runs the first
  and last three denoising steps at 16-bit activations to protect quality.
- **Edge has no published FP8 checkpoint.** At 9 GB it does not need one on this card. You
  can export one with `quantize_fp8_checkpoint` from cosmos-framework (`make framework`,
  then its `quantization/` cookbook) if you want to study the speed and quality trade.
- **NVFP4** (4-bit) is offered only inside the NIM container, for the Reasoner.
- The model cards state that only BF16 is officially tested.

## Earlier Cosmos models

These predate Cosmos 3 and each does one job. Figures are NVIDIA's unless noted; none were
run here, and the first four are gated repos you need to accept terms for on Hugging Face.

| Model | Job | Weights | VRAM | Fits? |
|---|---|---|---|---|
| Cosmos-Reason2-2B | video reasoning (VLM) | 4.9 GB | 24 GB minimum per NVIDIA | Yes |
| Cosmos-Reason2-8B | video reasoning (VLM) | 17.5 GB | 32 GB minimum per NVIDIA; a third-party DGX Spark test measured 17.5 GB loaded | Likely, with the GPU otherwise idle |
| Cosmos-Reason2-32B | video reasoning (VLM) | | | No |
| Cosmos-Predict2.5-2B | text/image/video to video | 4.1 GB per variant | no reliable figure found; the older Predict2-2B is listed at 26 to 33 GB | Possibly, with offloading |
| Cosmos-Transfer2.5-2B | control video to photoreal video | about 5 GB per control | 65.4 GB | No |
| Cosmos-Predict2.5-14B | video generation | | | No |
| Cosmos-Policy-\*-Predict2-2B | robot policy (LIBERO, RoboCasa, ALOHA) | | 6.0 to 8.9 GB | Yes |
| Cosmos-Embed1 (224p / 336p / 448p) | video and text embeddings for search and curation | 2.4 GB | small | Yes |
| Cosmos tokenizers | video compression | small | small | Yes |

For new work, Cosmos3-Edge covers what Reason2-2B and Predict2-2B did in one checkpoint
that uses less memory than either.

## When memory runs out

In order of how much speed each costs:

1. `--tile-vae` (generation, dynamics, policy): the video decoder sets the peak at 480p and
   above; tiling removes it.
2. Lower `--size` or `--frames`.
3. `--actions-only` (policy): never decode the imagined video.
4. `--offload model`: components take turns on the GPU. Needs the weights in RAM.
5. `--offload sequential`: one layer at a time. Slowest by far.
6. `COSMOS_VRAM_CAP_GIB=12` in the environment makes a run fail cleanly at that limit
   instead of taking memory another job on the same GPU needs.

## Sources

- [Cosmos3 NIM support matrix](https://github.com/NVIDIA/cosmos-framework/blob/main/.agents/skills/cookbooks/cosmos3/nim/support-matrix.md) for every Cosmos 3 floor above
- [FP8 quantization cookbook](https://github.com/NVIDIA/cosmos-framework/blob/main/.agents/skills/cookbooks/cosmos3/quantization/README.md) and [inference memory options](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/inference.md)
- [Cosmos prerequisites](https://docs.nvidia.com/cosmos/latest/prerequisites.html) (Reason2 minimums), [DGX Spark test of Reason2](https://dev.classmethod.jp/en/articles/dgx-spark-cosmos-reason2-structured-reasoning/), [Transfer2.5 model card](https://build.nvidia.com/nvidia/cosmos-transfer2_5-2b/modelcard), [Predict2 model matrix](https://docs.nvidia.com/cosmos/latest/predict2/model_matrix.html)
- [Cosmos-Policy-LIBERO-Predict2-2B model card](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B)
