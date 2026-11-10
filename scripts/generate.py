"""Generate video with the Cosmos 3 Generator tower (world simulation / future prediction).

    python scripts/generate.py                       # Edge, bundled image -> 5 s of 480p video
    python scripts/generate.py --frames 49 --size 256
    python scripts/generate.py --image scene.jpg --prompt "The forklift backs away from the shelf."
    python scripts/generate.py --model nano --offload sequential --prompt "A robot arm stacks two cubes."

Edge needs a starting image (or omit --image and --prompt to use the bundled example).
Nano also does text-to-video, but on a 32 GB card it needs --offload and a lot of system RAM;
see docs/vram-fit.md.
"""

from __future__ import annotations

import argparse
import json

import torch

from common import OFFLOAD_MODES, apply_offload, asset, measure, output_path, resolve_model

# 16:9 sizes for each resolution tier.
SIZES = {256: (320, 192), 480: (832, 480), 720: (1280, 720)}


def load_pipeline(model_name: str, offload: str = "none", guardrails: bool = False, tile_vae: bool = False):
    from diffusers import Cosmos3OmniPipeline

    pipe = Cosmos3OmniPipeline.from_pretrained(
        resolve_model(model_name),
        torch_dtype=torch.bfloat16,
        # The guardrail is a separate ~35 GB download holding ~10 GiB of extra models.
        enable_safety_checker=guardrails,
    )
    if tile_vae:
        # Encode/decode video in overlapping tiles. The VAE, not the transformer, sets the
        # memory peak at 480p and above; tiling trades a little speed for a much lower peak.
        pipe.vae.enable_tiling()
    return apply_offload(pipe, offload)


def set_flow_shift(pipe, flow_shift: float) -> None:
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=flow_shift, use_karras_sigmas=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="edge")
    parser.add_argument("--image", help="first frame; omit for text-to-video (Nano only)")
    parser.add_argument("--prompt")
    parser.add_argument("--size", type=int, default=480, choices=sorted(SIZES))
    parser.add_argument("--frames", type=int, default=121, help="Edge supports 50-150")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offload", default="none", choices=OFFLOAD_MODES)
    parser.add_argument("--tile-vae", action="store_true", help="lower peak VRAM at 480p and above")
    parser.add_argument("--guardrails", action="store_true", help="enable NVIDIA's safety checker")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from diffusers.utils import export_to_video, load_image

    image_path, prompt = args.image, args.prompt
    if image_path is None and prompt is None:
        # NVIDIA's example: an image plus a JSON-structured ("upsampled") prompt.
        image_path = str(asset(args.model, "assets/example_i2v_input.jpg"))
        prompt = json.dumps(json.loads(asset(args.model, "assets/example_i2v_prompt.json").read_text()))
    negative = json.dumps(json.loads(asset(args.model, "assets/negative_prompt.json").read_text()))

    with measure("load"):
        pipe = load_pipeline(args.model, args.offload, args.guardrails, args.tile_vae)
    set_flow_shift(pipe, 12.0)

    width, height = SIZES[args.size]
    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative,
        num_frames=args.frames,
        height=height,
        width=width,
        fps=float(args.fps),
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
        add_resolution_template=False,
        add_duration_template=False,
    )
    if image_path:
        kwargs["image"] = load_image(image_path)

    with measure("generate") as stats:
        result = pipe(**kwargs)

    out = args.out or output_path(f"{args.model}_{'i2v' if image_path else 't2v'}_{args.size}p.mp4")
    export_to_video(result.video, str(out), fps=args.fps, macro_block_size=1)
    clip_seconds = args.frames / args.fps
    print(f"Saved {out}")
    print(f"{args.frames} frames ({clip_seconds:.1f} s of video) in {stats['seconds']:.1f} s: "
          f"{clip_seconds / stats['seconds']:.2f}x real time")


if __name__ == "__main__":
    main()
