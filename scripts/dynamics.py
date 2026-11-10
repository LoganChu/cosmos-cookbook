"""Action-conditioned world modelling with the Cosmos 3 Generator.

forward   image + action chunks -> video    "what will I see if the robot does this?"
inverse   video                 -> actions  "what actions produced this motion?"

    python scripts/dynamics.py forward    # UMI gripper rollout from NVIDIA's example actions
    python scripts/dynamics.py inverse    # recover a 60-step driving trajectory from dashcam video

Both use the examples bundled with the model repo. Action prompts are plain task
descriptions; unlike video generation they are not JSON-upsampled.
"""

from __future__ import annotations

import argparse
import json

import torch

from common import OFFLOAD_MODES, asset, measure, output_path
from generate import load_pipeline, set_flow_shift


def forward_dynamics(pipe, model: str, steps: int) -> None:
    """Roll the world forward chunk by chunk, feeding each chunk's last frame into the next."""
    from diffusers import CosmosActionCondition
    from diffusers.utils import export_to_video, load_image

    if "nano" in model:
        spec_file, frame_file = ("assets/example_action_fd_agibotworld_action_chunks.json",
                                 "assets/example_action_fd_agibotworld_first_frame.png")
    else:
        spec_file, frame_file = ("assets/example_action_fd_umi_action_chunks.json",
                                 "assets/example_action_fd_umi_first_frame.png")
    spec = json.loads(asset(model, spec_file).read_text())
    frame = load_image(str(asset(model, frame_file)))
    fps, chunk_size = int(spec.get("fps", 20)), int(spec.get("action_chunk_size", 16))

    frames = []
    for i, chunk in enumerate(spec["action_chunks"]):
        with measure(f"chunk {i}: {chunk_size} actions") as stats:
            result = pipe(
                prompt=spec.get("prompt", ""),
                action=CosmosActionCondition(
                    mode="forward_dynamics",
                    chunk_size=chunk_size,
                    domain_name=spec["domain_name"],
                    resolution_tier=int(spec.get("image_size", 256)),
                    raw_actions=torch.tensor(chunk, dtype=torch.float32),
                    image=frame,
                    view_point=spec.get("view_point", "ego_view"),
                ),
                fps=fps,
                num_inference_steps=steps,
                guidance_scale=1.0,
                generator=torch.Generator(device="cuda").manual_seed(i),
                use_system_prompt=False,
            )
        frames.extend(result.video[1:])  # drop the conditioning frame
        frame = result.video[-1]
        budget = chunk_size / fps
        print(f"  simulated {budget:.2f} s of robot time in {stats['seconds']:.2f} s "
              f"({budget / stats['seconds']:.2f}x real time)")

    out = output_path(f"{model}_forward_dynamics.mp4")
    export_to_video(frames, str(out), fps=fps, macro_block_size=1)
    print(f"Saved {out} ({len(frames)} frames, embodiment {spec['domain_name']})")


def inverse_dynamics(pipe, model: str, steps: int) -> None:
    """Label a video with the actions that explain it (9-D ego motion for the AV examples)."""
    from diffusers import CosmosActionCondition
    from diffusers.utils import load_video

    for name in ("av_0", "av_1"):
        video = load_video(str(asset(model, f"assets/example_action_id_{name}_input.mp4")))
        with measure(f"{name}: {len(video)} frames"):
            result = pipe(
                prompt="You are an autonomous vehicle planning system.",
                action=CosmosActionCondition(
                    mode="inverse_dynamics",
                    chunk_size=60,
                    domain_name="av",
                    resolution_tier=480,
                    video=video,
                    view_point="ego_view",
                ),
                fps=10,
                num_inference_steps=steps,
                guidance_scale=1.0,
                generator=torch.Generator(device="cuda").manual_seed(0),
                use_system_prompt=False,
            )
        action = result.action[0].float().cpu()
        out = output_path(f"{model}_inverse_dynamics_{name}.json")
        out.write_text(json.dumps({"shape": list(action.shape), "data": action.tolist()}, indent=2) + "\n")
        print(f"  Saved {out}: {tuple(action.shape)} (steps x action dims)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("forward", "inverse"))
    parser.add_argument("--model", default="edge")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--offload", default="none", choices=OFFLOAD_MODES)
    parser.add_argument("--tile-vae", action="store_true", help="lower peak VRAM at 480p")
    args = parser.parse_args()

    with measure("load"):
        pipe = load_pipeline(args.model, args.offload, tile_vae=args.tile_vae)
    set_flow_shift(pipe, 10.0)
    (forward_dynamics if args.mode == "forward" else inverse_dynamics)(pipe, args.model, args.steps)


if __name__ == "__main__":
    main()
