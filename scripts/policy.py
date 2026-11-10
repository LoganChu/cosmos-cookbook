"""Run a Cosmos 3 robot policy: camera frames + instruction -> a chunk of robot actions.

The policy is a "world action model": it denoises the next 16 actions jointly with the
video it expects to see while executing them. This script replays a recorded DROID
episode and re-plans every chunk, the way a receding-horizon controller would, and
reports whether each plan arrives inside the control budget.

    python scripts/policy.py                          # Edge policy, 3 re-plans
    python scripts/policy.py --replans 6 --steps 8    # fewer denoising steps = faster
    python scripts/policy.py --prompt "Put the marker in the cup."

Needs `make framework` (sample episode and action statistics live in NVIDIA's repo).
"""

from __future__ import annotations

import argparse
import json

import torch
from PIL import Image, ImageOps

from common import OFFLOAD_MODES, ROOT, measure, output_path
from generate import load_pipeline, set_flow_shift

FRAMEWORK = ROOT / "third_party" / "cosmos-framework"
EPISODE = FRAMEWORK / ".agents/skills/cookbooks/cosmos3/generator/action/assets/droid_lerobot_example/videos"
STATS = FRAMEWORK / "cosmos_framework/data/generator/action/normalizer_stats/droid_lerobot_stats.json"
CAMERAS = {
    "wrist": "observation.image.wrist_image_left",
    "exterior_1": "observation.image.exterior_image_1_left",
    "exterior_2": "observation.image.exterior_image_2_left",
}
CANVAS = (640, 540)  # DROID policies see one image: wrist camera on top, two exterior views below
CONTROL_HZ = 15      # DROID control rate
CHUNK = 16           # actions predicted per call


def concat_view(frames: dict[str, Image.Image]) -> Image.Image:
    width, height = CANVAS
    top = height // 2
    canvas = Image.new("RGB", CANVAS)
    for name, size, pos in (
        ("wrist", (width, top), (0, 0)),
        ("exterior_1", (width // 2, height - top), (0, top)),
        ("exterior_2", (width // 2, height - top), (width // 2, top)),
    ):
        canvas.paste(ImageOps.fit(frames[name], size, method=Image.Resampling.BICUBIC), pos)
    return canvas


def denormalize(actions: torch.Tensor) -> torch.Tensor:
    """Map model outputs in [-1, 1] back to robot units using DROID's 1st/99th percentiles."""
    stats = json.loads(STATS.read_text())
    q01, q99 = torch.tensor(stats["q01"]), torch.tensor(stats["q99"])
    return 0.5 * (actions + 1.0) * (q99 - q01) + q01


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="edge-policy", help="edge-policy or nano-policy")
    parser.add_argument("--prompt", default="Pick up the object and place it in the target container.")
    parser.add_argument("--replans", type=int, default=3)
    parser.add_argument("--steps", type=int, default=30, help="denoising steps per plan")
    parser.add_argument("--offload", default="none", choices=OFFLOAD_MODES)
    parser.add_argument("--tile-vae", action="store_true", help="lower peak VRAM")
    parser.add_argument("--actions-only", action="store_true",
                        help="skip decoding the imagined video; a robot only needs the actions")
    args = parser.parse_args()

    if not STATS.exists():
        raise SystemExit("Run `make framework` first: the sample episode comes from NVIDIA's cosmos-framework repo.")

    from diffusers import CosmosActionCondition
    from diffusers.utils import export_to_video, load_video

    episode = {name: load_video(str(EPISODE / folder / "chunk-000" / "file-000.mp4"))
               for name, folder in CAMERAS.items()}
    length = min(len(v) for v in episode.values())

    with measure("load"):
        pipe = load_pipeline(args.model, args.offload, tile_vae=args.tile_vae)
    set_flow_shift(pipe, 5.0)

    budget = CHUNK / CONTROL_HZ
    print(f"\nTask: {args.prompt}")
    print(f"Each plan covers {CHUNK} actions = {budget:.2f} s of motion at {CONTROL_HZ} Hz.\n")

    imagined, plans = [], []
    for i in range(args.replans):
        t = i * CHUNK
        if t >= length:
            break
        observation = concat_view({name: video[t] for name, video in episode.items()})
        with measure(f"plan {i} from frame {t}") as stats:
            result = pipe(
                prompt=args.prompt,
                action=CosmosActionCondition(
                    mode="policy",
                    chunk_size=CHUNK,
                    domain_name="droid_lerobot",
                    resolution_tier=480,
                    image=observation,
                    view_point="concat_view",
                ),
                fps=CONTROL_HZ,
                num_inference_steps=args.steps,
                guidance_scale=1.0,
                generator=torch.Generator(device="cuda").manual_seed(0),
                use_system_prompt=False,
                output_type="latent" if args.actions_only else "pil",
            )
        actions = denormalize(result.action[0].float().cpu())
        plans.append(actions.tolist())
        if not args.actions_only:
            imagined.extend(result.video[1:])
        # The first plan includes one-off warm-up, so judge real-time on the later ones.
        verdict = "in budget" if stats["seconds"] <= budget else f"{stats['seconds'] / budget:.1f}x over budget"
        step = actions[0]
        print(f"  {verdict}; first action: move xyz {[round(v, 4) for v in step[:3].tolist()]} m, "
              f"gripper {step[9].item():.2f}")

    print()
    if imagined:
        video_out = output_path(f"{args.model}_imagined_rollout.mp4")
        export_to_video(imagined, str(video_out), fps=CONTROL_HZ, macro_block_size=1)
        print(f"Saved {video_out} (the future the policy imagined while planning)")
    actions_out = output_path(f"{args.model}_actions.json")
    actions_out.write_text(json.dumps(plans) + "\n")
    print(f"Saved {actions_out} ({len(plans)} plans x {CHUNK} steps x 10: xyz delta, 6-D rotation, gripper)")


if __name__ == "__main__":
    main()
