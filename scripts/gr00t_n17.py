"""Run NVIDIA Isaac GR00T N1.7: camera frames + robot state + instruction -> a chunk of actions.

GR00T N1.7 is a 3B "dual-system" VLA: a truncated Qwen3-VL backbone (Cosmos-Reason2-2B)
reads the scene, and a flow-matching DiT action head denoises a chunk of future actions.
This script replays a recorded DROID episode that ships with Isaac-GR00T, compares the
predicted chunks with what the robot actually did (open loop), and times each call.

Run it with the Isaac-GR00T environment, not the cookbook's .venv:

    third_party/Isaac-GR00T/.venv/bin/python scripts/gr00t_n17.py
    third_party/Isaac-GR00T/.venv/bin/python scripts/gr00t_n17.py --describe
    third_party/Isaac-GR00T/.venv/bin/python scripts/gr00t_n17.py --denoising-steps 1 2 4 8
    third_party/Isaac-GR00T/.venv/bin/python scripts/gr00t_n17.py --model nvidia/GR00T-N1.7-DROID

Every GR00T checkpoint names the gated repo nvidia/Cosmos-Reason2-2B for the backbone's
config, processor and initial weights. The Hub's public file listing gives its config,
processor and tokenizer files the same git blob hashes as Qwen/Qwen3-VL-2B-Instruct, and
the GR00T checkpoint holds all 1031 model tensors (no missing keys), so it overwrites the
initial backbone weights. --backbone-source (default: the open Qwen repo) therefore stands
in. Pass --backbone-source nvidia/Cosmos-Reason2-2B once your account has accepted the gate.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch

GR00T = Path(__file__).resolve().parents[1] / "third_party" / "Isaac-GR00T"
GATED_BACKBONE = "nvidia/Cosmos-Reason2-2B"


def redirect_backbone(source: str) -> None:
    """Load the backbone's config/processor/initial weights from `source` instead of the gated repo."""
    import gr00t  # noqa: F401  (installs its own from_pretrained wrappers first)
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    for cls in (Qwen3VLForConditionalGeneration, Qwen3VLProcessor):
        original = cls.from_pretrained.__func__

        def patched(klass, name, *args, _original=original, **kwargs):
            return _original(klass, source if str(name) == GATED_BACKBONE else name, *args, **kwargs)

        cls.from_pretrained = classmethod(patched)


def build_in_bfloat16() -> None:
    """Gr00tPolicy builds the model in float32, then casts to bfloat16; that peak needs >12 GB of RAM.
    Building it in bfloat16 from the start gives the same weights with roughly half the peak."""
    import gr00t.policy.gr00t_policy as policy_module
    from transformers import AutoModel

    class AutoModelBF16:
        @staticmethod
        def from_pretrained(path):
            return AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16)

    policy_module.AutoModel = AutoModelBF16


def count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def describe(model) -> None:
    head, cfg, vlm = model.action_head, model.config, model.backbone.model
    slots = cfg.max_num_embodiments
    projectors = {name: count(getattr(head, name)) for name in ("state_encoder", "action_encoder", "action_decoder")}
    rows = [
        ("total", count(model)),
        ("VLM backbone (as truncated)", count(model.backbone)),
        ("  vision tower", count(vlm.visual)),
        (f"  language model ({len(vlm.language_model.layers)} layers kept)", count(vlm.language_model)),
        ("action head", count(head)),
        ("  DiT", count(head.model)),
        ("  VL self-attention + layernorm", count(head.vl_self_attention) + count(head.vlln)),
        (f"  embodiment projectors ({slots} slots)", sum(projectors.values())),
        *[(f"    {name}", n) for name, n in projectors.items()],
        ("    one embodiment's share", sum(projectors.values()) // slots),
    ]
    print("\nParameters")
    for name, n in rows:
        print(f"  {name:<42} {n / 1e6:>10.2f} M")
    dit = cfg.diffusion_model_cfg
    print("\nAction head config")
    print(f"  DiT: {dit['num_layers']} layers, {dit['num_attention_heads']} heads x {dit['attention_head_dim']} dims "
          f"= {dit['num_attention_heads'] * dit['attention_head_dim']} wide, class {type(head.model).__name__}")
    print(f"  hidden_size {cfg.hidden_size}, input_embedding_dim {cfg.input_embedding_dim}, "
          f"backbone_embedding_dim {cfg.backbone_embedding_dim}")
    print(f"  action_horizon {cfg.action_horizon}, max_state_dim {cfg.max_state_dim}, "
          f"max_action_dim {cfg.max_action_dim}, embodiment slots {slots}")
    print(f"  default denoising steps {cfg.num_inference_timesteps}, backbone select_layer {cfg.select_layer}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="nvidia/GR00T-N1.7-3B", help="HF repo or local checkpoint directory")
    parser.add_argument("--dataset", default=str(GR00T / "demo_data" / "droid_sample"), help="LeRobot-format dataset")
    parser.add_argument("--embodiment", default="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT")
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--calls", type=int, default=12, help="timed calls per setting, spread over the episode")
    parser.add_argument("--denoising-steps", type=int, nargs="+", default=None,
                        help="flow-matching steps to benchmark (default: 1 2 4 and the checkpoint's own)")
    parser.add_argument("--backbone-source", default="Qwen/Qwen3-VL-2B-Instruct",
                        help=f"repo that stands in for the gated {GATED_BACKBONE}")
    parser.add_argument("--describe", action="store_true", help="print parameter counts and action head config")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.utils import parse_observation_gr00t
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    if args.backbone_source != GATED_BACKBONE:
        redirect_backbone(args.backbone_source)
    build_in_bfloat16()
    model_dir = args.model if Path(args.model).is_dir() else snapshot_download(args.model, allow_patterns=[
        "*.json", "*.safetensors"])
    tag = EmbodimentTag.resolve(args.embodiment)

    started = time.perf_counter()
    policy = Gr00tPolicy(embodiment_tag=tag, model_path=model_dir, device="cuda")
    print(f"\nLoaded {args.model} in {time.perf_counter() - started:.1f} s "
          f"({torch.cuda.memory_allocated() / 2**30:.2f} GB of weights on GPU, bfloat16)")
    if args.describe:
        describe(policy.model)

    modality = policy.get_modality_config()
    episode = LeRobotEpisodeLoader(dataset_path=args.dataset, modality_configs=modality)[args.episode]
    inputs_only = {k: v for k, v in modality.items() if k != "action"}
    action_keys = modality["action"].modality_keys
    horizon = len(modality["action"].delta_indices)
    recorded = {k: np.vstack(list(episode[f"action.{k}"])) for k in action_keys}

    def observe(t: int) -> dict:
        step = extract_step_data(episode, t, inputs_only, tag)
        flat = {f"state.{k}": v for k, v in step.states.items()}
        flat |= {f"video.{k}": np.array(v) for k, v in step.images.items()}
        flat |= {k: step.text for k in modality["language"].modality_keys}
        return parse_observation_gr00t(flat, modality)

    obs = observe(0)
    print(f"\nEpisode {args.episode}: {len(episode)} steps. Instruction: {obs['language'][policy.language_key][0][0]!r}")
    print("Inputs (batch, time, ...):")
    for group in ("video", "state"):
        for key, value in obs[group].items():
            print(f"  {group}.{key:<18} {value.dtype!s:<8} {value.shape}")
    actions, _ = policy.get_action(obs)  # also the warm-up call
    print("Outputs (batch, horizon, dims):")
    for key, value in actions.items():
        print(f"  action.{key:<17} {value.dtype!s:<8} {value.shape}")
    print(f"One call plans {horizon} actions; the padded head works on "
          f"{policy.model.config.action_horizon} x {policy.model.config.max_action_dim}.")

    default_steps = policy.model.action_head.num_inference_timesteps
    settings = args.denoising_steps or sorted({1, 2, 4, default_steps})
    starts = np.linspace(0, len(episode) - horizon, args.calls).astype(int)
    observations = [observe(int(t)) for t in starts]  # decode video up front so only the policy is timed
    print(f"\nOpen loop over {len(starts)} calls from steps {starts[0]}..{starts[-1]} "
          f"(MSE in recorded action units, whole {horizon}-step chunk):")
    print(f"  {'steps':>5} {'median ms':>10} {'min ms':>8} {'max ms':>8} {'calls/s':>8} {'MSE':>9} {'peak VRAM':>10}")
    for steps in settings:
        policy.model.action_head.num_inference_timesteps = steps
        policy.get_action(observations[0])
        torch.cuda.reset_peak_memory_stats()
        times, errors = [], {k: [] for k in action_keys}
        for t, observation in zip(starts, observations):
            torch.manual_seed(0)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            actions, _ = policy.get_action(observation)
            torch.cuda.synchronize()
            times.append(1000 * (time.perf_counter() - tick))
            for k in action_keys:
                errors[k].append((actions[k][0] - recorded[k][t:t + horizon]) ** 2)
        per_key = {k: float(np.mean(v)) for k, v in errors.items()}
        mse = float(np.mean(np.concatenate([np.stack(errors[k]) for k in action_keys], axis=-1)))
        median = statistics.median(times)
        note = "  <- checkpoint default" if steps == default_steps else ""
        print(f"  {steps:>5} {median:>10.1f} {min(times):>8.1f} {max(times):>8.1f} {1000 / median:>8.1f} "
              f"{mse:>9.5f} {torch.cuda.max_memory_allocated() / 2**30:>7.2f} GB{note}")
        if steps == default_steps:
            default_per_key = per_key
    if default_steps in settings:
        print(f"MSE by action key at {default_steps} steps: "
              + ", ".join(f"{k} {v:.5f}" for k, v in default_per_key.items()))
    print("Timing covers the whole policy.get_action call: image preprocessing, VLM, denoising, un-normalising.")


if __name__ == "__main__":
    main()
