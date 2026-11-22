"""Serve Cosmos3-Edge-Policy-DROID to the RoboLab simulator, with CFG settable per request.

This is NVIDIA's reference DROID policy server (cosmos_framework.scripts.action_policy_server_robolab)
with one change: each request may carry its own sampling settings instead of the ones fixed at
startup, so `scripts/sim_droid.py` can change classifier-free guidance between plans.

Optional request keys, next to the usual observation and prompt:

    guidance           CFG scale. 1.0 turns CFG off and skips the unconditional pass (about half
                       the compute); larger values push the plan harder toward the instruction.
    guidance_interval  [low, high] in the sampler's 0-1000 timestep scale: CFG applies only to
                       denoising steps inside it. None applies it at every step. NVIDIA serves the
                       Edge policy with [960, 1001]: with 4 steps and shift 5 the step timesteps are
                       about 1000, 938, 833 and 625, so only the first, noisiest step gets CFG.
    num_steps          denoising steps per plan
    seed               sampling seed for this plan
    negative           a negative prompt: CFG's second pass is conditioned on it instead of on an
                       empty caption, so the plan is pushed toward the instruction and away from
                       this, v = v_negative + guidance * (v_instruction - v_negative). It is
                       formatted exactly as the instruction is (the JSON prompt), so the two passes
                       differ only in the instruction text. Has no effect at guidance 1.0, which
                       skips the second pass.
    decode_video       also return the video the policy imagined while planning

Replies carry `action` (32 x 8: seven absolute joint angles in radians, then the gripper, 0 open
and 1 closed), `server_ms`, the `settings` used, and `video` when decode_video was set.

Run it with cosmos-framework's environment (`make sim-setup` builds it):

    make sim-server
    third_party/cosmos-framework/.venv/bin/python scripts/sim_policy_server.py --guidance 1.0
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import threading
import time
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

CHECKPOINT = "nvidia/Cosmos3-Edge-Policy-DROID"
# What NVIDIA's model card passes when serving the Edge policy.
EDGE_GUIDANCE_INTERVAL = (960.0, 1001.0)


def parse_interval(values: list[str] | None) -> tuple[float, float] | None:
    if not values or values == ["none"]:
        return None
    if len(values) != 2:
        raise argparse.ArgumentTypeError("--guidance-interval takes two numbers, or 'none'")
    low, high = (float(v) for v in values)
    return (low, high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=CHECKPOINT, help="Hugging Face repo or local export")
    parser.add_argument("--host", default="127.0.0.1", help="bind address; 0.0.0.0 exposes it to the network")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--guidance", type=float, default=3.0, help="default CFG scale when a request sets none")
    parser.add_argument("--guidance-interval", nargs="+", default=["960", "1001"],
                        help="default CFG timestep window: two numbers, or 'none' for every step")
    parser.add_argument("--steps", type=int, default=4, help="default denoising steps per plan")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Shared-GPU guard, same convention as the other scripts: fail at the cap instead of
    # crowding out another job.
    import torch

    if os.environ.get("COSMOS_VRAM_CAP_GIB") and torch.cuda.is_available():
        cap = float(os.environ["COSMOS_VRAM_CAP_GIB"]) * 1024**3
        torch.cuda.set_per_process_memory_fraction(min(1.0, cap / torch.cuda.mem_get_info()[1]))

    from cosmos_framework.scripts import action_policy_server_robolab as reference

    class SteerablePolicyService(reference.RobolabPolicyService):
        """Reference service whose sampling settings can be overridden per request."""

        def __init__(self, server_args: reference.RobolabServerArgs) -> None:
            super().__init__(server_args)
            self._override_lock = threading.Lock()

        def _settings_for(self, obs: dict[str, Any]) -> dict[str, Any]:
            settings: dict[str, Any] = {}
            if obs.get("guidance") is not None:
                guidance = float(obs["guidance"])
                if not 0.0 <= guidance <= 20.0:
                    raise ValueError(f"guidance must be between 0 and 20, got {guidance}")
                settings["guidance"] = guidance
            if "guidance_interval" in obs:
                interval = obs["guidance_interval"]
                if interval is None:
                    settings["guidance_interval"] = None
                else:
                    low, high = (float(v) for v in interval)
                    if not 0.0 <= low <= high:
                        raise ValueError(f"guidance_interval must satisfy 0 <= low <= high, got {interval}")
                    settings["guidance_interval"] = (low, high)
            if obs.get("num_steps") is not None:
                steps = int(obs["num_steps"])
                if not 1 <= steps <= 100:
                    raise ValueError(f"num_steps must be between 1 and 100, got {steps}")
                settings["num_steps"] = steps
            if obs.get("seed") is not None:
                settings["seed"] = int(obs["seed"])
                settings["deterministic_seed"] = True
            if obs.get("decode_video") is not None:
                settings["decode_video"] = bool(obs["decode_video"])
            if obs.get("negative") is not None and not isinstance(obs["negative"], str):
                raise ValueError("negative must be a string")
            return settings

        def _build_sample(self, obs: dict[str, Any]) -> dict[str, Any]:
            sample = super()._build_sample(obs)
            if obs.get("negative"):
                # Run the negative through the same transform as the instruction, so it gets the
                # same JSON wrapping (view description, duration, fps); only the task text differs.
                negative = super()._build_sample({**obs, "prompt": obs["negative"]})
                sample[self._negative_key] = negative["ai_caption"]
            return sample

        @property
        def _negative_key(self) -> str:
            return "neg_" + getattr(self.model, "input_caption_key", "ai_caption")

        def _generate(self, sample: dict[str, Any], seed: int) -> dict[str, Any]:
            if self._negative_key not in sample:
                return super()._generate(sample, seed)
            # The reference _generate, plus has_negative_prompt: the model then tokenizes
            # data_batch["neg_ai_caption"] for the unconditional pass instead of "".
            data_batch = reference._build_data_batch_from_sample(sample)
            with torch.inference_mode():
                return self.model.generate_samples_from_batch(
                    data_batch,
                    guidance=self.cfg.guidance,
                    guidance_interval=(
                        list(self.cfg.guidance_interval) if self.cfg.guidance_interval is not None else None
                    ),
                    seed=[seed],
                    num_steps=self.cfg.num_steps,
                    shift=self.cfg.shift,
                    has_negative_prompt=True,
                )

        def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
            with self._override_lock:
                default_cfg = self.cfg
                self.cfg = dataclasses.replace(default_cfg, **self._settings_for(obs))
                try:
                    start = time.monotonic()
                    reply = super().infer(obs)
                    reply["server_ms"] = (time.monotonic() - start) * 1000.0
                    reply["settings"] = {
                        "guidance": self.cfg.guidance,
                        "guidance_interval": (
                            list(self.cfg.guidance_interval) if self.cfg.guidance_interval is not None else None
                        ),
                        "num_steps": self.cfg.num_steps,
                        "negative": obs.get("negative") or None,
                    }
                    return reply
                finally:
                    self.cfg = default_cfg

    server_args = reference.RobolabServerArgs(
        checkpoint_path=args.checkpoint,
        host=args.host,
        port=args.port,
        guidance=args.guidance,
        guidance_interval=parse_interval(args.guidance_interval),
        num_steps=args.steps,
        seed=args.seed,
        # The Edge policy was trained on JSON-formatted prompts (NVIDIA's serving command for it
        # passes the same flag); the training config inside the checkpoint is not always found.
        format_prompt_as_json=True,
    )
    # serve() builds the stock service; build ours instead and hand it to the same transport.
    service = SteerablePolicyService(server_args)
    server_cls = reference._load_openpi_websocket_policy_server()
    print(f"[sim-server] Cosmos policy ready on ws://{args.host}:{args.port} "
          f"(defaults: guidance {args.guidance}, interval {parse_interval(args.guidance_interval)}, "
          f"{args.steps} steps)", flush=True)
    server_cls(policy=service, host=args.host, port=args.port, metadata={}).serve_forever()


if __name__ == "__main__":
    main()
