"""Train and take apart a miniature world-action model built like GR00T 2 (docs/gr00t2.md).

    python scripts/wam.py describe                  # token layout and parameter counts
    python scripts/wam.py train                     # ~10 min: learn both robots from demonstrations
    python scripts/wam.py eval                      # closed-loop success per robot and colour
    python scripts/wam.py imagine                   # PNG: the future it imagined vs what happened
    python scripts/wam.py bench                     # denoising schedules, KV cache, bf16, compile
    python scripts/wam.py adapt --shots 5           # LoRA few-shot on a task it was never shown

    python scripts/wam.py train --no-video --name vla           # same network, actions only
    python scripts/wam.py train --action-frac 0.02 --name wam2  # 2% of episodes have action labels
    python scripts/wam.py train --patch 8 --name turbo          # 4x fewer video tokens

Training covers three tasks: point/reach, point/below and arm/reach. That leaves
arm/below (a skill only the other robot was shown) and the "left" skill (shown to nobody)
for `eval` to test zero-shot and for `adapt` to learn from a few demonstrations.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch

import wam_toy as toy
from common import output_path
from wam_model import MiniWAM, WAMConfig, add_lora, merge_lora

DEVICE = "cuda"
TRAINED = (("point", "reach"), ("point", "below"), ("arm", "reach"))


def ckpt_path(name: str):
    return output_path("miniwam") / f"{name}.pt"


def save(model: MiniWAM, name: str, meta: dict) -> None:
    ckpt_path(name).parent.mkdir(parents=True, exist_ok=True)
    meta = json.loads(json.dumps(meta, default=str))
    torch.save({"cfg": vars(model.cfg), "state": model.state_dict(), "meta": meta}, ckpt_path(name))
    print(f"Saved {ckpt_path(name)}")


def load(name: str) -> MiniWAM:
    blob = torch.load(ckpt_path(name), map_location=DEVICE, weights_only=True)
    model = MiniWAM(WAMConfig(**blob["cfg"])).to(DEVICE)
    model.load_state_dict(blob["state"])
    return model.eval()


def frames_at(demos_or_states, cfg: WAMConfig, world_at) -> torch.Tensor:
    """Stack rendered frames for a list of world states into (B, V, F, 3, H, W)."""
    return torch.stack([toy.render(world_at(i), cfg.image) for i in demos_or_states], 2)


def draw_batch(demos: toy.Demos, cfg: WAMConfig, idx: torch.Tensor, t: torch.Tensor):
    """Training windows starting at step t of episodes idx: frames, state, actions, masks, text."""
    hist_t = [(t - cfg.frame_stride * (cfg.hist - 1 - i)).clamp(min=0) for i in range(cfg.hist)]
    fut_t = [t + cfg.frame_stride * (i + 1) for i in range(cfg.fut)]
    hist = frames_at(hist_t, cfg, lambda ti: demos.at(idx, ti))
    fut = frames_at(fut_t, cfg, lambda ti: demos.at(idx, ti)) if cfg.fut else None
    steps = t[:, None] + torch.arange(cfg.horizon, device=t.device)
    actions = demos.actions[idx[:, None], steps]
    dims = torch.tensor(toy.ACTION_DIMS, device=t.device)[demos.emb[idx]]
    mask = (torch.arange(cfg.max_action, device=t.device) < dims[:, None]) & demos.labelled[idx, None]
    mask = mask[:, None, :].expand(-1, cfg.horizon, -1).float()
    return hist, fut, toy.proprio(demos.at(idx, t)), actions, mask, toy.encode_text(demos.color[idx], demos.skill[idx]), demos.emb[idx]


@torch.no_grad()
def rollout(model: MiniWAM, n: int, emb: str, skill: str, seed: int = 100, execute: int = 8,
            amp: bool = True, **plan_kwargs) -> dict:
    """Closed loop: plan a chunk, execute its first `execute` actions, look again, repeat."""
    cfg = model.cfg
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    emb_ids = torch.full((n,), toy.EMBODIMENTS.index(emb), device=DEVICE)
    color = torch.randint(3, (n,), device=DEVICE, generator=gen)
    skill_ids = torch.full((n,), toy.SKILLS.index(skill), device=DEVICE)
    world = toy.sample_world(n, emb_ids, gen)
    goal = toy.goal_of(world, color, skill_ids)
    text = model.encode_text(toy.encode_text(color, skill_ids))
    states = [world]
    for t in range(0, toy.EPISODE_LEN, execute):
        past = [states[max(t - cfg.frame_stride * (cfg.hist - 1 - i), 0)] for i in range(cfg.hist)]
        hist = frames_at(past, cfg, lambda w: w)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            actions, _ = model.plan(hist, toy.proprio(world), None, emb_ids, text=text, generator=gen, **plan_kwargs)
        for a in actions[:, :execute].unbind(1):
            world = toy.step(world, a)
            states.append(world)
    tip = toy.gripper(world)
    return {"success": ((tip - goal).norm(dim=-1) < toy.SUCCESS_RADIUS).float().mean().item(),
            "mean_distance": (tip - goal).norm(dim=-1).mean().item()}


def success_table(model: MiniWAM, n: int = 256, **kw) -> dict:
    table = {f"{e}/{s}": rollout(model, n, e, s, **kw) for e in toy.EMBODIMENTS for s in toy.SKILLS}
    print(f"  {'':8}" + "".join(f"{s:>8}" for s in toy.SKILLS) + "     (closed-loop success; * = never trained)")
    for e in toy.EMBODIMENTS:
        cells = "".join(f"{table[f'{e}/{s}']['success']:>7.0%}{' ' if (e, s) in TRAINED else '*'}" for s in toy.SKILLS)
        print(f"  {e:8}{cells}")
    table["trained_mean"] = sum(table[f"{e}/{s}"]["success"] for e, s in TRAINED) / len(TRAINED)
    return table


# --- commands ---------------------------------------------------------------------------
def cmd_describe(args) -> None:
    info = MiniWAM(WAMConfig(patch=args.patch, fut=0 if args.no_video else 4)).describe()
    print(json.dumps(info, indent=2))
    # The same arithmetic for the real model, from the white paper's numbers.
    print("\nGR00T 2, for scale (white paper, Sec. 2 and I.1):")
    print("  224x320 image -> Wan 2.1 VAE /8 -> 28x40 latent -> 2x2 patches -> 14x20 = 280 tokens per latent frame per camera")
    print("  Turbo: Wan 2.2 VAE /16 -> 14x20 latent, a 4x smaller grid, same 2B transformer")
    print("  one shared 2B-parameter transformer; 48-action chunks (80 for the G1 humanoid)")


def cmd_train(args) -> None:
    torch.manual_seed(args.seed)
    cfg = WAMConfig(patch=args.patch, fut=0 if args.no_video else 4, dim=args.dim, depth=args.depth)
    model = MiniWAM(cfg).to(DEVICE)
    demos = toy.record_demos(args.episodes, DEVICE, TRAINED, seed=args.seed, horizon=cfg.horizon,
                             action_frac=args.action_frac)
    print(f"{len(demos)} demonstration episodes, {int(demos.labelled.sum())} with action labels; "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    warmup = 300
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    gen = torch.Generator(device=DEVICE).manual_seed(args.seed)
    start, log = time.perf_counter(), []
    for step in range(1, args.steps + 1):
        idx = torch.randint(len(demos), (args.batch,), device=DEVICE, generator=gen)
        t = torch.randint(toy.EPISODE_LEN, (args.batch,), device=DEVICE, generator=gen)
        batch = draw_batch(demos, cfg, idx, t)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = model.loss(*batch)
        # Average the action loss over the episodes that have action labels, not the whole batch.
        has_actions = batch[4].sum((1, 2)) > 0
        losses["action"] = losses["action"][has_actions] if has_actions.any() else losses["action"] * 0
        total = sum(v.mean() for v in losses.values())
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(), sched.step()
        if step % 500 == 0 or step == args.steps:
            row = {"step": step, **{k: round(v.mean().item(), 4) for k, v in losses.items()}}
            row["seconds"] = round(time.perf_counter() - start, 1)
            log.append(row)
            print(row)
    model.eval()
    print(f"\nTrained in {time.perf_counter() - start:.0f} s, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    table = success_table(model)
    save(model, args.name, {"args": vars(args), "log": log, "success": table})


def cmd_eval(args) -> None:
    model = load(args.name)
    v, a = (int(x) for x in args.schedule.lower().rstrip("a").split("v"))
    print(f"{args.name}, schedule {args.schedule}, execute {args.execute} of {model.cfg.horizon} actions per plan")
    success_table(model, args.episodes, video_steps=v, action_steps=a, use_cache=not args.no_cache, execute=args.execute)


@torch.no_grad()
def cmd_imagine(args) -> None:
    """Plan once from the first frame, then execute the whole chunk open loop and compare
    the frames the model imagined with the frames the simulator actually produced."""
    from PIL import Image

    model = load(args.name)
    cfg = model.cfg
    if not cfg.fut:
        raise SystemExit("This checkpoint was trained with --no-video: it has no imagination to show.")
    gen = torch.Generator(device=DEVICE).manual_seed(args.seed)
    color = torch.randint(3, (args.examples,), device=DEVICE, generator=gen)
    skill = torch.tensor([toy.SKILLS.index(TRAINED[i % len(TRAINED)][1]) for i in range(args.examples)], device=DEVICE)
    emb = torch.tensor([toy.EMBODIMENTS.index(TRAINED[i % len(TRAINED)][0]) for i in range(args.examples)], device=DEVICE)
    world = toy.sample_world(args.examples, emb, gen)
    hist = frames_at([world] * cfg.hist, cfg, lambda w: w)
    v, a = (int(x) for x in args.schedule.lower().rstrip("a").split("v"))
    actions, imagined = model.plan(hist, toy.proprio(world), toy.encode_text(color, skill), emb,
                                   video_steps=v, action_steps=a, generator=gen)
    actual = []
    for i, act in enumerate(actions.unbind(1), start=1):
        world = toy.step(world, act)
        if i % cfg.frame_stride == 0:
            actual.append(world)
    actual = frames_at(actual[: cfg.fut], cfg, lambda w: w)
    # One block per example: rows = imagined top, actual top, imagined wrist, actual wrist.
    rows = []
    for b in range(args.examples):
        for view in range(cfg.views):
            for source in (imagined, actual):
                rows.append(torch.cat([hist[b, view, -1], *source[b, view].float().unbind(0)], -1))
        rows.append(torch.full_like(rows[-1][:, :4], -1.0))
    grid = ((torch.cat(rows, 1).clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
    image = Image.fromarray(grid).resize((grid.shape[1] * 6, grid.shape[0] * 6), Image.Resampling.NEAREST)
    out = output_path("miniwam") / f"{args.name}_imagine_{args.schedule}.png"
    image.save(out)
    err = (imagined.float() - actual).abs().mean().item()
    tasks = ", ".join(f"{toy.EMBODIMENTS[e]}: {toy.SKILLS[k]} {toy.COLORS[c]}"
                      for e, k, c in zip(emb.tolist(), skill.tolist(), color.tolist()))
    print(f"Saved {out}\nExamples top to bottom: {tasks}")
    print("Per example, 4 rows: top camera imagined / actual, wrist camera imagined / actual.")
    print(f"Columns: current frame, then +{cfg.frame_stride}, +{2 * cfg.frame_stride}, ... control steps.")
    print(f"Mean absolute pixel error, imagined vs actual: {err:.3f} (pixel range is 2.0)")


@torch.no_grad()
def cmd_bench(args) -> None:
    model = load(args.name)
    cfg = model.cfg
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    emb = torch.zeros(args.batch, dtype=torch.long, device=DEVICE)
    world = toy.sample_world(args.batch, emb, gen)
    hist = frames_at([world] * cfg.hist, cfg, lambda w: w)
    state, text_ids = toy.proprio(world), toy.encode_text(emb, emb)

    def latency(amp: bool, **kw) -> float:
        times = []
        for i in range(args.warmup + args.calls):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                model.plan(hist, state, text_ids, emb, **kw)
            torch.cuda.synchronize()
            if i >= args.warmup:
                times.append((time.perf_counter() - start) * 1000)
        return statistics.median(times)

    schedules = [(4, 4, True), (2, 4, False), (2, 4, True), (2, 2, True), (1, 2, True), (1, 1, True)]
    if not cfg.fut:
        schedules = [(1, a, c) for a, c in ((4, False), (4, True), (2, True), (1, True))]
    results = []
    for v, a, cache in schedules:
        kw = dict(video_steps=v, action_steps=a, use_cache=cache)
        row = {"schedule": f"{v}v{a}a" if cfg.fut else f"{a}a", "kv_cache": cache,
               "ms_fp32": latency(False, **kw), "ms_bf16": latency(True, **kw)}
        row["success"] = statistics.mean(rollout(model, args.episodes, e, s, **kw)["success"] for e, s in TRAINED)
        results.append(row)
    if args.compile:
        eager = model.forward
        model.forward = torch.compile(eager)
        for row, (v, a, cache) in zip(results, schedules):
            row["ms_bf16_compiled"] = latency(True, video_steps=v, action_steps=a, use_cache=cache)
        model.forward = eager
    print(f"\n{args.name}: one plan at batch {args.batch}, median of {args.calls} calls; success over {args.episodes} "
          f"episodes x 3 trained tasks\n")
    cols = list(results[0])
    print("  ".join(f"{c:>16}" for c in cols))
    for row in results:
        print("  ".join(f"{row[c]:>16.1%}" if c == "success" else f"{row[c]:>16.2f}" if isinstance(row[c], float)
                        else f"{str(row[c]):>16}" for c in cols))
    budget = cfg.horizon / 2 / 30 * 1000
    print(f"\nFor scale: executing {cfg.horizon // 2} actions per plan at 30 Hz gives a budget of {budget:.0f} ms per plan.")


def cmd_adapt(args) -> None:
    """Few-shot LoRA on a task the model was never shown, following the white paper's Sec. A.3 recipe."""
    task = tuple(args.task.split("/"))
    torch.manual_seed(args.seed)
    model = load(args.name)
    cfg = model.cfg
    print("Before adaptation:")
    before = success_table(model, args.episodes)
    demos = toy.record_demos(args.shots, DEVICE, (task,), seed=1000 + args.seed, horizon=cfg.horizon)
    idx_all = torch.arange(args.shots, device=DEVICE).repeat_interleave(toy.EPISODE_LEN)
    t_all = torch.arange(toy.EPISODE_LEN, device=DEVICE).repeat(args.shots)

    # Sample windows in proportion to the frozen model's loss, so updates go to the hard phases.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        batch = draw_batch(demos, cfg, idx_all, t_all)
        weight = torch.stack([sum(model.loss(*batch).values()) for _ in range(8)]).mean(0).float()
    if args.uniform:
        weight = torch.ones_like(weight)

    params = [p for p in model.parameters() if p.requires_grad] if args.full else add_lora(model, args.rank)
    print(f"\n{args.shots} demonstration(s) of {args.task}; training "
          f"{sum(p.numel() for p in params) / 1e6:.2f}M parameters ({'full fine-tune' if args.full else f'LoRA rank {args.rank}'})")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    model.train()
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        pick = torch.multinomial(weight, args.batch, replacement=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = model.loss(*draw_batch(demos, cfg, idx_all[pick], t_all[pick]))
        total = sum(v.mean() for v in losses.values())
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        if step % 100 == 0 or step == args.steps:
            print({"step": step, **{k: round(v.mean().item(), 4) for k, v in losses.items()}})
    seconds = time.perf_counter() - start
    if not args.full:
        merge_lora(model)
    model.eval()
    print(f"\nAfter adaptation ({seconds:.0f} s):")
    after = success_table(model, args.episodes)
    key = args.task
    print(f"\n{key}: {before[key]['success']:.0%} -> {after[key]['success']:.0%}.  "
          f"Originally trained tasks: {before['trained_mean']:.0%} -> {after['trained_mean']:.0%}")
    save(model, f"{args.name}_{key.replace('/', '-')}_{args.shots}shot", {"args": vars(args), "before": before, "after": after})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name, fn, **defaults):
        p = sub.add_parser(name, help=fn.__doc__)
        p.set_defaults(fn=fn)
        p.add_argument("--name", default="base", help="checkpoint name under outputs/miniwam/")
        return p

    p = command("describe", cmd_describe)
    p.add_argument("--patch", type=int, default=4)
    p.add_argument("--no-video", action="store_true")

    p = command("train", cmd_train)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--episodes", type=int, default=4000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--patch", type=int, default=4, help="8 gives 4x fewer video tokens (the Turbo trade)")
    p.add_argument("--no-video", action="store_true", help="action-only policy: no future frames, no video loss")
    p.add_argument("--action-frac", type=float, default=1.0,
                   help="fraction of episodes that keep action labels; the rest are video only")
    p.add_argument("--seed", type=int, default=0)

    p = command("eval", cmd_eval)
    p.add_argument("--schedule", default="2v4a", help="video steps and action steps, e.g. 2v4a, 1v2a, 4v4a")
    p.add_argument("--no-cache", action="store_true", help="recompute video tokens on action-only steps")
    p.add_argument("--execute", type=int, default=8, help="actions executed before re-planning")
    p.add_argument("--episodes", type=int, default=256)

    p = command("imagine", cmd_imagine)
    p.add_argument("--schedule", default="2v4a")
    p.add_argument("--examples", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)

    p = command("bench", cmd_bench)
    p.add_argument("--calls", type=int, default=50)
    p.add_argument("--batch", type=int, default=1, help="plans per call; large batches show compute, not overhead")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--compile", action="store_true", help="also time torch.compile (adds a minute of compilation)")

    p = command("adapt", cmd_adapt)
    p.add_argument("--task", default="arm/left", help="embodiment/skill to learn, e.g. arm/left, arm/below, point/left")
    p.add_argument("--shots", type=int, default=5)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--full", action="store_true", help="fine-tune every weight instead of LoRA")
    p.add_argument("--uniform", action="store_true", help="sample windows uniformly, not by frozen-model loss")
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
