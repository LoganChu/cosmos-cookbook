"""A miniature world-action model with the structure the GR00T 2 white paper describes.

One transformer denoises future video and an action chunk together (flow matching, one
loss each). What is copied from the paper, at toy scale:

    video tokens      past frames enter clean as context, future frames are generated;
                      a learned view embedding says which camera a token came from
    language          a frozen text encoder, read through cross-attention
    state / actions   small per-embodiment MLPs in, a per-embodiment decoder out;
                      the transformer itself is shared by every robot
    attention         video tokens attend to video; action tokens attend to everything,
                      including the future frames being generated
    inference         "2v4a": two video steps, four action steps. Action-only steps reuse
                      the attention keys and values of the last video step.
    adaptation        rank-32 LoRA on every linear layer of the transformer, heads frozen

What is NOT copied: the real model is a 2B-parameter Cosmos-Predict2.5 backbone working on
Wan 2.1 VAE latents with a Cosmos-Reason1 text encoder. Here the transformer is about
5M parameters, "latents" are raw 4x4 pixel patches and the text encoder is a frozen
lookup table. See docs/gr00t2.md for the side-by-side.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WAMConfig:
    image: int = 32
    patch: int = 4            # larger patch = fewer video tokens (the "Turbo" trade)
    views: int = 2
    hist: int = 2             # past frames given as context
    fut: int = 4              # future frames generated; 0 turns the model into an action-only policy
    frame_stride: int = 4     # control steps between video frames
    horizon: int = 16         # actions per chunk
    dim: int = 256
    depth: int = 6
    heads: int = 4
    text_dim: int = 64
    vocab: int = 24
    max_state: int = 6
    max_action: int = 3
    embodiments: int = 4      # slots; unused ones are spare rows for a new robot

    @property
    def patches(self) -> int:
        return (self.image // self.patch) ** 2

    @property
    def patch_dim(self) -> int:
        return 3 * self.patch**2


class EmbodimentMLP(nn.Module):
    """A two-layer MLP with one row of weights per robot, selected by embodiment id."""

    def __init__(self, n: int, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(n, d_in, d_hidden) * d_in**-0.5)
        self.b1 = nn.Parameter(torch.zeros(n, 1, d_hidden))
        self.w2 = nn.Parameter(torch.randn(n, d_hidden, d_out) * d_hidden**-0.5)
        self.b2 = nn.Parameter(torch.zeros(n, 1, d_out))

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:  # x: (B, T, d_in)
        h = F.silu(torch.bmm(x, self.w1[emb]) + self.b1[emb])
        return torch.bmm(h, self.w2[emb]) + self.b2[emb]


def timestep_features(t: torch.Tensor, dim: int) -> torch.Tensor:
    freqs = torch.exp(-math.log(1000.0) * torch.arange(dim // 2, device=t.device) / (dim // 2))
    args = t[:, None].float() * 1000.0 * freqs
    return torch.cat([args.sin(), args.cos()], -1)


class Block(nn.Module):
    """Self-attention, cross-attention to text, MLP. Runs a context stream (video + state)
    and an action stream through the same weights; only the action stream sees both."""

    def __init__(self, cfg: WAMConfig):
        super().__init__()
        d = cfg.dim
        self.heads = cfg.heads
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.xq, self.xkv, self.xproj = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def _split(self, x: torch.Tensor, parts: int):
        b, n, _ = x.shape
        return x.view(b, n, parts, self.heads, -1).permute(2, 0, 3, 1, 4)  # (parts, B, heads, N, d_head)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2).flatten(2)

    def _rest(self, x: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        q = self._split(self.xq(self.norm2(x)), 1)[0]
        k, v = self._split(self.xkv(text), 2)
        x = x + self.xproj(self._merge(F.scaled_dot_product_attention(q, k, v)))
        return x + self.mlp(self.norm3(x))

    def forward(self, ctx, act, text, cached=None):
        """ctx is None on an action-only step, in which case `cached` supplies its keys/values."""
        if ctx is not None:
            q, k, v = self._split(self.qkv(self.norm1(ctx)), 3)
            ctx = self._rest(ctx + self.proj(self._merge(F.scaled_dot_product_attention(q, k, v))), text)
            cached = (k, v)
        qa, ka, va = self._split(self.qkv(self.norm1(act)), 3)
        keys, values = torch.cat([cached[0], ka], 2), torch.cat([cached[1], va], 2)
        act = self._rest(act + self.proj(self._merge(F.scaled_dot_product_attention(qa, keys, values))), text)
        return ctx, act, cached


class MiniWAM(nn.Module):
    def __init__(self, cfg: WAMConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        # Frozen "text encoder": stands in for the frozen VLM that encodes the instruction.
        self.text_encoder = nn.Embedding(cfg.vocab, cfg.text_dim).requires_grad_(False)
        self.text_proj = nn.Linear(cfg.text_dim, d)
        self.patch_in = nn.Linear(cfg.patch_dim, d)
        self.patch_out = nn.Linear(d, cfg.patch_dim)
        self.space_pos = nn.Parameter(torch.randn(cfg.patches, d) * 0.02)
        self.frame_pos = nn.Parameter(torch.randn(cfg.hist + cfg.fut, d) * 0.02)
        self.view_emb = nn.Parameter(torch.randn(cfg.views, d) * 0.02)
        self.action_pos = nn.Parameter(torch.randn(cfg.horizon, d) * 0.02)
        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        # Embodiment-specific parts: the only weights a new robot does not share.
        self.state_in = EmbodimentMLP(cfg.embodiments, cfg.max_state, d, d)
        self.action_in = EmbodimentMLP(cfg.embodiments, cfg.max_action, d, d)
        self.action_out = EmbodimentMLP(cfg.embodiments, d, d, cfg.max_action)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.depth))
        self.norm_ctx, self.norm_act = nn.LayerNorm(d), nn.LayerNorm(d)
        nn.init.zeros_(self.patch_out.weight), nn.init.zeros_(self.patch_out.bias)

    # --- tokens -------------------------------------------------------------------------
    def patchify(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, V, F, 3, H, W) -> (B, V, F, patches, patch_dim)"""
        p = self.cfg.patch
        b, v, f, c, h, w = frames.shape
        x = frames.view(b, v, f, c, h // p, p, w // p, p).permute(0, 1, 2, 4, 6, 3, 5, 7)
        return x.reshape(b, v, f, -1, c * p * p)

    def unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        p, s = self.cfg.patch, self.cfg.image // self.cfg.patch
        b, v, f = tokens.shape[:3]
        x = tokens.view(b, v, f, s, s, 3, p, p).permute(0, 1, 2, 5, 3, 6, 4, 7)
        return x.reshape(b, v, f, 3, s * p, s * p)

    def _time(self, t: torch.Tensor) -> torch.Tensor:
        return self.time_mlp(timestep_features(t, self.cfg.dim).to(self.patch_in.weight.dtype))

    def encode_text(self, text_ids: torch.Tensor) -> torch.Tensor:
        """Frozen encoder + projection. Reusable for as long as the instruction is unchanged."""
        return self.text_proj(self.text_encoder(text_ids))

    def context_tokens(self, hist, fut_noisy, state, t_video, emb) -> torch.Tensor:
        cfg = self.cfg
        b = hist.shape[0]
        frames = hist if fut_noisy is None else torch.cat([hist, fut_noisy], 2)
        x = self.patch_in(self.patchify(frames))                                    # (B, V, F, P, d)
        x = x + self.space_pos + self.frame_pos[: x.shape[2], None] + self.view_emb[:, None, None]
        level = torch.zeros(b, x.shape[2], device=x.device)                          # past frames are clean: t = 0
        if fut_noisy is not None:
            level[:, cfg.hist:] = t_video[:, None]
        x = x + self._time(level.flatten()).view(b, 1, x.shape[2], 1, -1)
        return torch.cat([x.flatten(1, 3), self.state_in(state[:, None], emb)], 1)

    # --- one denoising pass -------------------------------------------------------------
    def forward(self, hist, fut_noisy, state, act_noisy, t_video, t_action, text, emb, cache=None):
        """Predict flow velocities (noise minus data) for future video and for the action chunk.

        Pass `cache` (from an earlier call) to run an action-only step: the video and state
        tokens are not recomputed and no video velocity is returned.
        """
        cfg = self.cfg
        act = self.action_in(act_noisy, emb) + self.action_pos + self._time(t_action)[:, None]
        ctx = None if cache is not None else self.context_tokens(hist, fut_noisy, state, t_video, emb)
        new_cache = []
        for i, block in enumerate(self.blocks):
            ctx, act, kv = block(ctx, act, text, None if cache is None else cache[i])
            new_cache.append(kv)
        v_action = self.action_out(self.norm_act(act), emb)
        v_video = None
        if ctx is not None and fut_noisy is not None:
            b = ctx.shape[0]
            video = self.norm_ctx(ctx[:, :-1]).view(b, cfg.views, cfg.hist + cfg.fut, cfg.patches, -1)
            v_video = self.unpatchify(self.patch_out(video[:, :, cfg.hist:]))
        return v_video, v_action, new_cache

    # --- training -----------------------------------------------------------------------
    def loss(self, hist, fut, state, actions, action_mask, text_ids, emb):
        """Flow matching: x_t = (1 - t) * data + t * noise, and the target is noise - data.
        Video and actions get independent noise levels so they can be denoised on different
        schedules at inference. `action_mask` zeroes padded dimensions and unlabelled episodes."""
        b = hist.shape[0]
        t_v, t_a = torch.rand(b, device=hist.device), torch.rand(b, device=hist.device)
        eps_a = torch.randn_like(actions)
        act_noisy = (1 - t_a)[:, None, None] * actions + t_a[:, None, None] * eps_a
        fut_noisy = eps_v = None
        if self.cfg.fut:
            eps_v = torch.randn_like(fut)
            tv = t_v.view(b, 1, 1, 1, 1, 1)
            fut_noisy = (1 - tv) * fut + tv * eps_v
        v_video, v_action, _ = self(hist, fut_noisy, state, act_noisy, t_v, t_a, self.encode_text(text_ids), emb)
        err = ((v_action.float() - (eps_a - actions)) ** 2) * action_mask
        per_sample = {"action": err.sum((1, 2)) / action_mask.sum((1, 2)).clamp(min=1)}
        if v_video is not None:
            per_sample["video"] = ((v_video.float() - (eps_v - fut)) ** 2).mean((1, 2, 3, 4, 5))
        return per_sample

    # --- inference ----------------------------------------------------------------------
    @torch.no_grad()
    def plan(self, hist, state, text_ids, emb, video_steps=2, action_steps=4, use_cache=True,
             text=None, generator=None):
        """Denoise an action chunk and the imagined future. `video_steps` v, `action_steps` a
        is the paper's "v-v-a-a" notation: 2v4a runs four passes, and every second one moves
        only the actions. With use_cache those passes skip the video tokens entirely."""
        cfg = self.cfg
        b, dev = hist.shape[0], hist.device
        text = self.encode_text(text_ids) if text is None else text
        act = torch.randn(b, cfg.horizon, cfg.max_action, device=dev, generator=generator)
        fut = None
        if cfg.fut:
            video_steps = max(1, min(video_steps, action_steps))
            fut = torch.randn(b, cfg.views, cfg.fut, 3, cfg.image, cfg.image, device=dev, generator=generator)
        every = action_steps // video_steps if cfg.fut else action_steps
        t_v, cache = torch.ones(b, device=dev), None
        for k in range(action_steps):
            t_a = torch.full((b,), 1 - k / action_steps, device=dev)
            joint = k % every == 0
            if joint or not use_cache:
                v_video, v_action, cache = self(hist, fut, state, act, t_v, t_a, text, emb)
                if joint and fut is not None:
                    fut = fut - v_video.to(fut.dtype) / video_steps
                    t_v = t_v - 1 / video_steps
            else:
                _, v_action, _ = self(None, None, None, act, None, t_a, text, emb, cache=cache)
            act = act - v_action.to(act.dtype) / action_steps
        return act.clamp(-1, 1), fut

    def describe(self) -> dict:
        count = lambda *mods: sum(p.numel() for m in mods for p in (m.parameters() if isinstance(m, nn.Module) else [m]))
        cfg = self.cfg
        return {
            "config": asdict(cfg),
            "tokens": {
                "per frame per camera": cfg.patches,
                "context video": cfg.views * cfg.hist * cfg.patches,
                "generated video": cfg.views * cfg.fut * cfg.patches,
                "state": 1,
                "action": cfg.horizon,
            },
            "parameters": {
                "shared transformer": count(self.blocks, self.patch_in, self.patch_out, self.time_mlp, self.text_proj,
                                            self.norm_ctx, self.norm_act, self.space_pos, self.frame_pos,
                                            self.view_emb, self.action_pos),
                "per-embodiment heads, each robot": count(self.state_in, self.action_in, self.action_out) // cfg.embodiments,
                "frozen text encoder": count(self.text_encoder),
            },
        }


# --- LoRA -------------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank update: W + (alpha / r) * B @ A."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(torch.randn(rank, base.in_features) * base.in_features**-0.5)
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))  # zero init: starts as a no-op
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.scale

    def merged(self) -> nn.Linear:
        self.base.weight.data += (self.b @ self.a) * self.scale
        return self.base


def add_lora(model: MiniWAM, rank: int = 32, alpha: float | None = None) -> list[nn.Parameter]:
    """Freeze the model and put adapters on every linear layer inside the transformer blocks.
    The text encoder and the per-embodiment state/action heads stay frozen, as in the paper."""
    model.requires_grad_(False)
    for block in model.blocks:
        for parent in list(block.modules()):
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(parent, name, LoRALinear(child, rank, alpha or rank).to(child.weight.device))
    return [p for p in model.parameters() if p.requires_grad]


def merge_lora(model: MiniWAM) -> None:
    """Fold adapters into the base weights so the adapted model is served like the original."""
    for block in model.blocks:
        for parent in list(block.modules()):
            for name, child in list(parent.named_children()):
                if isinstance(child, LoRALinear):
                    setattr(parent, name, child.merged())
