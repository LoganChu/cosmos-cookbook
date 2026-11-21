"""A tiny tabletop world for the miniature world-action model (see docs/gr00t2.md).

Two robots share one table with a red, a green and a blue block:

    point   a free-floating gripper. State: (x, y).           Action: 2-D position delta.
    arm     a 3-link planar arm.     State: sin/cos of joints. Action: 3-D joint delta.

Each is seen by two cameras, a fixed top view and a "wrist" view centred on the gripper.
A task is a skill plus a colour, given as a sentence: "reach the green block", "stop below
the red block", "stop left of the blue block". Everything is batched on the GPU, so
thousands of episodes simulate and render in milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

EMBODIMENTS = ("point", "arm")
COLORS = ("red", "green", "blue")
RGB = torch.tensor([[0.9, 0.15, 0.15], [0.15, 0.8, 0.2], [0.2, 0.35, 0.95]])
SKILLS = ("reach", "below", "left")
OFFSETS = torch.tensor([[0.0, 0.0], [0.0, -0.14], [-0.14, 0.0]])   # goal = block centre + offset
TEMPLATES = (
    ("reach the {} block", "go to the {} cube", "move to {}", "touch the {} one"),
    ("stop below the {} block", "go under the {} cube", "park below {}"),
    ("stop left of the {} block", "go left of {}", "park left of the {} cube"),
)
VOCAB = {w: i + 1 for i, w in enumerate(sorted({w for ts in TEMPLATES for t in ts for w in t.split()} - {"{}"} | set(COLORS)))}
TEXT_LEN = 6

MAX_STATE, MAX_ACTION = 6, 3      # every robot's vectors are zero-padded to these sizes
ACTION_DIMS = (2, 3)              # real action size per embodiment
POINT_STEP, JOINT_STEP = 0.04, 0.08   # one full-scale action moves this far (table units, radians)
ARM_BASE = (0.5, -0.05)
ARM_LINKS = (0.40, 0.35, 0.30)
EPISODE_LEN = 48                  # control steps per episode
SUCCESS_RADIUS = 0.06
WRIST_HALF = 0.25                 # the wrist camera sees a 0.5 x 0.5 window around the gripper


@dataclass
class World:
    """Batched world state. `q` is used by the arm, `p` by the point robot; `emb` picks which."""

    blocks: torch.Tensor   # (N, 3, 2) block centres, in colour order
    q: torch.Tensor        # (N, 3) joint angles
    p: torch.Tensor        # (N, 2) point-robot position
    emb: torch.Tensor      # (N,) 0 = point, 1 = arm

    def clone(self) -> "World":
        return World(self.blocks, self.q.clone(), self.p.clone(), self.emb)


def arm_joints(q: torch.Tensor) -> torch.Tensor:
    """Forward kinematics: (N, 3) angles -> (N, 4, 2) positions of base, two elbows and tip."""
    pts = [q.new_tensor(ARM_BASE).expand(q.shape[0], 2)]
    angle = torch.zeros_like(q[:, 0])
    for i, length in enumerate(ARM_LINKS):
        angle = angle + q[:, i]
        pts.append(pts[-1] + length * torch.stack([angle.cos(), angle.sin()], -1))
    return torch.stack(pts, 1)


def gripper(w: World) -> torch.Tensor:
    return torch.where(w.emb[:, None] == 1, arm_joints(w.q)[:, -1], w.p)


def proprio(w: World) -> torch.Tensor:
    """Robot state as the policy sees it, padded to MAX_STATE."""
    point = torch.cat([w.p * 2 - 1, w.p.new_zeros(len(w.p), MAX_STATE - 2)], -1)
    arm = torch.cat([w.q.sin(), w.q.cos()], -1)
    return torch.where(w.emb[:, None] == 1, arm, point)


def sample_world(n: int, emb: torch.Tensor, gen: torch.Generator) -> World:
    dev = emb.device
    rand = lambda *shape: torch.rand(*shape, generator=gen, device=dev)
    blocks = 0.15 + 0.7 * rand(n, 3, 2)
    for _ in range(50):  # resample scenes whose blocks overlap
        d = torch.cdist(blocks, blocks) + 10 * torch.eye(3, device=dev)
        bad = d.amin((1, 2)) < 0.22
        if not bad.any():
            break
        blocks[bad] = 0.15 + 0.7 * rand(int(bad.sum()), 3, 2)
    q = torch.stack([torch.pi * (0.3 + 0.4 * rand(n)), 1.6 * (rand(n) - 0.5) + 0.6, 1.6 * (rand(n) - 0.5) + 0.6], -1)
    return World(blocks, q, 0.1 + 0.8 * rand(n, 2), emb)


def step(w: World, action: torch.Tensor) -> World:
    """Apply one action (N, MAX_ACTION) in [-1, 1]."""
    a = action.float().clamp(-1, 1)
    arm = (w.emb == 1)[:, None]
    q = torch.where(arm, w.q + JOINT_STEP * a, w.q)
    p = torch.where(arm, w.p, (w.p + POINT_STEP * a[:, :2]).clamp(0, 1))
    return World(w.blocks, q, p, w.emb)


def goal_of(w: World, color: torch.Tensor, skill: torch.Tensor) -> torch.Tensor:
    return w.blocks[torch.arange(len(color)), color] + OFFSETS.to(color.device)[skill]


def expert(w: World, goal: torch.Tensor) -> torch.Tensor:
    """The demonstrator: head straight for the goal point. Returns (N, MAX_ACTION)."""
    point = torch.cat([(goal - w.p) / POINT_STEP, w.p.new_zeros(len(goal), 1)], -1)
    # Arm: damped least-squares inverse kinematics on the tip position.
    joints = arm_joints(w.q)
    r = joints[:, -1:, :] - joints[:, :3, :]                 # tip relative to each joint
    jac = torch.stack([-r[..., 1], r[..., 0]], 1)            # (N, 2, 3)
    err = (goal - joints[:, -1]).unsqueeze(-1)
    jjt = jac @ jac.transpose(1, 2) + 0.01 * torch.eye(2, device=err.device)
    arm = (jac.transpose(1, 2) @ torch.linalg.solve(jjt, err)).squeeze(-1) / JOINT_STEP
    act = torch.where(w.emb[:, None] == 1, arm, point)
    scale = act.abs().amax(-1, keepdim=True).clamp(min=1.0)  # keep direction when saturating
    return act / scale


def _disc(grid, centre, radius, soft):
    return torch.sigmoid((radius - (grid - centre[:, None, None, :]).norm(dim=-1)) / soft)


def render(w: World, size: int = 32) -> torch.Tensor:
    """Render both cameras: (N, 2 views, 3, size, size) in [-1, 1]. View 0 = top, 1 = wrist."""
    n, dev = len(w.emb), w.emb.device
    tip = gripper(w)
    axis = (torch.arange(size, device=dev) + 0.5) / size
    unit = torch.stack(torch.meshgrid(axis, axis, indexing="xy"), -1)          # (S, S, 2) in [0, 1]
    unit = torch.stack([unit[..., 0], 1 - unit[..., 1]], -1)                   # y up
    views = []
    for centre, half in ((tip.new_full((n, 2), 0.5), 0.5), (tip, WRIST_HALF)):
        grid = centre[:, None, None, :] + (unit[None] - 0.5) * 2 * half
        soft = half / size
        img = torch.full((n, size, size, 3), 0.08, device=dev)
        joints = arm_joints(w.q)
        for i in range(3):  # arm links: distance from each pixel to the segment
            a, b = joints[:, i, None, None, :], joints[:, i + 1, None, None, :]
            t = (((grid - a) * (b - a)).sum(-1) / ((b - a) ** 2).sum(-1)).clamp(0, 1)
            dist = (grid - (a + t[..., None] * (b - a))).norm(dim=-1)
            alpha = torch.sigmoid((0.018 - dist) / soft) * (w.emb == 1)[:, None, None]
            img = img * (1 - alpha[..., None]) + 0.55 * alpha[..., None]
        for c in range(3):
            alpha = _disc(grid, w.blocks[:, c], 0.06, soft)[..., None]
            img = img * (1 - alpha) + RGB.to(dev)[c] * alpha
        alpha = _disc(grid, tip, 0.035, soft)[..., None]
        img = img * (1 - alpha) + alpha
        views.append(img.permute(0, 3, 1, 2))
    return torch.stack(views, 1) * 2 - 1


def encode_text(color: torch.Tensor, skill: torch.Tensor) -> torch.Tensor:
    """Instruction token ids (N, TEXT_LEN) for each task, with a random phrasing."""
    ids = torch.zeros(len(color), TEXT_LEN, dtype=torch.long)
    for i, (c, s) in enumerate(zip(color.tolist(), skill.tolist())):
        phrasing = TEMPLATES[s][int(torch.randint(len(TEMPLATES[s]), (1,)))]
        words = phrasing.format(COLORS[c]).split()
        ids[i, : len(words)] = torch.tensor([VOCAB[word] for word in words])
    return ids.to(color.device)


@dataclass
class Demos:
    """Recorded expert episodes. Frames are re-rendered from states when a batch is drawn."""

    blocks: torch.Tensor   # (N, 3, 2)
    q: torch.Tensor        # (N, T + 1, 3)
    p: torch.Tensor        # (N, T + 1, 2)
    actions: torch.Tensor  # (N, T, MAX_ACTION)
    emb: torch.Tensor      # (N,)
    color: torch.Tensor    # (N,)
    skill: torch.Tensor    # (N,)
    labelled: torch.Tensor  # (N,) False = video-only episode, like human video without actions

    def __len__(self) -> int:
        return len(self.emb)

    def at(self, idx: torch.Tensor, t: torch.Tensor) -> World:
        return World(self.blocks[idx], self.q[idx, t], self.p[idx, t], self.emb[idx])


def record_demos(n: int, device, tasks, seed: int = 0, horizon: int = 16, action_frac: float = 1.0) -> Demos:
    """Roll out the expert on a mix of (embodiment, skill) tasks; block colour is random."""
    gen = torch.Generator(device=device).manual_seed(seed)
    pairs = torch.tensor([(EMBODIMENTS.index(e), SKILLS.index(s)) for e, s in tasks], device=device)
    pick = pairs[torch.randint(len(pairs), (n,), generator=gen, device=device)]
    emb, skill = pick[:, 0], pick[:, 1]
    color = torch.randint(3, (n,), generator=gen, device=device)
    w = sample_world(n, emb, gen)
    goal = goal_of(w, color, skill)
    qs, ps, acts = [w.q], [w.p], []
    for _ in range(EPISODE_LEN + horizon):
        a = expert(w, goal)
        w = step(w, a)
        qs.append(w.q), ps.append(w.p), acts.append(a)
    labelled = torch.rand(n, generator=gen, device=device) < action_frac
    return Demos(w.blocks, torch.stack(qs, 1), torch.stack(ps, 1), torch.stack(acts, 1), emb, color, skill, labelled)
