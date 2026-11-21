# Cosmos3-Edge-Policy-DROID in simulation

This puts the Edge DROID policy in control of a simulated robot: you type an instruction, the
policy plans, the arm moves, and you can change the classifier-free guidance (CFG) scale between
plans. The robot is the DROID setup the policy was trained on, a Franka arm with a Robotiq gripper,
a wrist camera and two over-the-shoulder cameras. It stands in one of
[RoboLab](https://github.com/NVlabs/RoboLab)'s 120 tabletop scenes, NVIDIA's benchmark for
exactly this kind of policy.

## How it fits together

```
 terminal 2: make sim                                   terminal 1: make sim-server
 ┌───────────────────────────────────────────┐          ┌──────────────────────────────────────────┐
 │ scripts/sim_droid.py   (RoboLab env)      │          │ scripts/sim_policy_server.py             │
 │                                           │  ws://   │   NVIDIA's reference DROID policy server │
 │ Isaac Sim 6.0.1: scene, robot, physics    │ ───────► │   + per-request CFG, steps, seed, video  │
 │ 3 cameras → one 640x540 image             │ image,   │                                          │
 │ your instruction, /cfg, /steps ...        │ joints,  │ Cosmos3-Edge-Policy-DROID (4B)           │
 │                                           │ prompt,  │   flow-matches 32 actions and 33 frames  │
 │ executes 32 joint targets at 15 Hz        │ ◄─────── │   of video together                      │
 └───────────────────────────────────────────┘ actions  └──────────────────────────────────────────┘
```

Each plan sends one image (the wrist camera on top, the two shoulder cameras side by side below,
as in training), the seven joint angles and gripper opening, and the instruction. The policy
returns 32 rows of eight numbers: absolute joint angles in radians and the gripper (0 open,
1 closed), 2.13 s of motion at DROID's 15 Hz. The arm executes them, then the policy plans again
from the new camera images. Simulated time pauses while the policy thinks, so a slow setting
makes the run take longer but does not change what the robot does.

The two halves run in separate environments because they cannot share one: Isaac Sim 6 needs
Python 3.12 and its own PyTorch, while the policy server pins Python 3.13 and PyTorch 2.13.

## Setup

```bash
make sim-setup      # about 40 GB: RoboLab + Isaac Sim 6.0.1 + Isaac Lab 3.0 beta, and the server env
```

Three things in that target are workarounds found while building this, worth knowing if
something breaks:

- **Isaac Sim 6, not RoboLab's default 5.0.** Isaac Sim 5.0 crashes on this machine's 595
  driver: its Warp library cannot find `cuDeviceGetUuid` and the renderer segfaults while
  creating the first stage. RoboLab supports 6.0.1 as an opt-in for recent drivers.
- **`websockets` installed by hand.** RoboLab's dependency override for it only covers Python
  below 3.12, so its Isaac Sim 6 environment comes without the package its policy clients import.
- **Four extra packages for the server.** NVIDIA's server module imports `webdataset`, `pandas`,
  `wandb` and `multi-storage-client`, which normally arrive with the (much larger) training extra.

The first `make sim-server` also downloads the Wan2.2 video VAE (2.7 GB) that the policy uses to
encode and decode video.

## Run it

```bash
make sim-server                      # terminal 1; ready in about 10 s once the weights are cached
make sim                             # terminal 2; TASK=RubiksCubeAndBananaTask for another scene
```

An Isaac Sim window shows the scene, a second window shows what the policy sees (with
`/imagine on`, what it imagined next to it), and you type at the `>` prompt:

| Command | Effect |
|---|---|
| `put the banana in the bowl` | Act on this instruction until `--max-plans` (12, about 26 s), `/stop`, or a new instruction. Typing a new one interrupts the current run. |
| `/cfg 6` | CFG scale from the next plan on, without interrupting. `1` turns CFG off. |
| `/interval all` | Apply CFG at every denoising step. `/interval edge` restores NVIDIA's setting (only the first step), `/interval 900 1001` sets a custom window. |
| `/steps 8` | Denoising steps per plan (default 4). |
| `/horizon 16` | Execute only the first 16 of the 32 actions before re-planning: more reactive, twice the planning. |
| `/imagine on` | Also return the video the policy imagined; shown beside the real view and saved per plan. Costs 2 to 4 s per plan. |
| `/seed 3`, `/seed random` | Fix or free the sampling seed. |
| `/default`, `/default vague` | Run the task's own instruction (each task has a default and usually `vague` and `specific` variants). |
| `/plans 20` | Plans per instruction. |
| `/reset`, `/stop`, `/status`, `/help`, `/quit` | |

Each run is saved in `outputs/sim/<time>_<task>/`: `*_scene.mp4` from a fixed camera,
`*_views.mp4` with what the policy saw (and imagined), `*_imagined_planNN.mp4` per plan when
imagining, and one line per run in `log.jsonl` with the settings, plan latencies and outcome.

Batch mode runs every instruction × CFG scale × seed from a freshly reset scene, headless, and
stops each run early when the task's success condition is met:

```bash
make sim-batch PROMPTS='"Pick up the banana and place it in the bowl" "Put the fruit in the bowl"' \
               CFG="1 3 6" SEEDS="0 1 2"
make sim-grid                        # the newest batch as one labeled video
```

**Without a display that can run Isaac Sim** (a remote desktop, for example), batch mode and
`make sim-grid` are the whole workflow: everything renders offscreen, and `sim-grid` stitches
the batch into `grid_scene.mp4` with one row per instruction (and seed) and one column per CFG
scale. Each tile is labeled with its CFG scale, and the outcome appears at the moment the goal is
met or the run ends. `make sim-grid VIEWS=1` uses the policy's camera views instead, which with
`--imagine` show the real views next to the imagined ones. The interactive prompt also works
headless, saving each run's videos: run it without `--viz kit`, as
`OMNI_KIT_ACCEPT_EULA=Y third_party/RoboLab/.venv-60/bin/python scripts/sim_droid.py`.

Tasks are listed with pictures and instructions in
`third_party/RoboLab/robolab/tasks/README.md`. Success detection is the task's own: in
`BananaInBowlTask` it is "banana in the bowl and released", whatever you typed.

## What the CFG scale does here

At each denoising step the policy predicts a velocity twice, once with your instruction and once
with an empty one, and moves along

```
v = v_empty + g · (v_instruction − v_empty)
```

At `g = 1` that is just the instruction-conditioned prediction, and the server skips the empty
pass. Larger `g` exaggerates whatever the instruction changes. Because video and actions are
denoised together, CFG pushes both: the imagined future and the motion that goes with it.

NVIDIA serves the Edge policy with guidance 3 applied only for timesteps between 960 and 1001.
With 4 steps and the sampler's shift of 5, the steps sit near timesteps 1000, 938, 833 and 625, so
only the first, noisiest step gets CFG. That step fixes the coarse plan, where the instruction
matters most, and the remaining three refine it. `/interval all` applies CFG at every step.

Cost per plan, measured on the RTX 5090 shared with another job:

| Setting | Server time per plan |
|---|---|
| CFG off (`/cfg 1`) | 0.43 to 0.53 s |
| CFG 3, NVIDIA's window (default) | 0.50 to 0.60 s |
| CFG 3, every step (`/interval all`) | 0.75 s |
| CFG 6, every step | 0.82 s |
| CFG 3, 8 steps | 0.97 s |
| any of the above with `/imagine on` | 2 to 4.5 s more (decoding 33 frames) |

All but the last fit the 2.13 s a 32-action plan lasts, so on a real robot the policy would keep
up. The first plan after the server starts takes 7 to 16 s while it compiles.

First results on `BananaInBowlTask` ("Pick up the banana and place it in the bowl"), up to 12
plans (26 s), three sampling seeds per scale, same starting scene
(`make sim-batch CFG="1 3 6" SEEDS="0 1 2"`):

| CFG scale | Succeeded | Seconds to success (seeds 0, 1, 2) | Server time per plan |
|---|---|---|---|
| 1 (off) | 3/3 | 24.1, 9.9, 11.3 | 0.43 s |
| 3 (NVIDIA's setting) | 3/3 | 21.2, 9.7, 11.3 | 0.51 s |
| 6 | 3/3 | 12.0, 8.1, 12.5 | 0.51 s |

On this task the scale makes no difference you can separate from chance. All nine runs
succeeded, and the spread between seeds is larger than any difference between scales. A seed
also does not reproduce a run exactly, because GPU sampling and physics are not bit-deterministic:
an earlier CFG 1, seed 0 run was still holding the banana when its 26 s ran out. A banana and a
bowl on an empty table is close to the training distribution. Differences, if any, are more
likely on cluttered scenes, on `vague` instructions, or on instructions that ask for something
other than the obvious action. Those are the experiments this setup is for, with more seeds than
three.

A first look at that, in `RubiksCubeAndBananaTask` (a cube, a banana and a bowl), seed 0, one
run per cell, up to 16 plans (34 s):

| Instruction | CFG 1 | CFG 6 |
|---|---|---|
| "Put the rubiks cube in the bowl" | cube in the bowl, then picked up the banana too | cube in the bowl, banana left alone |
| "Put the banana in the bowl" | banana in the bowl, then went for the cube | banana in the bowl, cube left alone |
| "Put the cube and the banana in the bowl" | cube in; banana still in the gripper at 34 s | both in the bowl at 17.7 s |

At CFG 6 the robot did what was named and stopped; at CFG 1 it carried on with the other object,
as if drifting back to the scene's usual task. That is the effect CFG is meant to have, but one
run per cell is an anecdote, not a measurement. In `BananaInBowlTask`, "Push the bowl to the left"
and "Pick up the bowl" sent the arm to the bowl rather than the banana at every scale, but it
shoved or grabbed the bowl rather than cleanly pushing or lifting it.

## Memory on this machine

| | RAM | VRAM |
|---|---|---|
| Policy server | 4.3 GB (5.5 GB while loading) | 9 GB; 13 GB once `/imagine` has decoded video (PyTorch keeps the buffers) |
| Simulator, one scene with three 1280x720 cameras | 9.3 GB | 8.5 GB |

Together that is 18 to 22 GB of the card's 32 GB. The server caps itself at 14 GB of VRAM
(`COSMOS_VRAM_CAP_GIB`), so on a shared GPU it fails with its own out-of-memory error rather
than crowding out another job.

Both `make` targets run in memory-capped scopes (12 GB server, 14 GB simulator) so running out of
RAM kills that job and not the desktop session. The first launches are the exception to the
9.3 GB figure: Isaac Sim compiles and caches shaders, which here took 19 CPU-minutes and was
killed at 10 GB, then at 13 GB on the next attempt. After that, every run fit under 12 GB. If a
first launch is killed, stop `sim-server` while it runs, or simply run it again: the cache
persists. A scene with new materials compiles its shaders on first use.

If a launch prints nothing for more than a minute, kill it and start it again. That happened once
here, when a batch started within a second of the previous simulator exiting: the new process sat
in Isaac Sim's startup with no output and no CPU use, and the next launch was fine. Leave a few
seconds between back-to-back runs in your own scripts.

## Caveats

- **Isaac Lab 3.0 is a beta**, and RoboLab's published results are on Isaac Sim 5.0. Contact
  physics differs between versions, so success rates here are not comparable to the RoboLab
  leaderboard. NVIDIA lists a known Robotiq 2F-85 issue for the 3.0 beta; grasps worked in the
  runs above.
- **The policy was trained on real DROID data**, so the simulator is already out of distribution
  for it. Instructions far from DROID's style of short tabletop commands may give aimless motion.
- **The GUI path is the same code** as the headless runs above, plus `cv2.imshow` and Isaac Sim's
  own window, but it was not exercised while building this, and Isaac Sim's window does not run
  over this machine's remote desktop. Headless runs plus `make sim-grid` cover the same ground.
- `make policy` (`scripts/policy.py`) is a different path: it calls the Diffusers pipeline, which
  treats this checkpoint as the base model's 10-D end-effector action space. The server here uses
  NVIDIA's reference preprocessing (8-D joint positions, robot state, JSON prompt), which is how
  the checkpoint was trained.

## Sources

- [RoboLab](https://github.com/NVlabs/RoboLab) and its [Isaac Sim 6 notes](https://github.com/NVlabs/RoboLab/blob/main/docs/isaac_sim_6.md)
- [cosmos-framework DROID policy server](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/action_policy_droid_server.md), the source of `sim_policy_server.py`'s reference service
- [Cosmos3-Edge-Policy-DROID model card](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID)
