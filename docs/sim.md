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
| `/negative put the cube in the bowl` | CFG's second pass uses this instead of an empty prompt, so the plan is pushed away from it (see below). `/negative off` restores the empty prompt. |
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

Batch mode runs every instruction × CFG scale × negative prompt × seed from a freshly reset
scene, headless, and stops each run early when the task's success condition is met:

```bash
make sim-batch PROMPTS='"Pick up the banana and place it in the bowl" "Put the fruit in the bowl"' \
               CFG="1 3 6" SEEDS="0 1 2"
make sim-grid                        # the newest batch as one labeled video
```

**Without a display that can run Isaac Sim** (a remote desktop, for example), batch mode and
`make sim-grid` are the whole workflow: everything renders offscreen, and `sim-grid` stitches
the batch into `grid_scene.mp4` with one row per instruction (and seed) and one column per CFG
scale (and negative prompt). Each tile is labeled with its settings, and the outcome appears at
the moment the goal is met or the run ends. In scenes with several objects to place, each
object's placement appears when it happens instead: green if the instruction named it, gold if
not. `make sim-grid VIEWS=1` uses the policy's camera views instead, which with
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

### Negative prompts

The empty prompt in the second pass can be replaced by a negative prompt (`/negative TEXT`,
`--negative`, or `NEGATIVE=` for `make sim-batch`):

```
v = v_negative + g · (v_instruction − v_negative)
```

The push is then along "what the instruction asks, minus what the negative asks" instead of
"what the instruction asks, minus nothing in particular". In a scene with a cube and a banana,
"Put the banana in the bowl" with the negative "Put the rubiks cube in the bowl" should push
away from the cube specifically. The policy was trained for the empty prompt, not for negatives:
the published DROID recipe (`action_policy_droid_nano`; Edge's is not published) replaces the
caption with an empty string 10% of the time. So this tests whether two instruction-conditioned
predictions differ in a direction CFG can use. The server wraps the negative in the same JSON prompt as the
instruction, so the two passes differ only in the task text. A check that this holds: at CFG 6,
a negative identical to the instruction gives exactly the CFG 1 actions, since
`v_i + g · (v_i − v_i) = v_i`. At `g = 1` the negative is never used, and a batch skips those
combinations. A negative adds 40 to 80 ms per plan over the empty prompt (a longer prompt to
encode).

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

### Negative prompt ablation

Same scene, same two single-object instructions, each with the *other* object's instruction as
the negative prompt, against the usual empty prompt. Four seeds per cell, 16 plans (34 s) per
run, NVIDIA's CFG window (first denoising step only), 40 runs in all:

```bash
make sim-batch TASK=RubiksCubeAndBananaTask CFG="1 3 6" SEEDS="0 1 2 3" PLANS=16 \
  PROMPTS='"Put the rubiks cube in the bowl" "Put the banana in the bowl"' \
  NEGATIVE='none "Put the rubiks cube in the bowl" "Put the banana in the bowl"'
```

When the requested object was in the bowl, in seconds, seeds 0 to 3 (– never):

| Instruction | CFG 1 | CFG 3 vs. empty | CFG 3 vs. other object | CFG 6 vs. empty | CFG 6 vs. other object |
|---|---|---|---|---|---|
| "Put the rubiks cube in the bowl" | 6.5, 16.6, 9.5, 9.9 | 8.6, 7.5, 8.9, 7.7 | 8.5, –, 11.5, 9.7 | 7.5, 8.5, 15.7, 8.1 | 23.3, 10.6, 11.1, 10.2 |
| "Put the banana in the bowl" | –, –, 26.3, 11.8 | 12.4, 14.1, 12.6, 9.7 | –, 24.7, 15.4, 12.6 | 11.2, 9.3, 10.8, 9.3 | 18.5, 22.2, 24.1, 11.3 |

- **The negative prompt made the robot slower, not more selective.** With the empty prompt the
  requested object went in 16 times out of 16 (median 9.3 s); with the other object as the
  negative, 14 out of 16 (median 12.0 s among those placed). The negative run was the slower of the
  two in 14 of the 16 same-seed pairs. (Seeds pair runs only loosely, because physics is not
  bit-deterministic.) Both misses were hesitation, not the wrong object: one run held the banana
  over the bowl for 30 s without letting go, and the other hovered over the cube without closing
  the gripper.
- **There was little wrong-object behaviour left for it to remove.** At CFG 3 and 6 the robot
  picked up the object it was not asked for in 1 run of 16 with the empty prompt (cube
  instruction, CFG 3, seed 0: lifted the banana after placing the cube) and 1 of 16 with the
  negative (banana instruction, CFG 6, seed 0: lifted the cube at 32 s, after taking the banana
  back out). That drift is a CFG 1 effect: in both measured CFG 1 runs of the cube instruction
  (seeds 2 and 3), the robot placed the cube, then lifted the banana about 13 s later. Seeds 2 and 3
  were measured (an object 5 cm above its start counts as picked up). Seeds 0 and 1 were judged
  from the video, because the lift tracking was added after they ran.
- CFG 1 placed the banana only 2 times out of 4; every CFG 3 and 6 setting placed it at least 3
  times out of 4.
- A placed banana was taken back out in 4 runs (two with each kind of second pass): the policy
  has no notion of being done, and DROID's training episodes stop when the task is complete.
- Cost: 0.50 s per plan against the empty prompt, 0.54 s against a negative, 0.42 s at CFG 1.

A likely reason, not tested here: the update is `v_instruction + (g − 1) · (v_instruction −
v_second_pass)`. Against the empty prompt, the difference holds everything the instruction asks
for: reach, grasp, carry to the bowl, and which object. Against "put the *other* object in the
bowl", the shared part (pick something up and put it in the bowl) cancels, so CFG amplifies only
"this object, not that one" and no longer pushes the plan toward getting on with the task. With
NVIDIA's window, that push lands only on the first denoising step, where the coarse plan is set.

#### Negative prompt = just an object's name

Two more variants, each against a bare noun, so "put … in the bowl" is not in the negative and
stays in what CFG amplifies:

- **The other object's name:** "Put the rubiks cube in the bowl" against "The banana", and the
  banana instruction against "The rubiks cube". This pushes away from the wrong object without
  cancelling the pick-and-place.
- **The requested object's own name:** "Put the rubiks cube in the bowl" against "The rubiks
  cube". This cancels the object instead and leaves only "put it in the bowl" to amplify. It is a
  contrast, not something you would use.

Same scene, seeds and settings, 16 runs each. Seconds until the requested object was in the bowl,
seeds 0 to 3:

| Instruction | CFG 3 vs. the other object's name | CFG 6 vs. the other object's name | CFG 3 vs. its own name | CFG 6 vs. its own name |
|---|---|---|---|---|
| "Put the rubiks cube in the bowl" | 8.5, 22.1, 11.2, 9.6 | 9.6, 10.2, –, 9.8 | 8.4, 8.1, 8.8, 9.9 | 6.9, 24.3, 8.6, 15.1 |
| "Put the banana in the bowl" | 14.1, 9.7, 9.3, 11.0 | 14.8, 24.4, 11.7, 19.1 | 9.7, 15.5, –, – | 10.4, –, –, 15.5 |

Every second pass side by side (CFG 3 and 6 pooled, 16 runs each; CFG 1 has no second pass):

| Second pass | Requested object placed | Median seconds to place it | Picked up the object not asked for |
|---|---|---|---|
| none: instruction only (CFG 1) | 6/8 | 10.8 | 2 of the 4 measured runs |
| empty prompt | 16/16 | 9.3 | 1/16 |
| the other object's name | 15/16 | 11.0 | 0/16; never moved it more than 1.4 cm |
| the other object's instruction | 14/16 | 12.0 | 1/16 |
| the requested object's own name | 12/16 | 9.8 | 2/16, plus one that shoved it 6 cm |

- **The other object's name did what the other object's instruction was meant to.** The robot
  went for the requested object every time and never disturbed the other one (all 16 runs
  measured). It was also more reliable and quicker than with the full instruction as the negative:
  15 of 16 placed against 14, median 11.0 s against 12.0 s. The one miss carried the cube to the
  bowl and let go just past its far edge.
- **It still did not beat the empty prompt.** The empty prompt placed 16 of 16, sooner (median
  9.3 s; the name-negative run was slower in 13 of the 16 same-seed pairs). At CFG 3 and 6 it had
  also already almost stopped the robot going for the other object (1 of 16). In this scene there
  was little drift left to remove, so pushing away from the other object bought nothing
  measurable and cost a couple of seconds.
- **Cancelling the object instead brought back instruction-only behaviour.** With its own name as
  the negative, the cube instruction at CFG 3 went on to pick up the banana after placing the cube
  in 2 of 4 runs and shoved it in a third, as at CFG 1. The banana went in only 4 of 8 times
  (CFG 1: 2 of 4), fumbled every time: dropped on the bowl's rim, set down in front of the bowl,
  pushed or dragged without lifting. With the cube's name as the negative instead, the banana went
  in 8 of 8. That fits the guess that the object part carries *how* to handle the object as well
  as which one it is.

Read together, the empty prompt's difference carries both what to do and which object to do it
with, and amplifying both is what made CFG 3 and 6 better than instruction only here:

- Cancelling the action part (the other object's instruction) cost speed and occasionally the
  task.
- Cancelling the object part (its own name) brought back instruction-only behaviour.
- The other object's name kept both and removed the remaining drift, at a small cost in speed.

A negative prompt should pay off where the empty prompt still drifts at CFG 3: more distractors,
or vaguer instructions. With four seeds per cell, all of this is suggestive rather than settled.

To put chosen second passes side by side from several batches:
`make sim-grid RUN="<folder> <folder> ..." NEGATIVE='none "The banana" "The rubiks cube"'`.

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
