# How Cosmos 3 works, and how it is used for real-time Physical AI

This is a working explanation, written from NVIDIA's model cards, the `NVIDIA/cosmos` and
`NVIDIA/cosmos-framework` repositories, and the Cosmos 3 technical report. Where a point is
general robotics practice rather than something NVIDIA states, it says so.

## The one-paragraph version

A robot, a car or a camera system needs three things from a model: to **understand** the
scene, to **predict** what happens next, and to **act**. Earlier Cosmos releases shipped
these as separate models (Reason, Predict, Transfer, Policy). Cosmos 3 puts them in one
network. The same weights answer a question about a video, generate the next five seconds
of that video, tell you what a robot arm will see if it moves a certain way, and output the
motor commands to pick something up. Which of those it does depends only on what you feed
in and what you ask it to fill in.

## Two towers in one model

Cosmos 3 is a Mixture-of-Transformers: two transformer "towers" that run side by side and
share attention layers and a common 3-D position encoding for space and time.

| Tower | How it generates | Takes | Produces | Role |
|---|---|---|---|---|
| **Reasoner** | autoregressive, one token at a time, like a chat model | text, images, video | text | perception, question answering, planning, grounding (points, boxes) |
| **Generator** | diffusion: starts from noise and denoises over 4 to 50 steps | text, images, video, audio, actions | images, video, audio, actions | prediction, simulation, control |

Text is discrete, so next-token prediction suits it. Pixels, sound and joint commands are
continuous, so iterative denoising suits them. Cosmos 3 keeps each mechanism for the
modality it fits instead of forcing everything through one.

Information flows one way, from Reasoner to Generator. The Reasoner reads the prompt and
any visual context first; the Generator then denoises while attending to what the Reasoner
computed. That has two practical consequences:

- **The Reasoner can run alone.** Loading only that tower is the cheapest way to use the
  model, and it is what vLLM and Transformers do.
- **The Generator always runs both.** That is why generation needs far more memory than
  reasoning for the same checkpoint.

Around the towers sit three smaller parts, visible as folders in each checkpoint: a
`vision_encoder` that turns frames into tokens for the Reasoner, a `vae` that compresses
video into a compact latent grid the Generator denoises (and decodes back to pixels), and
on Nano and Super a `sound_tokenizer`. Edge has no audio.

## Actions are just another modality

An action is a `(T, D)` array: `T` timesteps, `D` numbers per step. `D` depends on the
body. A single robot arm is 10-D (3 for position change, 6 for rotation, 1 for the
gripper), a two-arm setup is 20-D, a humanoid 29-D, a car or a free camera 9-D.

The Generator denoises that array the same way it denoises video latents. So the three
classic robotics problems become one model with different parts held fixed:

| Task | Given (clean) | Generated (from noise) | Question it answers |
|---|---|---|---|
| Video generation | text, optional first frame | video | What could this scene look like? |
| **Forward dynamics** | current frame + actions | video | What will I see if I do this? |
| **Inverse dynamics** | video | actions | What was done to produce this motion? |
| **Policy** | current frame + instruction | actions **and** video | What should I do, and what do I expect to see? |

The policy row is why NVIDIA calls these *world action models*. The policy does not map
pixels straight to motor commands. It imagines the next second of video and the actions
that produce it together, so the action plan is tied to a predicted outcome.

## Three sizes

| | Edge | Nano | Super |
|---|---|---|---|
| Parameters | 4B | 16B | 64B |
| Checkpoint on disk | 9 GB | 35 GB (21 GB as FP8) | much larger |
| Video | 256p, 480p | up to 720p, with sound | up to 720p, with sound |
| Meant for | on the robot: Jetson, one RTX GPU | a workstation or one data-centre GPU | data-centre clusters |
| Typical job | real-time policy and perception | post-training base, simulation, data generation | highest quality data; teacher for distillation |

The sizes are designed to work as a pipeline, not as alternatives: Super generates data
and teaches, Nano is fine-tuned and used as a simulator, Edge is what gets deployed.

## How this turns into Physical AI workflows

**1. Making training data (offline, quality over speed).**
Robot data is expensive: every example is a person teleoperating a real machine. World
models multiply it. Text- or image-to-video invents new scenes. *Transfer* takes a cheap
simulator render (depth, edges, segmentation) and repaints it as photoreal video with the
geometry preserved, which closes much of the sim-to-real gap. Inverse dynamics then labels
any video, including generated or internet video, with the actions that explain it, turning
unlabelled footage into policy training data.

**2. Testing a policy without a robot (interactive).**
Forward dynamics is a learned simulator. Feed it a policy's actions and it returns the
resulting frames; feed those back to the policy and you have a closed loop with no
hardware and no hand-built 3-D scene. NVIDIA's `Cosmos3-Nano-Sim-*` checkpoints are
fine-tuned for this: multi-camera driving with LiDAR, two-arm manipulation, and
sim-to-real transfer running autoregressively.

**3. Acting (real time, on the machine).**
A post-trained policy such as `Cosmos3-Edge-Policy-DROID` runs on the robot's own
computer and turns camera frames plus an instruction into motor commands.

**4. Understanding (real time or batch).**
The Reasoner watches video and answers in text: is the aisle blocked, did the grasp
succeed, what is the pedestrian about to do. It is used as a live monitor on fixed
cameras, as the high-level planner that breaks "tidy the table" into steps for a policy,
and as an automatic critic that filters generated training video for physical
plausibility.

These chain into a loop: generate and label data (1), train a policy, evaluate it in the
world model (2), deploy it (3), monitor it (4), and send failures back to (1).

## What "real time" means here

A diffusion model that takes a second or more per call sounds too slow to control a robot.
Four ideas make it work.

**Action chunking.** The policy does not produce one action per call. It produces a chunk,
16 or 32 future actions. At a 15 Hz control rate a 16-action chunk covers 1.07 s of
motion. The robot executes the chunk while the next one is being computed. The rule for
real time is therefore:

> time to compute a chunk  ≤  chunk length ÷ control rate

NVIDIA reports this as a *real-time factor* (budget ÷ latency); 1.0 or more keeps up.
For Edge-Policy-DROID with 32-action chunks they measure 1.5 s per chunk on a Jetson AGX
Thor T5000, which is 4.2x real time at 5 Hz and 1.4x at 15 Hz. `scripts/policy.py`
measures the same thing on your GPU.

**Fewer denoising steps.** Cost is linear in steps. Video for people to look at uses 20 to
50. NVIDIA's real-time policy numbers use 4. Distillation makes that a property of the
weights: the `-4Step` Super checkpoints and `Sim-Auto` are trained to match a many-step
teacher in four steps, which NVIDIA reports as 17 to 25x faster.

**Do less work per step.** Low resolution (policies run at 256p to 480p), the 4B model
instead of the 16B one, FP8 or NVFP4 weights, compiled runtimes (TensorRT, vLLM), and
caching: the Reasoner's pass over the prompt is computed once and reused for every
denoising step instead of being repeated each time.

**Split by timescale.** This is common robotics practice rather than a Cosmos feature, and
it is how the pieces usually fit together:

| Layer | Rate | What runs |
|---|---|---|
| Planner | about 1 Hz or on events | Reasoner: "what is the next subtask?" |
| Policy | 1 to 15 chunks per second | Generator: a chunk of actions |
| Controller | 100s of Hz | classical control tracking the chunk; no neural network |

Not everything has to be real time. Data generation and distillation are offline, where
Super's quality matters more than its speed. Only the deployed policy and any live monitor
have a deadline, and those are exactly what Edge is sized for.

## Limits worth keeping in mind

These models approximate physics from video; they do not compute it. NVIDIA's cards list
object morphing, implausible contacts and drift over long rollouts, and state that outputs
are not physically accurate simulation or safety-certified decisions. A generated rollout
is evidence about what a policy might do, not proof.

## Sources

- [NVIDIA/cosmos](https://github.com/NVIDIA/cosmos) README and `docs/reference/models.md`
- [Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge), [Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) and [Cosmos3-Edge-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID) model cards
- [Cosmos 3 technical report](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf)
- [NVIDIA developer blog on Cosmos 3](https://developer.nvidia.com/blog/develop-physical-ai-reasoning-world-and-action-models-with-nvidia-cosmos-3/)
