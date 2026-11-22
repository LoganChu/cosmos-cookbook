"""Drive Cosmos3-Edge-Policy-DROID in simulation with your own instructions and CFG scale.

The robot is the DROID setup the policy was trained on (Franka arm, Robotiq gripper, a wrist camera
and two over-the-shoulder cameras) in one of RoboLab's tabletop scenes, simulated by Isaac Sim.
For each plan, the three camera views, the joint angles and your instruction go to
`scripts/sim_policy_server.py`, which returns 32 joint-angle targets (2.1 s of motion at 15 Hz).
The arm executes them (or the first --horizon of them), then the policy plans again. Simulated time
pauses while the policy thinks, so slow settings change speed, not behaviour.

Start the policy server first, in its own terminal:

    make sim-server

Interactive: an Isaac Sim window shows the scene (`--viz kit`; without it Isaac Lab 3 runs headless
and you follow along in the saved videos), a second window shows what the policy sees, and you
type here.

    make sim                                   # BananaInBowlTask
    make sim TASK=RubiksCubeAndBananaTask

    > put the banana in the bowl               plan and act until /stop, a new instruction, or --max-plans
    > /cfg 5                                   CFG scale from the next plan on (1 = off)
    > /negative put the cube in the bowl       CFG pushes away from this instead of from an empty prompt
    > /interval all                            CFG at every denoising step (/interval edge: NVIDIA's 960 1001)
    > /steps 8                                 denoising steps per plan
    > /horizon 16                              actions executed before re-planning (1 to 32)
    > /imagine on                              also show and save the video the policy imagined
    > /reset, /default, /status, /help, /quit

Batch, headless: every instruction x CFG scale x seed, each from a freshly reset scene, stopping
early when the task's success condition is met.

    make sim-batch PROMPTS='"put the banana in the bowl" "push the bowl to the left"' CFG="1 3 6" SEEDS="0 1 2"
    make sim-batch TASK=RubiksCubeAndBananaTask CFG="1 3 6" \
        PROMPTS='"Put the banana in the bowl" "Put the rubiks cube in the bowl"' \
        NEGATIVE='none "Put the banana in the bowl" "Put the rubiks cube in the bowl"'

With several negative prompts, a batch skips the combinations where the negative cannot change
anything: CFG 1 (no second pass) and a negative identical to the instruction.

Videos and a log land in outputs/sim/<time>_<task>/.
"""

from __future__ import annotations

import argparse
import json
import queue
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROBOLAB = ROOT / "third_party" / "RoboLab"
sys.path.insert(0, str(ROBOLAB))

import cv2  # noqa: E402  RoboLab: cv2 must be imported before isaaclab
from isaaclab.app import AppLauncher  # noqa: E402

CHUNK = 32          # actions per plan from the Edge DROID policy
CONTROL_HZ = 15     # DROID control rate; RoboLab steps physics at 120 Hz with decimation 8
LIFTED_M = 0.05     # an object this far above where it started counts as picked up
EDGE_INTERVAL = (960.0, 1001.0)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="BananaInBowlTask",
                    help="RoboLab task (scene); see third_party/RoboLab/robolab/tasks/README.md")
parser.add_argument("--server", default="ws://127.0.0.1:8000", help="policy server URI")
parser.add_argument("--cfg", type=float, nargs="+", default=[3.0],
                    help="CFG scale; several values in batch mode run each one")
parser.add_argument("--interval", nargs="+", default=["edge"],
                    help="CFG timestep window: 'edge' (960 1001), 'all', or two numbers")
parser.add_argument("--steps", type=int, default=4, help="denoising steps per plan")
parser.add_argument("--horizon", type=int, default=CHUNK, help="actions executed before re-planning (1-32)")
parser.add_argument("--max-plans", type=int, default=12, help="plans per instruction before stopping (12 = 26 s)")
parser.add_argument("--negative", nargs="+", action="extend", default=None,
                    help="negative prompt: CFG's second pass uses it instead of an empty caption; several in "
                         "batch mode run each ('none' = the usual empty caption)")
parser.add_argument("--seed", type=int, nargs="+", default=None,
                    help="fixed sampling seed; several values in batch mode run each one (batch default: 0)")
parser.add_argument("--imagine", action="store_true", help="return and save the video the policy imagined")
parser.add_argument("--prompt", nargs="+", action="extend", default=None,
                    help="batch mode: run these instructions, record, report, exit")
parser.add_argument("--keep-going", action="store_true",
                    help="batch mode: keep acting after the task's success condition is met")
parser.add_argument("--no-record", action="store_true", help="do not save videos")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
# Progress lines should appear as they happen even when output is piped or tee'd to a log.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Everything below needs the running app.
import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from policies.cosmos3.client import Cosmos3Client  # noqa: E402
from robolab.core.environments.config import parse_env_cfg  # noqa: E402
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.observations.observation_utils import unpack_viewport_cams  # noqa: E402
from robolab.core.task.instruction import normalize_instruction_fields  # noqa: E402
from robolab.core.utils.video_utils import VideoWriter  # noqa: E402
from robolab.core.world.world_state import get_world  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402
from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD  # noqa: E402


def parse_interval(words: list[str]) -> tuple[float, float] | None:
    if words in (["edge"], ["default"]):
        return EDGE_INTERVAL
    if words in (["all"], ["none"], ["off"]):
        return None
    if len(words) == 2:
        low, high = float(words[0]), float(words[1])
        if 0 <= low <= high:
            return (low, high)
    raise ValueError("interval takes 'edge', 'all', or two numbers low <= high (0-1000 timestep scale)")


@dataclass
class Settings:
    guidance: float
    interval: tuple[float, float] | None
    steps: int
    horizon: int
    max_plans: int
    imagine: bool
    seed: int | None
    negative: str = ""  # CFG's second pass: "" is the empty caption the model was trained with

    def describe(self) -> str:
        window = "every step" if self.interval is None else f"timesteps {self.interval[0]:g}-{self.interval[1]:g}"
        seed = "random" if self.seed is None else self.seed
        against = f"negative '{self.negative}'" if self.negative else "no negative prompt"
        return (f"cfg {self.guidance:g} ({window}, {against}), {self.steps} steps, execute {self.horizon}/{CHUNK} "
                f"actions per plan, up to {self.max_plans} plans, seed {seed}, "
                f"imagine {'on' if self.imagine else 'off'}")


class SteerableClient(Cosmos3Client):
    """RoboLab's Cosmos 3 client, plus per-request sampling settings for sim_policy_server.py."""

    def __init__(self, settings: Settings, uri: str) -> None:
        self.settings = settings
        self.plans = 0
        self.reply: dict = {}
        self.round_trip_ms = 0.0
        super().__init__(remote_uri=uri)

    def _needs_refresh(self, env_id: int) -> bool:
        self.open_loop_horizon = self.settings.horizon
        return super()._needs_refresh(env_id)

    def _pack_request(self, extracted_obs: dict, instruction) -> dict:
        request = super()._pack_request(extracted_obs, instruction)
        s = self.settings
        request.update(
            guidance=s.guidance,
            guidance_interval=list(s.interval) if s.interval is not None else None,
            num_steps=s.steps,
            decode_video=s.imagine,
        )
        if s.seed is not None:
            request["seed"] = s.seed
        if s.negative:
            request["negative"] = s.negative
        return request

    def _query_server(self, request: dict) -> object:
        start = time.monotonic()
        reply = super()._query_server(request)
        self.round_trip_ms = (time.monotonic() - start) * 1000.0
        return reply

    def _unpack_response(self, response: object) -> np.ndarray:
        self.reply = response if isinstance(response, dict) else {}
        self.plans += 1
        return super()._unpack_response(response)

    def policy_view(self, obs) -> np.ndarray:
        """The single image the policy sees: wrist camera on top, the two shoulder cameras below."""
        extracted = self._extract_observation(obs, env_id=0)
        wrist = extracted["wrist_image"]
        size = (self._image_h // 2, self._image_w // 2)
        halves = []
        for key in ("left_image", "right_image"):
            image = torch.from_numpy(extracted[key]).permute(2, 0, 1).unsqueeze(0).float()
            image = F.interpolate(image, size=size, mode="bilinear")
            halves.append(image.squeeze(0).permute(1, 2, 0).numpy().astype(wrist.dtype))
        return np.concatenate((wrist, np.concatenate(halves, axis=1)))


def slug(text: str, limit: int = 48) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:limit] or "instruction"


def make_success_check(terminations):
    """Return the task's success test (any non-timeout termination) as a callable on the env.

    Also returns one test per object when a success term covers several (RubiksCubeAndBananaTask:
    "cube and banana in the bowl"), so a run can report which of them ended up there.
    """
    terms, per_object = [], {}
    for name in list(vars(terminations)):
        term = getattr(terminations, name)
        if name.startswith("_") or term is None or not hasattr(term, "func"):
            continue
        if not getattr(term, "time_out", False):
            terms.append((name, term))
            objects = term.params.get("object") if isinstance(term.params, dict) else None
            if isinstance(objects, (list, tuple)) and len(objects) > 1:
                for obj in objects:
                    params = {**term.params, "object": [obj], "logical": "all"}
                    where = term.params.get("container", "goal")
                    per_object[obj] = (where, lambda env, f=term.func, p=params: bool(f(env, **p)[0]))
        # Interactive control: nothing ends or freezes the episode on its own.
        setattr(terminations, name, None)

    def check(env) -> bool:
        return any(bool(term.func(env, **term.params)[0]) for _name, term in terms)

    return check, [name for name, _ in terms], per_object


class Session:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        auto_register_droid_envs(task=[args.task], cameras=WRIST_LEFT_RIGHT_HEAD)
        envs = get_envs(task=[args.task])
        if not envs:
            raise SystemExit(f"Unknown task {args.task!r}; see third_party/RoboLab/robolab/tasks/README.md")
        env_cfg = parse_env_cfg(envs[0], device=args.device, num_envs=1, seed=0)
        self.success_check, success_terms, self.object_checks = make_success_check(env_cfg.terminations)
        env_cfg.recorders = None  # RoboLab's HDF5 episode recording is for benchmark runs
        self.env, self.env_cfg = create_env(env_cfg, device=args.device, num_envs=1, policy="cosmos3")
        # Isaac Lab 2 has sim.has_gui(); Lab 3 makes it a property and drops args.headless.
        has_gui = self.env.sim.has_gui
        self.headless = not (has_gui() if callable(has_gui) else has_gui)
        self.instructions = normalize_instruction_fields(self.env_cfg._instruction_variants)
        self.out_dir = ROOT / "outputs" / "sim" / f"{time.strftime('%Y%m%d-%H%M%S')}_{args.task}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "log.jsonl"
        self.run_count = 0
        self.check_success = bool(success_terms)
        self.track_objects = True
        self.client = SteerableClient(settings, args.server)
        self.timeline = omni.timeline.get_timeline_interface()
        self.reset()
        print(f"\n[sim] Task {args.task}. Its own instructions, for reference:")
        for name, text in self.instructions.items():
            print(f"[sim]   {name:<9} {text}")
        if not success_terms:
            print("[sim] This task defines no success condition; runs end on /stop or --max-plans.")
        print(f"[sim] Settings: {settings.describe()}")
        print(f"[sim] Output: {self.out_dir}\n")

    def reset(self) -> None:
        self.env.reset_eval_state()
        self.obs, _ = self.env.reset()
        self.obs, _ = self.env.reset()  # as RoboLab's runner does: the first reset leaves stale sensors
        self.client.reset()

    def render_idle(self) -> None:
        self.env.sim.render()

    def object_positions(self) -> dict[str, torch.Tensor]:
        """Where each of the goal's objects is (scene frame, metres); empty if they cannot be read."""
        if not self.object_checks or not self.track_objects:
            return {}
        try:
            world = get_world(self.env)
            return {obj: world.get_pose(obj, env_id=0)[0].float().cpu() for obj in self.object_checks}
        except Exception as exc:  # an object that is not a rigid body, say
            print(f"[sim]   object tracking disabled: {exc}")
            self.track_objects = False
            return {}

    def run(self, instruction: str, label: str = "", poll=None) -> dict:
        """Act on one instruction until max plans, success (batch), or a command from `poll`."""
        s = self.settings
        self.run_count += 1
        self.client.reset()
        self.client.plans = 0
        negative = f"_neg_{slug(s.negative, 24)}" if s.negative else ""
        name = f"{self.run_count:02d}_{slug(instruction)}_cfg{s.guidance:g}{negative}{label}"
        record = not args.no_record
        scene_video = VideoWriter(str(self.out_dir / f"{name}_scene.mp4"), CONTROL_HZ) if record else None
        views_video = VideoWriter(str(self.out_dir / f"{name}_views.mp4"), CONTROL_HZ) if record else None
        print(f"[sim] Run {name}: '{instruction}'  ({s.describe()})")

        latencies, success_step, step, stop_reason, imagined, pending = [], None, 0, "max plans", None, None
        placed: dict[str, int | None] = {obj: None for obj in self.object_checks}
        # Did the robot pick up or shove an object, whether or not it ended up in the goal?
        start_pos = self.object_positions()
        lifted_at: dict[str, int | None] = {obj: None for obj in start_pos}
        max_moved = {obj: 0.0 for obj in start_pos}
        max_lift = {obj: 0.0 for obj in start_pos}
        budget_ms = s.horizon / CONTROL_HZ * 1000.0
        try:
            while simulation_app.is_running():
                while not self.timeline.is_playing():
                    simulation_app.update()
                if poll is not None and (pending := poll()) is not None:
                    stop_reason = "interrupted"
                    break
                plans_before = self.client.plans
                if plans_before >= s.max_plans and self.client._needs_refresh(0):
                    break
                result = self.client.infer(self.obs, instruction, env_id=0)
                if self.client.plans != plans_before:
                    reply = self.client.reply
                    server_ms = float(reply.get("server_ms", self.client.round_trip_ms))
                    latencies.append(server_ms)
                    verdict = "keeps up" if server_ms <= budget_ms else f"is {server_ms / budget_ms:.1f}x too slow"
                    if server_ms > 3 * budget_ms and not latencies[:-1]:
                        verdict += " (the server's first plan includes a one-time compile)"
                    used = reply.get("settings", {}).get("guidance", s.guidance)
                    print(f"[sim]   plan {self.client.plans} (cfg {used:g}): {server_ms:.0f} ms on the server, "
                          f"{self.client.round_trip_ms:.0f} ms round trip; budget for {s.horizon} actions "
                          f"is {budget_ms:.0f} ms, so it {verdict} in real time")
                    imagined = reply.get("video") if s.imagine else None
                    if imagined is not None and record:
                        clip = VideoWriter(str(self.out_dir / f"{name}_imagined_plan{self.client.plans:02d}.mp4"),
                                           CONTROL_HZ)
                        for frame in np.asarray(imagined):
                            clip.write(frame)
                        clip.release()
                action = torch.as_tensor(result["action"], device=self.env.device, dtype=torch.float32)
                self.obs, *_ = self.env.step(action.unsqueeze(0))
                step += 1
                for obj, pos in self.object_positions().items():
                    offset = pos - start_pos[obj]
                    max_moved[obj] = max(max_moved[obj], float(torch.linalg.norm(offset)))
                    max_lift[obj] = max(max_lift[obj], float(offset[2]))
                    if lifted_at[obj] is None and offset[2] > LIFTED_M:
                        lifted_at[obj] = step
                        print(f"[sim]   {obj} picked up after {step / CONTROL_HZ:.1f} s")

                if self.check_success and success_step is None:
                    try:
                        if self.success_check(self.env):
                            success_step = step
                            print(f"[sim]   task success condition met after {step / CONTROL_HZ:.1f} s "
                                  f"({self.client.plans} plans)")
                    except Exception as exc:  # a task predicate that cannot run outside the benchmark loop
                        print(f"[sim]   success check disabled: {exc}")
                        self.check_success = False
                if self.check_success:
                    for obj, (where, test) in self.object_checks.items():
                        if placed[obj] is None and test(self.env):
                            placed[obj] = step
                            print(f"[sim]   {obj} in the {where} after {step / CONTROL_HZ:.1f} s")
                if success_step is not None and args.prompt and not args.keep_going:
                    stop_reason = "success"
                    break

                if record or not self.headless:
                    view = self.client.policy_view(self.obs)
                    if imagined is not None:
                        frames = np.asarray(imagined)
                        index = min(self.client._counters.get(0, 0), len(frames) - 1)
                        dream = cv2.resize(frames[index], (view.shape[1], view.shape[0]))
                        view = np.concatenate((view, dream), axis=1)
                    if not self.headless:
                        # One window; with /imagine on, the imagined frame is on the right.
                        cv2.imshow("policy view", cv2.cvtColor(view, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                    if record:
                        views_video.write(view)
                        scene_video.write(unpack_viewport_cams(self.obs, env_id=0).get("combined_image"))
        finally:
            for video in (scene_video, views_video):
                if video is not None:
                    video.release()

        at_end = {}
        if self.check_success and self.object_checks:
            at_end = {obj: test(self.env) for obj, (_where, test) in self.object_checks.items()}
        summary = {
            "run": name,
            "task": args.task,
            "instruction": instruction,
            # The success check is always the task's goal, whatever was typed; sim_grid.py uses this
            # to tell "the task's own instruction failed" from "a different instruction ran".
            "task_instructions": list(self.instructions.values()),
            "settings": asdict(s),
            "plans": self.client.plans,
            "sim_seconds": round(step / CONTROL_HZ, 2),
            "success": success_step is not None if self.check_success else None,
            "success_after_s": round(success_step / CONTROL_HZ, 2) if success_step else None,
            # Per object, for tasks whose goal covers several: when it first met the goal's test
            # (seconds, or None), and whether it still did when the run ended.
            "objects_placed_s": {obj: round(at / CONTROL_HZ, 2) if at is not None else None
                                 for obj, at in placed.items()},
            "objects_placed_at_end": at_end,
            # And whether the robot handled each one: first time it was 5 cm above its start
            # (seconds, or None), and the most it was moved or raised (cm).
            "objects_lifted_s": {obj: round(at / CONTROL_HZ, 2) if at is not None else None
                                 for obj, at in lifted_at.items()},
            "objects_max_moved_cm": {obj: round(100 * m, 1) for obj, m in max_moved.items()},
            "objects_max_lift_cm": {obj: round(100 * m, 1) for obj, m in max_lift.items()},
            "median_plan_ms": round(float(np.median(latencies)), 1) if latencies else None,
            "stopped": stop_reason,
        }
        with self.log_path.open("a") as log:
            log.write(json.dumps(summary) + "\n")
        outcome = {True: "success", False: "no success", None: "no success check"}[summary["success"]]
        print(f"[sim] Done ({stop_reason}): {summary['plans']} plans, {summary['sim_seconds']} s simulated, "
              f"{outcome}, median plan {summary['median_plan_ms']} ms")
        return {"summary": summary, "pending": pending}


HELP = """\
  <text>                 act on this instruction (replaces the current one)
  /cfg X                 CFG scale from the next plan on; 1 turns CFG off (and halves compute)
  /negative TEXT|off     CFG's second pass uses TEXT instead of an empty prompt: push away from it
  /interval edge|all|A B CFG timestep window: edge = 960 1001 (NVIDIA's Edge setting), all = every step
  /steps N               denoising steps per plan
  /horizon N             actions executed before re-planning, 1 to 32
  /plans N               plans per instruction before stopping
  /seed N|random         sampling seed
  /imagine on|off        also show and save the video the policy imagined
  /default [name]        act on the task's own instruction (default, vague, specific, ...)
  /stop                  stop acting, keep the scene as it is
  /reset                 put the scene back to its start
  /status, /help, /quit"""


def apply_setting(settings: Settings, command: str, words: list[str]) -> None:
    if command == "/cfg":
        value = float(words[0])
        if not 0 <= value <= 20:
            raise ValueError("cfg must be between 0 and 20")
        settings.guidance = value
    elif command == "/negative":
        text = " ".join(words)
        settings.negative = "" if text.lower() in ("", "off", "none") else text
    elif command == "/interval":
        settings.interval = parse_interval(words)
    elif command == "/steps":
        settings.steps = max(1, min(100, int(words[0])))
    elif command == "/horizon":
        settings.horizon = max(1, min(CHUNK, int(words[0])))
    elif command == "/plans":
        settings.max_plans = max(1, int(words[0]))
    elif command == "/seed":
        settings.seed = None if words[0] == "random" else int(words[0])
    elif command == "/imagine":
        settings.imagine = words[0] in ("on", "1", "true", "yes")
    else:
        raise KeyError(command)


def interactive(session: Session) -> None:
    commands: queue.Queue[str] = queue.Queue()

    def read_stdin() -> None:
        while True:
            try:
                commands.put(input().strip())
            except EOFError:
                commands.put("/quit")
                return

    threading.Thread(target=read_stdin, daemon=True).start()
    print("[sim] Type an instruction, or /help.")
    print("> ", end="", flush=True)

    def next_command() -> str | None:
        try:
            line = commands.get_nowait()
        except queue.Empty:
            return None
        return line or None

    def poll_during_run() -> str | None:
        # Setting changes apply on the next plan without interrupting; anything else interrupts.
        while (line := next_command()) is not None:
            if handle(line, running=True):
                return line
        return None

    def handle(line: str, running: bool) -> bool:
        """Apply a command. Returns True if it needs the current run to stop."""
        command, *words = line.split()
        if not line.startswith("/"):
            return True
        try:
            if command in ("/quit", "/exit", "/stop", "/reset", "/default"):
                return True
            if command == "/help":
                print(HELP)
            elif command == "/status":
                print(f"[sim] {session.settings.describe()}")
            else:
                apply_setting(session.settings, command, words)
                note = " (from the next plan)" if running else ""
                print(f"[sim] {session.settings.describe()}{note}")
        except KeyError:
            print(f"[sim] Unknown command {command}; /help lists them.")
        except (ValueError, IndexError) as exc:
            print(f"[sim] {command}: {exc}")
        return False

    pending: str | None = None
    while simulation_app.is_running():
        line = pending or next_command()
        pending = None
        if line is None:
            session.render_idle()
            time.sleep(1 / 60)
            continue
        command, *words = line.split()
        if command in ("/quit", "/exit"):
            break
        if command == "/reset":
            session.reset()
            print("[sim] Scene reset.")
        elif command == "/stop":
            print("[sim] Stopped.")
        elif command == "/default" or not line.startswith("/"):
            if command == "/default":
                field = words[0] if words else "default"
                if field not in session.instructions:
                    print(f"[sim] This task's instructions are: {', '.join(session.instructions)}")
                    print("> ", end="", flush=True)
                    continue
                line = session.instructions[field]
            pending = session.run(line, poll=poll_during_run)["pending"]
        else:
            handle(line, running=False)
        if pending is None:
            print("> ", end="", flush=True)


def batch(session: Session) -> None:
    rows, skipped = [], 0
    seeds = args.seed or [0]
    negatives = ["" if n.strip().lower() in ("", "none") else n for n in (args.negative or [""])]
    for instruction in args.prompt:
        for guidance in args.cfg:
            for negative in negatives:
                # Combinations a negative prompt cannot change: CFG 1 skips the second pass, and a
                # negative equal to the instruction makes both passes the same.
                if negative and (guidance == 1.0 or negative.strip().lower() == instruction.strip().lower()):
                    skipped += 1
                    continue
                for seed in seeds:
                    session.settings.guidance = guidance
                    session.settings.negative = negative
                    session.settings.seed = seed
                    session.reset()
                    label = f"_seed{seed}" if len(seeds) > 1 else ""
                    rows.append(session.run(instruction, label=label)["summary"])
    if skipped:
        print(f"\n[sim] Skipped {skipped} instruction/CFG/negative combinations where the negative prompt "
              f"cannot change anything (CFG 1, or a negative identical to the instruction).")
    print("\n[sim] Summary")
    show_negative = any(row["settings"]["negative"] for row in rows)
    show_objects = any(row["objects_placed_s"] for row in rows)
    header = f"  {'cfg':>5}  {'seed':>4}  {'success':>8}  {'after s':>7}  {'plans':>5}  {'ms/plan':>7}"
    if show_objects:
        header += f"  {'placed (s)':<28}"
    print(header + "  instruction" + ("  | negative" if show_negative else ""))
    for row in rows:
        success = {True: "yes", False: "no", None: "n/a"}[row["success"]]
        line = (f"  {row['settings']['guidance']:>5g}  {row['settings']['seed']:>4}  {success:>8}  "
                f"{str(row['success_after_s'] or '-'):>7}  {row['plans']:>5}  {str(row['median_plan_ms']):>7}")
        if show_objects:
            placed = ", ".join(f"{obj} {at}" for obj, at in row["objects_placed_s"].items() if at is not None)
            line += f"  {placed or '-':<28}"
        line += f"  {row['instruction']}"
        if show_negative:
            line += f"  | {row['settings']['negative'] or '-'}"
        print(line)
    if len(rows) > len(args.cfg) and all(row["success"] is not None for row in rows):
        print("\n[sim] Success rate by CFG scale (all instructions and seeds)")
        for guidance in args.cfg:
            for negative in negatives:
                hits = [row["success"] for row in rows if row["settings"]["guidance"] == guidance
                        and row["settings"]["negative"] == negative]
                if hits:
                    against = f", negative '{negative}'" if negative else ""
                    print(f"  cfg {guidance:g}{against}: {sum(hits)}/{len(hits)}")
    print(f"\n[sim] Videos and log.jsonl in {session.out_dir}")


def main() -> None:
    settings = Settings(
        guidance=args.cfg[0],
        interval=parse_interval(args.interval),
        steps=args.steps,
        horizon=max(1, min(CHUNK, args.horizon)),
        max_plans=args.max_plans,
        imagine=args.imagine,
        seed=args.seed[0] if args.seed else (0 if args.prompt else None),
        negative=next((n for n in args.negative or [] if n.strip().lower() not in ("", "none")), ""),
    )
    session = Session(settings)
    try:
        if args.prompt:
            batch(session)
        else:
            interactive(session)
    finally:
        session.client.close()
        session.env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
