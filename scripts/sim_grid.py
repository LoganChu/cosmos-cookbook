"""Stitch one sim batch into a single labeled comparison video, for watching without Isaac Sim.

Reads outputs/sim/<run>/log.jsonl written by scripts/sim_droid.py and lays the runs out as a grid:
one row per instruction (and seed), one column per CFG scale (and negative prompt, when a batch used
them), each tile labeled with its settings and what happened: whether the task's goal was met or, in
scenes with several objects to place, when each one was placed. Shorter runs hold their last frame
until the longest ends.

    python3 scripts/sim_grid.py                      # newest folder in outputs/sim
    python3 scripts/sim_grid.py outputs/sim/20261120-122455_BananaInBowlTask
    python3 scripts/sim_grid.py --views              # the policy's camera views instead of the scene camera
    python3 scripts/sim_grid.py A B                  # two batches (say, more seeds of the same runs) in one grid
    python3 scripts/sim_grid.py A B --negatives none "The banana"   # only these second passes
                                                     # (with /imagine on: real on the left, imagined on the right)

Needs only Python and ffmpeg. Writes grid_scene.mp4 or grid_views.mp4 into the (first) folder.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
GREEN, GOLD, RED = "0x7CFC00", "0xFFD700", "0xFF8C69"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(path)], capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def wrap(text: str, width: int) -> str:
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return "\n".join(lines + [line])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="*", help="outputs/sim/<run> folders (default: the newest)")
    parser.add_argument("--views", action="store_true", help="use the policy's camera views")
    parser.add_argument("--tile-width", type=int, default=0,
                        help="pixels per tile (default 576, 640 with --views, narrower when there are over 3 columns)")
    parser.add_argument("--goal", nargs="+", default=None,
                        help="instructions that ask for the task's own goal (default: as logged by sim_droid.py; "
                             "for older logs without that field, every instruction counts)")
    parser.add_argument("--negatives", nargs="+", default=None,
                        help="keep only runs whose negative prompt is one of these ('none' = the empty prompt, "
                             "which includes every CFG 1 run)")
    args = parser.parse_args()

    folders = [Path(f) for f in args.folders] or [max((ROOT / "outputs" / "sim").glob("*/log.jsonl"),
                                                      key=lambda p: p.stat().st_mtime).parent]
    folder = folders[0]
    kind = "views" if args.views else "scene"
    runs = []
    for source in folders:
        for line in (source / "log.jsonl").read_text().splitlines():
            if line.strip():
                run = json.loads(line)
                run["video"] = source / f"{run['run']}_{kind}.mp4"
                runs.append(run)
    runs = [r for r in runs if r["video"].exists()]
    if args.negatives:
        keep = {"" if n.strip().lower() == "none" else n for n in args.negatives}
        runs = [r for r in runs if (r["settings"].get("negative") or "") in keep]
    if not runs:
        raise SystemExit(f"No {kind} videos listed in {', '.join(str(f / 'log.jsonl') for f in folders)}")

    # Rows: instruction (x seed when there are several). Columns: CFG scale, then which negative
    # prompt: within a row its negatives are numbered in order of appearance (0 = none), so "the
    # other object's instruction" lines up in one column though its text differs per row.
    def negative(r: dict) -> str:
        return r["settings"].get("negative") or ""

    def row_key(r: dict) -> tuple:
        return (r["instruction"], r["settings"]["seed"])

    order = list(dict.fromkeys(r["instruction"] for r in runs))
    rows_keys = sorted(dict.fromkeys(row_key(r) for r in runs), key=lambda k: (order.index(k[0]), k[1]))
    slot = {}
    for key in rows_keys:
        texts = dict.fromkeys(negative(r) for r in runs if row_key(r) == key and negative(r))
        slot.update({(*key, text): number for number, text in enumerate(texts, 1)})

    def column(r: dict) -> tuple:
        return (r["settings"]["guidance"], slot[(*row_key(r), negative(r))] if negative(r) else 0)

    columns = sorted(dict.fromkeys(column(r) for r in runs))
    any_negative = any(negative(r) for r in runs)
    show_seed = len({seed for _, seed in rows_keys}) > 1
    cells = {(*row_key(r), column(r)): r for r in runs}

    tile_w = args.tile_width or (640 if args.views else 576)
    if not args.tile_width and len(columns) > 3:
        tile_w = 1920 // len(columns) // 2 * 2  # keep the grid within 1920 pixels
    sample = runs[0]["video"]
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", str(sample)],
                           capture_output=True, text=True, check=True).stdout.strip().split(",")
    tile_h = round(tile_w * int(probe[1]) / int(probe[0]) / 2) * 2
    length = max(duration(r["video"]) for r in runs)

    tmp = Path(tempfile.mkdtemp(prefix="sim_grid_"))
    inputs, filters, labels = [], [], []
    size = min(26, max(14, tile_w // 34))
    for row, (instruction, seed) in enumerate(rows_keys):
        for col, (guidance, _slot) in enumerate(columns):
            run = cells.get((instruction, seed, (guidance, _slot)))
            index = len(labels)
            outcomes: list[tuple[str, float, str]] = []  # (text, when it appears, colour)
            if run is None:  # this combination was not run: a black tile
                inputs += ["-f", "lavfi", "-t", f"{length:.2f}", "-i", f"color=c=black:s={tile_w}x{tile_h}:r=15"]
                settings = "not run"
                filters.append(f"[{index}:v]null[t{index}]")
            else:
                path = run["video"]
                clip = duration(path)
                inputs += ["-i", str(path)]
                settings = f"CFG {guidance:g}"
                if any_negative:  # say what CFG pushed away from
                    if guidance == 1:
                        settings += ": instruction only, no second pass"
                    elif negative(run):
                        settings += f" away from “{negative(run)}”"
                    else:
                        settings += " away from the empty prompt"
                if args.views and run["settings"].get("imagine"):
                    settings += "   |   left: the cameras   right: what the policy imagined"
                settings = wrap(settings, int(tile_w / (size * 0.6)) - 2)
                # The outcome appears when it happens: on success, or when the run's time is up.
                # The success check is the task's goal; for any other instruction, say only how long
                # it ran, and mention the task goal if it happened anyway.
                goals = args.goal or run.get("task_instructions")
                asked_for_goal = goals is None or instruction.strip().lower() in {g.strip().lower() for g in goals}
                placed = run.get("objects_placed_s") or {}
                if len(placed) > 1:
                    # Several objects: when each was placed. Green if the instruction names it, gold if
                    # the robot placed something it was not asked for.
                    def named(obj: str) -> bool:
                        return any(part in instruction.lower() for part in obj.lower().split("_") if len(part) > 2)

                    lifted = run.get("objects_lifted_s") or {}
                    events = []
                    for obj, at in placed.items():
                        label = obj.replace("_", " ")
                        if not named(obj) and lifted.get(obj) is not None:  # went for the other object
                            events.append((f"{label} picked up at {lifted[obj]} s (not asked)",
                                           float(lifted[obj]), GOLD))
                        if at is not None:
                            later = "" if run.get("objects_placed_at_end", {}).get(obj, True) else ", later taken out"
                            events.append((f"{label} placed at {at} s{later}", float(at),
                                           GREEN if named(obj) else GOLD))
                        elif named(obj):
                            events.append((f"{label} not placed in {run['sim_seconds']} s", clip, RED))
                    outcomes += sorted(events, key=lambda event: event[1])
                elif not asked_for_goal:
                    if run["success"]:
                        outcomes.append((f"(task's own goal met at {run['success_after_s']} s)",
                                         float(run["success_after_s"]), GOLD))
                    else:
                        outcomes.append((f"ran {run['sim_seconds']} s", clip, "white"))
                elif run["success"]:
                    outcomes.append((f"task goal met at {run['success_after_s']} s",
                                     float(run["success_after_s"]), GREEN))
                elif run["success"] is False:
                    outcomes.append((f"task goal not met in {run['sim_seconds']} s", clip, RED))
                else:
                    outcomes.append((f"stopped after {run['sim_seconds']} s", clip, "white"))
                filters.append(f"[{index}:v]scale={tile_w}:{tile_h},setsar=1,fps=15,"
                               f"tpad=stop_mode=clone:stop_duration={max(0.0, length - clip) + 0.1:.2f}[t{index}]")
            style = f"fontfile={FONT}:fontsize={size}:box=1:boxcolor=black@0.6:boxborderw=6"
            settings_file = tmp / f"settings{index}.txt"
            settings_file.write_text(settings)
            chain = f"[t{index}]drawtext={style}:textfile={settings_file}:fontcolor=white:x=8:y=h-th-12"
            # Outcomes stack from the top of the tile, each appearing when it happens.
            for number, (text, at, color) in enumerate(outcomes):
                outcome_file = tmp / f"outcome{index}_{number}.txt"
                outcome_file.write_text(text)
                chain += (f",drawtext={style}:textfile={outcome_file}:fontcolor={color}:x=8:"
                          f"y={10 + number * (size + 18)}:enable='gte(t,{at:.2f})'")
            filters.append(f"{chain}[l{index}]")
            labels.append((row, col, index))

    # A title bar above each row with its instruction.
    bar_h = min(56, max(36, tile_w // 14))
    row_streams = []
    for row, (instruction, seed) in enumerate(rows_keys):
        tiles = [f"[l{i}]" for r, _, i in labels if r == row]
        title_file = tmp / f"title{row}.txt"
        title = f"“{instruction}”" + (f"   seed {seed}" if show_seed else "")
        title_file.write_text(wrap(title, 30 * len(columns) * tile_w // 576))
        stacked = (f"{''.join(tiles)}hstack=inputs={len(tiles)}[h{row}]" if len(tiles) > 1
                   else f"{tiles[0]}null[h{row}]")
        filters.append(stacked)
        filters.append(f"[h{row}]pad=iw:ih+{bar_h}:0:{bar_h}:color=0x202020,"
                       f"drawtext=fontfile={BOLD}:textfile={title_file}:fontsize={bar_h // 2}:fontcolor=white:"
                       f"x=12:y=({bar_h}-th)/2[r{row}]")
        row_streams.append(f"[r{row}]")
    if len(row_streams) > 1:
        filters.append(f"{''.join(row_streams)}vstack=inputs={len(row_streams)},format=yuv420p[out]")
    else:
        filters.append(f"{row_streams[0]}format=yuv420p[out]")

    out = folder / f"grid_{kind}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(filters),
                    "-map", "[out]", "-t", f"{length + 0.1:.2f}", "-c:v", "libx264", "-crf", "23",
                    "-preset", "veryfast", "-movflags", "+faststart", str(out)], check=True)
    print(f"{out}  ({len(rows_keys)} rows x {len(columns)} columns, {length:.1f} s)")


if __name__ == "__main__":
    main()
