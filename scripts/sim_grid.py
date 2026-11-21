"""Stitch one sim batch into a single labeled comparison video, for watching without Isaac Sim.

Reads outputs/sim/<run>/log.jsonl written by scripts/sim_droid.py and lays the runs out as a grid:
one row per instruction (and seed), one column per CFG scale, each tile labeled with its settings
and whether the task's goal was met. Shorter runs hold their last frame until the longest ends.

    python3 scripts/sim_grid.py                      # newest folder in outputs/sim
    python3 scripts/sim_grid.py outputs/sim/20261120-122455_BananaInBowlTask
    python3 scripts/sim_grid.py --views              # the policy's camera views instead of the scene camera
                                                     # (with /imagine on: real on the left, imagined on the right)

Needs only Python and ffmpeg. Writes grid_scene.mp4 or grid_views.mp4 into the same folder.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
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
    parser.add_argument("folder", nargs="?", help="an outputs/sim/<run> folder (default: the newest)")
    parser.add_argument("--views", action="store_true", help="use the policy's camera views")
    parser.add_argument("--tile-width", type=int, default=0, help="pixels per tile (default 576, 640 with --views)")
    parser.add_argument("--goal", nargs="+", default=None,
                        help="instructions that ask for the task's own goal (default: as logged by sim_droid.py; "
                             "for older logs without that field, every instruction counts)")
    args = parser.parse_args()

    folder = Path(args.folder) if args.folder else max((ROOT / "outputs" / "sim").glob("*/log.jsonl"),
                                                        key=lambda p: p.stat().st_mtime).parent
    runs = [json.loads(line) for line in (folder / "log.jsonl").read_text().splitlines() if line.strip()]
    kind = "views" if args.views else "scene"
    runs = [r for r in runs if (folder / f"{r['run']}_{kind}.mp4").exists()]
    if not runs:
        raise SystemExit(f"No {kind} videos listed in {folder}/log.jsonl")

    # Rows: instruction (x seed when there are several); columns: CFG scale.
    rows_keys = list(dict.fromkeys((r["instruction"], r["settings"]["seed"]) for r in runs))
    columns = list(dict.fromkeys(r["settings"]["guidance"] for r in runs))
    show_seed = len({seed for _, seed in rows_keys}) > 1
    cells = {(r["instruction"], r["settings"]["seed"], r["settings"]["guidance"]): r for r in runs}

    tile_w = args.tile_width or (640 if args.views else 576)
    sample = folder / f"{runs[0]['run']}_{kind}.mp4"
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", str(sample)],
                           capture_output=True, text=True, check=True).stdout.strip().split(",")
    tile_h = round(tile_w * int(probe[1]) / int(probe[0]) / 2) * 2
    length = max(duration(folder / f"{r['run']}_{kind}.mp4") for r in runs)

    tmp = Path(tempfile.mkdtemp(prefix="sim_grid_"))
    inputs, filters, labels = [], [], []
    for row, (instruction, seed) in enumerate(rows_keys):
        for col, guidance in enumerate(columns):
            run = cells.get((instruction, seed, guidance))
            index = len(labels)
            outcome, outcome_at, color = "", 0.0, "white"
            if run is None:  # this combination was not run: a black tile
                inputs += ["-f", "lavfi", "-t", f"{length:.2f}", "-i", f"color=c=black:s={tile_w}x{tile_h}:r=15"]
                settings = "not run"
                filters.append(f"[{index}:v]null[t{index}]")
            else:
                path = folder / f"{run['run']}_{kind}.mp4"
                clip = duration(path)
                inputs += ["-i", str(path)]
                settings = f"CFG {guidance:g}"
                if args.views and run["settings"].get("imagine"):
                    settings += "   |   left: the cameras   right: what the policy imagined"
                # The outcome appears when it happens: on success, or when the run's time is up.
                # The success check is the task's goal; for any other instruction, say only how long
                # it ran, and mention the task goal if it happened anyway.
                goals = args.goal or run.get("task_instructions")
                asked_for_goal = goals is None or instruction.strip().lower() in {g.strip().lower() for g in goals}
                if not asked_for_goal:
                    if run["success"]:
                        outcome, outcome_at, color = f"(task's own goal met at {run['success_after_s']} s)", \
                            float(run["success_after_s"]), "0xFFD700"
                    else:
                        outcome, outcome_at = f"ran {run['sim_seconds']} s", clip
                elif run["success"]:
                    outcome, outcome_at, color = f"task goal met at {run['success_after_s']} s", \
                        float(run["success_after_s"]), "0x7CFC00"
                elif run["success"] is False:
                    outcome, outcome_at, color = f"task goal not met in {run['sim_seconds']} s", clip, "0xFF8C69"
                else:
                    outcome, outcome_at = f"stopped after {run['sim_seconds']} s", clip
                filters.append(f"[{index}:v]scale={tile_w}:{tile_h},setsar=1,fps=15,"
                               f"tpad=stop_mode=clone:stop_duration={max(0.0, length - clip) + 0.1:.2f}[t{index}]")
            size = min(26, max(14, tile_w // 34))
            style = f"fontfile={FONT}:fontsize={size}:box=1:boxcolor=black@0.6:boxborderw=6"
            settings_file = tmp / f"settings{index}.txt"
            settings_file.write_text(settings)
            chain = f"[t{index}]drawtext={style}:textfile={settings_file}:fontcolor=white:x=8:y=h-th-12"
            if outcome:
                outcome_file = tmp / f"outcome{index}.txt"
                outcome_file.write_text(outcome)
                chain += (f",drawtext={style}:textfile={outcome_file}:fontcolor={color}:x=8:y=h-2*th-30:"
                          f"enable='gte(t,{outcome_at:.2f})'")
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
    print(f"{out}  ({len(rows_keys)} rows x {len(columns)} CFG scales, {length:.1f} s)")


if __name__ == "__main__":
    main()
