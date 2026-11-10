"""Use the Reasoner as a live monitor: ask the same question about each window of a video.

This is the shape of a real-time perception loop. A camera fills a window of a few
seconds, the model answers about it, and the answer has to arrive before the next
window is ready. The script reports that margin as a real-time factor.

    python scripts/watch.py                                  # bundled dashcam clip
    python scripts/watch.py --video cam.mp4 --window 4 --question "Is anyone in the marked zone?"
    python scripts/watch.py --video rtsp://camera/stream --model nano

A file is read up front and replayed in windows; it is not paced to wall-clock time.
"""

from __future__ import annotations

import argparse
import time

import torch

from common import asset
from reason import ask, load_reasoner, read_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="edge")
    parser.add_argument("--video", help="file path or stream URL")
    parser.add_argument("--question", default="You are a driving monitor. In one sentence: what is the "
                                               "vehicle doing, and is there any hazard?")
    parser.add_argument("--window", type=float, default=2.0, help="seconds of video per question")
    parser.add_argument("--fps", type=float, default=2.0, help="frames per second shown to the model")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--thinking", action="store_true", help="let the model reason first (slower)")
    args = parser.parse_args()

    video = args.video or str(asset(args.model, "assets/example_action_id_av_0_input.mp4"))
    frames, native_fps = read_video(video)
    per_window = max(1, round(args.window * native_fps))
    processor, model = load_reasoner(args.model)

    latencies = []
    for start in range(0, len(frames) - per_window + 1, per_window):
        window = frames[start:start + per_window]
        t0 = time.perf_counter()
        reply, _ = ask(processor, model, args.question, frames=window, native_fps=native_fps,
                       sample_fps=args.fps, max_new_tokens=args.max_new_tokens, thinking=args.thinking)
        torch.cuda.synchronize()
        latency = time.perf_counter() - t0
        latencies.append(latency)
        print(f"[{start / native_fps:5.1f}-{(start + per_window) / native_fps:5.1f} s] "
              f"{latency:.2f} s  {reply.strip()}")

    if len(latencies) > 1:
        latencies = latencies[1:]  # the first call includes warm-up
    mean = sum(latencies) / len(latencies)
    print(f"\nMean latency {mean:.2f} s per {args.window:.1f} s window: "
          f"real-time factor {args.window / mean:.1f} "
          f"({'keeps up' if mean <= args.window else 'falls behind'})")


if __name__ == "__main__":
    main()
