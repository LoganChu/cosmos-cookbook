"""Run the Cosmos 3 Reasoner tower in process: text, image or video in, text out.

Only the autoregressive tower is loaded, so this is the cheapest way to try a model.

    python scripts/reason.py                                   # Edge, bundled robot scene
    python scripts/reason.py --image my.jpg --prompt "Is the gripper holding anything?"
    python scripts/reason.py --model nano --video clip.mp4 --fps 4 --prompt "What happens next?"
    python scripts/reason.py --model nano --fp8               # Nano's calibrated FP8 weights
"""

from __future__ import annotations

import argparse
import json

import torch

from common import asset, measure, resolve_model


def load_reasoner(model_name: str, fp8: bool = False):
    """Return (processor, model). Edge and Nano/Super use different model classes."""
    from transformers import AutoModelForImageTextToText, AutoProcessor, Cosmos3OmniForConditionalGeneration

    model_id = resolve_model(model_name)
    checkpoint = model_id
    if fp8:
        # FP8 ships as the `fp8` branch of the Nano and Super repos. ModelOpt restores the
        # calibrated weights and scales; it is not quantizing at load time.
        from huggingface_hub import snapshot_download
        from modelopt.torch import opt as modelopt_opt
        from modelopt.torch.quantization.backends import fp8_per_tensor_gemm  # noqa: F401

        modelopt_opt.enable_huggingface_checkpointing()
        checkpoint = snapshot_download(model_id, revision="fp8")

    processor = AutoProcessor.from_pretrained(checkpoint)
    cls = AutoModelForImageTextToText if "Edge" in model_id else Cosmos3OmniForConditionalGeneration
    model = cls.from_pretrained(checkpoint, dtype=torch.bfloat16, device_map="cuda")
    return processor, model


def read_video(path: str) -> tuple[list, float]:
    """Decode a video file or stream URL into PIL frames plus its native frame rate."""
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        return [f.to_image() for f in container.decode(stream)], float(stream.average_rate)


def ask(processor, model, prompt: str, *, image: str | None = None, frames: list | None = None,
        native_fps: float | None = None, sample_fps: float = 4.0,
        max_new_tokens: int = 512, thinking: bool = True) -> tuple[str, int]:
    """Send one chat turn. Returns the reply and the number of generated tokens.

    For video, pass every decoded frame with its native frame rate; the processor then
    subsamples to `sample_fps` and stamps the frames with correct times.
    """
    from transformers.video_utils import VideoMetadata

    content, processor_kwargs = [], {}
    if frames is not None:
        content.append({"type": "video", "video": frames})
        processor_kwargs = {"fps": sample_fps,
                            "video_metadata": VideoMetadata(total_num_frames=len(frames), fps=native_fps)}
    elif image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})

    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        enable_thinking=thinking, processor_kwargs=processor_kwargs,
    ).to(model.device, torch.bfloat16)
    out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    new_tokens = out[0][inputs.input_ids.shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True), len(new_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="edge", help="edge, nano, or a repo ID / local path")
    parser.add_argument("--image")
    parser.add_argument("--video")
    parser.add_argument("--fps", type=float, default=4.0, help="video sampling rate (NVIDIA recommends 4)")
    parser.add_argument("--prompt")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-thinking", action="store_true", help="skip the chain-of-thought; much faster (Edge only; Nano ignores it)")
    parser.add_argument("--fp8", action="store_true", help="Nano/Super only; needs `make fp8-deps`")
    args = parser.parse_args()

    prompt, image, frames, native_fps = args.prompt, args.image, None, None
    if args.video:
        frames, native_fps = read_video(args.video)
    elif image is None:
        example = json.loads(asset(args.model, "assets/example_reasoning_prompt.json").read_text())
        image = str(asset(args.model, "assets/example_reasoning_input.png"))
        prompt = prompt or example["prompt"]
    prompt = prompt or "Describe what is happening and what will happen next."

    with measure("load"):
        processor, model = load_reasoner(args.model, args.fp8)
    with measure("generate") as stats:
        reply, n_tokens = ask(processor, model, prompt, image=image, frames=frames, native_fps=native_fps,
                              sample_fps=args.fps, max_new_tokens=args.max_new_tokens,
                              thinking=not args.no_thinking)
    print(f"\n{reply}\n")
    print(f"{n_tokens} tokens at {n_tokens / stats['seconds']:.1f} tok/s")


if __name__ == "__main__":
    main()
