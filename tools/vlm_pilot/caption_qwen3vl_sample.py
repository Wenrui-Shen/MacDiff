#!/usr/bin/env python3
"""Caption one rendered skeleton frame sequence with Qwen3-VL on Ubuntu."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


PERSON2_BLUE = np.asarray((66, 145, 255), dtype=np.uint8)
MAIN_PARTS = {"head", "torso", "left_arm", "right_arm", "left_leg", "right_leg"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", type=Path, required=True)
    parser.add_argument(
        "--prompt_path",
        type=Path,
        default=Path(__file__).with_name("skeleton_motion_prompt_v0.txt"),
    )
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--sample_fps", type=float, default=4.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--attn_implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default=None,
    )
    return parser.parse_args()


def extract_json(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("The model response does not contain a JSON object")
    return json.loads(cleaned[start:end + 1])


def determine_actor_count(frames_dir, frame_paths):
    """Use renderer metadata, or visible blue pixels as a fallback."""
    metadata_path = frames_dir.parent / "render_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        actor_count = metadata.get("visible_actor_count")
        if actor_count in (1, 2):
            return actor_count, "render_metadata"

    for frame_path in frame_paths:
        pixels = np.asarray(Image.open(frame_path).convert("RGB"))
        if np.any(np.all(pixels == PERSON2_BLUE, axis=-1)):
            return 2, "blue_pixel_fallback"
    return 1, "blue_pixel_fallback"


def validate_caption(caption, expected_actors):
    """Return semantic errors that JSON parsing alone cannot detect."""
    errors = []
    if not isinstance(caption, dict):
        return ["caption must be a JSON object"]

    if caption.get("actors") != expected_actors:
        errors.append("actors must equal the color-derived count %d" % expected_actors)

    for key in (
        "main_actor", "main_part", "motion", "beginning", "middle", "end",
        "interaction", "text",
    ):
        value = caption.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append("%s must be a non-empty string" % key)

    if caption.get("main_part") not in MAIN_PARTS:
        errors.append("main_part must be one of: %s" % ", ".join(sorted(MAIN_PARTS)))

    allowed_actors = {"red"} if expected_actors == 1 else {"red", "blue", "both"}
    if caption.get("main_actor") not in allowed_actors:
        errors.append("main_actor must be one of: %s" % ", ".join(sorted(allowed_actors)))

    summary_text = caption.get("text")
    if isinstance(summary_text, str) and len(summary_text.split()) > 35:
        errors.append("text must contain at most 35 English words")
    return errors


def main():
    args = parse_args()
    try:
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install transformers>=4.57.0, accelerate, "
            "qwen-vl-utils==0.0.14, Pillow, and a CUDA-compatible PyTorch build."
        ) from exc

    frame_paths = sorted(args.frames_dir.glob("frame_*.png"))
    if len(frame_paths) < 2:
        raise ValueError("Expected at least two frame_*.png files in %s" % args.frames_dir)
    frame_urls = [path.resolve().as_uri() for path in frame_paths]
    actor_count, actor_count_source = determine_actor_count(args.frames_dir, frame_paths)
    actor_description = (
        "one person (red only; no blue skeleton is visible)"
        if actor_count == 1
        else "two people (both red and blue skeletons are visible)"
    )
    prompt = args.prompt_path.read_text(encoding="utf-8")
    prompt = prompt.replace("{ACTOR_COUNT}", str(actor_count))
    prompt = prompt.replace("{ACTOR_DESCRIPTION}", actor_description)

    load_kwargs = {"dtype": "auto", "device_map": "auto"}
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kwargs)
    processor = AutoProcessor.from_pretrained(args.model)

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": frame_urls,
                "sample_fps": args.sample_fps,
                "min_pixels": 6 * 32 * 32,
                "max_pixels": 360 * 32 * 32,
                "total_pixels": 12288 * 32 * 32,
            },
            {"type": "text", "text": prompt},
        ],
    }]

    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    if videos is not None:
        videos, video_metadata = zip(*videos)
        videos, video_metadata = list(videos), list(video_metadata)
    else:
        video_metadata = None
    inputs = processor(
        text=chat_text,
        images=images,
        videos=videos,
        video_metadata=video_metadata,
        return_tensors="pt",
        do_resize=False,
        **video_kwargs,
    )
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
    raw_response = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    result = {
        "model": args.model,
        "frames_dir": str(args.frames_dir.resolve()),
        "num_frames": len(frame_paths),
        "sample_fps": args.sample_fps,
        "actor_count_from_color": actor_count,
        "actor_count_source": actor_count_source,
        "prompt_path": str(args.prompt_path.resolve()),
        "raw_response": raw_response,
    }
    try:
        caption = extract_json(raw_response)
        semantic_errors = validate_caption(caption, actor_count)
        result["caption"] = caption
        result["text"] = caption.get("text")
        if semantic_errors:
            result["status"] = "invalid_content"
            result["errors"] = semantic_errors
        else:
            result["status"] = "accepted"
    except (ValueError, json.JSONDecodeError) as exc:
        result["caption"] = None
        result["status"] = "invalid_json"
        result["error"] = str(exc)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
