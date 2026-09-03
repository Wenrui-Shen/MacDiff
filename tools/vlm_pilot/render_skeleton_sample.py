#!/usr/bin/env python3
"""Render one label-free skeleton sample as a frame sequence for a video VLM.

The renderer reads only ``x_train``/``x_test`` from the processed MAMP NPZ.
It intentionally never reads ``y_train`` or ``y_test``.  Random feeder crop,
rotation, flip, and noise augmentation are not applied.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


NTU_EDGES = (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (23, 8), (24, 25), (25, 12),
)
NTU_EDGES = tuple((a - 1, b - 1) for a, b in NTU_EDGES)

ROOT_JOINT = 1  # NTU joint 2: SpineMid

BACKGROUND = (18, 20, 24)
PANEL_BACKGROUND = (25, 28, 34)
GRID = (62, 68, 78)
TEXT = (232, 235, 240)
MUTED_TEXT = (164, 170, 182)
PERSON_COLORS = (
    (239, 76, 76),   # person 1: red
    (66, 145, 255),  # person 2: blue
)
BONE_WIDTH = 4
JOINT_RADIUS = 4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, help="MAMP NPZ, e.g. NTU60_XSub.npz")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--gif_duration_ms", type=int, default=180)
    parser.add_argument(
        "--temporal_smooth",
        choices=("none", "savgol"),
        default="savgol",
        help="Smooth every joint trajectory before frame sampling (default: savgol).",
    )
    parser.add_argument(
        "--median_window",
        type=int,
        default=3,
        help="Odd median-filter window used to suppress one-frame spikes.",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=5,
        help="Odd Savitzky-Golay window applied to the full valid sequence.",
    )
    parser.add_argument("--smooth_polyorder", type=int, default=2)
    parser.add_argument(
        "--max_interp_gap",
        type=int,
        default=2,
        help="Maximum internal missing-joint gap to linearly interpolate.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use a synthetic walking-and-arm-raising sequence to test rendering only.",
    )
    return parser.parse_args()


def load_npz_sample(path, split, sample_index):
    if path is None:
        raise ValueError("--data_path is required unless --demo is used")
    key = "x_train" if split == "train" else "x_test"
    with np.load(str(path), mmap_mode="r") as archive:
        if key not in archive:
            raise KeyError("%s does not contain %s" % (path, key))
        data = np.asarray(archive[key][sample_index], dtype=np.float32)

    return normalize_sample(data)


def normalize_sample(data):
    """Convert one processed sample to valid-frame ``[T, M, 25, 3]`` form."""
    data = np.asarray(data, dtype=np.float32)

    if data.ndim == 2 and data.shape[-1] == 150:
        data = data.reshape(data.shape[0], 2, 25, 3)
    elif data.ndim == 4 and data.shape == (3, data.shape[1], 25, 2):
        data = data.transpose(1, 3, 2, 0)
    elif data.ndim == 4 and data.shape[-2:] == (25, 3):
        pass
    else:
        raise ValueError("Unsupported sample shape: %r" % (data.shape,))

    valid = np.any(np.abs(data) > 1e-8, axis=(1, 2, 3))
    data = data[valid]
    if len(data) == 0:
        raise ValueError("Selected sample contains no valid skeleton frame")
    return data


def make_demo_sample(num_frames=72):
    frames = []
    for frame in range(num_frames):
        u = frame / max(num_frames - 1, 1)
        phase = u * math.pi * 4
        root = np.array([-0.8 + 1.6 * u, 1.05, 0.18 * math.sin(u * math.pi * 2)])
        raise_amount = 0.5 - 0.5 * math.cos(u * math.pi * 2)
        stride = 0.18 * math.sin(phase)
        pose = np.repeat(root[None], 25, axis=0)

        def set_joint(index, x, y, z=0.0):
            pose[index] = root + np.array([x, y, z])

        set_joint(0, 0, -0.18); set_joint(1, 0, 0); set_joint(20, 0, 0.38)
        set_joint(2, 0, 0.58); set_joint(3, 0, 0.83)
        set_joint(4, -0.28, 0.38); set_joint(5, -0.48, 0.18, -0.03)
        set_joint(6, -0.58, -0.04, -0.02); set_joint(7, -0.62, -0.12, -0.02)
        set_joint(21, -0.66, -0.14, -0.02); set_joint(22, -0.58, -0.10, 0.05)
        set_joint(8, 0.28, 0.38)
        set_joint(9, 0.48 - 0.10 * raise_amount, 0.18 + 0.48 * raise_amount, 0.04 * raise_amount)
        set_joint(10, 0.58 - 0.34 * raise_amount, -0.04 + 0.86 * raise_amount, 0.08 * raise_amount)
        set_joint(11, 0.62 - 0.40 * raise_amount, -0.12 + 0.93 * raise_amount, 0.10 * raise_amount)
        set_joint(23, 0.66 - 0.42 * raise_amount, -0.14 + 0.96 * raise_amount, 0.10 * raise_amount)
        set_joint(24, 0.58 - 0.35 * raise_amount, -0.10 + 0.91 * raise_amount, 0.16 * raise_amount)
        set_joint(12, -0.17, -0.18); set_joint(13, -0.18, -0.58, stride)
        set_joint(14, -0.18, -0.98, -stride); set_joint(15, -0.18, -1.03, 0.15 - stride)
        set_joint(16, 0.17, -0.18); set_joint(17, 0.18, -0.58, -stride)
        set_joint(18, 0.18, -0.98, stride); set_joint(19, 0.18, -1.03, 0.15 + stride)
        local_pose = pose - root[None]
        pose2 = local_pose.copy()
        pose2[:, 0] *= -1
        pose2 += root[None] + np.array([1.0 - 0.35 * u, 0.0, 0.55])[None]
        frames.append(np.stack([pose, pose2], axis=0))
    return np.asarray(frames, dtype=np.float32)


def uniformly_sample(sequence, num_frames):
    indices = np.rint(np.linspace(0, len(sequence) - 1, num_frames)).astype(np.int64)
    return sequence[indices], indices


def active_joint_mask(sequence):
    return np.any(np.abs(sequence) > 1e-8, axis=-1)


def _validate_odd_window(name, value):
    if value < 1 or value % 2 == 0:
        raise ValueError("%s must be a positive odd integer" % name)


def _contiguous_true_runs(mask):
    """Yield half-open intervals for contiguous true regions in a 1-D mask."""
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    return zip(starts.tolist(), stops.tolist())


def _interpolate_short_joint_gaps(sequence, max_gap):
    """Fill only short internal gaps; never create an absent actor at an edge."""
    result = np.asarray(sequence, dtype=np.float32).copy()
    if max_gap <= 0:
        return result, 0

    active = active_joint_mask(result)
    filled = 0
    for person in range(result.shape[1]):
        for joint in range(result.shape[2]):
            valid = active[:, person, joint]
            missing = ~valid
            for start, stop in _contiguous_true_runs(missing):
                gap = stop - start
                if (
                    gap > max_gap
                    or start == 0
                    or stop == len(result)
                    or not valid[start - 1]
                    or not valid[stop]
                ):
                    continue
                left = result[start - 1, person, joint]
                right = result[stop, person, joint]
                denominator = float(gap + 1)
                for offset in range(1, gap + 1):
                    weight = offset / denominator
                    result[start + offset - 1, person, joint] = (
                        (1.0 - weight) * left + weight * right
                    )
                valid[start:stop] = True
                filled += gap
    return result, filled


def _median_filter_segment(segment, window):
    length = len(segment)
    effective = min(window, length if length % 2 else length - 1)
    if effective < 3:
        return segment.copy()
    half = effective // 2
    padded = np.pad(segment, ((half, half), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, effective, axis=0
    )
    return np.median(windows, axis=-1).astype(np.float32)


def _savgol_weights(window, polyorder, position):
    """Return weights that evaluate a local polynomial at ``position``."""
    x = np.arange(window, dtype=np.float64) - float(position)
    design = np.vander(x, N=polyorder + 1, increasing=True)
    return np.linalg.pinv(design)[0]


def _savgol_filter_segment(segment, window, polyorder):
    """Dependency-free, zero-phase Savitzky-Golay smoothing for one joint."""
    length = len(segment)
    effective = min(window, length if length % 2 else length - 1)
    if effective <= polyorder or effective < 3:
        return segment.copy()

    half = effective // 2
    result = np.empty_like(segment, dtype=np.float32)
    center_weights = _savgol_weights(effective, polyorder, half)
    if length >= effective:
        windows = np.lib.stride_tricks.sliding_window_view(
            segment, effective, axis=0
        )
        result[half:length - half] = np.einsum(
            "tcw,w->tc", windows, center_weights
        ).astype(np.float32)

    # At sequence boundaries, fit the same polynomial to the nearest complete
    # window instead of padding.  This avoids both temporal lag and flat edges.
    for index in range(half):
        weights = _savgol_weights(effective, polyorder, index)
        result[index] = weights @ segment[:effective]
    start = length - effective
    for index in range(length - half, length):
        position = index - start
        weights = _savgol_weights(effective, polyorder, position)
        result[index] = weights @ segment[start:]
    return result


def temporal_smooth_sequence(
    sequence,
    *,
    method="savgol",
    median_window=3,
    smooth_window=5,
    smooth_polyorder=2,
    max_interp_gap=2,
):
    """Smooth full per-person joint trajectories without crossing missing gaps."""
    if method not in ("none", "savgol"):
        raise ValueError("Unsupported temporal smoothing method: %s" % method)
    if max_interp_gap < 0:
        raise ValueError("max_interp_gap must be non-negative")
    if method == "none":
        return np.asarray(sequence, dtype=np.float32).copy(), {
            "method": "none",
            "applied_before_sampling": False,
            "interpolated_joint_frames": 0,
            "filtered_joint_segments": 0,
        }

    _validate_odd_window("median_window", median_window)
    _validate_odd_window("smooth_window", smooth_window)
    if smooth_polyorder < 0 or smooth_polyorder >= smooth_window:
        raise ValueError("smooth_polyorder must be in [0, smooth_window)")

    result, filled = _interpolate_short_joint_gaps(sequence, max_interp_gap)
    active = active_joint_mask(result)
    segments = 0
    for person in range(result.shape[1]):
        for joint in range(result.shape[2]):
            for start, stop in _contiguous_true_runs(active[:, person, joint]):
                segment = result[start:stop, person, joint]
                segment = _median_filter_segment(segment, median_window)
                segment = _savgol_filter_segment(
                    segment, smooth_window, smooth_polyorder
                )
                result[start:stop, person, joint] = segment
                segments += 1

    # Preserve truly missing joints as exact zeros.  Short gaps filled above are
    # active now and intentionally remain visible.
    result[~active] = 0
    return result, {
        "method": method,
        "applied_before_sampling": True,
        "median_window": int(median_window),
        "smooth_window": int(smooth_window),
        "smooth_polyorder": int(smooth_polyorder),
        "max_interp_gap": int(max_interp_gap),
        "interpolated_joint_frames": int(filled),
        "filtered_joint_segments": int(segments),
    }


def primary_root(sequence):
    active = active_joint_mask(sequence)
    root = np.zeros((len(sequence), 3), dtype=np.float32)
    last = np.zeros(3, dtype=np.float32)
    for frame in range(len(sequence)):
        found = False
        for person in range(sequence.shape[1]):
            if active[frame, person, ROOT_JOINT]:
                last = sequence[frame, person, ROOT_JOINT]
                found = True
                break
        root[frame] = last if found or frame else np.zeros(3, dtype=np.float32)
    return root


def root_centered_span(sequence):
    mask = active_joint_mask(sequence)
    points = sequence[mask]
    extent = np.percentile(np.abs(points), 99, axis=0)
    return float(max(np.max(extent) * 2.3, 1e-3))


def projected(point, view):
    x, y, z = point
    if view.endswith("xy"):
        return np.array([x, y])
    if view.endswith("zy"):
        return np.array([z, y])
    if view.endswith("xz"):
        return np.array([x, z])
    raise ValueError("Unsupported projection: %s" % view)


def load_font(size):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


class Panel:
    def __init__(self, box, title, view, center, span):
        self.box = box
        self.title = title
        self.view = view
        self.center = np.asarray(center, dtype=np.float32)
        self.span = float(span)

    def map_point(self, point):
        left, top, right, bottom = self.box
        header = 34
        pad = 18
        width = right - left - 2 * pad
        height = bottom - top - header - 2 * pad
        p = projected(point, self.view)
        c = projected(self.center, self.view)
        scale = min(width, height) / self.span
        return (
            left + (right - left) / 2 + (p[0] - c[0]) * scale,
            top + header + height / 2 - (p[1] - c[1]) * scale,
        )


def make_panels(width, height, local_span):
    cols = 3
    panels = []
    specs = (
        ("FRONT: ROOT-CENTERED X-Y", "local_xy", np.zeros(3), local_span),
        ("SIDE: ROOT-CENTERED Z-Y", "local_zy", np.zeros(3), local_span),
        ("TOP: ROOT-CENTERED X-Z", "local_xz", np.zeros(3), local_span),
    )
    for col in range(cols):
        box = (
            col * width // cols,
            0,
            (col + 1) * width // cols,
            height,
        )
        panels.append(Panel(box, *specs[col]))
    return panels


def draw_grid(draw, panel):
    left, top, right, bottom = panel.box
    draw.rectangle((left + 2, top + 2, right - 2, bottom - 2), fill=PANEL_BACKGROUND, outline=GRID, width=1)
    for frac in (0.25, 0.5, 0.75):
        x = left + int((right - left) * frac)
        y = top + 34 + int((bottom - top - 34) * frac)
        draw.line((x, top + 34, x, bottom), fill=GRID, width=1)
        draw.line((left, y, right, y), fill=GRID, width=1)


def render_frame(sequence, roots, frame_index, panels, width, height, font, actor_count):
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    local_pose = sequence[frame_index] - roots[frame_index][None, None, :]
    active = active_joint_mask(sequence)[frame_index]

    for panel in panels:
        draw_grid(draw, panel)
        left, top, right, _ = panel.box
        draw.text((left + 10, top + 9), panel.title, font=font, fill=TEXT)
        pose = local_pose

        for person in range(pose.shape[0]):
            color = PERSON_COLORS[min(person, len(PERSON_COLORS) - 1)]
            for a, b in NTU_EDGES:
                if not active[person, a] or not active[person, b]:
                    continue
                draw.line((*panel.map_point(pose[person, a]), *panel.map_point(pose[person, b])),
                          fill=color, width=BONE_WIDTH)
            for joint in range(25):
                if not active[person, joint]:
                    continue
                x, y = panel.map_point(pose[person, joint])
                draw.ellipse(
                    (x - JOINT_RADIUS, y - JOINT_RADIUS,
                     x + JOINT_RADIUS, y + JOINT_RADIUS),
                    fill=color,
                )

    if actor_count == 1:
        footer = "ONE PERSON: the red skeleton is repeated in all three views."
    else:
        footer = "TWO PEOPLE: red is person 1 and blue is person 2; views repeat them."
    draw.text((12, height - 22), footer, font=font, fill=MUTED_TEXT)
    return image


def render_sample_frames(
    source,
    num_frames=32,
    width=960,
    height=360,
    font=None,
    temporal_smooth="savgol",
    median_window=3,
    smooth_window=5,
    smooth_polyorder=2,
    max_interp_gap=2,
):
    """Render one normalized or raw sample to an in-memory PIL frame list."""
    if num_frames < 2:
        raise ValueError("num_frames must be at least 2")
    if width % 3:
        raise ValueError("width must be divisible by 3")

    source = normalize_sample(source)
    smoothed, smoothing_info = temporal_smooth_sequence(
        source,
        method=temporal_smooth,
        median_window=median_window,
        smooth_window=smooth_window,
        smooth_polyorder=smooth_polyorder,
        max_interp_gap=max_interp_gap,
    )
    sampled, frame_indices = uniformly_sample(smoothed, num_frames)
    sampled_active = active_joint_mask(sampled)
    person_active_frames = sampled_active.any(axis=2).sum(axis=0)
    blue_skeleton_visible = bool(
        len(person_active_frames) > 1 and person_active_frames[1] > 0
    )
    visible_actor_count = 2 if blue_skeleton_visible else 1
    roots = primary_root(sampled)
    local = sampled - roots[:, None, None, :]
    # Keep missing all-zero joints missing after centering; otherwise subtracting
    # the red root would turn an absent blue skeleton into phantom span points.
    local[~sampled_active] = 0
    local_span = root_centered_span(local)
    panels = make_panels(width, height, local_span)
    font = load_font(14) if font is None else font
    frames = [
        render_frame(
            sampled, roots, index, panels, width, height, font,
            visible_actor_count,
        )
        for index in range(num_frames)
    ]
    render_info = {
        "num_valid_frames": int(len(source)),
        "num_rendered_frames": int(num_frames),
        "source_frame_indices": frame_indices.tolist(),
        "visible_actor_count": visible_actor_count,
        "blue_skeleton_visible": blue_skeleton_visible,
        "active_rendered_frames_per_person": person_active_frames.astype(int).tolist(),
        "local_span": local_span,
        "temporal_smoothing": smoothing_info,
    }
    return frames, render_info


def main():
    args = parse_args()
    source = make_demo_sample() if args.demo else load_npz_sample(args.data_path, args.split, args.sample_index)
    frames, render_info = render_sample_frames(
        source,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        temporal_smooth=args.temporal_smooth,
        median_window=args.median_window,
        smooth_window=args.smooth_window,
        smooth_polyorder=args.smooth_polyorder,
        max_interp_gap=args.max_interp_gap,
    )
    visible_actor_count = render_info["visible_actor_count"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    # Reusing an output directory must not leave frames from an earlier,
    # longer render (for example, frame_016.png through frame_063.png).
    for stale_frame in frames_dir.glob("frame_*.png"):
        stale_frame.unlink()
    for index, image in enumerate(frames):
        image.save(frames_dir / ("frame_%03d.png" % index), optimize=True)
    frames[0].save(
        args.output_dir / "preview.gif",
        save_all=True,
        append_images=frames[1:],
        duration=args.gif_duration_ms,
        loop=0,
        optimize=False,
    )
    frames[len(frames) // 2].save(args.output_dir / "preview_middle.png", optimize=True)

    metadata = {
        "source": "synthetic_demo" if args.demo else str(args.data_path.resolve()),
        "split": None if args.demo else args.split,
        "sample_index": None if args.demo else args.sample_index,
        "labels_read": False,
        "input_stage": "processed_npz_before_feeder_augmentation",
        "num_valid_frames": render_info["num_valid_frames"],
        "num_rendered_frames": render_info["num_rendered_frames"],
        "source_frame_indices": render_info["source_frame_indices"],
        "layout": [
            "front_xy_root_centered",
            "side_zy_root_centered",
            "top_xz_root_centered",
        ],
        "root_centering": "subtract primary actor NTU joint 2 (SpineMid) independently at every frame",
        "temporal_smoothing": render_info["temporal_smoothing"],
        "global_translation_available": False,
        "person_colors": {"person_1": "red", "person_2": "blue"},
        "actor_count_rule": "one if no blue skeleton is rendered; two if a blue skeleton is rendered",
        "visible_actor_count": visible_actor_count,
        "blue_skeleton_visible": render_info["blue_skeleton_visible"],
        "active_rendered_frames_per_person": render_info["active_rendered_frames_per_person"],
        "bone_width_px": BONE_WIDTH,
        "joint_radius_px": JOINT_RADIUS,
        "render_size": [args.width, args.height],
    }
    (args.output_dir / "render_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
