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

LEFT_JOINTS = {4, 5, 6, 7, 12, 13, 14, 15, 21, 22}
RIGHT_JOINTS = {8, 9, 10, 11, 16, 17, 18, 19, 23, 24}
ROOT_JOINT = 1  # NTU joint 2: SpineMid

BACKGROUND = (18, 20, 24)
PANEL_BACKGROUND = (25, 28, 34)
GRID = (62, 68, 78)
TEXT = (232, 235, 240)
MUTED_TEXT = (164, 170, 182)
ROOT_TRACE = (185, 112, 255)
PERSON_COLORS = (
    {"left": (255, 151, 67), "right": (62, 207, 232), "torso": (238, 240, 244)},
    {"left": (245, 106, 164), "right": (116, 221, 128), "torso": (190, 194, 204)},
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, help="MAMP NPZ, e.g. NTU60_XSub.npz")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--gif_duration_ms", type=int, default=180)
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
        frames.append(np.stack([pose, np.zeros_like(pose)], axis=0))
    return np.asarray(frames, dtype=np.float32)


def uniformly_sample(sequence, num_frames):
    indices = np.rint(np.linspace(0, len(sequence) - 1, num_frames)).astype(np.int64)
    return sequence[indices], indices


def active_joint_mask(sequence):
    return np.any(np.abs(sequence) > 1e-8, axis=-1)


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


def robust_xyz_bounds(sequence):
    mask = active_joint_mask(sequence)
    points = sequence[mask]
    lo = np.percentile(points, 1, axis=0)
    hi = np.percentile(points, 99, axis=0)
    center = (lo + hi) / 2
    span = float(max(np.max(hi - lo), 1e-3)) * 1.15
    return center, span


def projected(point, view, azimuth=-math.pi / 4, elevation=math.radians(20)):
    x, y, z = point
    if view.endswith("xy"):
        return np.array([x, y])
    if view.endswith("zy"):
        return np.array([z, y])
    if view == "world_xz":
        return np.array([x, z])
    horizontal = math.cos(azimuth) * x - math.sin(azimuth) * z
    depth_axis = math.sin(azimuth) * x + math.cos(azimuth) * z
    vertical = math.cos(elevation) * y + math.sin(elevation) * depth_axis
    return np.array([horizontal, vertical])


def joint_group(index):
    if index in LEFT_JOINTS:
        return "left"
    if index in RIGHT_JOINTS:
        return "right"
    return "torso"


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


def make_panels(width, height, world_center, world_span, local_center, local_span):
    cols, rows = 3, 2
    panels = []
    specs = (
        ("FRONT: X-Y WORLD", "world_xy", world_center, world_span),
        ("SIDE: Z-Y WORLD", "world_zy", world_center, world_span),
        ("TOP: X-Z WORLD", "world_xz", world_center, world_span),
        ("FRONT: ROOT-CENTERED", "local_xy", local_center, local_span),
        ("SIDE: ROOT-CENTERED", "local_zy", local_center, local_span),
        ("FIXED OBLIQUE 3D", "world_xyz", world_center, world_span),
    )
    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            box = (
                col * width // cols,
                row * height // rows,
                (col + 1) * width // cols,
                (row + 1) * height // rows,
            )
            panels.append(Panel(box, *specs[index]))
    return panels


def draw_grid(draw, panel):
    left, top, right, bottom = panel.box
    draw.rectangle((left + 2, top + 2, right - 2, bottom - 2), fill=PANEL_BACKGROUND, outline=GRID, width=1)
    for frac in (0.25, 0.5, 0.75):
        x = left + int((right - left) * frac)
        y = top + 34 + int((bottom - top - 34) * frac)
        draw.line((x, top + 34, x, bottom), fill=GRID, width=1)
        draw.line((left, y, right, y), fill=GRID, width=1)


def draw_dashed_polyline(draw, points, fill, width=3, dash=8):
    if len(points) < 2:
        return
    for a, b in zip(points[:-1], points[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        ux, uy = dx / length, dy / length
        position = 0.0
        visible = True
        while position < length:
            end = min(position + dash, length)
            if visible:
                draw.line((a[0] + ux * position, a[1] + uy * position,
                           a[0] + ux * end, a[1] + uy * end), fill=fill, width=width)
            visible = not visible
            position = end


def render_frame(sequence, roots, frame_index, panels, width, height, font):
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    world_pose = sequence[frame_index]
    local_pose = world_pose - roots[frame_index][None, None, :]
    active = active_joint_mask(sequence)[frame_index]

    for panel in panels:
        draw_grid(draw, panel)
        left, top, right, _ = panel.box
        draw.text((left + 10, top + 9), panel.title, font=font, fill=TEXT)
        pose = local_pose if panel.view.startswith("local") else world_pose

        if panel.view == "world_xz":
            trace = [panel.map_point(root) for root in roots[:frame_index + 1]]
            draw_dashed_polyline(draw, trace, ROOT_TRACE)

        for person in range(pose.shape[0]):
            colors = PERSON_COLORS[min(person, len(PERSON_COLORS) - 1)]
            for a, b in NTU_EDGES:
                if not active[person, a] or not active[person, b]:
                    continue
                group_a, group_b = joint_group(a), joint_group(b)
                group = group_a if group_a == group_b else (
                    "left" if "left" in (group_a, group_b) else
                    "right" if "right" in (group_a, group_b) else "torso"
                )
                draw.line((*panel.map_point(pose[person, a]), *panel.map_point(pose[person, b])),
                          fill=colors[group], width=5)
            for joint in range(25):
                if not active[person, joint]:
                    continue
                x, y = panel.map_point(pose[person, joint])
                radius = 5 if joint == ROOT_JOINT else 3
                color = ROOT_TRACE if joint == ROOT_JOINT else colors[joint_group(joint)]
                draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                             fill=color, outline=BACKGROUND, width=1)

        if panel.view.startswith("local"):
            cx, cy = panel.map_point(np.zeros(3, dtype=np.float32))
            draw.line((cx - 7, cy, cx + 7, cy), fill=ROOT_TRACE, width=2)
            draw.line((cx, cy - 7, cx, cy + 7), fill=ROOT_TRACE, width=2)

    draw.text((12, height - 22), "Purple: primary SpineMid / trajectory; no objects or scene are shown.",
              font=font, fill=MUTED_TEXT)
    return image


def main():
    args = parse_args()
    if args.num_frames < 2:
        raise ValueError("--num_frames must be at least 2")
    if args.width % 3 or args.height % 2:
        raise ValueError("--width must be divisible by 3 and --height by 2")

    source = make_demo_sample() if args.demo else load_npz_sample(args.data_path, args.split, args.sample_index)
    sampled, frame_indices = uniformly_sample(source, args.num_frames)
    roots = primary_root(sampled)
    world_center, world_span = robust_xyz_bounds(sampled)
    local = sampled - roots[:, None, None, :]
    local_center, local_span = robust_xyz_bounds(local)
    panels = make_panels(args.width, args.height, world_center, world_span, local_center, local_span)
    font = load_font(14)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(args.num_frames):
        image = render_frame(sampled, roots, index, panels, args.width, args.height, font)
        image.save(frames_dir / ("frame_%03d.png" % index), optimize=True)
        frames.append(image)
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
        "num_valid_frames": int(len(source)),
        "num_rendered_frames": int(args.num_frames),
        "source_frame_indices": frame_indices.tolist(),
        "layout": [
            "front_xy_world", "side_zy_world", "top_xz_world",
            "front_xy_root_centered", "side_zy_root_centered", "fixed_oblique_3d",
        ],
        "root_centering": "subtract primary actor NTU joint 2 (SpineMid) independently at every frame",
        "render_size": [args.width, args.height],
    }
    (args.output_dir / "render_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
