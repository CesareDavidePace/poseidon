#!/usr/bin/env python
"""Visualize Poseidon's learned Adaptive Frame Weighting coefficients.

The figure uses a zero-based, fixed y-axis and shows the uniform 1/T reference
so that small differences between weights are not visually exaggerated. A CSV
with the exact values is written next to the figure.
"""

import argparse
import csv
import os
import sys
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.zoo.posetrack.PoseTrack import PoseTrack
from datasets.transforms.build import reverse_transforms
from models.best.Poseidon import Poseidon
from posetimation import get_cfg, update_config
from utils.common import VAL_PHASE


def build_cfg(cfg_path, root_dir, frames=None, data_root=None):
    args = types.SimpleNamespace(
        cfg=os.path.abspath(cfg_path),
        rootDir=os.path.abspath(root_dir),
    )
    cfg = get_cfg(args)
    update_config(cfg, args)
    cfg.defrost()
    cfg.MODEL.PRETRAINED = ""
    if frames:
        cfg.WINDOWS_SIZE = int(frames)
    if data_root:
        cfg.DATASET.JSON_DIR = os.path.join(data_root, "jsons")
        cfg.DATASET.IMG_DIR = data_root
        cfg.DATASET.TEST_IMG_DIR = data_root
        cfg.VAL.ANNOT_DIR = os.path.join(data_root, "annotations", "val")
        cfg.VAL.COCO_BBOX_FILE = os.path.join(
            data_root, "detections", "detections_val.json"
        )
    cfg.freeze()
    return cfg


def canonical_path(path, data_root):
    path = os.path.normpath(path)
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(data_root, path))


def choose_instance(records, requested_path, data_root):
    """Choose the largest annotated person when an image has multiple people."""
    target = canonical_path(requested_path, data_root)
    candidates = []
    for idx, record in enumerate(records):
        image = canonical_path(record["image"], data_root)
        if image != target:
            continue
        scale = np.asarray(record["scale"], dtype=np.float32)
        candidates.append((float(np.prod(scale)), idx))
    if not candidates:
        raise ValueError(f"No validation record found for image: {requested_path}")
    candidates.sort(reverse=True)
    return candidates[0][1]


def crop_to_rgb(frame):
    restored = reverse_transforms(frame.unsqueeze(0).cpu())[0]
    image = restored.permute(1, 2, 0).numpy()
    return np.clip(image[..., ::-1] / 255.0, 0.0, 1.0)


def plot_cases(cases, out_path):
    num_cases = len(cases)
    num_frames = len(cases[0]["weights"])
    center_idx = num_frames // 2
    offsets = np.arange(num_frames) - center_idx
    uniform_weight = 1.0 / num_frames

    fig = plt.figure(figsize=(13.2, 2.75 * num_cases))
    outer = fig.add_gridspec(
        num_cases,
        2,
        width_ratios=(3.7, 1.8),
        hspace=0.48,
        wspace=0.22,
    )

    for row, case in enumerate(cases):
        frames_grid = outer[row, 0].subgridspec(1, num_frames, wspace=0.025)
        for col, (frame, offset, weight) in enumerate(
            zip(case["frames"], offsets, case["weights"])
        ):
            ax = fig.add_subplot(frames_grid[0, col])
            ax.imshow(frame)
            ax.set_xticks([])
            ax.set_yticks([])
            is_center = col == center_idx
            for spine in ax.spines.values():
                spine.set_color("#2563a6" if is_center else "#555555")
                spine.set_linewidth(3.0 if is_center else 1.2)
            ax.set_title(
                rf"$t{offset:+d}$" + f"\nAFW={weight:.4f}",
                fontsize=8.5,
            )
            if col == 0:
                ax.set_ylabel(case["label"], fontsize=10, fontweight="bold")

        ax = fig.add_subplot(outer[row, 1])
        colors = [
            "#2563a6" if frame_idx == center_idx else "#7aa6c2"
            for frame_idx in range(num_frames)
        ]
        ax.bar(np.arange(num_frames), case["weights"], color=colors, width=0.72)
        ax.axhline(
            uniform_weight,
            color="#b33a3a",
            linestyle="--",
            linewidth=1.2,
            label=rf"uniform $1/T={uniform_weight:.1f}$",
        )
        ax.set_ylim(0.0, 0.25)
        ax.set_xticks(np.arange(num_frames))
        ax.set_xticklabels([rf"$t{offset:+d}$" for offset in offsets])
        ax.set_ylabel("AFW coefficient")
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title(
            f"max deviation from uniform: {case['max_deviation']:.4f}",
            fontsize=9.5,
        )

    fig.suptitle(
        "Learned Adaptive Frame Weighting coefficients (absolute scale)",
        fontsize=13,
    )
    fig.text(
        0.77,
        0.012,
        "All bar charts use the same zero-based y-axis; values sum to one.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.055, right=0.99, top=0.93, bottom=0.06)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_csv(cases, out_path):
    csv_path = os.path.splitext(out_path)[0] + ".csv"
    num_frames = len(cases[0]["weights"])
    offsets = np.arange(num_frames) - num_frames // 2
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case",
                "image",
                "temporal_offset",
                "afw_coefficient",
                "uniform_reference",
                "deviation_from_uniform",
            ]
        )
        for case in cases:
            uniform_weight = 1.0 / len(case["weights"])
            for offset, weight in zip(offsets, case["weights"]):
                writer.writerow(
                    [
                        case["label"],
                        case["image"],
                        int(offset),
                        f"{weight:.8f}",
                        f"{uniform_weight:.8f}",
                        f"{weight - uniform_weight:+.8f}",
                    ]
                )
    return csv_path


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-root", default="/ssd2/Pace/Posetrack/data")
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Optional row labels; must match the number of selected images.",
    )
    parser.add_argument(
        "--out",
        default="paper/images/appendix_frame_weights.png",
    )
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.images):
        parser.error("--labels must contain one label for every --images entry")
    if args.frames < 3 or args.frames % 2 == 0:
        parser.error("--frames must be an odd integer of at least 3")

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    cfg = build_cfg(
        args.config,
        args.root_dir,
        frames=args.frames,
        data_root=args.data_root,
    )
    dataset = PoseTrack(cfg, phase=VAL_PHASE)
    selected = [
        choose_instance(dataset.data, image, args.data_root)
        for image in args.images
    ]
    print("Selected dataset indices:", selected)

    model = Poseidon(cfg, device=device, phase=VAL_PHASE).to(device).eval()
    checkpoint = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)

    cases = []
    for case_number, (requested_image, dataset_idx) in enumerate(
        zip(args.images, selected), start=1
    ):
        x, _, _, _ = dataset[dataset_idx]
        x = x.unsqueeze(0).to(device)
        _, frame_weights = model(x, return_weights=True)
        weights = frame_weights.detach().float().cpu().reshape(-1).numpy()
        weights = weights / weights.sum()
        uniform_weight = 1.0 / len(weights)
        case = {
            "label": (
                args.labels[case_number - 1]
                if args.labels
                else f"Case {case_number}"
            ),
            "image": canonical_path(requested_image, args.data_root),
            "frames": [
                crop_to_rgb(x[0, frame_idx]) for frame_idx in range(x.shape[1])
            ],
            "weights": weights,
            "max_deviation": float(np.max(np.abs(weights - uniform_weight))),
        }
        cases.append(case)
        print(
            f"Case {case_number}: {os.path.basename(requested_image)} -> ["
            + ", ".join(f"{weight:.6f}" for weight in weights)
            + f"], max |w-1/T|={case['max_deviation']:.6f}"
        )

    plot_cases(cases, args.out)
    csv_path = write_csv(cases, args.out)
    print(f"Saved figure: {args.out}")
    print(f"Saved numerical weights: {csv_path}")


if __name__ == "__main__":
    main()
