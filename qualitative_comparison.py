#!/usr/bin/env python
"""Create reproducible per-frame qualitative comparisons for Appendix B.

Each case is evaluated independently (batch size one). The left panel uses the
COCO-pretrained ViTPose backbone/head on the centre frame; the right panel uses
the trained temporal Poseidon checkpoint. This is a qualitative comparison
because the two models have different training, while the controlled
single-frame comparison remains the paper's ablation.
"""

import argparse
import os
import sys
import types

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.process.heatmaps_process import get_max_preds
from datasets.transforms.build import reverse_transforms
from datasets.zoo.posetrack.PoseTrack import PoseTrack
from models.best.Poseidon import Poseidon
from mmpose.apis import init_model
from posetimation import get_cfg, update_config
from utils.common import VAL_PHASE


JOINT_NAMES = [
    "nose", "left eye", "right eye", "left ear", "right ear",
    "left shoulder", "right shoulder", "left elbow", "right elbow",
    "left wrist", "right wrist", "left hip", "right hip",
    "left knee", "right knee", "left ankle", "right ankle",
]
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
PRED_COLOR = (0, 165, 255)
GT_COLOR = (0, 255, 0)
SELECTED_COLOR = (40, 40, 230)


def build_cfg(cfg_path, root_dir, frames, data_root):
    args = types.SimpleNamespace(
        cfg=os.path.abspath(cfg_path),
        rootDir=os.path.abspath(root_dir),
    )
    cfg = get_cfg(args)
    update_config(cfg, args)
    cfg.defrost()
    cfg.MODEL.PRETRAINED = ""
    cfg.WINDOWS_SIZE = int(frames)
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
    target = canonical_path(requested_path, data_root)
    candidates = []
    for idx, record in enumerate(records):
        if canonical_path(record["image"], data_root) != target:
            continue
        scale = np.asarray(record["scale"], dtype=np.float32)
        candidates.append((float(np.prod(scale)), idx))
    if not candidates:
        raise ValueError(f"No validation record found for image: {requested_path}")
    candidates.sort(reverse=True)
    return candidates[0][1]


def decode_crop_coords(heatmaps, crop_w, crop_h):
    coords, _ = get_max_preds(heatmaps.detach().float().cpu().numpy())
    hm_h, hm_w = heatmaps.shape[-2:]
    coords[..., 0] *= crop_w / hm_w
    coords[..., 1] *= crop_h / hm_h
    return coords[0]


def crop_to_bgr(frame):
    restored = reverse_transforms(frame.unsqueeze(0).cpu())[0]
    return (
        restored.permute(1, 2, 0).numpy().clip(0, 255).astype(np.uint8)
    )


def draw_pose(image, keypoints, visible, ground_truth, joint_idx):
    canvas = image.copy()
    for first, second in COCO_SKELETON:
        if visible[first] and visible[second]:
            cv2.line(
                canvas,
                tuple(np.round(keypoints[first]).astype(int)),
                tuple(np.round(keypoints[second]).astype(int)),
                PRED_COLOR,
                2,
                cv2.LINE_AA,
            )
    for idx, point in enumerate(keypoints):
        if visible[idx]:
            cv2.circle(
                canvas,
                tuple(np.round(point).astype(int)),
                3,
                PRED_COLOR,
                -1,
                cv2.LINE_AA,
            )
    for idx, point in enumerate(ground_truth):
        if visible[idx]:
            cv2.circle(
                canvas,
                tuple(np.round(point).astype(int)),
                5,
                GT_COLOR,
                1,
                cv2.LINE_AA,
            )
    cv2.circle(
        canvas,
        tuple(np.round(keypoints[joint_idx]).astype(int)),
        7,
        SELECTED_COLOR,
        2,
        cv2.LINE_AA,
    )
    return canvas


def add_label(image, text):
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        text,
        (7, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


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
    parser.add_argument("--joints", nargs="+", type=int, required=True)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument(
        "--out", default="paper/images/appendix_comparison.png"
    )
    args = parser.parse_args()
    if len(args.images) != len(args.joints):
        parser.error("--joints must contain one index per image")
    if args.labels and len(args.labels) != len(args.images):
        parser.error("--labels must contain one label per image")

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    cfg = build_cfg(args.config, args.root_dir, args.frames, args.data_root)
    dataset = PoseTrack(cfg, phase=VAL_PHASE)
    selected = [
        choose_instance(dataset.data, image, args.data_root)
        for image in args.images
    ]

    model = Poseidon(cfg, device=device, phase=VAL_PHASE).to(device).eval()
    checkpoint = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    single_frame_model = init_model(
        cfg.MODEL.CONFIG_FILE, cfg.MODEL.CHECKPOINT_FILE, device=device
    ).eval()

    rows = []
    for case_idx, (image_path, dataset_idx, joint_idx) in enumerate(
        zip(args.images, selected, args.joints)
    ):
        x, meta, _, _ = dataset[dataset_idx]
        x = x.unsqueeze(0).to(device)
        center_idx = x.shape[1] // 2
        temporal_heatmaps = model(x)
        single_heatmaps = single_frame_model.head(
            single_frame_model.backbone(x[:, center_idx])
        )
        crop_h, crop_w = x.shape[-2:]
        temporal_keypoints = decode_crop_coords(
            temporal_heatmaps, crop_w, crop_h
        )
        single_keypoints = decode_crop_coords(single_heatmaps, crop_w, crop_h)
        ground_truth = np.asarray(meta["joints"], dtype=np.float32)[:, :2]
        visible = (
            np.asarray(meta["joints_vis"], dtype=np.float32)[:, 0] > 0
        )
        single_error = float(
            np.linalg.norm(single_keypoints[joint_idx] - ground_truth[joint_idx])
        )
        temporal_error = float(
            np.linalg.norm(temporal_keypoints[joint_idx] - ground_truth[joint_idx])
        )
        crop = crop_to_bgr(x[0, center_idx])
        label = (
            args.labels[case_idx]
            if args.labels
            else f"{JOINT_NAMES[joint_idx]} case"
        )
        left = add_label(
            draw_pose(
                crop, single_keypoints, visible, ground_truth, joint_idx
            ),
            f"{label} | ViTPose | {single_error:.1f}px",
        )
        right = add_label(
            draw_pose(
                crop, temporal_keypoints, visible, ground_truth, joint_idx
            ),
            f"Poseidon | {temporal_error:.1f}px",
        )
        separator = np.full((crop_h, 5, 3), 255, dtype=np.uint8)
        rows.append(cv2.hconcat([left, separator, right]))
        print(
            f"{os.path.basename(image_path)} {JOINT_NAMES[joint_idx]}: "
            f"ViTPose={single_error:.3f}px, Poseidon={temporal_error:.3f}px"
        )

    row_width = max(row.shape[1] for row in rows)
    padded_rows = []
    for row in rows:
        if row.shape[1] < row_width:
            pad = np.full(
                (row.shape[0], row_width - row.shape[1], 3),
                255,
                dtype=np.uint8,
            )
            row = cv2.hconcat([row, pad])
        padded_rows.append(row)
    row_separator = np.full((5, row_width, 3), 255, dtype=np.uint8)
    combined = padded_rows[0]
    for row in padded_rows[1:]:
        combined = cv2.vconcat([combined, row_separator, row])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if not cv2.imwrite(args.out, combined):
        raise RuntimeError(f"Failed to save {args.out}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
