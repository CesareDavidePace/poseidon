#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poseidon multi-frame inference 
------------------------------
"""
import argparse
import os
import random

import cv2
import numpy as np
import torch
import yaml
from torchvision import transforms
from ultralytics import YOLO
import json
from models.best.Poseidon import Poseidon
from datasets.zoo.posetrack.pose_skeleton import (
    PoseTrack_Official_Keypoint_Ordering,
    PoseTrack_Keypoint_Pairs,
)


# ─────────────────────── Utility ───────────────────────
MAPPED_KP_NAMES = [
    'nose',
    'right_eye',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
]
USED_KP_IDX = [0, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
USED_KP_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (192, 192, 192), (128, 0, 128),
    (255, 165, 0), (128, 128, 0), (0, 128, 128), (75, 0, 130),
    (255, 105, 180), (0, 191, 255), (255, 223, 0), (165, 42, 42),
    (34, 139, 34)
]
COCO_ORDER = [
    0,  # nose  (missing → 0,0,0)
    1,  # left_eye   ↓
    2,  # right_eye  ↓      these three indices are *absent*
    3,  # left_ear   ↓      from PoseTrack-minus-eyes so they
    4,  # right_ear  ↓      will be padded.
    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16   # same as COCO
]

def to_coco(keypoints14: np.ndarray, conf: float = 2.0) -> list[int]:
    """
    Map the 14 predicted (x,y) pairs to a 17-*3* COCO list.
    The three missing keypoints are zero-filled.
    Visibility is always `conf` (2 = "visible") for predicted points.
    """
    coco = [0]*17*3
    # indices 5–16 in COCO are exactly your USED_KP_IDX[1:]
    available = {
        5:0, 6:1, 7:2, 8:3, 9:4, 10:5, 11:6, 12:7,
        13:8, 14:9, 15:10, 16:11
    }
    for coco_id, src_pos in available.items():
        x, y = keypoints14[src_pos]
        off = coco_id*3
        coco[off:off+3] = [float(x), float(y), conf]
    return coco

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    p = argparse.ArgumentParser("Poseidon multi-frame inference")
    p.add_argument("-c", "--config", required=True, help="path to YAML config")
    p.add_argument("-w", "--weights", required=True, help=".pt checkpoint")
    p.add_argument("-i", "--video_in", required=True, help="input video path")
    p.add_argument("-o", "--video_out", default="output.mp4",
                   help="where to write annotated video")
    p.add_argument("-n", "--window", type=int, default=5,
                   help="number of frames per inference window")
    p.add_argument("-s", "--step", type=int, default=1,
                   help="frame stride between samples")
    p.add_argument("--coco_json", default="sample/predictions.json",
               help="file to store COCO-format results")
    p.add_argument("-g", "--gpu", type=int, default=0, help="CUDA device index")
    return p.parse_args()


# ─────────────────────── Config ───────────────────────
def load_cfg(path: str):
    data = yaml.safe_load(open(path, "r"))

    class C:
        pass
    cfg = C()
    cfg.SEED = data.get("SEED", 42)

    cfg.MODEL = C()
    cfg.MODEL.METHOD = data["MODEL"]["METHOD"]
    cfg.MODEL.NUM_JOINTS = data["MODEL"]["NUM_JOINTS"]
    cfg.MODEL.IMAGE_SIZE = tuple(data["MODEL"]["IMAGE_SIZE"])
    cfg.MODEL.CONFIG_FILE = data["MODEL"].get("CONFIG_FILE")
    cfg.MODEL.CHECKPOINT_FILE = data["MODEL"].get("CHECKPOINT_FILE")
    cfg.MODEL.EMBED_DIM = data["MODEL"].get("EMBED_DIM", 256)
    cfg.WINDOWS_SIZE = data["MODEL"].get("WINDOWS_SIZE", 5)
    cfg.MODEL.HEATMAP_SIZE = data["MODEL"].get("HEATMAP_SIZE", (96, 72))
    cfg.MODEL.FREEZE_WEIGHTS = data["MODEL"].get("FREEZE_WEIGHTS", False)

    cfg.DATASET = C()
    cfg.DATASET.BBOX_ENLARGE_FACTOR = data["DATASET"].get(
        "BBOX_ENLARGE_FACTOR", 1.25)
    return cfg


# ─────────────────────── Video IO ───────────────────────
def make_writer(path: str, fps: float, size: tuple[int, int], is_color=True):
    """Return a cv2.VideoWriter whose codec matches the container."""
    ext = os.path.splitext(path)[1].lower()
    if ext in {".mp4", ".m4v"}:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")   # very safe
    elif ext in {".avi"}:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    elif ext in {".mov"}:
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
    else:
        raise ValueError(f"Unsupported video container: “{ext}”")

    writer = cv2.VideoWriter(path, fourcc, fps, size, is_color)
    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open VideoWriter for {path!r}. "
            "Your OpenCV build may lack the requested codec."
        )
    return writer


def preprocess_frame(frame, size):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, size)
    # These are the mean and std deviation of the ImageNet dataset.
    # It's a common practice to normalize images with these values when using
    # models pretrained on ImageNet.
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return tfm(frame)


def extract_kps(heatmaps, h_crop, w_crop):
    _, _, H, W = heatmaps.shape
    hm = heatmaps[0, USED_KP_IDX].view(len(USED_KP_IDX), -1)
    maxv, idx = hm.max(dim=1)
    ys = (idx // W).float() * (h_crop / H)
    xs = (idx % W).float() * (w_crop / W)
    return torch.stack([xs, ys], dim=1).cpu().numpy()


# ─────────────────────── Main loop ───────────────────────
def process_video(model, detector, device, cfg, args):
    cap = cv2.VideoCapture(args.video_in)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video {args.video_in!r}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W_img = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H_img = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if W_img == 0 or H_img == 0:
            raise RuntimeError("Failed to read frame dimensions from input video")

        out = make_writer(args.video_out, fps, (W_img, H_img))

        annotations = []
        image_id = 0
        ann_id   = 0
        buf: list[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            buf.append(frame)

            if len(buf) < args.window * args.step:
                continue

            sampled = buf[::args.step]
            center = sampled[len(sampled) // 2].copy()

            # ── Human detection on the centre frame ──
            dets = detector.predict(center, verbose=False)[0]
            for box in dets.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                w = (x2 - x1) * cfg.DATASET.BBOX_ENLARGE_FACTOR
                h = (y2 - y1) * cfg.DATASET.BBOX_ENLARGE_FACTOR

                x1c, y1c = int(max(cx - w / 2, 0)), int(max(cy - h / 2, 0))
                x2c, y2c = int(min(cx + w / 2, W_img)), int(min(cy + h / 2, H_img))
                w_crop, h_crop = x2c - x1c, y2c - y1c

                crops = [
                    preprocess_frame(f[y1c:y2c, x1c:x2c], cfg.MODEL.IMAGE_SIZE)
                    for f in sampled
                ]
                inp = torch.stack(crops).unsqueeze(0).to(device)

                with torch.no_grad():
                    hm = model(inp)

                kps = extract_kps(hm, h_crop, w_crop)

                # Create a mapping from keypoint names to their coordinates
                kp_map = {
                    name: (int(kps[i][0]) + x1c, int(kps[i][1]) + y1c)
                    for i, name in enumerate(MAPPED_KP_NAMES)
                }
                if 'nose' in kp_map:
                    kp_map['head_bottom'] = kp_map['nose']

                # Draw circles for each keypoint
                for i, (px, py) in enumerate(kps):
                    color = USED_KP_COLORS[i]
                    cv2.circle(center, (int(px) + x1c, int(py) + y1c), 2, color, -1)

                # Draw lines to connect the keypoints
                for kp1_name, kp2_name, _ in PoseTrack_Keypoint_Pairs:
                    if kp1_name in kp_map and kp2_name in kp_map:
                        pt1 = kp_map[kp1_name]
                        pt2 = kp_map[kp2_name]
                        cv2.line(center, pt1, pt2, (0, 255, 0), 1)
                
                # update json annotations
                coco_keypoints = to_coco(kps)
                # bbox = [x1, y1, width, height] as required by COCO
                bbox = [float(x1c), float(y1c), float(w_crop), float(h_crop)]

                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,           # person
                    "keypoints": coco_keypoints,
                    "num_keypoints": 14,
                    "bbox": bbox,
                    "area": float(w_crop*h_crop),
                    "iscrowd": 0
                })
                ann_id += 1

            # ── ensure frame size is still what we promised ──
            assert center.shape[0] == H_img and center.shape[1] == W_img

            out.write(center)
            image_id += 1
            buf = buf[args.step:]

        coco_dict = {
            "info": {"description": "Poseidon predictions"},
            "images": [
                {"id": i, "file_name": f"frame_{i:06d}.jpg"}
                for i in range(image_id)
            ],
            "annotations": annotations,
            "categories": [
                {
                    "id": 1,
                    "name": "person",
                    "keypoints": [
                        "nose","left_eye","right_eye","left_ear","right_ear",
                        "left_shoulder","right_shoulder","left_elbow","right_elbow",
                        "left_wrist","right_wrist","left_hip","right_hip",
                        "left_knee","right_knee","left_ankle","right_ankle"
                    ],
                    "skeleton": [
                        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                        [6, 8], [8, 10], [7, 9], [9, 11], [6, 7], [6, 12],
                        [7, 13], [12, 13]
                    ]
                }
            ]
        }
        with open(args.coco_json, "w") as f:
            json.dump(coco_dict, f, indent=4)
        print(f"✔ COCO file saved to {args.coco_json}")
        
    finally:
        cap.release()
        if 'out' in locals():
            out.release()
        cv2.destroyAllWindows()

    print(f"✔ Finished writing annotated video to “{args.video_out}”")
    
# ─────────────────────── Entrypoint ───────────────────────
def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    set_seed(cfg.SEED)
    
    # based on the args output create the output directory
    os.makedirs(os.path.dirname(args.video_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.coco_json), exist_ok=True)

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available() and args.gpu >= 0:
        device = f"cuda:{args.gpu}"
    else:
        device = "cpu"

    print("→ Using device:", device)

    model = Poseidon(cfg, phase="test", device=device, inference=True)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    detector = YOLO("yolov8s-pose.pt")  # or your own weights
    process_video(model, detector, device, cfg, args)
    print("✓ Done!")
    
    
if __name__ == "__main__":
    main()
