#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Temporal-consistency (trajectory / jitter) visualization for the Appendix (R2.2 / R4.5).

Per-frame mAP hides the main benefit of multi-frame modeling: temporal CONSISTENCY.
This script follows one person across consecutive frames of a video and plots, for a
chosen joint, its trajectory over time for:
    * per-frame     = independently applied COCO-pretrained ViTPose (default), or
                      optionally Poseidon with its temporal window collapsed
    * Poseidon      = the full temporal model
    * GT            = ground truth

Persons are associated across frames by nearest bounding-box center (PoseTrack GT records
carry no track id). For each candidate tracklet the script reports the body joint with the
largest mean error difference and saves: (a) the joint trajectory over time and (b) the
localization error over time. These are qualitative diagnostics, not a controlled
single-frame ablation. Figures go to --out.

Example
-------
  /home/pace/anaconda3/envs/poseidon/bin/python trajectory_viz.py \
      --config configs/posetrack21/configPoseidonVitH.yaml \
      --weights /ssd2/Pace/poseidon/poseidon/best_results/VITH/vith_model.pt \
      --gpu 1 --num-tracklets 8 --min-len 12 --out output_trajectories
"""

import os
import sys
import argparse
import types
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmpose.apis import init_model
from models.best.Poseidon import Poseidon
from datasets.zoo.posetrack.PoseTrack import PoseTrack
from datasets.process.heatmaps_process import get_final_preds
from posetimation import get_cfg, update_config
from utils.common import VAL_PHASE

JOINT_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_sho", "r_sho", "l_elb",
               "r_elb", "l_wri", "r_wri", "l_hip", "r_hip", "l_kne", "r_kne", "l_ank", "r_ank"]


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


def frame_num(path):
    return int(os.path.splitext(os.path.basename(path))[0])


def build_tracklets(records, min_len):
    """Greedy nearest-center association within each video. Returns list of tracklets,
    each a list of (frame_num, dataset_idx, center[2], joints_3d[J,3], joints_vis[J,3])."""
    by_video = defaultdict(list)
    for idx, r in enumerate(records):
        vid = os.path.dirname(r["image"])
        by_video[vid].append((frame_num(r["image"]), idx, np.asarray(r["center"], np.float32),
                              np.asarray(r["joints_3d"], np.float32),
                              np.asarray(r["joints_3d_vis"], np.float32),
                              np.asarray(r["scale"], np.float32)))
    tracklets = []
    for vid, items in by_video.items():
        items.sort(key=lambda t: t[0])
        frames = defaultdict(list)
        for it in items:
            frames[it[0]].append(it)
        ordered_fr = sorted(frames.keys())
        active = []  # each: {"pts":[items], "last_center":c, "last_frame":f}
        for fr in ordered_fr:
            persons = frames[fr]
            used = [False] * len(persons)
            for tr in active:
                thr = 0.5 * float(np.mean(tr["last_scale"])) * 200.0 + 20.0
                best, bd = -1, 1e9
                for pi, p in enumerate(persons):
                    if used[pi]:
                        continue
                    d = np.linalg.norm(p[2] - tr["last_center"])
                    if d < bd:
                        bd, best = d, pi
                if best >= 0 and bd < thr and (fr - tr["last_frame"]) <= 2:
                    used[best] = True
                    tr["pts"].append(persons[best])
                    tr["last_center"] = persons[best][2]
                    tr["last_scale"] = persons[best][5]
                    tr["last_frame"] = fr
            for pi, p in enumerate(persons):
                if not used[pi]:
                    active.append({"pts": [p], "last_center": p[2], "last_scale": p[5],
                                   "last_frame": fr})
        for tr in active:
            if len(tr["pts"]) >= min_len:
                tracklets.append(tr["pts"])
    tracklets.sort(key=len, reverse=True)
    return tracklets


@torch.no_grad()
def run_tracklet(model, dataset, tracklet, device, base_model=None):
    """Returns dict joint->(frames, gt[N,2], single[N,2], temporal[N,2], vis[N]).
    single-frame prediction: independent per-frame base ViTPose (base_model) if given,
    otherwise Poseidon with the window collapsed to the center frame (fair, same weights)."""
    idxs = [t[1] for t in tracklet]
    dl = DataLoader(Subset(dataset, idxs), batch_size=1, shuffle=False, num_workers=4)
    frames, gt, sing, temp, vis = [], [], [], [], []
    for (x, meta, _, _), t in zip(dl, tracklet):
        x = x.to(device, non_blocking=True)
        T = x.shape[1]; c = T // 2
        out_t = model(x)
        if base_model is not None:
            out_b = base_model.head(base_model.backbone(x[:, c]))  # independent per-frame ViTPose
        else:
            out_b = model(x[:, c:c + 1].expand(-1, T, -1, -1, -1).contiguous())
        center = meta["center"].numpy(); scale = meta["scale"].numpy()
        pt, _ = get_final_preds(out_t.float().cpu().numpy(), center, scale)  # orig coords
        pb, _ = get_final_preds(out_b.float().cpu().numpy(), center, scale)
        frames.append(t[0]); gt.append(t[3][:, :2]); vis.append(t[4][:, 0] > 0)
        temp.append(pt[0]); sing.append(pb[0])
    return {"frames": np.array(frames), "gt": np.array(gt), "single": np.array(sing),
            "temporal": np.array(temp), "vis": np.array(vis)}


# exclude head keypoints (nose/eyes/ears): COCO vs PoseTrack define these differently,
# so single-frame vs temporal differences there reflect keypoint definition, not jitter.
BODY_JOINTS = list(range(5, 17))  # shoulders..ankles


def pick_joint(res):
    """Body joint with the largest single->temporal error reduction (visible in >=60% frames)."""
    N, J = res["vis"].shape
    best_j, best_score = None, -1e9
    for j in BODY_JOINTS:
        m = res["vis"][:, j]
        if m.sum() < max(4, 0.6 * N):
            continue
        e_s = np.linalg.norm(res["single"][m, j] - res["gt"][m, j], axis=1)
        e_t = np.linalg.norm(res["temporal"][m, j] - res["gt"][m, j], axis=1)
        score = e_s.mean() - e_t.mean()
        if score > best_score:
            best_score, best_j = score, j
    return best_j, best_score


def plot_tracklet(res, j, out_png, title):
    m = res["vis"][:, j]
    fr = res["frames"]
    gt, sing, temp = res["gt"][:, j], res["single"][:, j], res["temporal"][:, j]
    e_s = np.full(len(fr), np.nan); e_t = np.full(len(fr), np.nan)
    e_s[m] = np.linalg.norm(sing[m] - gt[m], axis=1)
    e_t[m] = np.linalg.norm(temp[m] - gt[m], axis=1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    # (a) vertical coordinate over time (jitter view)
    sflabel = "ViTPose (per-frame)"
    ax[0].plot(fr[m], gt[m, 1], "-", color="black", lw=2, label="GT")
    ax[0].plot(fr, sing[:, 1], "--o", color="#d1495b", ms=3, lw=1.2, label=sflabel)
    ax[0].plot(fr, temp[:, 1], "-s", color="#2e6f95", ms=3, lw=1.5, label="Poseidon (temporal)")
    ax[0].set_xlabel("frame"); ax[0].set_ylabel(f"{JOINT_NAMES[j]} y-coord (px)")
    ax[0].set_title("(a) Joint trajectory"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    # (b) localization error over time
    ax[1].plot(fr, e_s, "--o", color="#d1495b", ms=3, lw=1.2, label=f"ViTPose per-frame (mean {np.nanmean(e_s):.1f}px)")
    ax[1].plot(fr, e_t, "-s", color="#2e6f95", ms=3, lw=1.5, label=f"Poseidon (mean {np.nanmean(e_t):.1f}px)")
    ax[1].set_xlabel("frame"); ax[1].set_ylabel("L2 error to GT (px)")
    ax[1].set_title("(b) Localization error"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--root-dir", default=".")
    ap.add_argument("--data-root", default="/ssd2/Pace/Posetrack/data")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--min-len", type=int, default=12, help="min tracklet length (frames)")
    ap.add_argument("--num-tracklets", type=int, default=8, help="how many top tracklets to render")
    ap.add_argument("--scan-tracklets", type=int, default=40, help="how many long tracklets to evaluate")
    ap.add_argument("--baseline", choices=["vitpose", "replicate"], default="vitpose",
                    help="single-frame reference: 'vitpose' = independent per-frame base ViTPose "
                         "(shows temporal jitter; domain-gap caveat); 'replicate' = same-weights, "
                         "context collapsed to center frame (fair, but nearly identical to Poseidon)")
    ap.add_argument("--out", default="output_trajectories")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

    cfg = build_cfg(args.config, args.root_dir, frames=args.frames, data_root=args.data_root)
    print("Building model...")
    model = Poseidon(cfg, device=device, phase=VAL_PHASE).to(device).eval()
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)

    base_model = None
    if args.baseline == "vitpose":
        print("Building base per-frame ViTPose (single-frame reference)...")
        base_model = init_model(cfg.MODEL.CONFIG_FILE, cfg.MODEL.CHECKPOINT_FILE, device=device).eval()

    dataset = PoseTrack(cfg, phase=VAL_PHASE)
    print("Building tracklets...")
    tracklets = build_tracklets(dataset.data, args.min_len)
    print(f"Found {len(tracklets)} tracklets (len>={args.min_len}). Evaluating top {args.scan_tracklets}...")

    scored = []
    for k, tr in enumerate(tracklets[:args.scan_tracklets]):
        res = run_tracklet(model, dataset, tr, device, base_model=base_model)
        j, score = pick_joint(res)
        if j is None:
            continue
        scored.append((score, k, j, res))
        if k % 5 == 0:
            print(f"  evaluated {k+1}/{min(args.scan_tracklets, len(tracklets))}")
    scored.sort(key=lambda s: -s[0])

    for rank, (score, k, j, res) in enumerate(scored[:args.num_tracklets]):
        vid = os.path.basename(os.path.dirname(dataset.data[tracklets[k][0][1]]["image"]))
        title = (f"Video {vid} | joint={JOINT_NAMES[j]} | len={len(res['frames'])} | "
                 f"mean err: single {np.nanmean(np.linalg.norm((res['single']-res['gt'])[res['vis']], axis=1)):.1f}px "
                 f"-> temporal {np.nanmean(np.linalg.norm((res['temporal']-res['gt'])[res['vis']], axis=1)):.1f}px")
        out_png = os.path.join(args.out, f"traj_{rank:02d}_{vid}_{JOINT_NAMES[j]}_gain{score:.1f}.png")
        plot_tracklet(res, j, out_png, title)
        print(f"  saved {out_png}")

    print(f"\nDone. {len(scored[:args.num_tracklets])} trajectory figures in {args.out}/")


if __name__ == "__main__":
    main()
