#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Efficiency benchmark for Poseidon (response to reviewers R2.1 / R2.3 / R4.4).

Measures, for each ViTPose backbone (S/B/L/H) and temporal window T (# frames):
  - #Params (total, backbone, added temporal modules)
  - GFLOPs   (fvcore, with thop fallback)
  - Latency  (ms per sequence)  and  FPS (sequences/s)  -- CUDA warmup + synchronize
  - Peak GPU memory (GB)        -- torch.cuda.max_memory_allocated

Notes
-----
* FLOPs/latency/memory depend ONLY on the architecture, not on trained weights,
  so this does not need the trained Poseidon checkpoints. By default it loads the
  base ViTPose weights referenced in each config (they are already in models/vitpose/);
  use --random-weights to skip loading them (identical FLOPs/timing).
* nn.MultiheadAttention is partially opaque to FLOP counters; attention FLOPs may be
  slightly under-counted. This is standard and noted in the paper table caption.
* One "sequence" = one forward pass that produces the pose for one center frame.

Examples
--------
  # Full sweep (all backbones x T in {3,5,7}) on GPU 0, save markdown + csv:
  python benchmark_efficiency.py --gpu 0 --out results/efficiency

  # Only ViT-H, only the reported T=5, mixed precision like training:
  python benchmark_efficiency.py --backbones H --frames 5 --amp --gpu 0
"""

import os
import sys
import time
import argparse
import warnings

import torch

# make sure repo root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from posetimation.config.defaults import _C
from models.best.Poseidon import Poseidon


CONFIG_BY_BACKBONE = {
    "S": "configs/posetrack21/configPoseidonVitS.yaml",
    "B": "configs/posetrack21/configPoseidonVitB.yaml",
    "L": "configs/posetrack21/configPoseidonVitL.yaml",
    "H": "configs/posetrack21/configPoseidonVitH.yaml",
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def build_cfg(yaml_path, num_frames, random_weights):
    cfg = _C.clone()
    cfg.defrost()
    cfg.merge_from_file(yaml_path)
    cfg.WINDOWS_SIZE = int(num_frames)
    if random_weights:
        # skip loading the (large) base ViTPose checkpoint; architecture is unchanged
        cfg.MODEL.CHECKPOINT_FILE = None
    cfg.freeze()
    return cfg


def count_params(model):
    """Return (total_M, backbone_M, added_modules_M)."""
    total = sum(p.numel() for p in model.parameters())
    backbone = sum(p.numel() for p in model.backbone.parameters())
    # temporal modules introduced by Poseidon on top of the ViTPose backbone+head
    added_mods = [
        model.feature_fusion,
        model.adaptive_weighting,
        model.cross_attention,
        model.self_attention,
        model.intermediate_layer_norms,
        model.layer_norm,
    ]
    added = 0
    for m in added_mods:
        added += sum(p.numel() for p in m.parameters())
    return total / 1e6, backbone / 1e6, added / 1e6


def make_input(batch, num_frames, img_h, img_w, device, dtype):
    # forward expects x: [B, num_frames, C, H, W]
    return torch.randn(batch, num_frames, 3, img_h, img_w, device=device, dtype=dtype)


# mapping of by_module() top-level scopes to reported groups
_MODULE_GROUPS = {
    "backbone_decoder": ["model", "extract_layers"],
    "msff":             ["feature_fusion"],
    "afw":              ["adaptive_weighting"],
    "cross_attn":       ["cross_attention", "self_attention"],
}


def _group_by_module(bym, total):
    """Aggregate fvcore by_module() top-level scopes into reported groups (GMACs)."""
    top = {k: v for k, v in bym.items() if k.count(".") == 0 and k != ""}
    groups = {g: sum(top.get(s, 0) for s in scopes) / 1e9
              for g, scopes in _MODULE_GROUPS.items()}
    accounted = sum(sum(top.get(s, 0) for s in scopes) for scopes in _MODULE_GROUPS.values())
    groups["other"] = max(total - accounted, 0) / 1e9  # norms / functional ops (softmax, interp, adds)
    return groups


def compute_flops(model, x, want_breakdown=False):
    """Return (GMACs, source, breakdown_or_None). GFLOPs ~= 2 * GMACs.
    breakdown is a dict {group: GMACs} when want_breakdown and fvcore is available."""
    # try fvcore first (more robust for transformers, and enables per-module breakdown)
    try:
        from fvcore.nn import FlopCountAnalysis
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fca = FlopCountAnalysis(model, x)
            fca.unsupported_ops_warnings(False)
            fca.uncalled_modules_warnings(False)
            macs = fca.total()
            breakdown = _group_by_module(fca.by_module(), macs) if want_breakdown else None
        return macs / 1e9, "fvcore", breakdown
    except Exception as e:
        print(f"  [flops] fvcore failed ({type(e).__name__}: {e}); trying thop...")
    try:
        from thop import profile
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            macs, _ = profile(model, inputs=(x,), verbose=False)
        return macs / 1e9, "thop", None
    except Exception as e:
        print(f"  [flops] thop failed ({type(e).__name__}: {e}); FLOPs will be N/A.")
        return None, None, None


@torch.no_grad()
def measure_latency(model, x, device, amp, warmup, iters):
    autocast = torch.cuda.amp.autocast(enabled=amp)
    # warmup
    for _ in range(warmup):
        with autocast:
            _ = model(x)
    torch.cuda.synchronize(device)

    # peak memory measured over the timed run
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        with autocast:
            _ = model(x)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    batch = x.shape[0]
    ms_per_seq = (elapsed / (iters * batch)) * 1000.0
    fps = (iters * batch) / elapsed
    return ms_per_seq, fps, peak_gb


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Poseidon efficiency benchmark")
    ap.add_argument("--backbones", nargs="+", default=["S", "B", "L", "H"],
                    choices=list(CONFIG_BY_BACKBONE.keys()),
                    help="Which ViTPose backbones to benchmark")
    ap.add_argument("--frames", nargs="+", type=int, default=[3, 5, 7],
                    help="Temporal window sizes (# frames)")
    ap.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    ap.add_argument("--batch", type=int, default=1, help="Batch size (sequences)")
    ap.add_argument("--amp", action="store_true", help="Use mixed precision (fp16 autocast)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--skip-flops", action="store_true", help="Skip GFLOPs computation")
    ap.add_argument("--per-module", action="store_true",
                    help="Also report per-module FLOPs breakdown (backbone/MSFF/AFW/Cross-Att) [fvcore only]")
    ap.add_argument("--flops-on-cpu", action="store_true",
                    help="Compute FLOPs on CPU (hardware-independent; avoids GPU OOM on shared clusters). "
                         "Timing/memory are still measured on the GPU.")
    ap.add_argument("--random-weights", action="store_true",
                    help="Do not load base ViTPose checkpoints (same FLOPs/timing, faster startup)")
    ap.add_argument("--out", type=str, default=None,
                    help="Output path prefix (writes <out>.md and <out>.csv)")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA is required for this benchmark"
    device = f"cuda:{args.gpu}"
    torch.backends.cudnn.benchmark = True

    rows = []
    module_rows = []
    for bk in args.backbones:
        yaml_path = CONFIG_BY_BACKBONE[bk]
        for T in args.frames:
            print(f"\n=== ViT-{bk} | T={T} frames ===")
            cfg = build_cfg(yaml_path, T, args.random_weights)
            img_w, img_h = cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1]  # [W, H] = [288, 384]

            model = Poseidon(cfg, device=device, phase="val", inference=True).to(device)
            model.eval()

            tot_m, bb_m, add_m = count_params(model)
            print(f"  Params: total={tot_m:.1f}M  backbone={bb_m:.1f}M  "
                  f"added-modules={add_m:.1f}M ({100*add_m/tot_m:.1f}% of total)")

            # FLOPs (fp32, batch=1) -- architecture cost, independent of AMP
            gmacs, flop_src, breakdown = (None, None, None)
            if not args.skip_flops:
                flops_dev = "cpu" if args.flops_on_cpu else device
                if args.flops_on_cpu:
                    model.to("cpu")
                x_flops = make_input(1, T, img_h, img_w, flops_dev, torch.float32)
                gmacs, flop_src, breakdown = compute_flops(
                    model, x_flops, want_breakdown=args.per_module)
                del x_flops
                if args.flops_on_cpu:
                    model.to(device)
                torch.cuda.empty_cache()
                if gmacs is not None:
                    print(f"  GMACs={gmacs:.1f}  (~GFLOPs={2*gmacs:.1f})  [{flop_src}]")
                if breakdown is not None:
                    tot = gmacs if gmacs else 1
                    print("  per-module GFLOPs (fvcore MACsx2): " + "  ".join(
                        f"{g}={2*v:.1f}({100*v/tot:.1f}%)" for g, v in breakdown.items()))
                    module_rows.append({
                        "backbone": f"ViT-{bk}", "frames": T,
                        **{g: round(2 * v, 1) for g, v in breakdown.items()},
                        **{g + "_pct": round(100 * v / tot, 1) for g, v in breakdown.items()},
                    })

            # latency / fps / memory
            dtype = torch.float32
            x = make_input(args.batch, T, img_h, img_w, device, dtype)
            ms, fps, peak_gb = measure_latency(
                model, x, device, args.amp, args.warmup, args.iters)
            print(f"  Latency={ms:.1f} ms/seq  FPS={fps:.1f}  PeakMem={peak_gb:.2f} GB"
                  f"  (amp={args.amp}, batch={args.batch})")

            rows.append({
                "backbone": f"ViT-{bk}",
                "frames": T,
                "input": f"{img_h}x{img_w}",
                "params_total_M": round(tot_m, 1),
                "params_backbone_M": round(bb_m, 1),
                "params_added_M": round(add_m, 1),
                "gflops": round(2 * gmacs, 1) if gmacs is not None else None,
                "gmacs": round(gmacs, 1) if gmacs is not None else None,
                "latency_ms": round(ms, 1),
                "fps": round(fps, 1),
                "peak_mem_gb": round(peak_gb, 2),
                "amp": args.amp,
                "batch": args.batch,
            })

            del model, x
            torch.cuda.empty_cache()

    # ---- report ---- #
    header = ["Backbone", "T", "Input", "Params(M)", "Backbone(M)", "Modules(M)",
              "GFLOPs", "Latency(ms)", "FPS", "PeakMem(GB)"]

    def fmt(r):
        g = r["gflops"] if r["gflops"] is not None else "N/A"
        return [r["backbone"], str(r["frames"]), r["input"], f'{r["params_total_M"]}',
                f'{r["params_backbone_M"]}', f'{r["params_added_M"]}', f'{g}',
                f'{r["latency_ms"]}', f'{r["fps"]}', f'{r["peak_mem_gb"]}']

    md_lines = ["| " + " | ".join(header) + " |",
                "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        md_lines.append("| " + " | ".join(fmt(r)) + " |")
    md = "\n".join(md_lines)

    print("\n\n================ RESULTS (markdown) ================\n")
    print(md)
    print(f"\n(FLOPs via fvcore MACs x2; AMP={args.amp}, batch={args.batch}, "
          f"device={device}, iters={args.iters}. Attention FLOPs may be slightly under-counted.)")

    # ---- per-module breakdown table ---- #
    mod_md = None
    if module_rows:
        mh = ["Backbone", "T", "Backbone+Dec", "MSFF", "AFW", "Cross+Self-Att", "Other", "Total"]
        mlines = ["| " + " | ".join(mh) + " |", "|" + "|".join(["---"] * len(mh)) + "|"]
        for r in module_rows:
            tot = r["backbone_decoder"] + r["msff"] + r["afw"] + r["cross_attn"] + r["other"]
            mlines.append("| " + " | ".join([
                r["backbone"], str(r["frames"]),
                f'{r["backbone_decoder"]} ({r["backbone_decoder_pct"]}%)',
                f'{r["msff"]} ({r["msff_pct"]}%)',
                f'{r["afw"]} ({r["afw_pct"]}%)',
                f'{r["cross_attn"]} ({r["cross_attn_pct"]}%)',
                f'{r["other"]} ({r["other_pct"]}%)',
                f'{round(tot,1)}']) + " |")
        mod_md = "\n".join(mlines)
        print("\n\n======= PER-MODULE GFLOPs BREAKDOWN (fvcore MACsx2) =======\n")
        print(mod_md)
        print("\n(MSFF is the dominant added cost; AFW is negligible. nn.MultiheadAttention "
              "matmuls are partially opaque to fvcore, so Cross/Self-Att are lower bounds.)")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out + ".md", "w") as f:
            f.write(md + "\n")
        if mod_md is not None:
            with open(args.out + "_modules.md", "w") as f:
                f.write(mod_md + "\n")
        import csv
        with open(args.out + ".csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved: {args.out}.md and {args.out}.csv")


if __name__ == "__main__":
    main()
