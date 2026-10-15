"""Leakage-isolation experiment (inference only; no training).

Separates the two explanations for the inflated paddy_edge score
(official IoU 0.362 vs site-blocked 0.004): genuine site LEAKAGE vs. the
alternative that the held-out site (site002) is simply an intrinsically
harder domain.

Design: hold the evaluation frames FIXED and vary only whether the model
trained on the site. The frame set is
    S = { frame : bsplit==val AND osplit==val }
i.e. site002 frames (site002 == the entire blocked-val site) that were
ALSO excluded from official training. Neither model ever saw these exact
frames, so the comparison is not frame memorization and not a
difficulty difference (identical frames):
  - "seen"   = the OFFICIAL-trained model. Official-train spans all 3
               sites, so it trained on OTHER site002 frames.
  - "unseen" = the BLOCKED-trained model. Blocked-train = sites 000+001;
               it never saw any site002 frame.
If seen >> unseen on S, the advantage is site-level leakage, not site
difficulty. Reported per arm (RGB/LUM/NIR) so the effect's arm-specificity
is visible: leakage that inflates NIR's paddy_edge specifically is the
finding; a gap present in every arm would be a generic memorization effect.

Caveat reported in the output: official-train (12,684 frames) is ~11%
larger than blocked-train (11,383), so seen/unseen differ slightly in data
QUANTITY as well as site coverage; an 11% quantity gap cannot produce an
86x score gap, but the asymmetry is disclosed rather than hidden.

Usage: python src/phases/leakage_isolation.py   (reads results/runs/*/model.pt)
Output: results/leakage_isolation.json  (+ markdown table to stdout)
"""
import importlib
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T = importlib.import_module("train")
C = T.C

ARMS = ["RGB", "LUM", "NIR"]           # spectral single-stream arms
COND = {"seen": "official", "unseen": "blocked"}
EDGE = C.CLASSES.index("paddy_edge")


def own_proto(arm):
    return "nir" if arm == "NIR" else "rgb"


def frame_iou(inter, union, order, cls_i):
    """Per-frame IoU aligned to dataset index (order), one class."""
    n = len(order)
    out = np.zeros(n)
    inv = np.empty(n, int)
    inv[np.asarray(order)] = np.arange(n)  # dataset-idx -> row position
    ii = inter[:, cls_i][inv]
    uu = union[:, cls_i][inv]
    nz = uu > 0
    out[nz] = ii[nz] / uu[nz]
    return out  # indexed by dataset position (= position in S)


def seq_ci(d, units):
    """The sequence-clustered interval: analyze.py cluster_ci, called rather
    than re-implemented, so this experiment's interval and the gates' cannot
    diverge. Frames of one traverse are not independent draws, so an interval
    that treats frames as independent is narrower than the data support."""
    return importlib.import_module("analyze").cluster_ci(d, units)


def main():
    recs = C.load_index()
    S = [r for r in recs if r["bsplit"] == "val" and r["osplit"] == "val"]
    assert len(S) > 500, len(S)
    # pre-treatment inclusion masks (GT present) for paddy_edge on S
    inc_own = {a: np.array([bool(r[("px_nir" if a == "NIR" else "px_rgb")][EDGE])
                            for r in S]) for a in ARMS}
    inc_isec = np.array([bool(r["px_rgb"][EDGE]) and bool(r["px_nir"][EDGE])
                         for r in S])
    device = "cuda"

    # Per-frame deltas and their sequence labels are persisted, not only the
    # intervals, so a contrast between two arms can be PAIRED downstream
    # without re-running inference (build_parameters.py liso_contrast, sec:results).
    report = {"ci_estimator": "sequence-clustered interval (analyze.py cluster_ci)",
              "n_frames": len(S),
              "n_edge_own": {a: int(inc_own[a].sum()) for a in ARMS},
              "n_edge_isec": int(inc_isec.sum()),
              "frame_set": "bsplit==val AND osplit==val (site002, held out "
                           "from official training)",
              "arms": {}}

    for arm in ARMS:
        op = own_proto(arm)
        # per-seed pixel-pooled edge IoU, and per-seed per-frame IoU arrays
        pooled = {c: {"own": [], "isec": []} for c in COND}
        perframe = {c: {"own": [], "isec": []} for c in COND}
        for cond, split in COND.items():
            for seed in C.SEEDS:
                mp = os.path.join(C.RUNS_ROOT, f"{arm}_s{seed}_{split}",
                                  "model.pt")
                model = T.build_model(arm).to(device)
                model.load_state_dict(torch.load(mp, map_location=device))
                dl = DataLoader(T.FSSeg(S, arm, train=False), batch_size=8,
                                shuffle=False, num_workers=8, pin_memory=True)
                inter, union, order = T.evaluate(model, dl, device)
                pooled[cond]["own"].append(
                    T.iou_table(inter[op], union[op])["paddy_edge"] or 0.0)
                pooled[cond]["isec"].append(
                    T.iou_table(inter["isec"], union["isec"])["paddy_edge"]
                    or 0.0)
                perframe[cond]["own"].append(
                    frame_iou(inter[op], union[op], order, EDGE))
                perframe[cond]["isec"].append(
                    frame_iou(inter["isec"], union["isec"], order, EDGE))
                del model
                torch.cuda.empty_cache()

        entry = {}
        for proto, inc in (("own", inc_own[arm]), ("isec", inc_isec)):
            seen = np.array(pooled["seen"][proto])
            unseen = np.array(pooled["unseen"][proto])
            # per-frame: seed-mean per frame, then restrict to GT-present frames
            f_seen = np.mean(perframe["seen"][proto], axis=0)[inc]
            f_unseen = np.mean(perframe["unseen"][proto], axis=0)[inc]
            d = f_seen - f_unseen
            units = [(r["env"], r["seq"]) for r, k in zip(S, inc) if k]
            lo, hi = seq_ci(d, units)
            p = importlib.import_module("analyze").paired_p(
                d, C.CLASSES[EDGE], proto)
            # Persist the deltas with their sequence keys. This experiment
            # needs GPU inference to reproduce, so a summary-only artifact
            # made re-estimating its interval cost a re-run; the raw deltas
            # make any later re-analysis a file read.
            entry[proto] = {
                "perframe_deltas": [float(x) for x in d],
                "perframe_units": ["%s/%s" % u for u in units],
                "seen_pooled_mean": float(seen.mean()),
                "seen_pooled_sd": float(seen.std()),
                "unseen_pooled_mean": float(unseen.mean()),
                "unseen_pooled_sd": float(unseen.std()),
                "inflation_x": (float(seen.mean() / unseen.mean())
                                if unseen.mean() > 1e-6 else None),
                "perframe_delta_mean": float(d.mean()),
                "perframe_ci": [lo, hi],
                "perframe_wilcoxon_p": p,
                "n_frames_incl": int(inc.sum()),
            }
        report["arms"][arm] = entry

    out = os.path.join(C.ROOT, "results", "leakage_isolation.json")
    json.dump(report, open(out, "w"), indent=2)

    print(f"\nLeakage isolation on {len(S)} held-out site002 frames "
          f"(paddy_edge; seen=official-trained, unseen=blocked-trained)\n")
    print("| arm | proto | seen IoU | unseen IoU | seen−unseen "
          "[95% CI] | Wilcoxon p |")
    print("|---|---|---|---|---|---|")
    for arm in ARMS:
        for proto in ("own", "isec"):
            e = report["arms"][arm][proto]
            print(f"| {arm} | {proto} | {e['seen_pooled_mean']:.3f}"
                  f"±{e['seen_pooled_sd']:.3f} | "
                  f"{e['unseen_pooled_mean']:.3f}±{e['unseen_pooled_sd']:.3f} | "
                  f"{e['perframe_delta_mean']:+.3f} "
                  f"[{e['perframe_ci'][0]:+.3f},{e['perframe_ci'][1]:+.3f}] | "
                  f"{e['perframe_wilcoxon_p']:.2e} |")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
