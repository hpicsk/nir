"""Generate the site-A leakage figure for the manuscript from committed results.

Same one-source-of-truth contract as build_parameters.py and build_tables.py:
the bars are the seed-mean per-frame IoUs that analyze_folds.py's
paired_protocols subtracts, read from the committed per-frame arrays of the
runs, so the difference between a configuration's two bars is its paddy-edge
inflation in the site-A row of tab:robustness(a). The site is the one where
the paddy-restricted cross-validation detects the leakage of near-infrared
(sec:concentration): seen bars come from its matched comparator, which trains
on other frames of the site, and unseen bars from its fold that holds the site
out. The plotted claim, that every configuration scores higher with the site
seen and near-infrared gains the most, is asserted by
test_figure_one_shows_the_site_a_inflations.

RNG-free: seed-mean per-frame IoUs averaged over a fixed frame set, no
resampling.

Usage: python src/build_figures.py   (reads results/; writes
paper/fig_leakage_isolation.pdf)
"""
import importlib
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C

A = importlib.import_module("analyze")
folds = importlib.import_module("analyze_folds")

OUT_PDF = os.path.join(C.ROOT, "paper", "fig_leakage_isolation.pdf")

# The site and the two protocols the figure pairs: the paddy-restricted fold
# that holds site A out, against the matched comparator that does not.
SITE = "site000"
SEEN, UNSEEN = "pofficial", folds.G_FOLD_PROTO[SITE]
CLS = "paddy_edge"

# manuscript vocabulary for the four input configurations
CONFIGS = [("RGB", "RGB"), ("LUM", "Luminance"), ("NIR", "NIR"),
           ("RGBNIR", "Fusion")]

# Okabe-Ito color-blind-safe pair (sky blue, vermillion). Their luminances
# differ, so the pair also separates in grayscale print. No hatch: a hatch
# pattern is tiled in the PDF, and a rasterised page shows the tile seams as
# stray dashes inside the bars.
C_UNSEEN, C_SEEN = "#56B4E9", "#D55E00"


def bars(runs, px):
    """Per configuration, (unseen, seen) mean per-frame IoU on the
    intersection ground truth, over the frames both protocols scored: the
    frame set of paired_protocols, so seen minus unseen is its inflation."""
    ci = C.CLASSES.index(CLS)
    out, frame_sets = {}, set()
    for arm, _ in CONFIGS:
        f = {}
        for proto in (UNSEEN, SEEN):
            r = [m for m in runs if m["arm"] == arm and m["protocol"] == proto
                 and m["train_env"] == "all" and m["_pf"] is not None]
            f[proto] = A.frame_iou_by_arm(r, "isec", ci, A.inc_own("isec"), px)
        shared = sorted(set(f[UNSEEN]) & set(f[SEEN]))
        frame_sets.add(tuple(shared))
        out[arm] = tuple(float(np.mean([f[p][k] for k in shared]))
                         for p in (UNSEEN, SEEN))
    assert len(frame_sets) == 1, (
        "the configurations are scored on different frames, so the bars of "
        "Fig. 1 do not share the frame count its caption prints")
    return out


def main():
    b = bars(folds.load_runs("mit_b0"), A.gt_px())
    unseen = [b[k][0] for k, _ in CONFIGS]
    seen = [b[k][1] for k, _ in CONFIGS]

    x = list(range(len(CONFIGS)))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    b1 = ax.bar([i - w / 2 for i in x], unseen, w,
                label="Site unseen in training",
                color=C_UNSEEN, edgecolor="black", linewidth=0.6)
    b2 = ax.bar([i + w / 2 for i in x], seen, w,
                label="Site seen in training",
                color=C_SEEN, edgecolor="black", linewidth=0.6)

    for group in (b1, b2):
        for r in group:
            ax.annotate(f"{r.get_height():.3f}",
                        (r.get_x() + r.get_width() / 2, r.get_height()),
                        ha="center", va="bottom", fontsize=8.5,
                        xytext=(0, 1.5), textcoords="offset points")

    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in CONFIGS], fontsize=10)
    ax.set_ylabel("Paddy-edge IoU, mean over frames", fontsize=10)
    ax.set_ylim(0, max(seen) * 1.20)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    fig.tight_layout()
    # No CreationDate: the default stamps the wall clock into the PDF, so a
    # regeneration differs from the committed file on every run and the
    # byte-identical freshness check can never pass on this
    # artifact. Suppressing it makes a diff here mean the plot changed.
    fig.savefig(OUT_PDF, bbox_inches="tight",
                metadata={"CreationDate": None})
    print("wrote", OUT_PDF)
    print("unseen", [round(v, 3) for v in unseen], "seen", [round(v, 3) for v in seen])


if __name__ == "__main__":
    main()
