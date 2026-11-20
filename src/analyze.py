"""Aggregate full-study runs into results/summary.md.

Statistics exactly as committed (config.py GATES):
  - Every number that enters a comparison is reported as mean +/- sd over
    the 5 committed seeds.
  - Paired per-frame IoU deltas: frames included by PRE-TREATMENT GT
    presence (per-frame GT pixel counts from the index, not prediction-
    dependent union>0):
      own/cross-label tests: GT present in that protocol's label set
      intersection tests:    GT present in BOTH label sets
    Per frame, IoU is seed-averaged within an arm first; the paired
    Wilcoxon runs across frames; the 95% CI on the mean delta treats the
    driving SEQUENCE as the independent unit (cluster_ci), because frames
    within a driving sequence are not independent draws. Over fewer than
    MIN_BOOT_UNITS sequences it is a cluster-robust t interval (t_ci), and
    over more a sequence-cluster bootstrap (boot_ci). The Wilcoxon stays
    frame-level as committed and is therefore anticonservative at these n;
    the cluster-corrected CI is the binding half of the gate below, and no
    p-value printed by this module should be read as cluster-robust.
  - Holm correction over the 5 target classes within each committed
    comparison family; alpha = config.ALPHA.
  - A directional claim requires Holm-corrected p < alpha AND CI
    excluding 0 on the INTERSECTION protocol, with the own-label delta
    direction-consistent; otherwise the verdict prints "comparable".
  - Controls (common_road; common_tree on official protocol only): any
    headline effect that also appears on controls is flagged suspect.

Comparison families (committed):
  F1  NIR  vs RGB     (blocked, joint)   - is NIR non-redundant?
  F2  NIR  vs LUM     (blocked, joint)   - does NIR beat its own format?
  F3  RGBNIR vs RGB   (blocked, joint)   - does adding NIR to RGB help?
  F4  same three, official protocol      - leakage-inflation comparison

Parameter ledger (every value below is fixed by design; nothing is fitted):
  CI_TAIL = 0.025        CHOSEN  two-sided 95% interval, the convention the
                                 criterion names; also its "excludes 0" line.
  MIN_BOOT_UNITS = 20    CHOSEN  fewest sequences an interval is bootstrapped
                                 over; below it the interval is t_ci's (the
                                 sensitivity is at the constant).
  BOOT_N = 1e6           CHOSEN  Monte Carlo error of a bootstrap endpoint
                                 far below the distance of either pooled
                                 endpoint from zero (derivation at the
                                 constant).
  BOOT_SEED = 0          CHOSEN  any fixed value; each interval re-seeds from
                                 its own inputs, so none depends on call order.
  BOOT_CELLS = 1e7       CHOSEN  memory bound on one resample block; the
                                 resamples drawn do not depend on it.

Usage: python src/analyze.py   (reads results/runs/*, data/processed/index.json)
"""
import glob
import json
import math
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

OUT_MD = os.path.join(C.ROOT, "results", "summary.md")

FAMILIES = [("F1", "NIR", "RGB"), ("F2", "NIR", "LUM"), ("F3", "RGBNIR", "RGB")]


def load_runs():
    """segformer/mit_b0 runs only. This is the headline analyzer (families
    F1-F4, phases A and C). The mit_b2 second backbone (phase F), the second
    architecture family (phase H) and the LOSO folds (phase E) have their own
    analyzer (analyze_folds.py); loading one model only here keeps them from being
    silently pooled into one table. Phase A and C metrics.json predate both
    keys, so a missing key means the headline model."""
    runs = []
    for mp in sorted(glob.glob(os.path.join(C.RUNS_ROOT, "*", "metrics.json"))):
        with open(mp) as f:
            m = json.load(f)
        if (m.get("backbone", "mit_b0"), m.get("arch", "segformer")) \
                != ("mit_b0", "segformer"):
            continue
        npz = os.path.join(os.path.dirname(mp), "perframe.npz")
        m["_pf"] = np.load(npz) if os.path.exists(npz) else None
        runs.append(m)
    return runs


def gt_px():
    """(env, seq, frame) -> (px_rgb[9], px_nir[9]) from the index."""
    recs = C.load_index()
    return {(r["env"], r["seq"], r["frame"]):
            (np.array(r["px_rgb"]), np.array(r["px_nir"])) for r in recs}


def fmt(v, nd=3):
    return "-" if v is None else f"{v:.{nd}f}"


def mean_sd(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    return f"{np.mean(vals):.3f}±{np.std(vals):.3f}"


def seq_units(keys):
    """(env,seq,frame) keys -> the (env,seq) resampling unit of each."""
    return [(k[0], k[1]) for k in keys]


BOOT_SEED = 0       # every interval is reproducible from its own inputs alone
CI_TAIL = 0.025     # two-sided 95% interval; also the gate's "excludes 0" line
BOOT_CELLS = 10000000   # resample-index cells held at once, bounding peak
                        # memory independently of how many units are resampled

# The fewest sequences an interval is bootstrapped over. The cluster bootstrap
# is consistent only as the number of clusters grows (Field and Welsh 2007,
# sec:criteria), and a percentile interval over a few clusters covers below its
# stated level, so below this count cluster_ci returns t_ci instead. Every
# interval the paper prints rests on at most 13 sequences or on at least 72,
# so any value from 14 to 72 sorts them as this one does
# (test_every_interval_over_few_sequences_is_a_t_interval).
MIN_BOOT_UNITS = 20

# Resample count of the bootstrap, which only the two pooled tests reach, over
# 76 and 72 sequences. At 1e6 the Monte Carlo error of either endpoint is below
# 1e-4 IoU, and the nearer pooled endpoint lies 0.023 from zero, so the
# resample draw cannot move a verdict. Raising it does not alter the committed
# criterion: "95% CI excluding 0" names the bootstrap quantile, and this
# constant is only the accuracy with which that quantile is evaluated.
BOOT_N = 1000000


def chunks(n, width):
    """(start, rows) blocks covering n resamples, each holding at most
    BOOT_CELLS index cells. Chunking a single seeded stream does not change
    the sequence of draws, so the interval is identical to an unchunked one."""
    rows = max(1, BOOT_CELLS // max(width, 1))
    return [(s, min(rows, n - s)) for s in range(0, n, rows)]


def boot_ci(deltas, units, n=BOOT_N, seed=BOOT_SEED):
    """Cluster bootstrap on the mean delta, as (lo, hi).

    The resampling unit is the sequence, not the frame. Consecutive frames of
    one traverse view the same ground, so they are
    not independent draws; an i.i.d. frame bootstrap treats ~10^3 correlated
    observations as ~10^3 independent ones and returns an interval narrower
    than the data support. Sequence is the finest unit the protocol of
    sec:methods treats as exchangeable - the same autocorrelation that makes a
    random frame-level SPLIT leak makes a frame-level INTERVAL too narrow, and
    a study whose subject is the first cannot use an estimator that ignores
    the second.

    `units` is required, not defaulted: a caller that has lost track of which
    frames cluster together must fail here rather than silently recover the
    frame-level interval this function exists to replace.

    The stream is seeded per call, not drawn from a module-level RNG. A
    shared stream makes an interval depend on how many intervals were
    computed before it, so the same deltas can land either side of a gate
    depending on call order - reproducibility failure of the same kind as
    seed drift, and invisible because every individual run is
    self-consistent.

    Resampling clusters and averaging the pooled frames is algebraically the
    ratio of the resampled cluster sums to the resampled cluster sizes, which
    is what the vectorized form below computes - identical draws, no loop. The
    draws are taken in chunks only to bound memory; chunking a single seeded
    stream leaves the sequence of draws unchanged."""
    d = np.asarray(deltas, float)
    g = defaultdict(list)
    for i, u in enumerate(units):
        g[u].append(i)
    idx = [np.array(v) for v in g.values()]
    sums = np.array([d[i].sum() for i in idx])
    sizes = np.array([len(i) for i in idx], float)
    rng = np.random.RandomState(seed)
    means = np.empty(n)
    for s, b in chunks(n, len(idx)):
        pick = rng.randint(0, len(idx), (b, len(idx)))
        means[s:s + b] = sums[pick].sum(1) / sizes[pick].sum(1)
    return (float(np.percentile(means, 100 * CI_TAIL)),
            float(np.percentile(means, 100 * (1 - CI_TAIL))))


def t_ci(deltas, units):
    """Cluster-robust t interval on the mean delta, as (lo, hi), with k - 1
    degrees of freedom for the k units.

    The mean is the frame-weighted one every margin prints: the ratio of the
    units' summed deltas to their summed frame counts. Its standard error is
    that ratio's under cluster sampling (Cochran 1977), each sequence one
    cluster: sqrt(k / (k - 1) * sum_g e_g^2) / N, where e_g sums the deviations
    of unit g's deltas from the mean and N counts the frames. With units of
    equal size it is the one-sample t interval over the units' means. The t
    quantile at k - 1 degrees of freedom rather than the normal one is what
    keeps an interval over three or four sequences from claiming a precision
    that so few units cannot estimate. One unit estimates no spread, so its
    interval is undefined and reads as spanning zero in every gate."""
    d = np.asarray(deltas, float)
    g = defaultdict(list)
    for i, u in enumerate(units):
        g[u].append(i)
    k = len(g)
    if k < 2:
        return float("nan"), float("nan")
    m = float(d.mean())
    e = np.array([float((d[np.array(v)] - m).sum()) for v in g.values()])
    se = math.sqrt(k / (k - 1) * float((e ** 2).sum())) / len(d)
    h = float(stats.t.ppf(1 - CI_TAIL, k - 1)) * se
    return m - h, m + h


def cluster_ci(deltas, units):
    """The interval of every reported comparison, as (lo, hi): t_ci over fewer
    than MIN_BOOT_UNITS sequences and boot_ci over more (sec:criteria)."""
    k = len(set(units))
    return (t_ci if k < MIN_BOOT_UNITS else boot_ci)(deltas, units)


def frame_iou_by_arm(runs_of_arm, proto, cls_i, include_mask_fn, px):
    """(env,seq,frame) -> seed-mean IoU for one arm/protocol/class,
    restricted to pre-treatment-included frames."""
    acc = defaultdict(list)
    for m in runs_of_arm:
        pf = m["_pf"]
        keys = list(zip(pf["env"], pf["seq"], pf["frame"]))
        inter = pf[f"inter_{proto}"][:, cls_i]
        union = pf[f"union_{proto}"][:, cls_i]
        for k, i_, u_ in zip(keys, inter, union):
            kk = (str(k[0]), str(k[1]), int(k[2]))
            if kk not in px or not include_mask_fn(px[kk], cls_i):
                continue
            acc[kk].append(i_ / u_ if u_ > 0 else 0.0)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def inc_own(proto):
    if proto == "rgb":
        return lambda p, c: p[0][c] > 0
    if proto == "nir":
        return lambda p, c: p[1][c] > 0
    return lambda p, c: p[0][c] > 0 and p[1][c] > 0  # isec


def own_proto(arm_x):
    """Which annotation the third directional condition reads for a family.

    Both arms are scored on one annotation, never each on its own. A paired
    difference needs the two IoUs measured against the same ground truth, and
    the two annotations mark a class on different frames (sec:methods). The
    one used is the annotation supervising arm_x. sec:methods states that rule
    in different terms, as the near-infrared annotation for the families
    near-infrared is an arm of. The two agree over FAMILIES as committed, and
    test_the_directional_rule_the_paper_states_is_the_one_the_code_applies is
    where that agreement is checked rather than asserted."""
    return "nir" if arm_x == "NIR" else "rgb"


def paired_p(d, cls, cell):
    """Paired Wilcoxon p over per-frame deltas. All-zero deltas are no
    evidence against the null, so p is 1 without calling scipy, whose normal
    approximation divides 0/0 there. For a target class that means both arms
    scored exactly 0 on every shared frame, which is worth a warning even
    though the cell scores as no difference; `cell` names the comparison in
    that warning."""
    d = np.asarray(d)
    if not np.any(d != 0):
        if cls in C.TARGETS:
            warnings.warn(f"{cls} ({cell}): all {len(d)} paired deltas are "
                          "zero, p set to 1.0")
        return 1.0
    return float(stats.wilcoxon(d).pvalue)


# Every frame count that evaluable() has compared with MIN_VAL_FRAMES in this
# process. minima_range.py reads it to find the minima that score the same
# classes.
EVAL_COUNTS = []


def evaluable(n_frames):
    """sec:protocols scores only evaluable classes: those whose ground truth
    marks them on at least MIN_VAL_FRAMES validation frames."""
    EVAL_COUNTS.append(n_frames)
    return n_frames >= C.MIN_VAL_FRAMES


def paired_family(runs, protocol, arm_x, arm_y, proto, px):
    """Per-class paired stats for arm_x - arm_y on one label protocol."""
    rx = [m for m in runs if m["arm"] == arm_x and m["protocol"] == protocol
          and m["train_env"] == "all" and m["_pf"] is not None]
    ry = [m for m in runs if m["arm"] == arm_y and m["protocol"] == protocol
          and m["train_env"] == "all" and m["_pf"] is not None]
    out = {}
    if not rx or not ry:
        return out
    for cls in C.TARGETS + ["common_road", "common_tree", "paddy_rice"]:
        ci = C.CLASSES.index(cls)
        fx = frame_iou_by_arm(rx, proto, ci, inc_own(proto), px)
        fy = frame_iou_by_arm(ry, proto, ci, inc_own(proto), px)
        shared = sorted(set(fx) & set(fy))
        if not evaluable(len(shared)):
            out[cls] = None
            continue
        d = np.array([fx[k] - fy[k] for k in shared])
        units = seq_units(shared)
        lo, hi = cluster_ci(d, units)
        p = paired_p(d, cls, proto)
        out[cls] = {"n": len(d), "k": len(set(units)), "mean": float(d.mean()),
                    "lo": lo, "hi": hi, "p": p,
                    "n_seeds": (len(rx), len(ry))}
    return out


def holm(pvals):
    """Holm-adjusted p-values (dict cls->p over the 5 targets)."""
    items = [(c, p) for c, p in pvals.items() if p is not None]
    items.sort(key=lambda t: t[1])
    m = len(items)
    adj, running = {}, 0.0
    for rank, (c, p) in enumerate(items):
        running = max(running, (m - rank) * p)
        adj[c] = min(1.0, running)
    return adj


def verdict(cls, stat_isec, stat_own, p_holm):
    """The directional gate.

    Three conditions, one more than the committed criterion (config.py
    gates) states: the own-label delta must run the same way as the
    intersection one. It is strictly conservative, so it can only withhold a
    ranking and never grant one, and on the committed runs no class passes the
    other two and fails it - asserted by
    test_the_directional_rule_the_paper_states_is_the_one_the_code_applies.
    The manuscript states all three,
    because all three produced the verdicts it prints; the ledger above is the
    record of what was committed and is not edited to match."""
    if stat_isec is None or stat_own is None:
        return "insufficient data"
    ci_excl = stat_isec["lo"] > 0 or stat_isec["hi"] < 0
    consistent = np.sign(stat_isec["mean"]) == np.sign(stat_own["mean"])
    if p_holm is not None and p_holm < C.ALPHA and ci_excl and consistent:
        return "DIRECTIONAL" + (" (+)" if stat_isec["mean"] > 0 else " (-)")
    return "comparable"


def table_overall(md, runs, protocol):
    md.append(f"\n### Per-class IoU, {protocol} protocol "
              "(own-label / intersection; mean±sd over seeds)\n")
    arms = ["RGB", "LUM", "NIR", "RGBNIR"]
    have = [a for a in arms if any(m["arm"] == a and m["protocol"] == protocol
                                   and m["train_env"] == "all" for m in runs)]
    md.append("| class | " + " | ".join(f"{a} own | {a} isec" for a in have) + " |")
    md.append("|---" * (1 + 2 * len(have)) + "|")
    for cls in C.CLASSES[1:]:
        row = [cls]
        for a in have:
            sel = [m for m in runs if m["arm"] == a and m["protocol"] == protocol
                   and m["train_env"] == "all"]
            own = "nir" if a == "NIR" else "rgb"
            row.append(mean_sd([m["iou"][own].get(cls) for m in sel]))
            row.append(mean_sd([m["iou"]["isec"].get(cls) for m in sel]))
        md.append("| " + " | ".join(row) + " |")


def table_families(md, runs, protocol, px):
    md.append(f"\n### Committed paired comparisons, {protocol} protocol\n")
    md.append("Per-frame paired deltas (seed-averaged per frame; "
              "pre-treatment GT inclusion). Holm over the 5 targets on the "
              "intersection protocol; gate = Holm p<"
              f"{C.ALPHA} AND CI excl. 0 AND own-label direction-consistent.\n")
    for fam, ax, ay in FAMILIES:
        s_isec = paired_family(runs, protocol, ax, ay, "isec", px)
        s_own = paired_family(runs, protocol, ax, ay, own_proto(ax), px)
        if not s_isec:
            md.append(f"**{fam}: {ax} − {ay}** — _runs missing_\n")
            continue
        p_holm = holm({c: (s_isec[c]["p"] if s_isec.get(c) else None)
                       for c in C.TARGETS})
        md.append(f"**{fam}: {ax} − {ay}** (own = {own_proto(ax)} labels)\n")
        md.append("| class | Δ isec [95% CI] (n frames, k sequences) | p | "
                  "Holm p | Δ own | verdict |")
        md.append("|---|---|---|---|---|---|")
        for cls in C.TARGETS + ["common_road", "common_tree", "paddy_rice"]:
            si, so = s_isec.get(cls), s_own.get(cls)
            if si is None:
                md.append(f"| {cls} | not evaluable (n<{C.MIN_VAL_FRAMES}) "
                          "| - | - | - | - |")
                continue
            ph = p_holm.get(cls)
            tag = verdict(cls, si, so, ph) if cls in C.TARGETS else \
                ("CONTROL-EFFECT?" if si["p"] < C.ALPHA
                 and (si["lo"] > 0 or si["hi"] < 0) else "control ok")
            ph_txt = fmt(ph, 4) if cls in C.TARGETS else "-"
            so_txt = f"{so['mean']:+.3f}" if so else "-"
            md.append(
                f"| {cls} | {si['mean']:+.3f} [{si['lo']:+.3f},{si['hi']:+.3f}] "
                f"(n={si['n']}, k={si['k']}) | {si['p']:.2e} | {ph_txt} | "
                f"{so_txt} | {tag} |")
        md.append("")


def main():
    runs = load_runs()
    px = gt_px()
    md = ["# Full-study summary — NIR traversability",
          "", f"_Auto-generated by src/analyze.py; {len(runs)} completed "
          "runs found. Gates: src/config.py, committed before the runs._"]
    n_by = defaultdict(int)
    for m in runs:
        n_by[(m["protocol"], m["train_env"] != "all", m["arm"])] += 1
    md.append("\nRun inventory: " + ", ".join(
        f"{k}: {v}" for k, v in sorted(n_by.items())))

    for protocol in ("blocked", "official"):
        if any(m["protocol"] == protocol and m["train_env"] == "all"
               for m in runs):
            md.append(f"\n## {protocol.capitalize()} protocol")
            table_overall(md, runs, protocol)
            table_families(md, runs, protocol, px)

    with open(OUT_MD, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
