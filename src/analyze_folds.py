"""Fold, backbone and architecture analyzer: gates E1-E3 (LOSO), F1-F2
(mit_b2), G1-G3 (Paddy-restricted LOSO + the pofficial comparator), H1-H2
(second architecture family).

Separate from analyze.py on purpose. analyze.py is the headline analyzer and loads
segformer/mit_b0 ONLY, so no two models are ever silently pooled into one
table. This file is where every phase-E/F/G/H gate written in
config.py is computed, and it computes them EXACTLY as committed -
the gate text in config.py is the specification, this file the
implementation, and neither may drift from the other.

The statistics are not re-implemented here. The sec:methods directional gate
(Holm p < ALPHA over the 5 targets AND a sequence-clustered 95% CI excluding 0
on the intersection protocol, with the own-label delta direction-consistent) is
imported from analyze.py and called, so one implementation serves both analyzers
and a change to the gate cannot land in one and miss the other.

Two pairing modes are needed, and only the second is new:
  ARM-paired    (analyze.py paired_family): same protocol, two arms. Used by E2,
                F2, G3 - "does NIR beat LUM here?"
  PROTOCOL-paired (paired_protocols below): same arm, two protocols, on the
                frames both evaluated. Used by E1, F1, G2 - "does the same
                model family score higher when the split leaks?"

Every pooled gate also prints a SITE-LEVEL reading (site_level below): the k
held-out sites as k observations. Pooling frames across folds cannot tell a
general effect from a site-specific one no matter how many frames it pools,
and on this dataset two committed positives that pass pooled do not
replicate between sites. The pooled pass/fail stays as committed; the
site-level line is reported beside it, never substituted for it.

INCOMPLETE PHASES ARE REPORTED AS INCOMPLETE. A gate whose runs are not all
on disk prints the exact shortfall and no verdict; it is never computed on a
seed subset, because a gate read on 1 of 5 seeds is not the committed
gate: it is read on the whole committed seed set, not the one that
finished first.

Usage: python src/analyze_folds.py   (reads results/runs/*, data/processed/index.json)
Output: results/folds.md
"""
import glob
import importlib
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

A = importlib.import_module("analyze")   # the gate implementation

OUT_MD = os.path.join(C.ROOT, "results", "folds.md")

# Fold <-> protocol map for phase E. Fold site002 is NOT re-run: the blocked
# protocol is split-identical to it (config ledger 2026-07-23, verified
# blocked-val == {site002} exactly), so the phase-A blocked runs ARE that
# fold. Named here so the substitution is explicit rather than assumed.
E_FOLD_PROTO = {"site000": "loso000", "site001": "loso001", "site002": "blocked"}
G_FOLD_PROTO = {f: "ploso" + f[len("site"):] for f in C.PLOSO_RUN_FOLDS}

ARMS = ["RGB", "NIR", "LUM", "RGBNIR"]
# E1/F1/G2 are about the paddy_edge leakage headline, which is the
# NIR-concentrated effect (phases A/C). Every arm is
# tabulated so the
# arm-specificity stays visible; the gate is declared on this one, stated
# here rather than chosen after seeing the table.
GATE_ARM = "NIR"


def load_runs(backbone, arch="segformer"):
    """All runs of one model, with per-frame arrays attached. Explicit
    backbone argument, never a default: pooling mit_b0 and mit_b2 into one
    table is the silent failure this file exists to avoid, and phase H's
    architecture is the same hazard one level up. `arch` defaults only
    because every pre-phase-H run is a segformer run."""
    runs = []
    for mp in sorted(glob.glob(os.path.join(C.RUNS_ROOT, "*", "metrics.json"))):
        with open(mp) as f:
            m = json.load(f)
        if (m.get("backbone", "mit_b0"),       # phase A-C predate both keys
                m.get("arch", "segformer")) != (backbone, arch):
            continue
        npz = os.path.join(os.path.dirname(mp), "perframe.npz")
        m["_pf"] = np.load(npz) if os.path.exists(npz) else None
        runs.append(m)
    return runs


def have(runs, **spec):
    """Runs matching a spec, for both selection and shortfall reporting."""
    return [m for m in runs if all(m.get(k) == v for k, v in spec.items())]


def key2site():
    with open(C.SPLITS_JSON) as f:
        sites = json.load(f)["sites"]
    return {m: s["site_id"] for s in sites for m in s["members"]}


def fixed_sets():
    """S_N for gate G2b (config ledger 2026-07-24b): frames of siteN in the
    PLOSO_ENV restriction that are official-VAL, i.e. frames NEITHER the
    pofficial model nor the plosoN model trained on. Keyed (env,seq,frame)
    to match the per-frame arrays."""
    recs = C.load_index()
    k2s = key2site()
    out = {f: set() for f in C.PLOSO_RUN_FOLDS}
    for r in recs:
        if r["env"] != C.PLOSO_ENV or r["osplit"] != "val":
            continue
        out[k2s[f"{r['osplit']}/{r['env']}/{r['seq']}"]].add(
            (r["env"], r["seq"], int(r["frame"])))
    return out


def paired_protocols(runs, arm, proto_x, proto_y, label_proto, px,
                     restrict=None):
    """Per-class paired stats for one arm, proto_x MINUS proto_y, over the
    frames both protocols evaluated (optionally restricted to `restrict`).

    The arm-paired counterpart is analyze.py paired_family; this varies the split
    instead of the modality, which is what E1/F1/G2 ask. Frames are matched
    by (env,seq,frame), so the two conditions are compared on identical
    frames and the delta cannot absorb a difference in which frames were
    scored."""
    rx = [m for m in runs if m["arm"] == arm and m["protocol"] == proto_x
          and m["train_env"] == "all" and m["_pf"] is not None]
    ry = [m for m in runs if m["arm"] == arm and m["protocol"] == proto_y
          and m["train_env"] == "all" and m["_pf"] is not None]
    out = {}
    if not rx or not ry:
        return out
    for cls in C.TARGETS + ["common_road", "common_tree", "paddy_rice"]:
        ci = C.CLASSES.index(cls)
        fx = A.frame_iou_by_arm(rx, label_proto, ci, A.inc_own(label_proto), px)
        fy = A.frame_iou_by_arm(ry, label_proto, ci, A.inc_own(label_proto), px)
        shared = sorted(set(fx) & set(fy))
        if restrict is not None:
            shared = [k for k in shared if k in restrict]
        if not A.evaluable(len(shared)):
            out[cls] = None
            continue
        d = np.array([fx[k] - fy[k] for k in shared])
        units = A.seq_units(shared)
        lo, hi = A.cluster_ci(d, units)
        p = A.paired_p(d, cls, label_proto)
        # k is the count of sequences behind (lo, hi), not a frame count; it
        # decides which interval analyze.py cluster_ci returns.
        out[cls] = {"n": len(d), "k": len(set(units)), "mean": float(d.mean()),
                    "lo": lo, "hi": hi, "p": p,
                    "n_seeds": (len(rx), len(ry))}
    return out


def paired_arm_inflation(runs, arm_x, arm_y, proto_x, proto_y, px,
                         cls="paddy_edge", label_proto="isec"):
    """How much more arm_x's protocol difference (proto_x minus proto_y) is
    than arm_y's, per frame, over the frames all four cells scored.

    paired_protocols gives each arm's difference with its own interval, and
    two arms' intervals overlap because each carries the frame-level variation
    the two share. Differencing the two per frame removes that variation,
    which is what makes 'arm X leaks more than arm Y' a tested claim rather
    than a comparison of two point estimates. The interval is analyze.py
    cluster_ci, the one every reported interval uses, and k is the count of
    sequences behind it."""
    ci = C.CLASSES.index(cls)

    def frames(arm, proto):
        r = [m for m in runs if m["arm"] == arm and m["protocol"] == proto
             and m["train_env"] == "all" and m["_pf"] is not None]
        return A.frame_iou_by_arm(r, label_proto, ci, A.inc_own(label_proto),
                                  px)

    f = {(a, p): frames(a, p) for a in (arm_x, arm_y)
         for p in (proto_x, proto_y)}
    shared = sorted(set.intersection(*(set(v) for v in f.values())))
    d = np.array([(f[arm_x, proto_x][k] - f[arm_x, proto_y][k])
                  - (f[arm_y, proto_x][k] - f[arm_y, proto_y][k])
                  for k in shared])
    units = A.seq_units(shared)
    lo, hi = A.cluster_ci(d, units)
    return {"n": len(d), "k": len(set(units)), "mean": float(d.mean()),
            "lo": lo, "hi": hi}


def sec7(stats_isec, stats_own, cls):
    """The committed directional gate, via analyze.py's implementation."""
    p_holm = A.holm({c: v["p"] for c, v in stats_isec.items()
                     if v is not None and c in C.TARGETS})
    return A.verdict(cls, stats_isec.get(cls), stats_own.get(cls),
                     p_holm.get(cls))


def seed_iou(runs, protocol, arm, cls, label_proto="isec"):
    """Per-seed pooled IoU for one cell; None entries kept out of the mean so
    an absent class is absent, never silently zero."""
    v = [m["iou"][label_proto][cls] for m in runs
         if m["protocol"] == protocol and m["arm"] == arm
         and m["train_env"] == "all"]
    return [x for x in v if x is not None]


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------

def floor_gate(md, runs, fold_proto, cls, floor, name, note):
    """E3 / G1: a per-fold IoU floor. Not a paired test - a collapse
    detector, so it is read on the seed mean of every arm."""
    md += [f"### {name}", "", note, "",
           f"Floor: `{cls}` intersection IoU >= {floor} on all "
           f"{len(fold_proto)} folds.", "",
           "| fold | protocol | " + " | ".join(ARMS) + " | fold mean | |",
           "|---|---|" + "---|" * (len(ARMS) + 2)]
    verdicts = []
    for fold, proto in fold_proto.items():
        cells, allv = [], []
        for arm in ARMS:
            v = seed_iou(runs, proto, arm, cls)
            cells.append(A.mean_sd(v) if v else "-")
            allv += v
        if not allv:
            md.append(f"| {fold} | `{proto}` | " + " | ".join(cells)
                      + " | - | no runs |")
            verdicts.append(None)
            continue
        fm = float(np.mean(allv))
        ok = fm >= floor
        verdicts.append(ok)
        md.append(f"| {fold} | `{proto}` | " + " | ".join(cells)
                  + f" | **{fm:.3f}** | {'PASS' if ok else '**FAIL**'} |")
    md.append("")
    if any(v is None for v in verdicts):
        md += ["**Verdict: not computed** - some folds have no runs.", ""]
    elif all(verdicts):
        md += [f"**{name} PASSES** on all folds.", ""]
    else:
        bad = [f for f, v in zip(fold_proto, verdicts) if not v]
        md += [f"**{name} FAILS** on {', '.join(bad)}. Reported as it fell; "
               "no fold is dropped from the phase it belongs to.", ""]
    return md, verdicts


def leakage_gate(md, runs, pairs, px, name, note, cls="paddy_edge",
                 restrict_by_fold=None):
    """E1 / F1 / G2: leaked protocol MINUS blocked protocol on `cls`, one arm
    at a time, on the frames both evaluated. `pairs` is fold -> (leaked,
    blocked) protocol names."""
    md += [f"### {name}", "", note, "",
           f"Per-frame paired delta on `{cls}` (leaked minus site-blocked), "
           f"intersection protocol. Gate declared on **{GATE_ARM}**; every "
           "arm tabulated so arm-specificity stays visible.", "",
           "| fold | arm | n frames | delta | 95% CI | verdict |",
           "|---|---|---|---|---|---|"]
    verdicts = {}
    for fold, (leaked, blocked) in pairs.items():
        restrict = None if restrict_by_fold is None else restrict_by_fold[fold]
        if restrict is not None and not restrict:
            md.append(f"| {fold} | - | 0 | - | - | not evaluable "
                      "(empty fixed set) |")
            continue
        for arm in ARMS:
            si = paired_protocols(runs, arm, leaked, blocked, "isec", px,
                                  restrict)
            so = paired_protocols(runs, arm, leaked, blocked,
                                  A.own_proto(arm), px, restrict)
            if not si or si.get(cls) is None:
                md.append(f"| {fold} | {arm} | - | - | - | insufficient data |")
                if arm == GATE_ARM:
                    verdicts[fold] = None
                continue
            s, v = si[cls], sec7(si, so, cls)
            md.append(f"| {fold} | {arm} | {s['n']} | {s['mean']:+.3f} | "
                      f"[{s['lo']:+.3f}, {s['hi']:+.3f}] | {v} |")
            if arm == GATE_ARM:
                verdicts[fold] = v
    md.append("")
    return md, verdicts


def pooled_arm_gate(md, runs, fold_proto, arm_x, arm_y, cls, px, name, note):
    """E2 / G3: arm_x minus arm_y on `cls`, per-frame deltas POOLED over all
    folds as committed. Pooling happens at the frame level - folds
    evaluate disjoint sites, so concatenating deltas double-counts nothing."""
    md += [f"### {name}", "", note, "",
           f"`{arm_x}` minus `{arm_y}` on `{cls}`, per-frame deltas pooled "
           f"over {len(fold_proto)} folds (pooling as committed; no fold "
           "dropped).", "",
           "| fold | protocol | n frames | delta | 95% CI | per-fold verdict |",
           "|---|---|---|---|---|---|"]
    pooled_d, missing = [], []
    for fold, proto in fold_proto.items():
        si = A.paired_family(runs, proto, arm_x, arm_y, "isec", px)
        so = A.paired_family(runs, proto, arm_x, arm_y,
                             A.own_proto(arm_x), px)
        if not si or si.get(cls) is None:
            md.append(f"| {fold} | `{proto}` | - | - | - | insufficient data |")
            missing.append(fold)
            continue
        s = si[cls]
        md.append(f"| {fold} | `{proto}` | {s['n']} | {s['mean']:+.3f} | "
                  f"[{s['lo']:+.3f}, {s['hi']:+.3f}] | {sec7(si, so, cls)} |")
        pooled_d.append((fold, proto, s))
    md.append("")
    if missing:
        md += [f"**Verdict: not computed** - pooling is committed over "
               f"all folds and {', '.join(missing)} "
               f"{'has' if len(missing) == 1 else 'have'} no usable runs. "
               "Computing it on the folds that finished would be a different, "
               "post-hoc gate.", ""]
        return md, None
    # pooled statistic on the concatenated per-frame deltas. Folds hold out
    # disjoint sites and a sequence belongs to one site, so concatenating the
    # keys cannot merge two folds into one sequence cluster.
    kd = [_pooled_deltas(runs, proto, arm_x, arm_y, cls, px)
          for _, proto, _ in pooled_d]
    keys = [k for ks, _ in kd for k in ks]
    d = np.concatenate([dd for _, dd in kd])
    units = A.seq_units(keys)
    lo, hi = A.cluster_ci(d, units)
    p = A.paired_p(d, cls, name)
    ci_excl = lo > 0 or hi < 0
    # The third condition of sec:criteria: the same sign on the single
    # annotation (the NIR annotation when NIR is compared), pooled the same way.
    own = A.own_proto(arm_x)
    d_own = np.concatenate([
        _pooled_deltas(runs, proto, arm_x, arm_y, cls, px, own)[1]
        for _, proto, _ in pooled_d])
    consistent = bool(np.sign(d_own.mean()) == np.sign(d.mean()))
    md += [f"Pooled: n={len(d)} frames over k={len(set(units))} sequences, "
           f"delta {d.mean():+.4f}, "
           f"95% CI [{lo:+.4f}, {hi:+.4f}], Wilcoxon p={p:.2e}, "
           f"delta on the {own} annotation {d_own.mean():+.4f}.", ""]
    ok = bool(p < C.ALPHA and ci_excl and consistent)
    # Name the direction: "CI excludes 0" is satisfied by a delta of either
    # sign, and a gate written expecting arm_x to lead reads as vindicated
    # when it is arm_x that trails.
    md += [f"**{name} {'PASSES' if ok else 'FAILS'}** "
           f"({'CI excludes 0 and p<' if ok else 'gate not cleared at alpha='}"
           f"{C.ALPHA}"
           f"{', in favor of `%s`' % (arm_x if d.mean() > 0 else arm_y) if ok else ''}"
           f").", ""]
    md += site_level(pooled_d, name)
    return md, ok


def site_level_stats(fold_means):
    """One-sample t on the per-fold means: each held-out site is ONE
    observation, so this tests whether an effect replicates BETWEEN sites
    rather than only within a pooled frame set. Returns None below 3 folds -
    a cross-validation of two folds cannot estimate a between-site spread."""
    m = np.asarray(list(fold_means), float)
    if len(m) < 3:
        return None
    se = float(m.std(ddof=1) / np.sqrt(len(m)))
    t = float(m.mean() / se) if se > 0 else float("nan")
    # The critical value is returned because it, not the effect, is what makes
    # a cross-validation this short report wide intervals: at k-1 df it is far above
    # the large-sample 1.96, so an interval containing zero here is what the
    # fold count buys before any effect is measured (sec:methods).
    tcrit = float(stats.t.ppf(0.975, df=len(m) - 1))
    h = tcrit * se
    return {"k": len(m), "folds": m, "mean": float(m.mean()), "t": t,
            "tcrit": tcrit,
            "p": float(2 * stats.t.sf(abs(t), df=len(m) - 1)),
            "lo": float(m.mean() - h), "hi": float(m.mean() + h)}


def site_level(pooled_d, name):
    """Markdown for site_level_stats. Reported, never gating: the pass/fail
    rule above stays exactly as committed in config.py. This is the
    statistic that separates a general effect from a site-specific one,
    which pooling cannot do at any frame count - see sec:methods."""
    s = site_level_stats([x["mean"] for _, _, x in pooled_d])
    if s is None:
        return [f"_Site-level reading of {name}: needs 3+ folds, "
                f"have {len(pooled_d)}. Not computed._", ""]
    return [f"Site-level ({s['k']} held-out sites, one observation each): "
            f"folds {', '.join(f'{x:+.3f}' for x in s['folds'])}; "
            f"mean {s['mean']:+.4f}, t={s['t']:.2f}, p={s['p']:.3f}, "
            f"95% CI [{s['lo']:+.4f}, {s['hi']:+.4f}]. "
            + ("Replicates across sites." if s["p"] < C.ALPHA
               else "Does NOT replicate across sites."), ""]


def _pooled_deltas(runs, protocol, arm_x, arm_y, cls, px, label="isec"):
    """Per-frame deltas for one fold as (keys, deltas), recomputed so the
    pooled test runs on frames rather than on per-fold summaries. The keys
    travel with the deltas because the CI treats the sequence as its unit
    (analyze.py cluster_ci) and cannot be formed from the deltas alone. `label`
    is the ground truth: the intersection, or a single annotation for the
    third condition."""
    ci = C.CLASSES.index(cls)
    rx = [m for m in runs if m["arm"] == arm_x and m["protocol"] == protocol
          and m["train_env"] == "all" and m["_pf"] is not None]
    ry = [m for m in runs if m["arm"] == arm_y and m["protocol"] == protocol
          and m["train_env"] == "all" and m["_pf"] is not None]
    fx = A.frame_iou_by_arm(rx, label, ci, A.inc_own(label), px)
    fy = A.frame_iou_by_arm(ry, label, ci, A.inc_own(label), px)
    shared = sorted(set(fx) & set(fy))
    return shared, np.array([fx[k] - fy[k] for k in shared])


def completeness(md, runs, title, specs):
    """Report the shortfall before any gate is read. A phase that is not all
    on disk gets no verdict - see module docstring."""
    md += [f"## {title}", "",
           "| protocol | backbone | runs | required | |",
           "|---|---|---|---|---|"]
    ok = True
    for proto, backbone, want in specs:
        # train_env "all" alone: the gates read no other run, so a run on
        # disk that the matrix does not hold cannot complete a phase.
        n = len([m for m in runs if m["protocol"] == proto
                 and m.get("train_env", "all") == "all"
                 and m.get("backbone", "mit_b0") == backbone])
        done = n >= want
        ok = ok and done
        md.append(f"| `{proto}` | {backbone} | {n} | {want} | "
                  f"{'complete' if done else '**awaiting %d**' % (want - n)} |")
    md.append("")
    return md, ok


def main():
    px = A.gt_px()
    b0, b2 = load_runs("mit_b0"), load_runs("mit_b2")
    hx = load_runs(C.BACKBONE_H, C.ARCH_H)
    md = ["# Fold, backbone and architecture gates (phases E, F, G, H)", "",
          "_Auto-generated by `src/analyze_folds.py`. Gates are specified "
          "in `src/config.py` (the criterion record) and computed "
          "here; the directional gate itself is imported from "
          "`src/analyze.py`, "
          "so both analyzers share one implementation._", "",
          "A gate whose runs are not all on disk is reported as incomplete "
          "and **not** computed on a seed subset.", ""]

    # ---------------- phase E ----------------
    md, e_ok = completeness(
        md, b0, "Phase E - leave-one-site-out",
        [("loso000", "mit_b0", 20), ("loso001", "mit_b0", 20),
         ("blocked", "mit_b0", 20)])
    if e_ok:
        md, _ = floor_gate(
            md, b0, E_FOLD_PROTO, "common_road", C.PLOSO_CONTROL_FLOOR, "E3",
            "Control: does leave-one-site-out simply kill everything? If it "
            "does, `paddy_edge`'s collapse cannot be read as leakage-specific.")
        md, _ = leakage_gate(
            md, b0, {f: ("official", p) for f, p in E_FOLD_PROTO.items()}, px,
            "E1", "Does the leakage inflation survive a re-split the claim "
            "did not choose? `paddy_edge` is evaluable on 2 of 3 folds "
            "(`site001` holds n=1, descriptive only).")
        md, _ = pooled_arm_gate(
            md, b0, E_FOLD_PROTO, "NIR", "LUM", "paddy_after_driving", px,
            "E2", "The study's one real positive, re-tested across sites. "
            "Pooled over all 3 folds exactly as committed - re-scoping "
            "it after seeing E3 fail would be post-hoc recalibration.")
    else:
        md += ["_Gates E1-E3 not computed: phase E incomplete._", ""]

    # ---------------- phase F ----------------
    md, f_ok = completeness(
        md, b2, "Phase F - mit_b2 second backbone",
        [("blocked", "mit_b2", 20), ("official", "mit_b2", 20)])
    if f_ok:
        md, _ = leakage_gate(
            md, b2, {"site002 (blocked)": ("official", "blocked")}, px, "F1",
            "Does the leakage finding reproduce on a ~7x larger encoder, or "
            "is it backbone-specific?")
        md, _ = pooled_arm_gate(
            md, b2, {"site002 (blocked)": "blocked"}, "NIR", "LUM",
            "paddy_after_driving", px, "F2",
            "Does the one real positive reproduce under mit_b2?")
    else:
        md += ["_Gates F1-F2 not computed: phase F incomplete. The mit_b0 "
               "headline is unaffected - it never depended on these runs._", ""]

    # ---------------- phase G ----------------
    md, g_ok = completeness(
        md, b0, "Phase G - Paddy-restricted LOSO + matched comparator",
        [(p, "mit_b0", 20) for p in list(G_FOLD_PROTO.values()) + ["pofficial"]])
    if g_ok:
        md, _ = floor_gate(
            md, b0, G_FOLD_PROTO, "paddy_after_driving", C.PLOSO_CONTROL_FLOOR,
            "G1", "The E3 re-test on a fold set whose environment composition "
            "matches by construction.")
        md, _ = leakage_gate(
            md, b0, {f: ("pofficial", p) for f, p in G_FOLD_PROTO.items()}, px,
            "G2a", "Matched pooled contrast: both conditions are "
            f"{C.PLOSO_ENV}-only, so the environment restriction cancels and "
            "the remaining difference is site blocking.")
        md, _ = leakage_gate(
            md, b0, {f: ("pofficial", p) for f, p in G_FOLD_PROTO.items()}, px,
            "G2b", "Frame-fixed paired test (seen and unseen networks scored "
            f"on the same frames, in the {C.PLOSO_ENV} regime): evaluation frames NEITHER condition "
            "trained on, so this is not frame memorization and not a "
            "difficulty difference. Governs if it disagrees with G2a. "
            f"Evaluable on {len(C.PLOSO_G2B_FOLDS)} of "
            f"{len(C.PLOSO_RUN_FOLDS)} folds, committed as such.",
            restrict_by_fold=fixed_sets())
        md, _ = pooled_arm_gate(
            md, b0, G_FOLD_PROTO, "NIR", "LUM", "paddy_after_driving", px,
            "G3", "Phase G's load-bearing deliverable: the E2 re-test on a "
            "fold set whose control is verified.")
        md, _ = pooled_arm_gate(
            md, b0, G_FOLD_PROTO, "RGBNIR", "RGB", "paddy_after_driving", px,
            "F3 under phase G (POST-HOC)",
            "NOT a committed gate. Phase G's committed gates are G1, "
            "G2a/G2b and G3; the RUN_MATRIX nonetheless ran all four arms on "
            "all three folds, so family F3 was computable from committed "
            "runs and simply never read. It is read here, with the omission "
            "disclosed rather than repaired: the runs are committed and only "
            "the contrast is post-hoc. sec:results reports it by its "
            "per-fold signs and its site-level test.")
    else:
        md += ["_Gates G1-G3 not computed: phase G incomplete._", ""]

    # ---------------- phase H ----------------
    # Deliberately the phase-F calls with the model swapped and nothing else,
    # so a divergence is attributable to the architecture (config ledger,
    # 2026-08-05).
    md, h_ok = completeness(
        md, hx, "Phase H - second architecture family",
        [("blocked", C.BACKBONE_H, 20), ("official", C.BACKBONE_H, 20)])
    if h_ok:
        md, _ = leakage_gate(
            md, hx, {"site002 (blocked)": ("official", "blocked")}, px, "H1",
            "Does the leakage finding reproduce on a different architecture "
            f"family ({C.ARCH_H}/{C.BACKBONE_H} - convolutional backbone, "
            "pyramid-pooling decoder), or is it specific to the "
            "SegFormer family?")
        md, _ = pooled_arm_gate(
            md, hx, {"site002 (blocked)": "blocked"}, "NIR", "LUM",
            "paddy_after_driving", px, "H2",
            "Does the one real positive reproduce off the SegFormer family?")
    else:
        md += ["_Gates H1-H2 not computed: phase H incomplete. The mit_b0 "
               "headline is unaffected - it never depended on these runs._",
               ""]

    with open(OUT_MD, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
