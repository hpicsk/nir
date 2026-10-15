"""Emit the GENERATED region of paper/params.tex from committed results.

One source of truth: every experimental number in the
manuscript body is a macro produced HERE, by recomputing from the committed
run artifacts through the SAME analyzer functions the gates and
test_assertions.py use - analyze.py's paired directional gate and analyze_folds.py's
fold/backbone gates - never by parsing a rendered .md table. A number and its
macro therefore cannot drift: change the analyzer, re-run this, the manuscript
follows.

Determinism: the confidence intervals route through analyze.py's cluster_ci.
Its t intervals are RNG-free, and its bootstrap seeds its stream per call from
its own inputs, so a macro does not depend on how many other macros were
computed before it and every regeneration emits byte-identical output. Point
estimates, fold means, shares, counts, and Wilcoxon verdicts are RNG-free.

Provenance: every macro's comment names the test_assertions.py
test that binds the same number to data. That test set is the committed claim
manifest, so the macro set is scoped to it rather than to every number the
analyzers can print.

Design constants: a value the code fixes (a seed count, a floor, a learning
rate, the resample count) has its source of truth in the module that fixes it,
so it is read from there and emitted here too, never mirrored by hand into the
manual region, where a change to the code would leave the prose stale.

Populations: every macro declares the population it is measured over, as a
block marker or a per-line override, and each population resolves to the
artifact holding its members. A count macro is tagged [count] and must be
printed beside its population's unit noun, which is the population's name.
test_every_macro_declares_the_population_it_is_measured_over reads these
declarations.

Usage: python src/build_parameters.py   (reads results/, src/; writes paper/params.tex)
Then re-run `pytest test_assertions.py -q` in the same commit.
"""
import importlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C

A = importlib.import_module("analyze")       # paired directional gate
folds = importlib.import_module("analyze_folds")  # fold/backbone gates

PARAMS = os.path.join(C.ROOT, "paper", "params.tex")
# The MANUAL region ends at this line; everything from here down is regenerated.
SENTINEL = ("% ============================ GENERATED REGION "
            "============================")

# CHOSEN: display precision of every IoU, margin and interval endpoint
# printed below, the digit the paired margins are resolved to.
DECIMALS = 3

# The sets a macro can be measured over. Each name is the unit noun a count
# drawn from that set is printed with, and each resolves to the artifact
# holding the members, as path#key where one file holds more than one set.
# "design" marks a value the code fixes rather than measures.
POPULATIONS = {
    "scene-matched frames": ("results/inventory.json#rgb_nir_correspondence",
                             "frames annotated on both modalities at one "
                             "instant, counted over the studied environments"),
    "driving environments": ("results/inventory.json#image_counts",
                             "the driving environments the dataset releases"),
    "sequence pairs": ("results/integrity_checks.json#gps_sequences",
                       "training-validation pairs of the sequence-level "
                       "split's sequences, every pair whose GPS footprints "
                       "the overlap check compared"),
    "GPS-footprint overlaps": ("results/integrity_checks.json#gps_leaks",
                               "training-validation sequence pairs of the "
                               "sequence-level split whose GPS footprints "
                               "overlap"),
    "frames": ("data/processed/index.json",
               "the strided scene-matched index, and every subset of it a "
               "comparison or a measurement is scored on"),
    "sequences": ("data/processed/splits_blocked.json#per_sequence",
                  "the recording sequences of the studied environments"),
    "sites": ("data/processed/splits_blocked.json#sites",
                    "the GPS sites the sequences cluster into"),
}

WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}

# Display names of the models the run matrix registers (config.py MODEL_TAG).
# An unregistered name is a KeyError, so a model added to the matrix cannot
# print under the name of the one it replaced.
DISPLAY = {"mit_b0": "MiT-B0", "mit_b2": "MiT-B2",
           "upernet": "UPerNet", "tu-convnextv2_tiny": "ConvNeXt-V2-T"}



def _load(*parts):
    with open(os.path.join(*parts)) as f:
        return json.load(f)


def pooled_iou(runs, arm, protocol, proto, cls):
    """Seed-mean pixel-pooled IoU for one cell (joint-env), same selection as
    test_assertions._fs_iou, with its standard deviation over the seeds, since
    sec:methods prints every IoU with that spread."""
    v = [m["iou"][proto].get(cls) for m in runs
         if m["arm"] == arm and m["protocol"] == protocol
         and m["train_env"] == "all"]
    v = [x for x in v if x is not None]
    return (float(np.mean(v)), float(np.std(v))) if v else None


def frame_t_ci(deltas):
    """The t interval that treats every frame as an independent observation,
    kept for one purpose: the manuscript states that on the same deltas it
    excludes zero where the interval over sequences spans it (sec:redundancy),
    and test_a_frame_level_interval_understates_the_one_over_sequences asserts
    that verdict through this function.

    It is analyze.py t_ci with each frame its own unit, so the two intervals
    differ ONLY in the unit, which is the whole point of the comparison.

    It produces no reported interval. Everything reported routes through
    analyze.py cluster_ci, which requires the units precisely so this cannot be
    reached by forgetting an argument."""
    return A.t_ci(deltas, range(len(deltas)))


def liso_contrast(arm_x, arm_y, proto):
    """src/phases/leakage_isolation.py's seen-minus-unseen gap for one arm PAIRED against another.

    The arms' marginal intervals overlap because each carries the frame-level
    variation the two share, so comparing them says little; pairing removes
    that variation and is what makes the ordering testable. Returns None when
    the two frame sets differ, which is the case on the own annotations for
    any pair involving near-infrared - that arm is scored on its own
    annotation and therefore on different frames. Reads the per-frame deltas
    leakage_isolation.py persists, so no GPU inference is repeated."""
    ex, ey = liso[arm_x][proto], liso[arm_y][proto]
    if ex["perframe_units"] != ey["perframe_units"]:
        return None
    u = [tuple(s.split("/")) for s in ex["perframe_units"]]
    d = np.array(ex["perframe_deltas"]) - np.array(ey["perframe_deltas"])
    lo, hi = A.cluster_ci(d, u)
    return {"mean": float(d.mean()), "lo": lo, "hi": hi,
            "n": len(d), "k": len(set(u))}


def pooled_over_folds(runs, protos, arm_x, arm_y, cls):
    """Keys and per-frame deltas concatenated over a fold set - the exact
    input analyze_folds.py pooled_arm_gate tests. The keys travel because the CI
    treats the sequence as its unit and cannot be formed from deltas alone."""
    kd = [folds._pooled_deltas(runs, p, arm_x, arm_y, cls, px) for p in protos]
    return ([k for ks, _ in kd for k in ks],
            np.concatenate([d for _, d in kd]))


def fold_mean(runs, proto, cls):
    """Pooled intersection IoU of one fold, averaged over all arms and seeds:
    exactly the value analyze_folds.py floor_gate reads for E3 / G1."""
    allv = []
    for arm in folds.ARMS:
        allv += folds.seed_iou(runs, proto, arm, cls)
    return float(np.mean(allv))


# -------------------------------------------------------------------------
# recompute (fixed order; the CI-bearing calls all go through A.cluster_ci)
# -------------------------------------------------------------------------
b0 = A.load_runs()                    # mit_b0 headline runs (phases A-C, E, G)
px = A.gt_px()

# structural (RNG-free reads)
integ = _load(C.ROOT, "results", "integrity_checks.json")
inv = _load(C.ROOT, "results", "inventory.json")
# The preliminary inventory walked the whole download, orchard included,
# because its job was to survey what AI Hub ships. The study runs on C.ENVS
# only (sec:materials: the orchard recordings carry none of the target
# classes), so both dataset counts the manuscript prints are re-anchored to
# C.ENVS here rather than taken as those whole-download totals. The
# inventory's own totals stay as it recorded them.
n_paired = sum(r.get("shared_frame_indices", 0)
               for r in inv["rgb_nir_correspondence"] if r["env"] in C.ENVS)
n_overlaps = sum(1 for lk in integ["gps_leaks"]
                 if lk["train"].split("/")[0] in C.ENVS
                 and lk["val"].split("/")[0] in C.ENVS)
# The pairs those overlaps are counted among: every training sequence of the
# check against every validation sequence, over the same environments.
n_seq_pairs = math.prod(sum(1 for q in integ["gps_sequences"][sp]
                            if q.split("/")[0] in C.ENVS)
                        for sp in ("train", "val"))
splits = _load(C.SPLITS_JSON)
geo = _load(C.PROCESSED, "geometry_probe.json")
field_site0_pct = 100.0 * geo["env_composition"]["Field"]["site000"]["share_of_env"]
sep_at = geo["separation_at_committed_k_m"]
sep_past = geo["separation_first_past_committed_k_m"]
liso = _load(C.ROOT, "results", "leakage_isolation.json")["arms"]
reg = _load(C.ROOT, "results", "camera_offset.json")
# Phase correlation returns an argmax on any pair of images, so the offset it
# reports needs a null: the same estimator on frames paired across driving
# sequences (sec:materials).
xpair = reg["cross_pair"]
# NIR's paddy-edge lead on the leaked split, and the same margin at the
# site-blocked protocol's held-out site
nir_edge_off = pooled_iou(b0, "NIR", "official", "isec", "paddy_edge")
rgb_edge_off = pooled_iou(b0, "RGB", "official", "isec", "paddy_edge")
fam_off = A.paired_family(b0, "official", "NIR", "RGB", "isec", px)
fam_blk = A.paired_family(b0, "blocked", "NIR", "RGB", "isec", px)
nir_rgb_edge_off = fam_off["paddy_edge"]["mean"]     # rank on the leaked split
nir_rgb_edge_blk = fam_blk["paddy_edge"]["mean"]     # no rank at the held-out site

# NIR minus the luminance control on the blocked split, whose evaluation
# frames are one site. On the untilled paddy surface its interval spans zero
# (sec:redundancy).
lum_blk = A.paired_family(b0, "blocked", "NIR", "LUM", "isec", px)
before = lum_blk["paddy_before_driving"]

# The same untilled-surface margin at each held-out site of the
# leave-one-site-out cross-validation, which sec:redundancy prints with its
# site-level test. paddy_before_driving is evaluable on all three phase-E
# folds, unlike the paddy-restricted cross-validation where one fold holds too
# few shared frames.
before_perfold = {f: A.paired_family(b0, p, "NIR", "LUM", "isec", px)
                  ["paddy_before_driving"]["mean"]
                  for f, p in folds.E_FOLD_PROTO.items()}
before_site = folds.site_level_stats(before_perfold.values())

# fusion minus RGB on the same single-site blocked split - the contrast the
# title asks for, and the one the protocol gate set never read
_f3 = A.paired_family(b0, "blocked", "RGBNIR", "RGB", "isec", px)
f3_blk = _f3["paddy_after_driving"]
f3_furrow = _f3["field_furrow"]     # the larger effect, and it runs the other way

# leakage isolation (leakage_isolation.py): identical held-out site002 frames,
# seen vs unseen. The luminance control's inflation paired against RGB's, which
# tab:margins prints and sec:concentration reports as spanning zero.
su_lum_rgb = liso_contrast("LUM", "RGB", "isec")

# The third criterion on the leave-one-site-out cross-validation's fold at the site with
# most dry-field frames. That fold falls below the floor, so sec:concentration prints
# the value with the reading the floor gives it.
e1_nir_s0 = folds.paired_protocols(b0, "NIR", "official",
                                   folds.E_FOLD_PROTO["site000"], "isec",
                                   px)["paddy_edge"]

# leave-one-site-out control (E3): common_road fold means, intersection GT
e3_s0 = fold_mean(b0, "loso000", "common_road")
e3_s1 = fold_mean(b0, "loso001", "common_road")
e3_s2 = fold_mean(b0, "blocked", "common_road")
# The other floor's class on each site-A fold: the tilled paddy surface on the
# leave-one-site-out fold and the road on the paddy-restricted fold, which
# sec:redundancy prints to say what the floor's class decides at site A.
e3_tilled_s0 = fold_mean(b0, "loso000", "paddy_after_driving")
g1_road_s0 = fold_mean(b0, folds.G_FOLD_PROTO["site000"], "common_road")

# The tilled paddy surface: the only target class evaluable on all three folds
# of BOTH cross-validations, so it is what E2, G3 and their site-level readings test.
WORK_CLS = "paddy_after_driving"

# E2 pooled over the leave-one-site-out folds, with the site-level test
# beside it. Both are emitted because the protocol criterion is the
# pooled one and the site-level statistic is what says whether it replicates;
# reporting either alone is what sec:results has to avoid.
e2_perfold = {f: A.paired_family(b0, p, "NIR", "LUM", "isec", px)[WORK_CLS]
              ["mean"] for f, p in folds.E_FOLD_PROTO.items()}
e2_keys, e2_pool = pooled_over_folds(b0, list(folds.E_FOLD_PROTO.values()),
                                     "NIR", "LUM", WORK_CLS)
e2_lo, e2_hi = A.cluster_ci(e2_pool, A.seq_units(e2_keys))
e2_k = len(set(A.seq_units(e2_keys)))   # sequences behind the pooled interval
e2_site = folds.site_level_stats(e2_perfold.values())

# Paddy-restricted successor (phase G, mit_b0)
g1_min = min(fold_mean(b0, p, WORK_CLS)
             for p in folds.G_FOLD_PROTO.values())
G_CLS = WORK_CLS                    # the only class evaluable on all 3 folds
G_PROTOS = list(folds.G_FOLD_PROTO.values())
g3_perfold = {f: A.paired_family(b0, p, "NIR", "LUM", "isec", px)
              [G_CLS]["mean"] for f, p in folds.G_FOLD_PROTO.items()}
g3_keys, g3_pool = pooled_over_folds(b0, G_PROTOS, "NIR", "LUM", G_CLS)
g3_lo, g3_hi = A.cluster_ci(g3_pool, A.seq_units(g3_keys))
g3_k = len(set(A.seq_units(g3_keys)))
g3_mean = float(g3_pool.mean())
g3_site = folds.site_level_stats(g3_perfold.values())

# fusion minus RGB rotated over the same three held-out sites: family F3
# read under phase G. Post-hoc as a CONTRAST (the runs are in the matrix;
# analyze_folds.py's "F3 under phase G" note). The contrast is read at every
# fold of both cross-validations, a set chosen by no outcome, and tab:folds
# prints all six fold means.
f3g_perfold = {f: A.paired_family(b0, p, "RGBNIR", "RGB", "isec", px)
               [G_CLS]["mean"] for f, p in folds.G_FOLD_PROTO.items()}
f3g_site = folds.site_level_stats(f3g_perfold.values())
# The same contrast on the leave-one-site-out cross-validation. Its fold at the site with most dry-field frames falls
# below the floor, so sec:redundancy prints it beside the paddy-restricted
# folds rather than in their place.
f3e_fold = {f: A.paired_family(b0, p, "RGBNIR", "RGB", "isec", px)[G_CLS]
            for f, p in folds.E_FOLD_PROTO.items()}
f3e_perfold = {f: v["mean"] for f, v in f3e_fold.items()}
# RGB over fusion on the field furrow at the other site the
# leave-one-site-out cross-validation scores the furrow on (the fold at the site with most dry-field frames holds too
# few furrow frames to pair).
f3e_furrow_s1 = A.paired_family(b0, folds.E_FOLD_PROTO["site001"], "RGBNIR",
                                "RGB", "isec", px)["field_furrow"]
# How many frames of the site with most dry-field frames carry the furrow, from the
# split's own per-site class counts: the reason the leave-one-site-out cross-validation
# cannot score the furrow there although the site holds most dry-field frames.
_site0 = next(x for x in splits["sites"] if x["site_id"] == "site000")
n_furrow_site_zero = _site0["class_frames"].get("field_furrow", 0)
g2 = folds.paired_protocols(b0, "NIR", "pofficial",
                            folds.G_FOLD_PROTO["site000"], "isec", px)
g2_nir_s0 = g2["paddy_edge"]["mean"]
# The frames and sequences behind that inflation, which Fig. 1 plots and
# sec:concentration and the abstract count
g2_n_s0, g2_k_s0 = g2["paddy_edge"]["n"], g2["paddy_edge"]["k"]

# Design constants, read from the modules that fix them (the docstring says
# why none is mirrored by hand), and the counts that describe the design's
# material rather than a result.
train = importlib.import_module("train")
headline = next(k for k, tag in C.MODEL_TAG.items() if tag == "")
larger = next(k for k, tag in C.MODEL_TAG.items()
              if tag and k[0] == headline[0])
second = next(k for k in C.MODEL_TAG if k[0] != headline[0])
gt_sets = {tuple(sorted(m["iou"])) for m in b0}
assert len(gt_sets) == 1, f"runs disagree on their ground truths: {gt_sets}"
n_ground_truths = len(gt_sets.pop())


def _n_params(arm, arch, backbone):
    return sum(p.numel() for p in
               train.build_model(arm, arch, backbone).parameters())


# The fusion surplus the paper states as a bound: the smallest ratio over the
# models the matrix registers, floored at the printed digit so the bound holds
# of every model rather than of the headline alone.
fusion_ratio = min(_n_params("RGBNIR", *k) / _n_params("RGB", *k)
                   for k in C.MODEL_TAG)
fusion_bound = math.floor(10 * fusion_ratio) / 10


# The untilled paddy surface, where tab:perclass's pooled IoUs put
# near-infrared above RGB: the paired margin that the three conditions read.
nir_rgb_before_blk = fam_blk["paddy_before_driving"]
# The in-field-tree control under the sequence-level split: the committed
# control rule marks an effect that also appears on a control class.
nir_rgb_tree_off = fam_off["common_tree"]
# Near-infrared over the luminance control on the tilled paddy surface, the
# site-blocked margin whose frame-level interval excludes zero while the
# cluster-bootstrap interval on the same deltas spans it (sec:redundancy).
_k_after, _d_after = folds._pooled_deltas(b0, "blocked", "NIR", "LUM",
                                          "paddy_after_driving", px)
after_frame = frame_t_ci(_d_after)

# The phase F and H contrasts are not computed here. Their verdicts reach the
# reader as cells of tab:robustness, which build_tables.py derives from
# the same runs, and the phase tests bind those cells directly; a second
# derivation of the same numbers here would be a copy nothing compares.


# The sequences behind every t interval the letter prints: the intervals of
# tab:margins computed above, and every interval tab:robustness prints, read
# through build_tables.py's own robustness(). sec:criteria prints the largest.
_rob_a, _rob_b = importlib.import_module("build_tables").robustness(px)
max_t_k = max(
    [fam_blk["paddy_edge"]["k"], nir_rgb_before_blk["k"], before["k"],
     f3_blk["k"], f3_furrow["k"], su_lum_rgb["k"], f3e_fold["site001"]["k"],
     e1_nir_s0["k"], nir_rgb_tree_off["k"]]
    + [infl[a]["k"] for _, infl in _rob_a
       for a in ["NIR"] + ["NIR-" + c for c in ("RGB", "LUM", "RGBNIR")]]
    + [m["k"] for _, m in _rob_b])
assert max_t_k < A.MIN_BOOT_UNITS <= min(e2_k, g3_k), (
    f"sec:criteria calls every interval but the two pooled ones a t interval, "
    f"but the t intervals rest on up to {max_t_k} sequences and the pooled ones "
    f"on {e2_k} and {g3_k}, against analyze.py MIN_BOOT_UNITS")

# -------------------------------------------------------------------------
# format + emit
# -------------------------------------------------------------------------
def iou(x):
    return f"{x:.{DECIMALS}f}\\xspace"   # unsigned magnitude, bare in prose


def iou_sd(x):
    """A (mean, sd) IoU as the tables print one, mean with its spread."""
    return f"\\ensuremath{{{x[0]:.{DECIMALS}f} \\pm {x[1]:.{DECIMALS}f}}}\\xspace"


def sgn(x):
    return f"\\ensuremath{{{x:+.{DECIMALS}f}}}"   # signed delta, text/math safe


def ci(lo, hi):
    return (f"\\ensuremath{{[{lo:+.{DECIMALS}f},\\,"
            f"{hi:+.{DECIMALS}f}]}}")


def num(x):
    return f"{x}"                        # bare integer/decimal for \num or \SI


def word(n):
    return f"{WORDS[n]}\\xspace"          # a count set in running text


def cmd(name, body, comment, pop=None, count=False):
    """One macro line. pop overrides its block's population; count marks a
    number of population members, printed beside the unit noun."""
    tags = (f" [population: {pop}]" if pop else "") + (" [count]" if count else "")
    return f"\\newcommand{{\\{name}}}{{{body}}}  % {comment}{tags}"


groups = [
    ("design constants, read from the code that fixes them", "design", [
        cmd("nModalityArms", word(len({r["arm"] for r in C.RUN_MATRIX})),
            "input configurations in the run matrix, src/config.py RUN_MATRIX"),
        cmd("nSeeds", word(len(C.SEEDS)),
            "random seeds behind each variance estimate, src/config.py SEEDS"),
        cmd("nTargetClasses", word(len(C.TARGETS)),
            "traversability target classes, src/config.py TARGETS"),
        cmd("nControls", word(len(C.CONTROLS)),
            "held-out non-target control classes, src/config.py CONTROLS"),
        cmd("nEnvironments", word(len(C.ENVS)),
            "driving environments this study uses, src/config.py ENVS"),
        cmd("nModelConfigs", word(len(C.MODEL_TAG)),
            "models the run matrix registers, src/config.py MODEL_TAG"),
        cmd("nGroundTruths", word(n_ground_truths),
            "ground truths every run is scored against, read off the runs"),
        cmd("floorIoU", f"{C.PLOSO_CONTROL_FLOOR:.2f}\\xspace",
            "the IoU floor of both cross-validations, src/config.py PLOSO_CONTROL_FLOOR"),
        cmd("alphaLevel", f"{C.ALPHA:.2f}\\xspace",
            "significance level, src/config.py ALPHA"),
        cmd("restrictedEnv", f"{C.PLOSO_ENV.lower()}\\xspace",
            "environment of the paddy-restricted cross-validation, src/config.py PLOSO_ENV"),
        cmd("backboneSmall", f"{DISPLAY[headline[1]]}\\xspace",
            "primary SegFormer encoder, src/config.py MODEL_TAG"),
        cmd("backboneLarge", f"{DISPLAY[larger[1]]}\\xspace",
            "second SegFormer encoder, src/config.py MODEL_TAG"),
        cmd("archSecond", f"{DISPLAY[second[0]]}/\\allowbreak "
            f"{DISPLAY[second[1]]}\\xspace",
            "second architecture family, src/config.py ARCH_H/BACKBONE_H; "
            "\\allowbreak lets the line break at the slash"),
        cmd("mergeRadiusM", f"{C.RADIUS_M:g}",
            "GPS track merge radius in meters, src/config.py RADIUS_M"),
        cmd("gpsPadDeg", f"\\num{{{C.PAD_DEG:.0e}}}".replace("e-0", "e-"),
            "padding of a GPS bounding box in degrees, src/config.py PAD_DEG"),
        cmd("minValFrames", f"{C.MIN_VAL_FRAMES}\\xspace",
            "validation frames a class needs to be scored under a partition, "
            "src/config.py MIN_VAL_FRAMES"),
        cmd("minTrainFrames", f"{C.MIN_TRAIN_FRAMES}\\xspace",
            "training frames a class needs, src/config.py MIN_TRAIN_FRAMES"),
        cmd("trainEpochs", f"{C.EPOCHS}\\xspace", "src/config.py EPOCHS"),
        cmd("fusionParamFactor", f"{fusion_bound:.1f}\\xspace",
            "lower bound on the fusion configuration's parameter count over a "
            "single-stream configuration's, over every model src/config.py "
            "registers, measured from src/train.py build_model; "
            "test_fusion_carries_a_parameter_surplus"),
        cmd("bootResamples", f"\\num{{{A.BOOT_N:.0e}}}".replace("e+0", "e"),
            "src/analyze.py BOOT_N; test_boot_resample_count_matches_the_analyzer"),
        cmd("ciLevel", f"{100 * (1 - 2 * A.CI_TAIL):g}",
            "confidence level of every interval, src/analyze.py CI_TAIL"),
    ]),
    ("the material the design is applied to", "sites", [
        cmd("nSites", word(len(splits["sites"])),
            "GPS sites the sequences cluster into; "
            "test_dataset_has_three_gps_supersites", count=True),
        cmd("nDatasetEnvironments",
            word(len({k.split("|")[1] for k in inv["image_counts"]})),
            "driving environments the dataset releases, orchard included; "
            "test_orchard_carries_no_target_class",
            pop="driving environments", count=True),
        cmd("nFurrowFramesSiteZero", num(n_furrow_site_zero),
            "strided frames of the site with most dry-field frames whose annotation "
            "carries the field furrow; "
            "test_fusion_loses_to_color_on_the_dry_field_class",
            pop="frames", count=True),
    ]),
    ("dataset scale and official-split leakage", "frames", [
        cmd("nPairedFrames", num(n_paired),
            "exactly paired RGB-NIR frames in the studied environments; "
            "test_studied_envs_paired_frame_count",
            pop="scene-matched frames", count=True),
        cmd("nOfficialPairs", num(n_seq_pairs),
            "training-validation sequence pairs of the official split, "
            "studied environments, the pairs the overlap check compared; "
            "test_studied_envs_official_split_leakage",
            pop="sequence pairs", count=True),
        cmd("nOfficialOverlaps", num(n_overlaps),
            "GPS footprint overlaps, official train/val, studied "
            "environments; test_studied_envs_official_split_leakage",
            pop="GPS-footprint overlaps", count=True),
        cmd("nIndexFrames", num(splits["stats"]["n_frames_total"]),
            "SAMPLED FRAMES - the strided sample of nPairedFrames that every "
            "protocol partitions, printed under its own noun because the two "
            "populations differ by the stride; "
            "test_blocked_split_share_and_class_floors", count=True),
    ]),
    ("headline: NIR paddy_edge inflates when the split leaks, and the lead "
     "does not survive site-blocking", "frames", [
        cmd("nirEdgeOfficial", iou_sd(nir_edge_off),
            "NIR paddy_edge intersection IoU with its seed SD, sequence-level split; "
            "test_fullstudy_paddy_edge_leakage_headline"),
        cmd("rgbEdgeOfficial", iou_sd(rgb_edge_off),
            "RGB paddy_edge intersection IoU with its seed SD, sequence-level split, so the "
            "reader is not left to subtract a pooled IoU from a paired mean; "
            "test_fullstudy_paddy_edge_leakage_headline"),
        cmd("nirRgbEdgeOfficial", sgn(nir_rgb_edge_off),
            "NIR minus RGB paddy_edge on the sequence-level split's own "
            "validation frames; " + "test_near_infrared_leads_color_on_the_paddy_edge_under_the_sequence_level_split"),
        cmd("nirRgbEdgeBlocked", sgn(nir_rgb_edge_blk),
            "NIR minus RGB paddy_edge, site-blocked; the three conditions "
            "rank neither camera; " + "test_neither_camera_is_better_on_the_paddy_edge_under_site_blocking"),
        cmd("nirRgbEdgeBlockedCI", ci(fam_blk["paddy_edge"]["lo"],
                                      fam_blk["paddy_edge"]["hi"]),
            "t-based 95% CI over the sequences, site-blocked; "
            "test_neither_camera_is_better_on_the_paddy_edge_under_site_blocking"),
        cmd("nirRgbBeforeBlocked", sgn(nir_rgb_before_blk["mean"]),
            "NIR minus RGB, untilled paddy, site-blocked, paired per frame "
            "on the intersection; test_near_infrared_is_better_than_color_on_no_class_under_site_blocking"),
        cmd("nirRgbBeforeBlockedCI", ci(nir_rgb_before_blk["lo"],
                                        nir_rgb_before_blk["hi"]),
            "t-based 95% CI over the sequences; same test"),
        cmd("nirRgbTreeOfficial", sgn(nir_rgb_tree_off["mean"]),
            "NIR minus RGB on the in-field-tree control class under the "
            "sequence-level split, paired per frame; "
            "test_the_control_rule_marks_the_leaked_split"),
        cmd("nirRgbTreeOfficialCI", ci(nir_rgb_tree_off["lo"],
                                       nir_rgb_tree_off["hi"]),
            "t-based 95% CI over the sequences; same test"),
    ]),
    ("NIR over the luminance control: no advantage once the interval treats "
     "sequences as independent", "frames", [
        cmd("nirLumBeforeBlocked", sgn(before["mean"]),
            "NIR minus luminance control, paddy_before_driving, blocked isec; "
            "test_fullstudy_f2_nir_over_lum_does_not_survive_clustering"),
        cmd("nirLumBeforeBlockedCI", ci(before["lo"], before["hi"]),
            "t-based 95% CI over the sequences; same test"),
        cmd("nirLumBeforeSiteP", f"{before_site['p']:.2f}",
            "site-level p for the same comparison over the 3 LOSO folds: "
            "the effect does not replicate between sites; "
            "test_the_untilled_margin_changes_sign_across_the_held_out_sites",
            pop="sites"),
        cmd("nirLumBeforeSiteZero", sgn(before_perfold["site000"]),
            "the same margin's fold mean at site A, the fold printed beside "
            "the most negative one so the sign change reads across all three "
            "folds; same test"),
        cmd("nirLumBeforeSiteLow", sgn(min(before_perfold.values())),
            "its most negative fold - the sign reverses across the folds; "
            "same test"),
        cmd("nirLumAfterBlockedFrameCI", ci(*after_frame),
            "the frame-level t-based 95% CI of NIR minus the luminance "
            "control, paddy_after_driving, blocked isec, which excludes zero; "
            "test_a_frame_level_interval_understates_the_one_over_sequences"),
    ]),
    ("fusion vs RGB: the contrast the title asks for", "frames", [
        cmd("fusionColorBlocked", sgn(f3_blk["mean"]),
            "RGBNIR minus RGB, paddy_after_driving, single-site blocked isec; "
            "test_fusion_gain_is_confined_to_one_site"),
        cmd("fusionColorBlockedCI", ci(f3_blk["lo"], f3_blk["hi"]),
            "t-based 95% CI over the sequences; same test"),
        cmd("fusionColorFurrow", sgn(-f3_furrow["mean"]),
            "RGB minus RGBNIR on field_furrow, blocked isec. Sign flipped "
            "with the sentence's subject, so that the value, its interval, "
            "and the neighboring paddy contrast all read on one convention; "
            "test_fusion_loses_to_color_on_the_dry_field_class"),
        cmd("fusionColorFurrowCI", ci(-f3_furrow["hi"], -f3_furrow["lo"]),
            "t-based 95% CI on the same flipped contrast; same test"),
        cmd("fusionColorSiteTwo", sgn(f3g_perfold["site002"]),
            "same contrast, Paddy-LOSO fold site002 (the blocked site); "
            "same test"),
        cmd("fusionColorSiteOne", sgn(f3g_perfold["site001"]),
            "fold site001; same test"),
        cmd("fusionColorSiteZero", sgn(f3g_perfold["site000"]),
            "fold site000 - the sign reverses; same test"),
        cmd("fusionColorUnrestrictedSiteOne", sgn(f3e_perfold["site001"]),
            "fusion minus RGB, tilled paddy, leave-one-site-out cross-validation, fold "
            "site001; test_fusion_gain_is_confined_to_one_site"),
        cmd("fusionColorUnrestrictedSiteOneCI",
            ci(f3e_fold["site001"]["lo"], f3e_fold["site001"]["hi"]),
            "its t-based 95% CI over the sequences; same test"),
        cmd("fusionColorUnrestrictedSiteZero", sgn(f3e_perfold["site000"]),
            "the same, fold site000, the fold below the floor; same test"),
        cmd("fusionColorFurrowSiteOne", sgn(-f3e_furrow_s1["mean"]),
            "RGB minus fusion on the field furrow, leave-one-site-out cross-validation, "
            "fold site001, on the flipped convention of fusionColorFurrow; "
            "test_fusion_loses_to_color_on_the_dry_field_class"),
        cmd("fusionColorSiteMean", sgn(f3g_site["mean"]),
            "mean over the 3 held-out sites, one observation each; same test",
            pop="sites"),
        cmd("fusionColorSiteP", f"{f3g_site['p']:.2f}",
            "site-level p, 2 df; same test", pop="sites"),
        cmd("siteLevelTCrit", f"{f3g_site['tcrit']:.2f}",
            "critical value of EVERY site-level test in standard errors: the "
            "97.5% t quantile at k-1 df for the k held-out sites. It is set "
            "by the fold count alone, so sec:methods prints it to say how "
            "coarse a site-level reading is before any effect is measured. "
            "The interval itself is not printed: at 2 df its endpoints are "
            "set by the fold count rather than by the data, so sec:results "
            "reports the site-level p and the per-fold signs instead; "
            "test_the_site_level_tests_report_their_own_resolution",
            pop="sites"),
        cmd("siteLevelDof", word(f3g_site["k"] - 1),
            "degrees of freedom of every site-level test, k-1 for the k "
            "held-out sites; same test", pop="sites"),
    ]),
    ("leakage isolation on identical held-out frames (src/phases/leakage_isolation.py)",
     "frames", [
        cmd("suLumRgb", sgn(su_lum_rgb["mean"]),
            "PAIRED: the luminance control's paddy-edge inflation minus RGB's "
            "on the intersection ground truth at site C, whose interval spans "
            "zero; test_no_inflation_is_detected_at_site_c_on_the_primary_model"),
        cmd("suLumRgbCI", ci(su_lum_rgb["lo"], su_lum_rgb["hi"]),
            "t-based 95% CI over the sequences; same test"),
    ]),
    ("the offset between the two cameras (src/phases/camera_offset.py)",
     "frames", [
        cmd("camShiftPx", num(abs(int(reg["median_dy"]))),
            "median vertical offset between the RGB and NIR "
            "cameras, px at the working resolution; "
            "test_the_two_cameras_are_not_co_registered"),
        cmd("camShiftXPx", num(abs(int(reg["median_dx"]))),
            "median horizontal offset between the RGB and NIR cameras, px at "
            "the working resolution; same test"),
        cmd("nOffsetFrames", num(reg["n_frames"]),
            "sampled frames that the two medians are taken over; same test",
            count=True),
        cmd("nOffsetFramesPerSite",
            num(importlib.import_module("phases.camera_offset")
                .REG_SAMPLE_PER_SITE),
            "frames drawn at random from each site for the offset, "
            "src/phases/camera_offset.py REG_SAMPLE_PER_SITE; same test",
            count=True),
        cmd("camShiftControlPx", num(abs(int(xpair["median_dy"]))),
            "the same estimator's median vertical offset on frames paired "
            "across driving sequences, px - the displacement it reports from "
            "no alignment at all; same test"),
        cmd("camShiftControlXPx", num(abs(int(xpair["median_dx"]))),
            "the same estimator's median horizontal offset on frames paired "
            "across driving sequences, px; same test"),
    ]),
    ("leave-one-site-out control and why the successor is restricted",
     "frames", [
        cmd("eThreeControlSiteZero", iou(e3_s0),
            "farm access road pooled IoU averaged over configurations and "
            "seeds, leave-one-site-out fold at site A (below the floor); "
            "test_the_committed_control_criterion_failed_on_site000"),
        cmd("eThreeControlSiteOne", iou(e3_s1),
            "the same at site B (holds); same test"),
        cmd("eThreeControlSiteTwo", iou(e3_s2),
            "the same at site C (holds); same test"),
        cmd("eThreeTilledSiteZero", iou(e3_tilled_s0),
            "tilled paddy surface pooled IoU averaged over configurations "
            "and seeds, leave-one-site-out fold at site A (above the floor); "
            "test_site_a_fails_on_the_road_with_the_environment_held_fixed"),
        cmd("fieldShareSiteZeroPct", f"{field_site0_pct:.1f}",
            "share of all Field frames in site000, percent - so the "
            "leave-one-site-out fold at site A trains on a minority of them; "
            "test_field_environment_is_concentrated_in_one_site"),
        cmd("sepCommittedM", f"{sep_at:.0f}",
            "cluster separation at the committed site count, meters; "
            "test_env_stratified_resplit_is_infeasible", pop="sequences"),
        cmd("sepPastM", f"{sep_past:.1f}",
            "separation one split past it, meters (re-split infeasible); "
            "same test", pop="sequences"),
        cmd("eTwoPooled", sgn(float(e2_pool.mean())),
            "NIR minus luminance control, tilled paddy, pooled over the 3 "
            "LOSO folds; test_pooled_positives_do_not_replicate_between_sites"),
        cmd("eTwoPooledCI", ci(e2_lo, e2_hi),
            "cluster-bootstrap 95% CI, excluding zero; "
            "same test"),
        cmd("maxTSequences", num(max_t_k),
            "most sequences behind any t interval the letter prints, every "
            "interval but the two pooled ones; "
            "test_every_interval_over_few_sequences_is_a_t_interval",
            pop="sequences", count=True),
        cmd("eTwoSequences", num(e2_k),
            "sequences the pooled leave-one-site-out comparison's bootstrap "
            "resamples; test_every_interval_over_few_sequences_is_a_t_interval",
            pop="sequences", count=True),
        cmd("eTwoSiteZero", sgn(e2_perfold["site000"]),
            "fold mean at the site with most dry-field frames; "
            "test_pooled_positives_do_not_replicate_between_sites"),
        cmd("eTwoSiteOne", sgn(e2_perfold["site001"]),
            "fold mean at the remaining site; same test"),
        cmd("eTwoSiteTwo", sgn(e2_perfold["site002"]),
            "fold mean at the site-blocked protocol's held-out site; "
            "same test"),
        cmd("eTwoSiteLevelP", f"{e2_site['p']:.2f}",
            "site-level p over the same 3 held-out sites, 2 df: the pooled "
            "positive does not replicate between sites; same test",
            pop="sites"),
        cmd("eOneNirSiteZero", sgn(e1_nir_s0["mean"]),
            "near-infrared's paddy-edge inflation, sequence-level split "
            "against the leave-one-site-out fold holding out the site with most "
            "dry-field frames; test_every_committed_criterion_reaches_the_reader"),
        cmd("eOneNirSiteZeroCI", ci(e1_nir_s0["lo"], e1_nir_s0["hi"]),
            "its t-based 95% CI over the sequences, which excludes zero; "
            "test_the_inflation_on_the_fold_below_the_floor_excludes_zero"),
    ]),
    ("Paddy-restricted leave-one-site-out successor (mit_b0)", "frames", [
        cmd("gOneMinFoldMean", iou(g1_min),
            "lowest tilled paddy surface pooled IoU averaged over "
            "configurations and seeds, paddy-restricted folds (all above the "
            "floor); test_phase_g1_control_holds_on_all_paddy_folds"),
        cmd("gOneRoadSiteZero", iou(g1_road_s0),
            "farm access road pooled IoU averaged over configurations and "
            "seeds, paddy-restricted fold at site A (below the floor); "
            "test_site_a_fails_on_the_road_with_the_environment_held_fixed"),
        cmd("gTwoNirEdgeSiteZero", sgn(g2_nir_s0),
            "NIR paddy_edge inflation vs matched comparator, site000 - the "
            "largest inflation among the input configurations; "
            "test_phase_g2_leakage_holds_at_site_a_against_the_matched_comparator"),
        cmd("gTwoSiteZeroFrames", num(g2_n_s0),
            "frames behind that inflation, which Fig. 1 averages over; "
            "test_figure_one_shows_the_site_a_inflations", count=True),
        cmd("gTwoSiteZeroSequences", word(g2_k_s0),
            "sequences those frames come from; same test",
            pop="sequences", count=True),
        cmd("gThreePooled", sgn(g3_mean),
            "NIR minus luminance control, paddy_after_driving, pooled over "
            "3 Paddy folds; "
            "test_phase_g3_positive_survives_pooled_but_is_carried_by_site000"),
        cmd("gThreePooledCI", ci(g3_lo, g3_hi),
            "cluster-bootstrap 95% CI; same test"),
        cmd("gThreeSequences", num(g3_k),
            "sequences the pooled paddy-restricted comparison's bootstrap "
            "resamples; test_every_interval_over_few_sequences_is_a_t_interval",
            pop="sequences", count=True),
        cmd("gThreeSiteZero", sgn(g3_perfold["site000"]),
            "per-fold delta, site000 - the fold that carries the pooled "
            "positive; same test"),
        cmd("gThreeSiteOne", sgn(g3_perfold["site001"]),
            "per-fold delta, site001 (runs negative); same test"),
        cmd("gThreeSiteTwo", sgn(g3_perfold["site002"]),
            "per-fold delta, site002; same test"),
        cmd("gThreeSiteLevelP", f"{g3_site['p']:.2f}",
            "site-level p, 2 df; same test", pop="sites"),
    ]),
]

lines = [
    SENTINEL,
    "% Emitted by src/build_parameters.py from committed results. DO NOT HAND-EDIT.",
    "% Regenerate:  python src/build_parameters.py"
    "   (then re-run pytest test_assertions.py)",
    "% Each macro's comment names the test_assertions.py test binding it to data.",
    "% Each block names the population its macros are measured over, and a",
    "% line overrides it with its own [population: ...]. [count] marks a count",
    "% of the population's members, printed beside the population's name.",
]
for name, (artifact, what) in POPULATIONS.items():
    lines.append(f"% population {name} = {artifact} : {what}")
lines.append("")
for title, pop, cmds in groups:
    lines.append(f"% ---- {title} ---- [population: {pop}]")
    lines.extend(cmds)
    lines.append("")

with open(PARAMS) as f:
    manual = f.read().split(SENTINEL)[0].rstrip() + "\n\n"
with open(PARAMS, "w") as f:
    f.write(manual + "\n".join(lines).rstrip() + "\n")

# echo for eyeball verification against results/summary.md and folds.md
print("wrote", PARAMS, "\n")
for title, pop, cmds in groups:
    print(f"# {title}")
    for c in cmds:
        print("  " + c.split("  % ")[0])
