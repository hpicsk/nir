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
                             "frames annotated on both cameras at one "
                             "instant, counted over the studied environments"),
    "driving environments": ("results/inventory.json#image_counts",
                             "the driving environments the dataset releases"),
    "sequence pairs": ("results/integrity_checks.json#gps_sequences",
                       "training-validation pairs of the sequences of the "
                       "dataset's split, every pair whose GPS bounding boxes "
                       "the overlap check compared"),
    "overlapping pairs": ("results/integrity_checks.json#gps_leaks",
                               "training-validation sequence pairs of the "
                               "dataset's split whose padded GPS bounding "
                               "boxes overlap"),
    "frames": ("data/processed/index.json",
               "the sampled frames, and every subset of them that a "
               "comparison or a measurement is scored on"),
    "sequences": ("data/processed/splits_blocked.json#per_sequence",
                  "the recording sequences of the studied environments"),
    "sites": ("data/processed/splits_blocked.json#sites",
                    "the recording sites that the sequences cluster into"),
}

# The manuscript's name of each driving environment the code keys by.
ENV_NAME = {"Paddy": "rice-paddy", "Field": "dry-field"}

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
n_seqs = {sp: sum(1 for q in integ["gps_sequences"][sp]
                  if q.split("/")[0] in C.ENVS) for sp in ("train", "val")}
n_seq_pairs = math.prod(n_seqs.values())
splits = _load(C.SPLITS_JSON)
geo = _load(C.PROCESSED, "geometry_probe.json")
field_site0_pct = 100.0 * geo["env_composition"]["Field"]["site000"]["share_of_env"]
sep_at = geo["separation_at_committed_k_m"]
sep_past = geo["separation_first_past_committed_k_m"]
reg = _load(C.ROOT, "results", "camera_offset.json")
minima = _load(C.ROOT, "results", "minima_range.json")
# Phase correlation returns an argmax on any pair of images, so the offset it
# reports needs a null: the same estimator on frames paired across driving
# sequences (sec:materials).
xpair = reg["cross_pair"]
# NIR's paddy-edge lead on the leaked split, and the same margin at the
# site-blocked protocol's held-out site
nir_edge_off = pooled_iou(b0, "NIR", "official", "isec", "paddy_edge")
rgb_edge_off = pooled_iou(b0, "RGB", "official", "isec", "paddy_edge")
fam_off = A.paired_family(b0, "official", "NIR", "RGB", "isec", px)
fam_off_nir = A.paired_family(b0, "official", "NIR", "RGB", "nir", px)
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
f3e_site = folds.site_level_stats(f3e_perfold.values())
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


# The band or the annotation (src/phases/annotation_diagnosis.py): how the two
# annotations mark the paddy edge (D1), NIR minus RGB on the halves of the
# dataset split's validation frames that their agreement splits (D2) and on
# the RGB annotation (D3), fusion minus RGB (D4), and the diagnostic network
# that receives the NIR image and trains on the RGB annotation (D5). The
# module's own JSON is the one source; nothing is recomputed here.
diag = _load(C.ROOT, "results", "annotation_diagnosis.json")
fus_edge_off = pooled_iou(b0, "RGBNIR", "official", "isec", "paddy_edge")
cross_edge_off = pooled_iou(b0, C.DIAG_ARM, "official", "isec", "paddy_edge")
# Every configuration's pooled paddy-edge IoU at site C, whose smallness
# sec:redundancy prints as a range.
edge_blk = [pooled_iou(b0, a, "blocked", "isec", "paddy_edge")[0]
            for a in ("RGB", "LUM", "NIR", "RGBNIR")]
d1 = diag["D1"]
s1 = diag["S1"]
DG = importlib.import_module("phases.annotation_diagnosis")
s1_upper = [s1[str(b)]["D2"]["upper"]["mean"] for b in DG.REG_BOUNDS]
# NIR's and RGB's pooled paddy-edge IoU at site C on the rice-paddy
# cross-validation.
edge_rice_c = {a: pooled_iou(b0, a, "ploso002", "isec", "paddy_edge")[0]
               for a in ("RGB", "NIR")}
diag_margins = [diag["D2"]["upper"], diag["D2"]["lower"],
                diag["D3"]["NIR-RGB_on_rgb"], diag["D4"]["RGBNIR-RGB"],
                diag["D5"]["band"], diag["D5"]["annotation"]]


# The sequences behind every t interval the letter prints: the intervals of
# tab:margins computed above, and every interval tab:robustness prints, read
# through build_tables.py's own robustness(). sec:criteria prints the largest.
_rob_a, _rob_b = importlib.import_module("build_tables").robustness(px)
max_t_k = max(
    [fam_blk["paddy_edge"]["k"], nir_rgb_before_blk["k"], before["k"],
     f3_blk["k"], f3_furrow["k"], f3e_fold["site001"]["k"],
     e1_nir_s0["k"], fam_off["paddy_edge"]["k"],
     fam_off["paddy_after_driving"]["k"]]
    + [m["n_sequences"] for m in diag_margins]
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
        cmd("nInputConfigs",
            word(len({r["arm"] for r in C.RUN_MATRIX} - {C.DIAG_ARM})),
            "input configurations compared at the held-out sites, "
            "src/config.py RUN_MATRIX without the diagnostic configuration"),
        cmd("nSeeds", word(len(C.SEEDS)),
            "random seeds behind each mean and standard deviation, "
            "src/config.py SEEDS"),
        cmd("nTargetClasses", word(len(C.TARGETS)),
            "target classes, src/config.py TARGETS"),
        cmd("nEnvironments", word(len(C.ENVS)),
            "driving environments that the study uses, src/config.py ENVS"),
        cmd("nModels", word(len(C.MODEL_TAG)),
            "models, src/config.py MODEL_TAG"),
        cmd("alphaLevel", f"{C.ALPHA:.2f}\\xspace",
            "significance level of every test, src/config.py ALPHA"),
        cmd("ricePaddy", f"{ENV_NAME[C.PLOSO_ENV]}\\xspace",
            "environment of the rice-paddy cross-validation, src/config.py "
            "PLOSO_ENV"),
        cmd("encoderPrimary", f"{DISPLAY[headline[1]]}\\xspace",
            "encoder of the primary model, src/config.py MODEL_TAG"),
        cmd("encoderLarge", f"{DISPLAY[larger[1]]}\\xspace",
            "larger SegFormer encoder, src/config.py MODEL_TAG"),
        cmd("archSecond", f"{DISPLAY[second[0]]}/\\allowbreak "
            f"{DISPLAY[second[1]]}\\xspace",
            "second architecture family, src/config.py ARCH_H and "
            "BACKBONE_H; \\allowbreak lets the line break at the slash"),
        cmd("mergeRadiusM", f"{C.RADIUS_M:g}",
            "merge radius of the site clustering in meters, src/config.py "
            "RADIUS_M"),
        cmd("gpsPadDeg", f"\\num{{{C.PAD_DEG:.0e}}}".replace("e-0", "e-"),
            "padding of a GPS bounding box in degrees, src/config.py "
            "PAD_DEG"),
        cmd("minValFrames", f"{C.MIN_VAL_FRAMES}\\xspace",
            "frames of a held-out site that must carry a class for the "
            "class to be scored, src/config.py MIN_VAL_FRAMES"),
        cmd("minTrainFrames", f"{C.MIN_TRAIN_FRAMES}\\xspace",
            "training frames of a class that the fold holding out site C "
            "needs, src/config.py MIN_TRAIN_FRAMES"),
        cmd("minValLo", f"{minima['val'][0]}",
            "lowest scoring minimum that selects the same fold and scores "
            "the same classes, src/minima_range.py"),
        cmd("minValHi", f"{minima['val'][1]}",
            "highest such scoring minimum, src/minima_range.py"),
        cmd("minTrainHi", f"{minima['train'][1]}",
            "highest training minimum that selects the same fold, "
            "src/minima_range.py"),
        cmd("trainEpochs", f"{C.EPOCHS}\\xspace", "training epochs of every network, src/config.py EPOCHS"),
        cmd("fusionParamFactor", f"{fusion_bound:.1f}\\xspace",
            "lowest ratio, over the models, of the fusion configuration's "
            "parameter count to that of a configuration with one encoder, "
            "measured from src/train.py build_model; "
            "test_fusion_carries_a_parameter_surplus"),
        cmd("ciLevel", f"{100 * (1 - 2 * A.CI_TAIL):g}",
            "confidence level of every interval, src/analyze.py CI_TAIL"),
    ]),
    ("the material the design is applied to", "sites", [
        cmd("nSites", word(len(splits["sites"])),
            "recording sites that the sequences cluster into; "
            "test_the_sequences_cluster_into_three_sites", count=True),
        cmd("nDatasetEnvironments",
            word(len({k.split("|")[1] for k in inv["image_counts"]})),
            "driving environments that the dataset releases, orchard "
            "included; test_orchard_carries_no_target_class",
            pop="driving environments", count=True),
        cmd("nFurrowFramesSiteA", num(n_furrow_site_zero),
            "sampled frames of site A that carry the furrow; "
            "test_fusion_loses_to_color_on_the_dry_field_class",
            pop="frames", count=True),
    ]),
    ("the dataset and the overlap of its split", "frames", [
        cmd("nSceneMatchedFrames", num(n_paired),
            "scene-matched frames of the two studied environments; "
            "test_the_scene_matched_frame_count",
            pop="scene-matched frames", count=True),
        cmd("nSequencePairs", num(n_seq_pairs),
            "training-validation sequence pairs of the dataset's split in "
            "the two studied environments; "
            "test_every_validation_sequence_overlaps_a_training_sequence",
            pop="sequence pairs", count=True),
        cmd("nOverlappingPairs", num(n_overlaps),
            "those pairs whose padded GPS bounding boxes overlap; "
            "test_every_validation_sequence_overlaps_a_training_sequence",
            pop="overlapping pairs", count=True),
        cmd("nSampledFrames", num(splits["stats"]["n_frames_total"]),
            "sampled frames, the frames that every split partitions, fewer "
            "than the scene-matched frames by the stride; "
            "test_the_site_c_fold_share_and_class_minima", count=True),
    ]),
    ("NIR and RGB on the paddy edge under the dataset's split and with site "
     "C held out", "frames", [
        cmd("nirEdgeSplit", iou_sd(nir_edge_off),
            "NIR's pooled paddy-edge IoU under the dataset's split, with "
            "its standard deviation over the seeds; "
            "test_the_paddy_edge_iou_of_nir_and_rgb_under_the_dataset_split"),
        cmd("rgbEdgeSplit", iou_sd(rgb_edge_off),
            "RGB's pooled paddy-edge IoU under the dataset's split, with "
            "its standard deviation over the seeds; "
            "test_the_paddy_edge_iou_of_nir_and_rgb_under_the_dataset_split"),
        cmd("nirRgbEdgeSplit", sgn(nir_rgb_edge_off),
            "margin of NIR over RGB on the paddy edge under the dataset's "
            "split; "
            "test_nir_is_better_than_rgb_on_the_paddy_edge_under_the_dataset_split"),
        cmd("nirRgbEdgeSiteC", sgn(nir_rgb_edge_blk),
            "margin of NIR over RGB on the paddy edge with site C held out, "
            "which the three conditions do not rank; "
            "test_neither_camera_is_better_on_the_paddy_edge_at_site_c"),
        cmd("nirRgbEdgeSiteCCI", ci(fam_blk["paddy_edge"]["lo"],
                                      fam_blk["paddy_edge"]["hi"]),
            "its 95% t interval over the sequences; same test"),
        cmd("nirRgbUntilledSiteC", sgn(nir_rgb_before_blk["mean"]),
            "margin of NIR over RGB on the untilled paddy surface with site "
            "C held out; test_nir_is_better_than_rgb_on_no_class_at_site_c"),
        cmd("nirRgbUntilledSiteCCI", ci(nir_rgb_before_blk["lo"],
                                        nir_rgb_before_blk["hi"]),
            "its 95% t interval over the sequences; same test"),
    ]),
    ("the band or the annotation (src/phases/annotation_diagnosis.py)",
     "frames", [
        cmd("nirRgbEdgeSplitCI", ci(fam_off["paddy_edge"]["lo"],
                                       fam_off["paddy_edge"]["hi"]),
            "95% t interval over the sequences of the margin of NIR over "
            "RGB on the paddy edge under the dataset's split; "
            "test_nir_is_better_than_rgb_on_the_paddy_edge_under_the_dataset_split"),
        cmd("nirRgbEdgeOnNir", sgn(fam_off_nir["paddy_edge"]["mean"]),
            "margin of NIR over RGB on the paddy edge under the dataset's "
            "split, scored on the NIR annotation; "
            "test_nir_is_better_than_rgb_on_the_paddy_edge_under_the_dataset_split"),
        cmd("nirRgbTilledSplit", sgn(fam_off["paddy_after_driving"]["mean"]),
            "margin of NIR over RGB on the tilled paddy surface under the "
            "dataset's split; "
            "test_nir_is_better_than_rgb_on_the_paddy_edge_under_the_dataset_split"),
        cmd("nirRgbTilledSplitCI", ci(fam_off["paddy_after_driving"]["lo"],
                                        fam_off["paddy_after_driving"]["hi"]),
            "its 95% t interval over the sequences; same test"),
        cmd("fusionEdgeSplit", iou_sd(fus_edge_off),
            "fusion's pooled paddy-edge IoU under the dataset's split, with "
            "its standard deviation over the seeds; "
            "test_the_band_or_the_annotation"),
        cmd("diagEdgeSplit", iou_sd(cross_edge_off),
            "the diagnostic configuration's pooled paddy-edge IoU under the "
            "dataset's split, with its standard deviation over the seeds; "
            "test_the_band_or_the_annotation"),
        cmd("edgeSiteCLow", iou(min(edge_blk)),
            "lowest pooled paddy-edge IoU of the four input configurations "
            "with site C held out; "
            "test_no_configuration_learns_the_paddy_edge_at_a_held_out_site"),
        cmd("edgeSiteCHigh", iou(max(edge_blk)),
            "highest such IoU; same test"),
        cmd("edgeSiteCRiceNir", iou(edge_rice_c["NIR"]),
            "NIR's pooled paddy-edge IoU at site C on the rice-paddy "
            "cross-validation; "
            "test_no_configuration_learns_the_paddy_edge_at_a_held_out_site"),
        cmd("edgeSiteCRiceRgb", iou(edge_rice_c["RGB"]),
            "RGB's pooled paddy-edge IoU there; same test"),
        cmd("regBoundPx", num(DG.SHIFT_PX),
            "half-width of the registration search, in pixels; "
            "test_the_registration_bound_leaves_the_upper_half_margin_above_zero", pop="design"),
        cmd("regBoundMidPx", num(sorted(DG.REG_BOUNDS)[1]),
            "the second of the three half-widths of the search; same test", pop="design"),
        cmd("regBoundWidePx", num(max(DG.REG_BOUNDS)),
            "the widest of them; same test", pop="design"),
        cmd("regAgreeVal", f"{d1['val']['paddy_edge']['agree_reg']:.2f}",
            "IoU between the two paddy-edge annotations over the validation "
            "frames after the shift that best aligns them over every class; "
            "same test"),
        cmd("regAgreeTrain", f"{d1['train']['paddy_edge']['agree_reg']:.2f}",
            "the same over the training frames; same test"),
        cmd("regSatValPct", f"{100 * s1[str(DG.SHIFT_PX)]['val']['saturation']:.0f}",
            "percent of the validation frames that mark the paddy edge in "
            "both annotations whose shift reaches the bound; same test"),
        cmd("regSatTrainPct", f"{100 * s1[str(DG.SHIFT_PX)]['train']['saturation']:.0f}",
            "the same for the training frames; same test"),
        cmd("regAgreeValWide", f"{s1[str(max(DG.REG_BOUNDS))]['val']['agree_reg']:.2f}",
            "the validation frames' IoU with the widest search; same test"),
        cmd("regAgreeTrainWide", f"{s1[str(max(DG.REG_BOUNDS))]['train']['agree_reg']:.2f}",
            "the training frames' IoU with the widest search; same test"),
        cmd("regSatValWidePct", f"{100 * s1[str(max(DG.REG_BOUNDS))]['val']['saturation']:.0f}",
            "percent of those validation frames whose shift reaches the "
            "widest bound; same test"),
        cmd("regSatTrainWidePct", f"{100 * s1[str(max(DG.REG_BOUNDS))]['train']['saturation']:.0f}",
            "the same for the training frames; same test"),
        cmd("regUpperLow", sgn(min(s1_upper)),
            "lowest margin of NIR over RGB on the half of the validation "
            "frames with the highest agreement, over the three half-widths; "
            "same test"),
        cmd("regUpperHigh", sgn(max(s1_upper)),
            "highest such margin; same test"),
        cmd("annCountRatio", f"{d1['all']['paddy_edge']['count_ratio']:.2f}",
            "sampled frames that the NIR annotation marks with the paddy "
            "edge, over those that the RGB annotation marks; "
            "test_the_two_annotations_mark_the_paddy_edge_alike_overall"),
        cmd("annAreaRatio", f"{d1['all']['paddy_edge']['area_ratio']:.2f}",
            "mean paddy-edge pixels per marking frame, NIR annotation over "
            "RGB annotation; same test"),
        cmd("annAgreeEdgeTrain",
            f"{d1['train']['paddy_edge']['agree_raw']:.2f}",
            "IoU between the two paddy-edge annotations over the training "
            "frames of the dataset's split; "
            "test_the_two_annotations_agree_less_on_the_validation_frames"),
        cmd("annAgreeEdgeVal", f"{d1['val']['paddy_edge']['agree_raw']:.2f}",
            "the same over its validation frames; same test"),
        cmd("annUpperMargin", sgn(diag["D2"]["upper"]["mean"]),
            "margin of NIR over RGB on the paddy edge under the dataset's "
            "split, on the validation frames whose agreement after the "
            "shift is at or above the median; "
            "test_the_lead_holds_where_the_two_annotations_agree"),
        cmd("annUpperMarginCI", ci(*diag["D2"]["upper"]["ci"]),
            "its 95% t interval over the sequences; same test"),
        cmd("annLowerMargin", sgn(diag["D2"]["lower"]["mean"]),
            "the same below the median; same test"),
        cmd("annLowerMarginCI", ci(*diag["D2"]["lower"]["ci"]),
            "its 95% t interval over the sequences; same test"),
        cmd("nirRgbEdgeOnRgb", sgn(diag["D3"]["NIR-RGB_on_rgb"]["mean"]),
            "margin of NIR over RGB on the paddy edge under the dataset's "
            "split, scored on the RGB annotation; "
            "test_the_lead_is_not_detected_on_the_rgb_annotation"),
        cmd("nirRgbEdgeOnRgbCI", ci(*diag["D3"]["NIR-RGB_on_rgb"]["ci"]),
            "its 95% t interval over the sequences; same test"),
        cmd("fusionRgbEdgeSplit", sgn(diag["D4"]["RGBNIR-RGB"]["mean"]),
            "margin of fusion over RGB on the paddy edge under the "
            "dataset's split; test_the_band_or_the_annotation"),
        cmd("fusionRgbEdgeSplitCI", ci(*diag["D4"]["RGBNIR-RGB"]["ci"]),
            "its 95% t interval over the sequences; same test"),
        cmd("diagRgbEdge", sgn(diag["D5"]["band"]["mean"]),
            "margin of the diagnostic configuration over RGB on the paddy "
            "edge under the dataset's split, which changes the input image "
            "alone; test_the_band_or_the_annotation"),
        cmd("diagRgbEdgeCI", ci(*diag["D5"]["band"]["ci"]),
            "its 95% t interval over the sequences; same test"),
        cmd("nirDiagEdge", sgn(diag["D5"]["annotation"]["mean"]),
            "margin of NIR over the diagnostic configuration on the paddy "
            "edge under the dataset's split, which changes the training "
            "annotation alone; test_the_band_or_the_annotation"),
        cmd("nirDiagEdgeCI", ci(*diag["D5"]["annotation"]["ci"]),
            "its 95% t interval over the sequences; same test"),
    ]),
    ("NIR over the luminance control at the held-out sites", "frames", [
        cmd("nirLumUntilledSiteC", sgn(before["mean"]),
            "margin of NIR over the luminance control on the untilled paddy "
            "surface with site C held out; "
            "test_nir_over_the_luminance_control_at_site_c_is_not_detected"),
        cmd("nirLumUntilledSiteCCI", ci(before["lo"], before["hi"]),
            "its 95% t interval over the sequences; same test"),
        cmd("nirLumUntilledSiteP", f"{before_site['p']:.2f}",
            "site-level p of the same margin over the three held-out sites "
            "of the leave-one-site-out cross-validation; "
            "test_the_untilled_margin_changes_sign_across_the_held_out_sites",
            pop="sites"),
        cmd("nirLumUntilledSiteA", sgn(before_perfold["site000"]),
            "the same margin at site A; same test"),
        cmd("nirLumUntilledSiteLow", sgn(min(before_perfold.values())),
            "the same margin at site B, the most negative of the three; "
            "same test"),
        cmd("nirLumTilledSiteCFrameCI", ci(*after_frame),
            "95% t interval of the margin of NIR over the luminance control "
            "on the tilled paddy surface with site C held out, treating "
            "frames as independent; "
            "test_a_frame_level_interval_understates_the_one_over_sequences"),
    ]),
    ("fusion over RGB at the held-out sites", "frames", [
        cmd("fusionRgbTilledSiteC", sgn(f3_blk["mean"]),
            "margin of fusion over RGB on the tilled paddy surface with "
            "site C held out; test_fusion_gain_is_confined_to_one_site"),
        cmd("fusionRgbTilledSiteCCI", ci(f3_blk["lo"], f3_blk["hi"]),
            "its 95% t interval over the sequences; same test"),
        cmd("rgbFusionFurrowSiteC", sgn(-f3_furrow["mean"]),
            "margin of RGB over fusion on the furrow with site C held out, "
            "signed as the sentence that prints it reads; "
            "test_fusion_loses_to_color_on_the_dry_field_class"),
        cmd("rgbFusionFurrowSiteCCI", ci(-f3_furrow["hi"], -f3_furrow["lo"]),
            "its 95% t interval over the sequences; same test"),
        cmd("fusionRgbTilledRiceSiteC", sgn(f3g_perfold["site002"]),
            "margin of fusion over RGB on the tilled paddy surface at site "
            "C on the rice-paddy cross-validation; "
            "test_fusion_gain_is_confined_to_one_site"),
        cmd("fusionRgbTilledRiceSiteB", sgn(f3g_perfold["site001"]),
            "the same at site B; same test"),
        cmd("fusionRgbTilledRiceSiteA", sgn(f3g_perfold["site000"]),
            "the same at site A; same test"),
        cmd("fusionRgbTilledLeaveOutSiteB", sgn(f3e_perfold["site001"]),
            "margin of fusion over RGB on the tilled paddy surface at site "
            "B on the leave-one-site-out cross-validation; "
            "test_fusion_gain_is_confined_to_one_site"),
        cmd("fusionRgbTilledLeaveOutSiteBCI",
            ci(f3e_fold["site001"]["lo"], f3e_fold["site001"]["hi"]),
            "its 95% t interval over the sequences; same test"),
        cmd("fusionRgbTilledLeaveOutSiteA", sgn(f3e_perfold["site000"]),
            "the same at site A; same test"),
        cmd("rgbFusionFurrowSiteB", sgn(-f3e_furrow_s1["mean"]),
            "margin of RGB over fusion on the furrow at site B on the "
            "leave-one-site-out cross-validation, signed as the sentence "
            "that prints it reads; "
            "test_fusion_loses_to_color_on_the_dry_field_class"),
        cmd("fusionRgbTilledRiceMean", sgn(f3g_site["mean"]),
            "mean of the three sites' margins of fusion over RGB on the "
            "tilled paddy surface on the rice-paddy cross-validation; "
            "test_fusion_gain_is_confined_to_one_site",
            pop="sites"),
        cmd("fusionRgbTilledRiceP", f"{f3g_site['p']:.2f}",
            "its site-level p; same test", pop="sites"),
        cmd("fusionRgbTilledLeaveOutP", f"{f3e_site['p']:.2f}",
            "site-level p of the margin of fusion over RGB on the tilled "
            "paddy surface on the leave-one-site-out cross-validation; same "
            "test",
            pop="sites"),
    ]),
    ("the offset between the two cameras (src/phases/camera_offset.py)",
     "frames", [
        cmd("camShiftPx", num(abs(int(reg["median_dy"]))),
            "median vertical offset between the RGB and the NIR image, in "
            "pixels at the input resolution; "
            "test_the_two_cameras_are_not_co_registered"),
        cmd("camShiftXPx", num(abs(int(reg["median_dx"]))),
            "median horizontal offset between them; same test"),
        cmd("nOffsetFrames", num(reg["n_frames"]),
            "sampled frames that the two medians are taken over; same test",
            count=True),
        cmd("camShiftControlPx", num(abs(int(xpair["median_dy"]))),
            "median vertical offset that phase correlation finds between an "
            "RGB and an NIR image of different sequences; same test"),
        cmd("camShiftControlXPx", num(abs(int(xpair["median_dx"]))),
            "the same horizontal offset; same test"),
    ]),
    ("the control class at site A, the sites, and the pooled "
     "leave-one-site-out comparison",
     "frames", [
        cmd("roadLeaveOutSiteA", iou(e3_s0),
            "pooled IoU on the farm access road at site A on the "
            "leave-one-site-out cross-validation, averaged over the four "
            "input configurations; "
            "test_the_road_scores_low_at_site_a_without_its_frames"),
        cmd("roadLeaveOutSiteB", iou(e3_s1),
            "the same at site B; same test"),
        cmd("roadLeaveOutSiteC", iou(e3_s2),
            "the same at site C; same test"),
        cmd("tilledLeaveOutSiteA", iou(e3_tilled_s0),
            "pooled IoU on the tilled paddy surface at site A on the "
            "leave-one-site-out cross-validation, averaged over the four "
            "input configurations; "
            "test_site_a_fails_on_the_road_with_the_environment_held_fixed"),
        cmd("dryFieldShareSiteAPct", f"{field_site0_pct:.1f}",
            "percent of the sampled dry-field frames that site A holds; "
            "test_field_environment_is_concentrated_in_one_site"),
        cmd("sepThreeSitesM", f"{sep_at:.0f}",
            "smallest distance between two sequences of different sites at "
            "the merge radius, in meters; "
            "test_no_fourth_site_separates_the_sequences", pop="sequences"),
        cmd("sepFourSitesM", f"{sep_past:.1f}",
            "the same distance where the clustering first returns a fourth "
            "site, in meters; same test", pop="sequences"),
        cmd("nirLumTilledLeaveOut", sgn(float(e2_pool.mean())),
            "margin of NIR over the luminance control on the tilled paddy "
            "surface, pooled over the folds of the leave-one-site-out "
            "cross-validation; "
            "test_pooled_positives_do_not_replicate_between_sites"),
        cmd("nirLumTilledLeaveOutCI", ci(e2_lo, e2_hi),
            "its 95% cluster-bootstrap interval; same test"),
        cmd("maxTSequences", num(max_t_k),
            "most sequences behind any t interval that the letter prints; "
            "test_every_interval_over_few_sequences_is_a_t_interval",
            pop="sequences", count=True),
        cmd("nLeaveOutSequences", num(e2_k),
            "sequences that the leave-one-site-out pooled interval "
            "resamples; "
            "test_every_interval_over_few_sequences_is_a_t_interval",
            pop="sequences", count=True),
        cmd("nirLumTilledLeaveOutSiteA", sgn(e2_perfold["site000"]),
            "the same margin at site A; "
            "test_pooled_positives_do_not_replicate_between_sites"),
        cmd("nirLumTilledLeaveOutSiteB", sgn(e2_perfold["site001"]),
            "the same at site B; same test"),
        cmd("nirLumTilledLeaveOutSiteC", sgn(e2_perfold["site002"]),
            "the same at site C; same test"),
        cmd("nirLumTilledLeaveOutP", f"{e2_site['p']:.2f}",
            "site-level p of the same margin over the three held-out sites; "
            "same test",
            pop="sites"),
        cmd("nirInflationLeaveOutSiteA", sgn(e1_nir_s0["mean"]),
            "NIR's paddy-edge inflation at site A on the leave-one-site-out "
            "cross-validation, against the dataset's split; "
            "test_every_committed_criterion_reaches_the_reader"),
    ]),
    ("the rice-paddy cross-validation", "frames", [
        cmd("tilledRiceSiteA", iou(g1_min),
            "pooled IoU on the tilled paddy surface at site A on the "
            "rice-paddy cross-validation, averaged over the four input "
            "configurations, the lowest of the three sites; "
            "test_the_tilled_surface_clears_the_control_on_every_rice_paddy_fold"),
        cmd("roadRiceSiteA", iou(g1_road_s0),
            "pooled IoU on the farm access road at site A on the rice-paddy "
            "cross-validation, averaged over the four input configurations; "
            "test_site_a_fails_on_the_road_with_the_environment_held_fixed"),
        cmd("nirInflationRiceSiteA", sgn(g2_nir_s0),
            "NIR's paddy-edge inflation at site A on the rice-paddy "
            "cross-validation, against the rice-paddy part of the dataset's "
            "split, the largest of the four input configurations; "
            "test_nir_inflation_at_site_a_on_the_rice_paddy_cross_validation"),
        cmd("nInflationFramesSiteA", num(g2_n_s0),
            "frames behind that inflation, which Fig. 1 averages over; "
            "test_figure_one_shows_the_site_a_inflations", count=True),
        cmd("nInflationSequencesSiteA", word(g2_k_s0),
            "sequences those frames come from; same test",
            pop="sequences", count=True),
        cmd("nirLumTilledRice", sgn(g3_mean),
            "margin of NIR over the luminance control on the tilled paddy "
            "surface, pooled over the folds of the rice-paddy "
            "cross-validation; "
            "test_the_pooled_rice_paddy_margin_is_carried_by_site_a"),
        cmd("nirLumTilledRiceCI", ci(g3_lo, g3_hi),
            "its 95% cluster-bootstrap interval; same test"),
        cmd("nRiceSequences", num(g3_k),
            "sequences that the rice-paddy pooled interval resamples; "
            "test_every_interval_over_few_sequences_is_a_t_interval",
            pop="sequences", count=True),
        cmd("nirLumTilledRiceSiteA", sgn(g3_perfold["site000"]),
            "the same margin at site A on the rice-paddy cross-validation; "
            "test_the_pooled_rice_paddy_margin_is_carried_by_site_a"),
        cmd("nirLumTilledRiceSiteB", sgn(g3_perfold["site001"]),
            "the same at site B; same test"),
        cmd("nirLumTilledRiceSiteC", sgn(g3_perfold["site002"]),
            "the same at site C; same test"),
        cmd("nirLumTilledRiceP", f"{g3_site['p']:.2f}",
            "site-level p of the same margin over the three held-out sites; "
            "same test", pop="sites"),
    ]),
]

lines = [
    SENTINEL,
    "% Emitted by src/build_parameters.py from committed results; not edited by hand.",
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
