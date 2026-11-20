"""Executable assertions binding report/manuscript claims to actual data,
plus integrity invariants promoted from the dataset measurements. Every test loads REAL artifacts from
results/ or src/ - never synthetic data - and every failure message
quotes the claim that is now wrong.

Run: pytest test_assertions.py -q
Full-study claims are appended here in the same commit that lands the
result they quote.
"""
import ast
import glob
import importlib
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(ROOT, "src"))
import config  # the protocol record  # noqa: E402

TARGETS = ["paddy_edge", "field_furrow", "field_levee",
           "paddy_before_driving", "paddy_after_driving"]


def _load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def _read_tex():
    """The manuscript body. Read when a claim is about what the paper SAYS -
    a number computed and macro'd but never printed is not reported."""
    with open(os.path.join(ROOT, "paper", "main.tex")) as f:
        return f.read()


def _read_table(name):
    """One generated table body. Read when a claim is about what a TABLE
    says - a caption sentence the generator derives from its own cells is
    a manuscript claim like any other, and nothing else in this file sees
    it, because it never appears in main.tex."""
    with open(os.path.join(ROOT, "paper", name)) as f:
        return f.read()


def _read_params():
    """The parameter macros, i.e. every number the paper prints. Read when a
    claim is about the reported values themselves rather than the data
    behind them."""
    with open(os.path.join(ROOT, "paper", "params.tex")) as f:
        return f.read()


# ---------------- dataset: reality claims ----------------------------

def test_frame_pairing_exact():
    corr = _load("inventory.json")["rgb_nir_correspondence"]
    # sequences with 0 shared frames are vacuously exempt; any sequence
    # WITH shared frames must carry the GPS/time checks and pass them
    bad = [c for c in corr if c.get("shared_frame_indices", 0) > 0
           and (c.get("max_dlat") != 0.0 or c.get("max_dlon") != 0.0
                or c.get("same_time_frac") != 1.0)]
    assert not bad, (
        "The dataset inventory established 'same frame index = same "
        "instant (identical "
        f"GPS/time)' but {len(bad)} sequences violate it, e.g. {bad[:2]}")


def test_shared_frame_count():
    # the inventory's own number, over the whole download it walked
    n = _load("integrity_checks.json")["n_shared_frames"]
    assert n == 82747, (
        "The dataset inventory established '82,747 exactly-paired "
        "RGB-NIR frames' "
        f"but integrity data shows {n}")


def test_the_scene_matched_frame_count():
    # the manuscript's number: the same count restricted to config.py ENVS,
    # which is where \nSceneMatchedFrames is re-anchored in build_parameters.py
    corr = _load("inventory.json")["rgb_nir_correspondence"]
    n = sum(r.get("shared_frame_indices", 0)
            for r in corr if r["env"] in config.ENVS)
    assert n == 79029, (
        "Manuscript sec. materials claims the two studied environments contribute "
        f"'79,029' frames annotated on both modalities, but inventory over "
        f"ENVS={config.ENVS} shows {n}")


def test_sequence_level_split_holds_whole_sequences():
    corr = _load("inventory.json")["rgb_nir_correspondence"]
    train = {(c["env"], c["seq"]) for c in corr if c["split"] == "train"}
    val = {(c["env"], c["seq"]) for c in corr if c["split"] == "val"}
    both = sorted(train & val)
    assert not both, (
        "Manuscript sec. materials claims the sequence-level split is 'drawn over "
        "whole recording sequences, and no sequence appears on both sides of "
        f"it', but {len(both)} sequences appear on both, e.g. {both[:3]}")


def test_orchard_carries_no_target_class():
    inv = _load("inventory.json")["instance_counts"]
    hits = {k: v for k, v in inv.items()
            if k.split("|")[1] == "Orchard" and k.split("|")[3] in TARGETS and v}
    assert not hits, (
        "Manuscript sec. materials claims 'the orchard recordings carry none "
        f"of the {len(TARGETS)} traversability classes', which is why "
        f"config.py ENVS={config.ENVS} drops them, but inventory shows "
        f"{hits}")


def test_evaluated_frames_are_daylight_clear_and_single_province():
    # file name schema, from the construction guideline:
    # NIA_AgricultureAD_{place}_{modality}_{position}_{region}_{time}_
    # {daylight}_{weather}_{seq}_total.json
    files = _load("inventory.json")["sequence_files"]
    conds = set()
    for key, name in files.items():
        if key.split("|")[1] not in config.ENVS:
            continue
        p = name.split("_")
        conds.add((p[5].lower(), p[7], p[8]))
    assert conds == {("gyeongsangnamdo", "day", "sunny")}, (
        "Manuscript sec. discussion states that the finding rests on the "
        "dataset's 'paddy and dry-field recordings, all made in daylight and "
        "clear weather in one Korean province, Gyeongsangnam-do', but the "
        f"evaluated sequences span {sorted(conds)}")

    # the night claim covers the whole released dataset rather than the studied
    # environments, so the daylight token is read over every sequence on disk
    released = {n.split("_")[7] for n in files.values()}
    assert released == {"day"}, (
        "Manuscript sec. discussion states that the dataset 'holds no night "
        "imagery', but the released sequences span "
        f"{sorted(released)}")


def test_paddy_water_untrainable():
    inv = _load("inventory.json")["instance_counts"]
    n = sum(v for k, v in inv.items()
            if k.startswith("train|") and k.endswith("|paddy_water"))
    assert n == 0, (
        "The study drops paddy_water as 'untrained/untrainable', "
        f"but the inventory shows {n} training instances")


# ---------------- dataset: integrity findings ------------------------

def test_official_split_site_leakage():
    # the inventory's own number, over the whole download it walked
    leaks = _load("integrity_checks.json")["gps_leaks"]
    assert len(leaks) == 276, (
        "The dataset integrity check established '276 GPS footprint "
        "overlaps' between "
        f"train and val sequences of the sequence-level split; data shows {len(leaks)}")


def test_every_validation_sequence_overlaps_a_training_sequence():
    """Manuscript sec. materials: 'In the two studied environments, each validation sequence overlaps one or more training sequences, and
    the 1066 training-validation sequence pairs include 270 overlapping
    pairs.' The count and the pairs
    it is counted among are read off the footprint check's own record, and
    every overlap has to be one of those pairs, or the first number is not a
    share of the second."""
    integ = _load("integrity_checks.json")
    # the manuscript's numbers: the check restricted to config.py ENVS, which
    # is where build_parameters.py re-anchors \nOverlappingPairs and
    # \nSequencePairs
    studied = {sp: {q for q in integ["gps_sequences"][sp]
                    if q.split("/")[0] in config.ENVS}
               for sp in ("train", "val")}
    leaks = [lk for lk in integ["gps_leaks"]
             if lk["train"].split("/")[0] in config.ENVS
             and lk["val"].split("/")[0] in config.ENVS]
    assert len(leaks) == 270, (
        "Manuscript sec. materials says 270 pairs of a training and a "
        "validation sequence overlap, but the overlaps over "
        f"ENVS={config.ENVS} number {len(leaks)}")
    pairs = len(studied["train"]) * len(studied["val"])
    assert pairs == 1066, (
        "Manuscript sec. materials counts the overlaps among '1066' "
        "training-validation sequence pairs, but the sequences the footprint "
        f"check compared over ENVS={config.ENVS} give {pairs}")
    outside = [lk for lk in leaks if lk["train"] not in studied["train"]
               or lk["val"] not in studied["val"]]
    assert not outside, (
        "manuscript sec. materials prints the overlaps as a count among "
        "the training-validation sequence pairs, but these overlaps name a "
        f"sequence the pairs do not hold: {outside[:3]}")
    lone = sorted(studied["val"] - {lk["val"] for lk in leaks})
    assert not lone, (
        "Manuscript sec. intro: every validation sequence was recorded on "
        "ground that a training sequence also covers, but these overlap none: "
        f"{lone}")


# ------------- full study: site-blocked split invariants (splits_blocked.py) -------

def _splits():
    with open(os.path.join(ROOT, "data", "processed", "splits_blocked.json")) as f:
        return json.load(f)


def test_blocked_split_zero_track_leakage():
    """Re-verified from the stored per-frame GPS tracks (inputs), not
    from the builder's own stored verdict."""
    sp = _splits()
    p = sp["params"]
    pts = {"train": [], "val": []}
    for member, split in sp["per_sequence"].items():
        pts[split].extend(sp["strided_tracks"][member])

    def xy(latlon):
        a = np.asarray(latlon)
        return np.stack(
            [a[:, 1] * p["m_per_deg_lon"] * np.cos(np.radians(p["lat0"])),
             a[:, 0] * p["m_per_deg_lat"]], axis=1)

    d, _ = cKDTree(xy(pts["train"])).query(xy(pts["val"]), k=1)
    assert float(d.min()) > p["radius_m"], (
        "config.py specifies a site-blocked split with no used val "
        f"frame within {p['radius_m']} m of a used train frame; recomputed "
        f"min cross-split distance is {float(d.min()):.2f} m")


def test_the_site_c_fold_share_and_class_minima():
    sp = _splits()
    st = sp["stats"]
    # Sec. methods prints the index size beside the share it is a share of,
    # because both populations are counted in 'frames' and a reader who takes
    # the share of the paired-frame count lands five times high.
    params = _read_params()
    m = re.search(r"\\newcommand\{\\nSampledFrames\}\{(\d+)\}", params)
    assert m and int(m.group(1)) == st["n_frames_total"], (
        "sec. methods states the site-blocked validation share as a share "
        f"'of the index'; the index holds {st['n_frames_total']} frames and the "
        f"macro says {m.group(1) if m else 'nothing'}")
    assert round(st["n_frames_val"] / st["n_frames_total"], 3) == \
        round(st["val_share"], 3), (
        "the share and the two counts sec. methods prints no longer agree, "
        "so a reader multiplying them does not land on the validation set")
    assert abs(st["val_share"] - config.VAL_TARGET) <= config.VAL_TOL, (
        f"Blocked val share {st['val_share']:.3f} outside protocol "
        f"{config.VAL_TARGET}+/-{config.VAL_TOL}")
    for cls, d in st["class_frames"].items():
        if cls in config.FLOOR_EXEMPT:
            continue
        assert d["val"] >= config.MIN_VAL_FRAMES \
            and d["train"] >= config.MIN_TRAIN_FRAMES, (
            f"Committed class floors (>= {config.MIN_VAL_FRAMES} "
            f"val / >= {config.MIN_TRAIN_FRAMES} train frames) violated "
            f"for {cls}: {d}")


def test_the_printed_minima_ranges_select_the_same_site_and_classes():
    """Manuscript sec. protocols: 'The fold and the scored classes stay the same for any scoring minimum
    from 29 to 54 frames in place of 50, and for any training minimum up to
    822 frames in place of 200.' The macros are read from
    results/minima_range.json, and each end of each range is recomputed here,
    each minimum varied with the other held fixed: the end itself keeps the
    selected site and the scored classes, and the next value beyond it does
    not."""
    M = importlib.import_module("minima_range")
    got = json.load(open(os.path.join(RES, "minima_range.json")))
    params = _read_params()
    for macro, value in (("minValLo", got["val"][0]), ("minValHi", got["val"][1]),
                         ("minTrainHi", got["train"][1])):
        m = re.search(rf"\\newcommand\{{\\{macro}\}}\{{(\d+)\}}", params)
        assert m and int(m.group(1)) == value, (
            f"sec. protocols prints \\{macro} as {m.group(1) if m else 'nothing'}, "
            f"but src/minima_range.py gives {value}")
    lo, hi = got["val"]
    t_hi = got["train"][1]
    assert lo <= config.MIN_VAL_FRAMES <= hi and config.MIN_TRAIN_FRAMES <= t_hi, (
        "the committed minima lie outside the ranges that sec. protocols prints")
    sites = _splits()["sites"]
    base = M.selected(sites, config.MIN_VAL_FRAMES, config.MIN_TRAIN_FRAMES)
    counts = M.evaluation_counts()

    def same(v, t):
        scored = all((n >= v) == (n >= config.MIN_VAL_FRAMES) for n in counts)
        return scored and M.selected(sites, v, t) == base
    assert same(lo, config.MIN_TRAIN_FRAMES) and same(hi, config.MIN_TRAIN_FRAMES), (
        f"a validation minimum of {lo} or {hi} changes the selected site or the "
        "scored classes, which sec. protocols says it keeps")
    assert not same(lo - 1, config.MIN_TRAIN_FRAMES), (
        f"a validation minimum of {lo - 1} also keeps the selected site and the "
        f"scored classes, so the printed range does not start at {lo}")
    assert not same(hi + 1, config.MIN_TRAIN_FRAMES), (
        f"a validation minimum of {hi + 1} also keeps the selected site and the "
        f"scored classes, so the printed range does not end at {hi}")
    assert same(config.MIN_VAL_FRAMES, 0) and same(config.MIN_VAL_FRAMES, t_hi), (
        f"a training minimum of 0 or {t_hi} changes the selected site, which "
        "sec. protocols says it keeps")
    assert not same(config.MIN_VAL_FRAMES, t_hi + 1), (
        f"a training minimum of {t_hi + 1} also selects the same site, so the "
        f"printed range does not end at {t_hi}")


def test_the_sequences_cluster_into_three_sites():
    sp = _splits()
    assert len(sp["sites"]) == 3, (
        "Report/manuscript claim: the dataset collapses to exactly 3 GPS "
        f"sites at 11 m track radius; split data shows {len(sp['sites'])}")


def test_index_class_floors_blocked():
    """Floors re-checked on RASTERIZED presence (what training sees),
    not only on polygon-label presence (what splits_blocked.py checked)."""
    with open(os.path.join(ROOT, "data", "processed", "index_stats.json")) as f:
        st = json.load(f)
    p = st["protocols"]["blocked"]
    for cls in config.CLASSES[1:]:
        if cls in config.FLOOR_EXEMPT:
            continue
        tr = p["train"]["classes"][cls]["frames_union"]
        va = p["val"]["classes"][cls]["frames_union"]
        assert va >= config.MIN_VAL_FRAMES \
            and tr >= config.MIN_TRAIN_FRAMES, (
            f"Committed floors (>= {config.MIN_VAL_FRAMES} val / "
            f">= {config.MIN_TRAIN_FRAMES} train) violated on the built "
            f"index for {cls}: train {tr}, val {va}")


def test_index_consistent_with_split():
    """Drift detector between splits_blocked.py (split) and
    src/phases/build_index.py (index): same
    universe, same stride => same frame totals per blocked split."""
    with open(os.path.join(ROOT, "data", "processed", "index_stats.json")) as f:
        st = json.load(f)
    sp = _splits()["stats"]
    n_tr = st["protocols"]["blocked"]["train"]["n_frames"]
    n_va = st["protocols"]["blocked"]["val"]["n_frames"]
    assert n_va == sp["n_frames_val"] \
        and n_tr + n_va == sp["n_frames_total"], (
        "Index and split disagree on blocked frame totals: index "
        f"{n_tr}+{n_va} vs split {sp['n_frames_total']} "
        f"(val {sp['n_frames_val']}) - one artifact is stale "
        "(ground-truth drift)")


def test_official_protocol_fully_site_contaminated():
    """Every site contains BOTH train and val sequences of the
    sequence-level split, i.e. that protocol evaluates entirely within
    training sites - the stronger form of the 276-overlap
    finding."""
    sp = _splits()
    for s in sp["sites"]:
        osplits = {m.split("/")[0] for m in s["members"]}
        assert osplits == {"train", "val"}, (
            "Claim: every GPS site spans both sides of the "
            f"sequence-level split (so it evaluates within site); {s['site_id']} spans "
            f"only {sorted(osplits)}")


# ------------- full study: model capacity (src/train.py) -------------
# The capacity claims in sec:methods are properties of build_model, not of
# any results file, so these two build the real networks. MODEL_TAG is the
# protocol record of which models ran, so a fourth model added to
# the matrix is checked here without editing this file.

def _param_counts(arch, backbone):
    train = importlib.import_module("train")
    return {a: sum(p.numel()
                   for p in train.build_model(a, arch, backbone).parameters())
            for a in ("RGB", "NIR", "LUM", "RGBNIR")}


def test_single_stream_arms_are_one_network_at_one_parameter_count():
    """Manuscript sec. methods: 'The NIR configuration receives the single
    channel of the NIR image copied into all three input channels' and the
    luminance control's grayscale is 'copied in the same way', so RGB, NIR and
    the luminance control are one network at one parameter count. Replicating
    a single band across three channels is what buys that, so an _input() that
    stopped replicating would change the stem and break the match."""
    for arch, backbone in config.MODEL_TAG:
        single = {a: n for a, n in _param_counts(arch, backbone).items()
                  if a != "RGBNIR"}
        assert len(set(single.values())) == 1, (
            "Manuscript sec. methods claims RGB, near-infrared, and the "
            "luminance control are 'one network at one parameter count', but "
            f"{arch}/{backbone} gives {single}")


def test_both_repeats_carry_more_parameters_than_the_primary_model():
    """Manuscript sec. methods: 'Both repeats carry more parameters than the
    primary model'. Compared configuration by configuration, over every model
    config.py registers; the primary model is the one whose run tag is
    empty."""
    counts = {k: _param_counts(*k) for k in config.MODEL_TAG}
    primary = next(k for k, tag in config.MODEL_TAG.items() if not tag)
    smaller = [(k, a) for k in counts if k != primary for a in counts[k]
               if counts[k][a] <= counts[primary][a]]
    assert not smaller, (
        "sec. methods says both repeats carry more parameters than the "
        f"primary model, but these do not: {smaller}")


def test_each_arm_trains_on_the_annotation_the_manuscript_names():
    """Manuscript sec. intro: 'NIR trains on the NIR annotation and the other
    configurations on the RGB annotation'; sec. methods: 'RGB, the luminance
    control and fusion train on the RGB annotation, so that the comparison
    of fusion with RGB isolates the added NIR input.' and a fifth, diagnostic configuration
    'receives the NIR image as the NIR configuration does and trains on the
    RGB annotation.'

    That sentence replaced one claiming the input bands were the only factor
    that varied between the matched three, which src/train.py SUP_MASK contradicts.
    The annotation is the second factor, so Methods names it. Asserted
    over every arm SUP_MASK defines rather than over the three the sentence
    names, so an arm added later has to be placed by the sentence too."""
    train = importlib.import_module("train")
    named = {"NIR": "mask_nir", "RGB": "mask_rgb",
             "LUM": "mask_rgb", "RGBNIR": "mask_rgb", config.DIAG_ARM: "mask_rgb"}
    assert train.SUP_MASK == named, (
        "Manuscript sec. methods claims near-infrared trains on the "
        "near-infrared annotation and the other three configurations and the "
        "diagnostic network on the RGB annotation, but src/train.py SUP_MASK "
        f"is {train.SUP_MASK}")
    ds = train.FSSeg([], config.DIAG_ARM, train=False)
    rec = {"img_rgb": None, "img_nir": _index()[0]["img_nir"]}
    rec["img_nir"] = os.path.join(config.STUDY_DATA, rec["img_nir"])
    if os.path.exists(rec["img_nir"]):
        ref = train.FSSeg([], "NIR", train=False)._input(rec)
        assert np.array_equal(ds._input(rec), ref), (
            "Manuscript sec. methods says the diagnostic network receives the "
            "NIR image as the NIR configuration does, but src/train.py builds "
            "a different input for it")


def test_fusion_carries_a_parameter_surplus():
    """Manuscript sec. methods: 'On each model, fusion carries at least 1.8 times the parameters of a
    configuration with one encoder, so fusion cannot fall short of RGB for
    lack of capacity.'. That sentence
    discharges the two-stream BASELINE tag in src/train.py: the surplus runs in
    fusion's favor, so the fusion nulls are not a capacity handicap. Checked on
    every model in the matrix, not just the headline."""
    for arch, backbone in config.MODEL_TAG:
        c = _param_counts(arch, backbone)
        ratio = c["RGBNIR"] / c["RGB"]
        assert ratio >= 1.8, (
            "Manuscript sec. methods claims the fusion configuration carries "
            "'at least 1.8 times' a single-stream configuration's parameters "
            f"in every model reported, but {arch}/{backbone} gives "
            f"{ratio:.3f}x ({c['RGBNIR']} vs {c['RGB']})")


def test_every_module_constant_carries_a_dof_tag():
    """Every numeric value a module fixes at module level is a degree of
    freedom, and the module says which kind: MEASURED (by which script),
    CHOSEN (structural, with why) or FITTED (to which target). A fit's
    evidence is what it reproduces beyond the parameters it spent, so an
    untagged constant leaves that count unknowable.

    The constants are read off each module's AST under src/, not listed, so a
    constant added tomorrow is covered the day it is added. A tag counts in the
    module docstring on a line naming the constant, on the assignment line, or
    in the comment block directly above it."""
    tag = re.compile(r"\b(MEASURED|CHOSEN|FITTED)\b")
    untagged = []
    for path in sorted(glob.glob(os.path.join(ROOT, "src", "**", "*.py"),
                                 recursive=True)):
        src = open(path).read()
        tree, lines = ast.parse(src), src.splitlines()
        doc = (ast.get_docstring(tree) or "").splitlines()
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            vals = (node.value.elts if isinstance(node.value, ast.Tuple)
                    else [node.value])
            if not all(isinstance(v, ast.Constant) and type(v.value) in (int, float)
                       for v in vals):
                continue
            for t in node.targets:
                if not (isinstance(t, ast.Name) and t.id.isupper()):
                    continue
                name = t.id
                above, k = [], node.lineno - 2
                while k >= 0 and lines[k].lstrip().startswith("#"):
                    above.append(lines[k])
                    k -= 1
                if (any(re.search(r"\b%s\b" % name, d) and tag.search(d) for d in doc)
                        or tag.search(lines[node.lineno - 1])
                        or any(tag.search(a) for a in above)):
                    continue
                untagged.append(f"{os.path.relpath(path, ROOT)}:{node.lineno} {name}")
    assert not untagged, (
        "these module constants carry no MEASURED, CHOSEN or FITTED tag, so "
        "nothing says whether each was measured, chosen or fitted:\n    "
        + "\n    ".join(untagged))


def test_every_baseline_tag_records_the_tuning_the_runs_received():
    """Every fixed architecture, encoder, optimizer or reference design
    carries a BASELINE tag naming the choice, the obvious alternative in one
    sentence, the tuning each configuration received, and where the
    objection is discharged. The redundancy claim is negative, so it is only as
    strong as that last field.

    The tuning field is derived here from what the runs record rather than
    taken from the tag: a tag saying one configuration was tried is true only
    while every run's metrics.json records the same hyperparameters. The tags
    are read from every module under src/ except config.py, whose amendment
    text is the dated record of what was committed and keeps its tags as they
    were written then."""
    tags = []
    for path in sorted(glob.glob(os.path.join(ROOT, "src", "**", "*.py"),
                                 recursive=True)):
        if os.path.basename(path) == "config.py":
            continue
        doc = ast.get_docstring(ast.parse(open(path).read())) or ""
        flat = " ".join(doc.split())
        tags += [(os.path.relpath(path, ROOT), t)
                 for t in re.findall(r"BASELINE: (.+?)(?= BASELINE: |$)", flat)]
    assert len(tags) >= 4, (
        f"only {len(tags)} BASELINE tags found under src/; the architecture, "
        "optimizer, pretraining and fusion choices each owe one")

    hyper = {(m["epochs"], m["batch"], m["crop"], m["lr"], m["weight_decay"])
             for m in (json.load(open(p)) for p in
                       glob.glob(os.path.join(RES, "runs", "*", "metrics.json")))}
    labels = set(re.findall(r"\\label\{([^}]+)\}", _read_tex()))
    for path, tag in tags:
        fields = [f.strip() for f in tag.split(" | ")]
        keys = [f.split(":")[0] for f in fields[1:4]]
        assert keys == ["obvious alternative", "tuning", "discharge"], (
            f"{path}: a BASELINE tag carries fields {keys}, not the obvious "
            f"alternative, the tuning, and the discharge: {tag[:120]}")
        tuning, discharge = fields[2], fields[3]
        if len(hyper) == 1:
            assert "one configuration" in tuning or "one fusion design" in tuning, (
                f"{path}: every run records one hyperparameter set {hyper}, "
                f"but the tag says '{tuning}'")
        slug = re.match(r"discharge: (sec:[\w-]+|app:[\w-]+)", discharge)
        assert slug and slug.group(1) in labels, (
            f"{path}: a BASELINE tag under a published claim discharges at "
            f"'{discharge[:60]}', which names no section of main.tex")


def test_figure_one_shows_the_site_a_inflations():
    """Caption of fig:leakage: 'A bar averages the per-frame IoUs of five networks, one per seed, over
    the 256 frames of site A that both annotations mark with
    the paddy edge and that no network of the figure trains on. Those frames
    come from three recording sequences.' and 'Every configuration
    scores higher with the site seen, NIR by the most.'

    The bars are src/build_figures.py's, and a configuration's two bars differ
    by its paddy-edge inflation at site A on the rice-paddy cross-validation,
    the site-A row of tab:robustness(a), so the figure and the table cannot
    report different leaks. The frame and sequence counts the caption prints
    are that inflation's."""
    fig = importlib.import_module("build_figures")
    runs, px = _folds.load_runs("mit_b0"), _folds.A.gt_px()
    bars = fig.bars(runs, px)
    assert sorted(bars) == sorted(_folds.ARMS), (
        "the caption of fig:leakage reads every configuration's bars, but the "
        f"figure plots {sorted(bars)} and the runs hold {sorted(_folds.ARMS)}")
    proto = _folds.G_FOLD_PROTO["site000"]
    gains = {}
    for arm, (unseen, seen) in bars.items():
        s = _folds.paired_protocols(runs, arm, "pofficial", proto, "isec",
                                    px)["paddy_edge"]
        assert math.isclose(seen - unseen, s["mean"], abs_tol=1e-9), (
            f"Fig. 1's two {arm} bars differ by {seen - unseen:+.4f}, but its "
            f"paddy-edge inflation at site A is {s['mean']:+.4f}")
        gains[arm] = seen - unseen
    assert min(gains.values()) > 0 and max(gains, key=gains.get) == "NIR", (
        "the caption of fig:leakage says every configuration scores higher "
        "with the site seen and NIR gains the most, but the gains are "
        f"{ {k: round(v, 3) for k, v in gains.items()} }")
    nir = _folds.paired_protocols(runs, "NIR", "pofficial", proto, "isec",
                                  px)["paddy_edge"]
    params = _read_params()
    frames = re.search(r"\\newcommand\{\\nInflationFramesSiteA\}\{(\d+)\}",
                       params).group(1)
    seqs = re.search(r"\\newcommand\{\\nInflationSequencesSiteA\}\{(\w+)\\xspace\}",
                     params).group(1)
    words = ast.literal_eval(re.search(
        r"^WORDS = (\{.*\})$",
        open(os.path.join(ROOT, "src", "build_parameters.py")).read(),
        re.M).group(1))
    assert int(frames) == nir["n"] and seqs == words[nir["k"]], (
        f"the caption of fig:leakage counts {frames} frames from {seqs} "
        f"sequences, but the site-A inflation rests on {nir['n']} frames from "
        f"{nir['k']} sequences")
    flat = " ".join(_read_tex().split())
    assert ("Every configuration scores higher with the site seen, NIR by "
            "the most") in flat, (
        "the caption of fig:leakage no longer states the takeaway this test "
        "checks")


# ------------- full study: leakage-aware headline claims (src/train.py,
# analyze.py) ----
# The manuscript's thesis (sec. intro, sec. results): under the site-blocked
# protocol near-infrared is better than RGB on no target class, and its
# paddy-edge lead under the sequence-level split is not detected there.
# Asserted on the protocol mit_b0 phase-A/C runs (phase-F mit_b2 and phase-H's
# second architecture excluded, as in analyze.py). Directional claims use
# analyze.py's protocol paired gate (Holm + the sequence-clustered interval on
# the intersection protocol), never a bare mean comparison.

def _study_runs():
    out = []
    for mp in glob.glob(os.path.join(RES, "runs", "*", "metrics.json")):
        with open(mp) as f:
            m = json.load(f)
        if (m.get("backbone", "mit_b0"), m.get("arch", "segformer")) \
                == ("mit_b0", "segformer"):
            out.append(m)
    return out


def _fs_iou(runs, arm, protocol, proto, cls):
    """Mean over seeds of pixel-pooled IoU (arm/protocol/class, joint-env)
    on label protocol `proto` in {rgb,nir,isec}."""
    vals = [m["iou"][proto].get(cls) for m in runs
            if m["arm"] == arm and m["protocol"] == protocol
            and m["train_env"] == "all"]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _fs_family_verdicts(arm_x, arm_y, protocol="blocked"):
    """Per-target verdicts from analyze.py's exact protocol gate."""
    A = importlib.import_module("analyze")
    runs, px = A.load_runs(), A.gt_px()
    op = "nir" if arm_x == "NIR" else "rgb"
    si = A.paired_family(runs, protocol, arm_x, arm_y, "isec", px)
    so = A.paired_family(runs, protocol, arm_x, arm_y, op, px)
    ph = A.holm({c: (si[c]["p"] if si.get(c) else None) for c in TARGETS})
    return {c: A.verdict(c, si.get(c), so.get(c), ph.get(c)) for c in TARGETS}


def test_the_paddy_edge_iou_of_nir_and_rgb_under_the_dataset_split():
    """sec. results headline: NIR paddy_edge is leakage-inflated on
    the sequence-level split (~0.339 isec) and collapses site-blocked (~0.015)."""
    runs = _study_runs()
    off = _fs_iou(runs, "NIR", "official", "isec", "paddy_edge")
    blk = _fs_iou(runs, "NIR", "blocked", "isec", "paddy_edge")
    assert off is not None and blk is not None and off > 0.25 and blk < 0.05, (
        "Manuscript sec. results: NIR paddy_edge inflates on the leaking "
        "sequence-level split and collapses site-blocked (0.339 -> 0.015 isec); "
        f"data shows official={off} blocked={blk}")
    # 'the sequence-level split, whose validation frames that carry the class
    # come from sites A and C': the sites of the frames that the paired
    # comparison scores, read through the split's own site table.
    A = importlib.import_module("analyze")
    keys, _ = _folds._pooled_deltas(_folds.load_runs("mit_b0"), "official",
                                    "NIR", "RGB", "paddy_edge", A.gt_px())
    k2s = _index_key2site()
    sites = {k2s[f"val/{env}/{seq}"] for env, seq, _f in keys}
    assert sites == {"site000", "site002"}, (
        "Manuscript sec. concentration says the sequence-level split's "
        "validation frames that carry the paddy edge come from sites A and C, "
        f"but they come from {sorted(sites)}")


def test_neither_camera_is_better_on_the_paddy_edge_at_site_c():
    """Manuscript sec. redundancy: 'At site C, NIR is better than RGB on none
    of the five target classes'. The verdict asserted is analyze.py verdict(),
    so the sentence follows the three conditions rather than the interval
    alone."""
    v = _fs_family_verdicts("NIR", "RGB")["paddy_edge"]
    assert v == "comparable", (
        "Manuscript sec. concentration reports that neither camera is better "
        "on the paddy edge under site blocking, but the three conditions return "
        f"'{v}'")
    A = importlib.import_module("analyze")
    s = A.paired_family(A.load_runs(), "blocked", "NIR", "RGB", "isec",
                        A.gt_px())["paddy_edge"]
    assert s["lo"] < 0 < s["hi"], (
        "Manuscript sec. concentration prints the site-blocked paddy-edge "
        "margin NIR - RGB with an interval spanning zero, but it is "
        f"[{s['lo']:+.4f},{s['hi']:+.4f}]")


def test_nir_is_better_than_rgb_on_the_paddy_edge_under_the_dataset_split():
    """Manuscript sec. split: 'Under the dataset's split, NIR is better than
    RGB on the paddy edge and on no other target class' and 'On the tilled
    paddy surface, NIR's pooled IoU also exceeds RGB's, but the per-frame
    comparison, which weights each frame equally, detects no difference', and
    fusion 'is better than RGB on the paddy edge too'. The abstract: 'NIR scores a higher intersection-over-union (IoU) than RGB on the paddy
    edge'."""
    v = _fs_family_verdicts("NIR", "RGB", protocol="official")
    wins = sorted(c for c, x in v.items() if x.startswith("DIRECTIONAL (+)"))
    assert wins == ["paddy_edge"], (
        "Manuscript sec. split: under the dataset's split near-infrared leads "
        "RGB on the paddy edge and on no other target class, but the "
        f"directional rule favors it on {wins}")
    runs = _study_runs()
    A = importlib.import_module("analyze")
    s_ = A.paired_family(A.load_runs(), "official", "NIR", "RGB", "isec",
                         A.gt_px())["paddy_after_driving"]
    assert (_fs_iou(runs, "NIR", "official", "isec", "paddy_after_driving")
            > _fs_iou(runs, "RGB", "official", "isec", "paddy_after_driving")
            and s_["lo"] < 0 < s_["hi"]), (
        "Manuscript sec. split: on the tilled surface NIR's pooled IoU exceeds "
        "RGB's while the per-frame interval spans zero, but the interval is "
        f"[{s_['lo']:+.3f}, {s_['hi']:+.3f}]")
    f = _fs_family_verdicts("RGBNIR", "RGB", protocol="official")
    assert f["paddy_edge"].startswith("DIRECTIONAL (+)"), (
        "Manuscript sec. split: fusion is better than RGB on the paddy edge "
        f"under the dataset's split, but the rule returns '{f['paddy_edge']}'")
    # 'Scored on the NIR annotation, as the third condition requires, its
    # paddy-edge margin over RGB is +0.207.'
    on_nir = A.paired_family(A.load_runs(), "official", "NIR", "RGB",
                             A.own_proto("NIR"), A.gt_px())["paddy_edge"]
    assert (A.own_proto("NIR") == "nir" and on_nir["mean"] > 0
            and _macro_value("nirRgbEdgeOnNir") == f"{on_nir['mean']:+.3f}"), (
        "sec. split prints NIR's paddy-edge margin on the NIR annotation as "
        f"{_macro_value('nirRgbEdgeOnNir')}, and the runs give {on_nir['mean']:+.4f}")


def test_nir_is_better_than_rgb_on_no_class_at_site_c():
    """Manuscript sec. redundancy: 'At site C, NIR is better than RGB on none
    of the five target classes ... NIR's pooled IoU exceeds RGB's on the
    untilled paddy surface, but the per-frame comparison detects no
    difference between the two.'"""
    v = _fs_family_verdicts("NIR", "RGB")
    wins = [c for c, x in v.items() if x.startswith("DIRECTIONAL (+)")]
    assert not wins, (
        "Manuscript sec. redundancy: under the site-blocked protocol "
        "near-infrared is better than RGB on none of the five target "
        f"classes, but the three conditions favor it on {wins}")
    A = importlib.import_module("analyze")
    s = A.paired_family(A.load_runs(), "blocked", "NIR", "RGB", "isec",
                        A.gt_px())["paddy_before_driving"]
    assert s["lo"] < 0 < s["hi"], (
        "Manuscript sec. redundancy: near-infrared's per-frame margin over "
        "RGB on the untilled paddy surface has an interval spanning zero, "
        f"but it is [{s['lo']:+.4f},{s['hi']:+.4f}]")


def test_nir_over_the_luminance_control_at_site_c_is_not_detected():
    """Manuscript sec. redundancy: 'At site C, NIR is not better than the luminance control on either paddy
    surface. On the tilled surface, the comparison detects no difference on
    any of the three models', and
    tab:margins prints the untilled surface's interval. Both surfaces are
    asserted through the three conditions, and the untilled one's interval as
    well."""
    v = _fs_family_verdicts("NIR", "LUM")
    for cls in ("paddy_after_driving", "paddy_before_driving"):
        assert not v[cls].startswith("DIRECTIONAL"), (
            "Manuscript sec. redundancy reports that near-infrared is not better "
            f"than the luminance control on either paddy surface, but on {cls} "
            f"the three conditions return '{v[cls]}'")
    A = importlib.import_module("analyze")
    s = A.paired_family(A.load_runs(), "blocked", "NIR", "LUM", "isec",
                        A.gt_px())["paddy_before_driving"]
    assert s["lo"] < 0 < s["hi"], (
        "Manuscript sec. redundancy says the paired comparison on the untilled "
        f"surface detects no difference, but its interval is [{s['lo']:+.4f},{s['hi']:+.4f}]")


def test_the_untilled_margin_changes_sign_across_the_held_out_sites():
    """Manuscript sec. redundancy: 'On the untilled surface, the
    leave-one-site-out margin of NIR over the luminance control is +0.012 at
    site A, -0.038 at site B, and +0.020 at site C'. The contrast is
    evaluable on all three leave-one-site-out folds, its sign reverses across
    them, and the site-level test does not separate it from zero."""
    runs = _folds.load_runs("mit_b0")
    px = _folds.A.gt_px()
    per = {f: _folds.A.paired_family(runs, p, "NIR", "LUM", "isec", px)
           ["paddy_before_driving"]["mean"]
           for f, p in _folds.E_FOLD_PROTO.items()}
    assert len(per) == 3 and all(v is not None for v in per.values()), (
        "manuscript sec. redundancy reports the untilled-surface margin at all "
        f"three folds, but it is evaluable on {len(per)}")
    s = _folds.site_level_stats(per.values())
    assert s["p"] > config.ALPHA, (
        "manuscript sec. redundancy states the untilled-surface margin does not "
        f"hold at every held-out site, but the site-level test gives "
        f"p={s['p']:.3f} on folds { {k: round(v, 4) for k, v in per.items()} }")
    assert min(per.values()) < 0 < max(per.values()), (
        "manuscript sec. redundancy reports the margin reverses sign across "
        f"the folds, but they are { {k: round(v, 4) for k, v in per.items()} }")
    for fold, macro in (("site000", "nirLumUntilledSiteA"),
                        ("site001", "nirLumUntilledSiteLow"),
                        ("site002", "nirLumUntilledSiteC")):
        printed = re.search(r"\\newcommand\{\\" + macro
                            + r"\}\{\\ensuremath\{([+-]\d\.\d+)\}\}",
                            _read_params()).group(1)
        assert f"{per[fold]:+.3f}" == printed, (
            f"sec. redundancy prints \\{macro} as the untilled-surface margin "
            f"at fold {fold}, but that fold's mean is {per[fold]:+.4f}")


def test_boot_resample_count_matches_the_analyzer():
    """Manuscript sec. criteria: 'comes from a cluster bootstrap'. tab:margins
    prints each pooled endpoint to the third decimal, so the resampling error
    of analyze.py BOOT_N resamples is read off the draw: a second stream of
    the same size moves neither pooled endpoint by half a unit of that
    decimal."""
    A = importlib.import_module("analyze")
    runs, px = _folds.load_runs("mit_b0"), _folds.A.gt_px()
    for fold_proto in (_folds.E_FOLD_PROTO, _folds.G_FOLD_PROTO):
        keys, d = _pooled_over_folds(runs, fold_proto.values(), "NIR", "LUM",
                                     "paddy_after_driving", px)
        u = A.seq_units(keys)
        first = A.boot_ci(d, u)
        second = A.boot_ci(d, u, seed=A.BOOT_SEED + 1)
        moved = max(abs(a - b) for a, b in zip(first, second))
        assert moved < 5e-4, (
            "tab:margins prints each pooled endpoint to the third decimal, but a "
            f"second draw moves it {moved:.5f}")


def _diag_pairs():
    """(units, per-frame deltas) behind each paddy-edge margin of
    sec. split that src/phases/annotation_diagnosis.py computes, recomputed
    here from the runs' per-frame counts and the committed agreement file
    rather than read from the module's JSON."""
    A = importlib.import_module("analyze")
    D = importlib.import_module("phases.annotation_diagnosis")
    px = A.gt_px()
    runs = [m for m in A.load_runs() if m["protocol"] == "official"]

    def fi(arm, proto):
        return A.frame_iou_by_arm([m for m in runs if m["arm"] == arm], proto,
                                  D.EDGE, A.inc_own(proto), px)

    def pair(fx, fy, keys=None):
        ks = sorted(set(fx) & set(fy) if keys is None
                    else set(keys) & set(fx) & set(fy))
        return A.seq_units(ks), np.array([fx[k] - fy[k] for k in ks])

    z = np.load(D.AGREE_NPZ)
    row = {(str(e), str(q), int(f)): i
           for i, (e, q, f) in enumerate(zip(z["env"], z["seq"], z["frame"]))}
    nir, rgb = fi("NIR", "isec"), fi("RGB", "isec")
    agree = {k: z["inter_reg"][row[k], D.EDGE] / z["union_reg"][row[k], D.EDGE]
             for k in nir}
    med = float(np.median(list(agree.values())))
    cross = fi(config.DIAG_ARM, "isec")
    return {
        "annUpperMarginCI": pair(nir, rgb, [k for k, a in agree.items()
                                            if a >= med]),
        "annLowerMarginCI": pair(nir, rgb, [k for k, a in agree.items()
                                            if a < med]),
        "nirRgbEdgeOnRgbCI": pair(fi("NIR", "rgb"), fi("RGB", "rgb")),
        "fusionRgbEdgeSplitCI": pair(fi("RGBNIR", "isec"), rgb),
        "diagRgbEdgeCI": pair(cross, rgb),
        "nirDiagEdgeCI": pair(nir, cross),
    }


def test_every_interval_over_few_sequences_is_a_t_interval():
    """Manuscript sec. criteria: 'Every interval below except two rests on at
    most 13 sequences and is a $t$ interval ..., because a cluster bootstrap,
    which resamples whole sequences, estimates the variance well only over
    many sequences ... The two exceptions pool all folds of a cross-validation and are
    cluster-bootstrap intervals of the margin of NIR over the luminance
    control on the tilled surface. The interval on the
    leave-one-site-out cross-validation resamples the 76 sequences with
    tilled-surface frames, some of them dry-field sequences whose frames show
    a tilled paddy, and the interval on the rice-paddy one resamples the 72
    rice-paddy sequences.'
    analyze.py makes the
    sort with
    MIN_BOOT_UNITS, which lies between the two counts.

    Both ends are derived. The intervals are every interval macro the
    manuscript prints, each recomputed here from the deltas behind it, and
    every interval tab:robustness prints, read through build_tables.py's own
    robustness(); a macro this map does not hold fails rather than going
    unchecked. The frame-level interval of sec. redundancy is the one that
    treats frames as independent, and its own test reads it."""
    A = importlib.import_module("analyze")
    runs, px = _folds.load_runs("mit_b0"), _folds.A.gt_px()

    def fam(proto, ax, ay, cls, sign=1):
        keys, d = _folds._pooled_deltas(runs, proto, ax, ay, cls, px)
        return A.seq_units(keys), sign * d

    def pooled(fold_proto):
        keys, d = _pooled_over_folds(runs, fold_proto.values(), "NIR", "LUM",
                                     "paddy_after_driving", px)
        return A.seq_units(keys), d

    behind = {
        "nirRgbEdgeSiteCCI": fam("blocked", "NIR", "RGB", "paddy_edge"),
        "nirRgbUntilledSiteCCI": fam("blocked", "NIR", "RGB",
                                     "paddy_before_driving"),
        "nirLumUntilledSiteCCI": fam("blocked", "NIR", "LUM",
                                     "paddy_before_driving"),
        "fusionRgbTilledSiteCCI": fam("blocked", "RGBNIR", "RGB",
                                    "paddy_after_driving"),
        "rgbFusionFurrowSiteCCI": fam("blocked", "RGBNIR", "RGB", "field_furrow",
                                   -1),
        "fusionRgbTilledLeaveOutSiteBCI": fam(
            _folds.E_FOLD_PROTO["site001"], "RGBNIR", "RGB",
            "paddy_after_driving"),
        "nirLumTilledLeaveOutCI": pooled(_folds.E_FOLD_PROTO),
        "nirLumTilledRiceCI": pooled(_folds.G_FOLD_PROTO),
        "nirRgbEdgeSplitCI": fam("official", "NIR", "RGB", "paddy_edge"),
        "nirRgbTilledSplitCI": fam("official", "NIR", "RGB",
                                     "paddy_after_driving"),
        **_diag_pairs(),
    }
    printed = set(re.findall(r"\\(\w+CI)\b", _read_tex()))
    printed.discard("nirLumTilledSiteCFrameCI")
    assert printed == set(behind), (
        "sec. criteria states which intervals are t-based, and this test "
        f"recomputes {sorted(behind)}, but the manuscript prints "
        f"{sorted(printed)}")
    params = _read_params()
    t_ks, boot_ks = [], {}
    for macro, (units, d) in behind.items():
        k = len(set(units))
        lo, hi = A.cluster_ci(d, units)
        want = (A.t_ci if k < A.MIN_BOOT_UNITS else A.boot_ci)(d, units)
        got = re.search(r"\\newcommand\{\\" + macro
                        + r"\}\{\\ensuremath\{\[([^,]+),\\,([^\]]+)\]\}\}",
                        params).groups()
        assert (lo, hi) == want and got == (f"{lo:+.3f}", f"{hi:+.3f}"), (
            f"\\{macro} prints [{got[0]}, {got[1]}], and the interval over its "
            f"{k} sequences is [{lo:+.4f}, {hi:+.4f}]")
        (t_ks.append(k) if k < A.MIN_BOOT_UNITS else
         boot_ks.__setitem__(macro, k))
    rows_a, rows_b = _gen_tables().robustness(px)
    t_ks += [infl_[k]["k"] for _, infl_ in rows_a
             for k in ["NIR"] + ["NIR-" + a for a in _gen_tables().CONTRAST_ARMS]]
    t_ks += [m["k"] for _, m in rows_b]
    assert sorted(boot_ks) == ["nirLumTilledLeaveOutCI", "nirLumTilledRiceCI"], (
        "sec. criteria says only the two comparisons that pool all folds are "
        f"bootstrapped, but {sorted(boot_ks)} rest on {A.MIN_BOOT_UNITS} or "
        "more sequences")
    assert max(t_ks) < A.MIN_BOOT_UNITS <= min(boot_ks.values()), (
        f"the t intervals rest on up to {max(t_ks)} sequences and the "
        f"bootstrap ones on {sorted(boot_ks.values())}, against the threshold "
        f"{A.MIN_BOOT_UNITS}")
    for macro, name in (("nirLumTilledLeaveOutCI", "nLeaveOutSequences"),
                        ("nirLumTilledRiceCI", "nRiceSequences")):
        n = re.search(r"\\newcommand\{\\" + name + r"\}\{(\d+)\}",
                      params).group(1)
        assert int(n) == boot_ks[macro], (
            f"sec. criteria prints \\{name} as {n} sequences, but "
            f"\\{macro} resamples {boot_ks[macro]}")
    # 'some of them dry-field sequences whose frames show a tilled paddy' and
    # 'the 72 rice-paddy sequences': the sequences that the leave-one-site-out
    # pool holds beyond the rice-paddy pool are dry-field sequences, and the
    # rice-paddy pool holds rice-paddy sequences only.
    seqs = {}
    for name, fp in (("loso", _folds.E_FOLD_PROTO), ("rice", _folds.G_FOLD_PROTO)):
        keys, _d = _pooled_over_folds(runs, fp.values(), "NIR", "LUM",
                                      "paddy_after_driving", px)
        seqs[name] = {(k[0], k[1]) for k in keys}
    extra = seqs["loso"] - seqs["rice"]
    assert (extra and {e for e, _q in extra} == {"Field"}
            and {e for e, _q in seqs["rice"]} == {"Paddy"}
            and seqs["rice"] <= seqs["loso"]), (
        "sec. criteria says the leave-one-site-out pool adds dry-field sequences "
        "to the rice-paddy sequences of the rice-paddy pool, but the extra "
        f"sequences are {sorted(extra)[:5]}")
    m = re.search(r"\\newcommand\{\\maxTSequences\}\{(\d+)\}", params)
    assert m and int(m.group(1)) == max(t_ks), (
        f"sec. criteria prints {m.group(1) if m else 'no count'} as the most "
        f"sequences behind a t interval, but the t intervals rest on up to "
        f"{max(t_ks)}")


def test_fusion_gain_is_confined_to_one_site():
    """Manuscript sec. redundancy: 'fusion is better than RGB on the tilled
    paddy surface' at site C, and 'Fusion's margin over RGB on the tilled surface is negative at site A on
    both cross-validations (Table~\\ref{tab:folds}), although the margin and
    its interval lie above zero at site B on the leave-one-site-out
    cross-validation (Table~\\ref{tab:margins}).' On both cross-validations
    the folds change sign and the site-level test does not separate them from
    zero.

    This is the claim that replaced a positive contribution, so both halves
    are asserted: the single-site gain is real (dropping it would hide what
    site C reports) and it does not replicate (asserting only the gain would
    republish the error)."""
    A = importlib.import_module("analyze")
    runs, px = A.load_runs(), A.gt_px()
    cls = "paddy_after_driving"
    blk = A.paired_family(runs, "blocked", "RGBNIR", "RGB", "isec", px)[cls]
    assert blk["lo"] > 0, (
        "manuscript reports a fusion gain on tilled paddy surface at the "
        f"blocked site, but the delta is {blk['mean']:+.3f} "
        f"CI [{blk['lo']:+.3f},{blk['hi']:+.3f}]")
    per_fold = {f: A.paired_family(runs, p, "RGBNIR", "RGB", "isec", px)
                [cls]["mean"] for f, p in _folds.G_FOLD_PROTO.items()}
    assert min(per_fold.values()) < 0 < max(per_fold.values()), (
        "manuscript states the fusion gain changes sign across held-out "
        f"sites, but the folds are "
        f"{ {k: round(v, 3) for k, v in per_fold.items()} }")
    assert per_fold["site000"] < 0, (
        "manuscript sec. redundancy states the paddy-restricted margin of "
        "fusion over RGB is negative at the site with most dry-field frames, "
        f"but it is {per_fold['site000']:+.3f}")
    s = _folds.site_level_stats(per_fold.values())
    assert s["p"] > config.ALPHA, (
        "manuscript states the fusion effect is indistinguishable from zero "
        f"site-to-site, but the site-level test gives p={s['p']:.3f} "
        f"CI [{s['lo']:+.3f},{s['hi']:+.3f}]")
    unres = {f: A.paired_family(runs, p, "RGBNIR", "RGB", "isec", px)[cls]
             for f, p in _folds.E_FOLD_PROTO.items()}
    road = {f: _fold_mean(runs, p, "common_road")
            for f, p in _folds.E_FOLD_PROTO.items()}
    below = [f for f, v in road.items() if v < config.PLOSO_CONTROL_FLOOR]
    assert below == ["site000"], (
        "manuscript sec. redundancy calls the leave-one-site-out fold at the "
        "site with most dry-field frames the one below the floor, but the folds "
        f"below it are {below}")
    # "Both margins on the folds above the floor" ranges over the folds the
    # floor admits, derived from the fold means rather than named, and "Both"
    # says there are two
    above = {f: s for f, s in unres.items() if f not in below}
    assert len(above) == 2 and all(s["lo"] > 0 for s in above.values()), (
        "manuscript sec. redundancy: 'fusion scores higher than RGB at site B "
        "and at site C', the folds above the floor, but those folds give "
        f"{ {f: (round(s['lo'], 4), round(s['hi'], 4)) for f, s in above.items()} }")
    assert unres["site000"]["mean"] < 0, (
        "manuscript tab:folds prints fusion's margin as negative on the "
        "leave-one-site-out fold at the site with most dry-field frames, but it is "
        f"{unres['site000']['mean']:+.3f}")
    # 'On both cross-validations, the three sites' mean margins of fusion
    # over RGB on the tilled surface change sign, so the site-level test does
    # not reject a mean of zero on either'
    loso = {f: v["mean"] for f, v in unres.items()}
    assert min(loso.values()) < 0 < max(loso.values()), (
        "sec. redundancy says the leave-one-site-out means of fusion over RGB "
        f"on the tilled surface change sign, but they are {loso}")
    s_e = _folds.site_level_stats(loso.values())
    assert (s_e["p"] > config.ALPHA
            and _macro_value("fusionRgbTilledLeaveOutP") == f"{s_e['p']:.2f}"), (
        "sec. redundancy prints the leave-one-site-out site-level p of fusion "
        f"over RGB as {_macro_value('fusionRgbTilledLeaveOutP')}, and the "
        f"folds give p = {s_e['p']:.3f}")


def test_fusion_loses_to_color_on_the_dry_field_class():
    """Manuscript sec. redundancy: 'RGB is better than fusion on the dry-field furrow', with an interval excluding zero, and 'RGB's margin over fusion on the furrow is -0.010 at site B on the
    leave-one-site-out cross-validation. The furrow is not scored at site A, the third site, because none of its
    dry-field frames carries the furrow and only 10 frames of its rice-paddy
    sequences do.' The
    set of folds scoring the furrow is derived from the runs, so a fold that
    starts scoring it fails the second sentence."""
    A = importlib.import_module("analyze")
    fam = A.paired_family(A.load_runs(), "blocked", "RGBNIR", "RGB", "isec",
                          A.gt_px())
    furrow, after = fam["field_furrow"], fam["paddy_after_driving"]
    assert furrow["hi"] < 0, (
        "manuscript sec. results reports RGB above fusion on field_furrow "
        f"with the interval excluding zero, but the delta is "
        f"{furrow['mean']:+.3f} CI [{furrow['lo']:+.3f},{furrow['hi']:+.3f}]")
    runs, px = A.load_runs(), A.gt_px()
    scored = {f: A.paired_family(runs, p, "RGBNIR", "RGB", "isec", px)
              .get("field_furrow") for f, p in _folds.E_FOLD_PROTO.items()}
    scored = {f: v for f, v in scored.items() if v is not None}
    sites = {x["site_id"]: x for x in json.load(open(config.SPLITS_JSON))["sites"]}
    furrow0 = sites["site000"]["class_frames"].get("field_furrow", 0)
    printed = re.search(r"\\newcommand\{\\nFurrowFramesSiteA\}\{(\d+)\}",
                        _read_params())
    assert printed and int(printed.group(1)) == furrow0 < config.MIN_VAL_FRAMES, (
        "manuscript sec. redundancy states only "
        f"{printed.group(1) if printed else None} frames of the site with most "
        f"dry-field frames carry the furrow, and the split counts {furrow0}")
    ci = config.CLASSES.index("field_furrow")
    envs = {r["env"] for r in config.load_index()
            if _index_key2site()[f"{r['osplit']}/{r['env']}/{r['seq']}"] == "site000"
            and (r["px_rgb"][ci] > 0 or r["px_nir"][ci] > 0)}
    assert envs == {"Paddy"}, (
        "manuscript sec. redundancy says every frame of site A that carries the "
        f"furrow is from a rice-paddy sequence, but they come from {envs}")
    assert set(scored) == {"site001", "site002"}, (
        "manuscript sec. redundancy states the leave-one-site-out cross-validation scores "
        "the furrow at the site-blocked protocol's site and the "
        f"remaining one only, but it scores {sorted(scored)}")
    assert scored["site001"]["mean"] > 0, (
        "manuscript sec. redundancy reports RGB's lead over fusion on the "
        "furrow reversing sign at the remaining site, but fusion minus "
        f"RGB there is {scored['site001']['mean']:+.3f}")


def test_pooled_positives_do_not_replicate_between_sites():
    """Manuscript sec. redundancy: 'Pooled over the folds, NIR is better than the luminance control on the
    tilled surface on both cross-validations ..., but the site-level test
    does not detect NIR's margin on either'. One site carries a
    pooled pass at any n, so the site-level test is the one a pooled value
    cannot stand in for.

    Asserted on both cross-validations at once - E2 (leave-one-site-out) and G3
    (rice paddy) - because the Discussion's prescription ('a
    leave-one-site-out cross-validation with a site-level test') rests on the
    pair, not on either alone."""
    runs = _folds.load_runs("mit_b0")
    px = _folds.A.gt_px()
    cls = "paddy_after_driving"
    for name, proto_map in (("E2", _folds.E_FOLD_PROTO),
                            ("G3", _folds.G_FOLD_PROTO)):
        per_fold = {f: _folds.A.paired_family(runs, p, "NIR", "LUM", "isec",
                                              px)[cls]["mean"]
                    for f, p in proto_map.items()}
        keys, d = _pooled_over_folds(runs, proto_map.values(), "NIR", "LUM",
                                     cls, px)
        lo, hi = _folds.A.cluster_ci(d, _folds.A.seq_units(keys))
        assert lo > 0, (
            f"manuscript states gate {name} passes pooled even after "
            f"sequence clustering, but the pooled delta is {d.mean():+.4f} "
            f"CI [{lo:+.4f},{hi:+.4f}]")
        s = _folds.site_level_stats(per_fold.values())
        assert s["p"] > config.ALPHA, (
            f"manuscript states the {name} pooled pass does not replicate "
            f"between sites, but the site-level test gives p={s['p']:.3f} on "
            f"folds { {k: round(v, 3) for k, v in per_fold.items()} }")


def test_the_two_cameras_are_not_co_registered():
    """Manuscript sec. materials: 'We estimate the offset that best aligns
    the two images by phase correlation on their gradient magnitudes. Over
    1200 frames drawn at random from the sampled frames, the median offset is
    7 px vertically and 6 px horizontally' and 'Applied to pairs of an RGB
    and an NIR image from different sequences, which have no alignment to
    find, phase correlation finds medians of only 1 px and 0 px'. The four
    medians are the \\camShiftPx,
    \\camShiftXPx, \\camShiftControlPx and \\camShiftControlXPx macros.

    Phase correlation returns an argmax on any two images, so the offset
    measures the cameras only where it exceeds the estimator's own null.
    Asserted against that null rather than against a threshold anyone chose,
    on both of the statistics the null returns: the matched displacement is
    the larger, and the matched correlation peak the stronger."""
    reg = _load("camera_offset.json")
    ctrl = reg["cross_pair"]
    per_site = importlib.import_module("phases.camera_offset").REG_SAMPLE_PER_SITE
    n_sites = len(_splits()["sites"])
    assert reg["n_frames"] == n_sites * per_site, (
        "manuscript sec. materials states the offset is a median over "
        f"frames drawn from the {n_sites} sites, {per_site} from each, but it "
        f"is taken over {reg['n_frames']} frames, so some site supplied fewer")
    assert abs(reg["median_dy"]) >= 2, (
        "manuscript sec. materials reports a vertical camera offset of about "
        f"{abs(reg['median_dy']):.0f} px, but the measured median is now "
        f"{reg['median_dy']:+.1f} px, so the registration limitation no "
        "longer holds")
    assert abs(reg["median_dy"]) > abs(ctrl["median_dy"]), (
        "manuscript sec. materials contrasts the measured offset with what the "
        "same estimator returns on frames paired across recording sequences, "
        f"but the matched median is {reg['median_dy']:+.1f} px against the "
        f"control's {ctrl['median_dy']:+.1f} px, so the displacement the "
        "paper reads as a camera offset is what the estimator returns from "
        "no alignment at all")
    assert abs(reg["median_dx"]) > abs(ctrl["median_dx"]), (
        "manuscript sec. materials reports the horizontal offset beside the "
        f"vertical one, against the control, but the matched median is "
        f"{reg['median_dx']:+.1f} px against the control's "
        f"{ctrl['median_dx']:+.1f} px")
    assert reg["median_peak"] > ctrl["median_peak"], (
        "manuscript sec. materials reads the measured offset as a property of "
        "the two cameras, but matched pairs correlate at "
        f"{reg['median_peak']:.5f} against {ctrl['median_peak']:.5f} on pairs "
        "sharing no scene, so the estimator finds no more structure in a "
        "matched pair than in a mismatched one")


def test_the_repeat_models_run_only_under_the_site_blocked_protocol_and_the_sequence_level_split():
    """Manuscript sec. protocols: 'The repeat models run only under the dataset's split and the
    leave-one-site-out fold that holds out site C.'. The repeat set is every model
    config.py registers other than the primary one, the model whose run tag is
    empty, so a third repeat added to the registry is checked too."""
    repeats = [k for k, tag in config.MODEL_TAG.items() if tag]
    assert len(repeats) == 2, (
        f"sec. methods names two repeat models, but config.py registers "
        f"{len(repeats)} besides the primary one")
    for arch, backbone in repeats:
        protos = {m["protocol"] for m in _folds.load_runs(backbone, arch)}
        assert protos == {"blocked", "official"}, (
            "sec. methods says the repeats run only under the site-blocked "
            "protocol and the sequence-level split, but "
            f"{arch}/{backbone} has runs under {sorted(protos)}")


def test_every_network_is_evaluated_once_at_the_end_of_its_training():
    """Manuscript sec. methods: every network 'is evaluated once, at the end
    of its training'. Read off src/train.py: the one call that scores the
    validation loader sits after the epoch loop, not inside it, so no epoch or
    checkpoint is chosen by its validation score."""
    tree = ast.parse(open(os.path.join(ROOT, "src", "train.py")).read())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "evaluate"]
    assert len(calls) == 1, (
        "sec. methods says every network is evaluated once, but src/train.py "
        f"calls evaluate() {len(calls)} times")
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.For)
             and isinstance(n.iter, ast.Call)
             and ast.unparse(n.iter) == "range(C.EPOCHS)"]
    assert len(loops) == 1, ("src/train.py no longer has one epoch loop, so "
                             "this check resolves nothing")
    inside = {id(n) for n in ast.walk(loops[0])}
    assert id(calls[0]) not in inside and calls[0].lineno > loops[0].end_lineno, (
        "sec. methods says each network is evaluated at the end of its "
        "training, but src/train.py calls evaluate() inside the epoch loop")


def test_the_site_level_tests_report_their_own_resolution():
    """Manuscript sec. criteria defines the site-level test as 'a one-sample
    $t$ test on the three held-out ... mean margins'.

    Every site-level p the paper prints misses the 0.05 level, which is the
    shape that reads as a demonstrated null. None is one: at three sites the
    critical value is more than double the large-sample 1.96. The paper does
    not print that value, because no claim rests on it; sec. results states
    each site-level p beside the pooled comparison it qualifies, and the
    reservation it takes rests on the sign of the sites' means.

    The interval behind each of those p values is computed and deliberately
    NOT printed: on 2 df its endpoints are set by the fold count rather than
    by the data, so quoting them would offer a width the cross-validation cannot
    support, and a reader cannot tell one apart in print from
    the intervals over sequences the same section quotes. Its half-width
    is still checked against the quantile below, because the sentence above is
    a claim about that arithmetic whether or not the endpoints reach the page.
    Checked against the analyzer's own quantile so a cross-validation with
    more folds re-derives it rather than keeping 4.30."""
    runs = _folds.load_runs("mit_b0")
    px = _folds.A.gt_px()
    cls = "paddy_after_driving"          # the class evaluable on every fold
    per_fold = {f: _folds.A.paired_family(runs, p, "RGBNIR", "RGB", "isec",
                                          px)[cls]["mean"]
                for f, p in _folds.G_FOLD_PROTO.items()}
    st = _folds.site_level_stats(per_fold.values())
    assert st["k"] == len(_splits()["sites"]), (
        "sec. methods says a site-level interval is taken over the dataset's "
        f"sites, but the test runs on {st['k']} of "
        f"{len(_splits()['sites'])}")
    half = (st["hi"] - st["lo"]) / 2
    se = float(np.std(st["folds"], ddof=1) / np.sqrt(st["k"]))
    assert abs(half - st["tcrit"] * se) < 1e-9, (
        "the site-level interval is the one-sample t interval at k-1 degrees "
        f"of freedom, but its half-width is {half:.6f} against "
        f"{st['tcrit'] * se:.6f}")
    gen = open(os.path.join(ROOT, "src", "build_parameters.py")).read()
    site_vars = set(re.findall(r"(\w+)\s*=\s*[\w.]*site_level_stats\(", gen))
    assert site_vars, ("build_parameters.py no longer builds any site-level "
                       "statistic, so this check resolves nothing")
    quoted = [n for n, v in re.findall(
        r'cmd\("(\w+)",\s*ci\((\w+)\["lo"\]', gen) if v in site_vars]
    assert not quoted, (
        "sec. results reports a site-level test by its p and its per-fold "
        "signs, never by its endpoints, because on 2 df those endpoints are "
        f"set by the fold count: {', '.join(quoted)} would print a width the "
        "cross-validation cannot support, indistinguishable on the page from the "
        "intervals over sequences the same section quotes. The set is "
        "read off which macros the generator builds from site_level_stats, "
        "not off their names: a fold-level bootstrap can be named for the "
        "site it holds out and is not one of these")


def test_manuscript_prints_both_halves_of_every_pooled_positive():
    """Manuscript sec. results: a criterion that passes on pooled frames is
    printed beside its site-level test, never on its own.

    A pooled pass reported without the site-level p reads as support
    for an effect the conclusions then decline to use, which is a contradiction
    on the page even when every individual number is correct. Both halves are
    macros, so what is checked is that the manuscript PRINTS them - a number
    computed and macro'd but never reported is not reported."""
    tex = _read_tex()
    for pooled, site in ((r"\nirLumTilledLeaveOutCI", r"\nirLumTilledLeaveOutP"),
                         (r"\nirLumTilledRiceCI", r"\nirLumTilledRiceP"),
                         (r"\fusionRgbTilledSiteCCI", r"\fusionRgbTilledRiceP")):
        assert pooled in tex, (
            f"manuscript sec. results must report {pooled}, the pooled "
            "outcome of a protocol criterion, but it is absent")
        assert site in tex, (
            f"manuscript sec. results reports {pooled} but not {site}, so a "
            "pooled positive is stated without the site-level test that "
            "says the conclusions do not rest on it")


def test_a_frame_level_interval_understates_the_one_over_sequences():
    """Manuscript sec. redundancy: 'On the tilled surface, the comparison
    detects no difference on any of the three models ..., although a $t$
    interval that treated frames rather than sequences as independent, ...,
    would exclude zero on the primary model.' The same site-C
    NIR-minus-luminance deltas give a frame-level t
    interval excluding zero and an interval over sequences spanning it. The
    assertion is that the two disagree in exactly that direction - a
    frame-level interval that did NOT understate would mean the paper's case
    study is misdescribed."""
    gen = importlib.import_module("build_parameters")
    keys, d = _folds._pooled_deltas(_folds.load_runs("mit_b0"), "blocked",
                                    "NIR", "LUM", "paddy_after_driving",
                                    _folds.A.gt_px())
    f_lo, f_hi = gen.frame_t_ci(d)
    s_lo, s_hi = _folds.A.cluster_ci(d, _folds.A.seq_units(keys))
    assert f_lo > 0 and s_lo < 0 < s_hi, (
        "manuscript sec. redundancy shows the frame-level interval excluding 0 "
        f"and the one over sequences spanning it, but frame=[{f_lo:+.4f},"
        f"{f_hi:+.4f}] sequence=[{s_lo:+.4f},{s_hi:+.4f}]")
    assert (s_hi - s_lo) > (f_hi - f_lo), (
        "manuscript sec. methods states 'a frame-level interval would be too "
        "narrow', but the frame-level and sequence-level intervals are "
        f"{f_hi - f_lo:.4f} and {s_hi - s_lo:.4f} wide")


# ------------- full study: phase-E control failure + why no re-split -------
# The 2026-07-24 config ledger quotes these numbers to justify (a) taking E3
# as failed on fold site000 and (b) rejecting an env-stratified re-split as
# infeasible. They are re-derived here so the justification cannot go stale.

def _geometry_probe():
    p = os.path.join(ROOT, "data", "processed", "geometry_probe.json")
    assert os.path.exists(p), "run geometry_probe.py first"
    with open(p) as f:
        return json.load(f)


def _index():
    """The per-frame index records, read as the protocols read them."""
    with open(config.INDEX_JSON) as f:
        return json.load(f)


def _fold_mean(runs, proto, cls):
    """Intersection IoU over every configuration and seed of one fold, the
    quantity the control floor reads (analyze_folds.py floor_gate)."""
    vals = []
    for arm in _folds.ARMS:
        vals += _folds.seed_iou(runs, proto, arm, cls)
    return float(np.mean(vals))


def test_the_pooled_margins_rest_on_the_field_heavy_fold():
    """Manuscript sec. redundancy: 'On both, site A has the largest mean
    margin', and Table~\\ref{tab:folds} prints the leave-one-site-out
    cross-validation's three site means, all positive."""
    b0 = _folds.load_runs("mit_b0")
    px = _folds.A.gt_px()
    for name, proto_map in (("leave-one-site-out", _folds.E_FOLD_PROTO),
                            ("paddy-restricted", _folds.G_FOLD_PROTO)):
        per = {f: _folds.A.paired_family(b0, p, "NIR", "LUM", "isec", px)
               ["paddy_after_driving"]["mean"] for f, p in proto_map.items()}
        assert max(per, key=per.get) == "site000", (
            f"manuscript sec. results claims the {name} cross-validation's largest fold "
            f"mean is at the site with most dry-field frames, but the folds are {per}")
        if name == "leave-one-site-out":
            assert min(per.values()) > 0, (
                "manuscript tab:folds prints the leave-one-site-out cross-validation's "
                f"fold means with one sign, but they are {per}")


def test_a_frame_enters_the_intersection_comparisons_when_both_annotations_mark_the_class():
    """Manuscript sec. protocols: the inflation's F_c is the set of frames
    'Let $F_c$ be the set of those frames whose two annotations both mark
    class $c$'; sec. criteria: 'A comparison of two configurations on one class takes the difference of
    their IoUs, each averaged over the seeds, on each frame whose ground
    truth marks that class. Its margin is the mean of those differences. On the intersection ground
    truth, the frames whose ground truth marks the class are the frames that
    both annotations mark.'

    The inclusion rule is read off the per-frame pixel counts of the two
    annotations. src/analyze.py then scores a frame whose two annotations mark
    the class on no common pixel as an IoU of zero, read off the runs: on a
    frame where no seed's intersection counts a pixel of the class that the
    prediction shares, the seed-mean per-frame IoU the comparisons use is
    exactly zero, including the frames whose union is empty."""
    A = _folds.A
    px = A.gt_px()
    ci = config.CLASSES.index("paddy_edge")
    inc = A.inc_own("isec")
    assert all(inc(v, ci) == (v[0][ci] > 0 and v[1][ci] > 0)
               for v in px.values()), (
        "sec. criteria says a frame enters an intersection comparison when "
        "both annotations mark the class, but the inclusion rule differs")
    runs = [m for m in A.load_runs() if m["arm"] == "NIR"
            and m["protocol"] == "official" and m["train_env"] == "all"]
    iou = A.frame_iou_by_arm(runs, "isec", ci, inc, px)
    zero_inter = set(iou)
    for m in runs:
        pf = m["_pf"]
        for k, i_ in zip(zip(pf["env"], pf["seq"], pf["frame"]),
                         pf["inter_isec"][:, ci]):
            if i_ > 0:
                zero_inter.discard((str(k[0]), str(k[1]), int(k[2])))
    assert zero_inter, ("no included frame has an empty intersection on the "
                        "paddy edge, so this check resolves nothing")
    nonzero = {k: iou[k] for k in zero_inter if iou[k] != 0.0}
    assert not nonzero, (
        "sec. criteria says a frame with no common pixel of the class enters "
        f"with an IoU of zero for both configurations, but {len(nonzero)} such "
        "frames score otherwise")


def test_every_configuration_draws_frames_with_the_same_weights():
    """Manuscript sec. methods: 'Every network uses the training settings of
    Table~\\ref{tab:config}, in which every configuration trained under one
    split draws its frames with the same rare-class sampling weights',
    which tab:config computes from the union of the two annotations. Each run
    records its sampler; runs that share a partition must record one
    sampler."""
    groups = defaultdict(set)
    for m in _study_runs():
        groups[(m["protocol"], m["train_env"])].add(
            json.dumps(m["sampler"], sort_keys=True))
    split = {k: len(v) for k, v in groups.items() if len(v) > 1}
    assert not split, (
        "manuscript sec. methods claims every configuration draws its training "
        f"frames with the same weights, but these partitions record several "
        f"samplers: {split}")


def test_every_run_used_the_same_training_settings():
    """Manuscript tab:config caption: 'Every setting is the same for every
    input configuration and every network.'"""
    keys = ("epochs", "batch", "crop", "lr", "weight_decay", "frame_stride")
    settings = {tuple(m[k] for k in keys) for m in _study_runs()}
    assert len(settings) == 1, (
        "manuscript tab:config claims every run used the same training "
        f"settings, but the runs record {sorted(settings)}")


def test_any_merge_radius_in_the_printed_range_returns_the_same_super_sites():
    """Manuscript sec. materials: 'Any merge radius from ... up to, but not including, ... returns the
    same sites.'. Single linkage returns three
    clusters exactly for radii between the merge height of the fourth cluster
    and the separation of the third, so the claim holds when the committed
    radius lies in that range and the probe's cut reproduces the committed
    sites (test_no_fourth_site_separates_the_sequences)."""
    g = _geometry_probe()
    low = g["separation_first_past_committed_k_m"]
    high = g["separation_at_committed_k_m"]
    assert low <= config.RADIUS_M < high, (
        f"the committed radius {config.RADIUS_M} m lies outside the printed "
        f"range [{low}, {high}) m")


def test_field_environment_is_concentrated_in_one_site():
    """Manuscript sec. materials and sec. protocols: site000 holds ~66% of all
    Field frames, so the site000 LOSO fold 'trains on the minority of the dry-field frames
    that sites B and C hold, and a low IoU at site A on this fold may
    reflect that scarcity as well as the new site', the reason the
    paddy-restricted cross-validation exists."""
    g = _geometry_probe()
    share = g["env_composition"]["Field"]["site000"]["share_of_env"]
    assert share > 0.60, (
        "config ledger 2026-07-24 claims site000 holds 66.4% of all Field "
        f"frames (cause of the E3 failure), but measured {share:.1%}")
    paddy = [g["env_composition"]["Paddy"][s]["share_of_env"]
             for s in ("site000", "site001", "site002")]
    assert max(paddy) - min(paddy) < 0.20, (
        "config ledger 2026-07-24 claims Paddy IS balanced across sites "
        f"(the premise of phase G), but shares are {paddy}")


def test_no_split_tests_a_gain_on_the_furrow_or_the_ridge_at_more_than_one_site_within_one_environment():
    """Manuscript sec. discussion: 'For the dry-field furrow and ridge, the only held-out test within one
    environment is the rice-paddy cross-validation at site C, the one site
    whose rice-paddy frames carry the two classes in at least 50 frames. No
    split is confined to dry fields, so the tests of the two classes at site
    B mix the two environments'; sec. materials: 'Each
    recording sequence carries one environment label, which its frames
    take.' The splits are every protocol
    of the run configuration, each read from its own runs: a split is within
    one environment when its validation frames are of one environment, and
    the sites at which it scores a class are the sites of the validation
    frames whose two annotations both mark the class. The frames that carry a
    class are read from both annotations of every sampled frame."""
    runs, px = _folds.A.load_runs(), _folds.A.gt_px()
    k2s = _index_key2site()
    recs = config.load_index()
    seq_site = {}
    for r in recs:
        seq_site.setdefault((r["env"], r["seq"]), set()).add(
            k2s[f"{r['osplit']}/{r['env']}/{r['seq']}"])
    assert all(len(v) == 1 for v in seq_site.values()), (
        "a recording sequence maps to more than one site")
    site_of = {k: next(iter(v)) for k, v in seq_site.items()}
    splits = sorted({r["protocol"] for r in config.RUN_MATRIX})
    envs_of, keys_of = {}, {}
    for p in splits:
        m = next((m for m in runs if m["protocol"] == p and m["_pf"] is not None),
                 None)
        assert m is not None, f"the run configuration's split {p} has no runs"
        keys_of[p] = [(str(e), str(s), int(f)) for e, s, f in
                      zip(m["_pf"]["env"], m["_pf"]["seq"], m["_pf"]["frame"])]
        envs_of[p] = {k[0] for k in keys_of[p]}
    within = [p for p in splits if len(envs_of[p]) == 1]
    for cls in ("field_furrow", "field_levee"):
        ci = config.CLASSES.index(cls)
        sites = set()
        for p in within:
            if _folds.A.paired_family(runs, p, "NIR", "RGB", "isec",
                                      px).get(cls) is None:
                continue
            sites |= {site_of[k[:2]] for k in keys_of[p] if k in px
                      and px[k][0][ci] > 0 and px[k][1][ci] > 0}
        assert sites == {"site002"}, (
            "sec. discussion says within one environment the furrow and the "
            f"ridge are scored on frames of site C alone, but {cls} is scored "
            f"on frames of {sorted(sites)} by the splits {within}")
    assert all(envs_of[p] == {"Paddy"} for p in within), (
        "sec. discussion says the splits within one environment are confined to "
        f"rice paddy, but they evaluate {[sorted(envs_of[p]) for p in within]}")
    train_envs = {r["train_env"] for r in config.RUN_MATRIX}
    for mp in glob.glob(os.path.join(config.RUNS_ROOT, "*", "metrics.json")):
        with open(mp) as f:
            train_envs.add(json.load(f).get("train_env", "all"))
    dry = [p for p in splits if envs_of[p] == {"Field"}]
    assert train_envs == {"all"} and not dry and config.PLOSO_ENV == "Paddy", (
        "sec. discussion says no split is confined to dry fields, but the "
        f"training environments are {sorted(train_envs)}, the splits evaluated "
        f"on dry fields alone are {dry}, and the one environment to which "
        f"train.py restricts a split is {config.PLOSO_ENV}")


def test_the_dry_field_frames_of_site_a_carry_no_furrow():
    """Manuscript sec. protocols: 'Site A holds 66.4% of the sampled dry-field
    frames'; sec. redundancy: 'none of its dry-field frames carries the furrow'. Site A is read as the site with most dry-field frames, and the
    claim is checked on both annotations of every sampled frame of that
    site's dry-field sequences."""
    comp = _geometry_probe()["env_composition"]
    heavy = max(comp["Field"], key=lambda st: comp["Field"][st]["share_of_env"])
    k2s = _index_key2site()
    field = [r for r in config.load_index() if r["env"] == "Field"
             and k2s[f"{r['osplit']}/{r['env']}/{r['seq']}"] == heavy]
    assert field, f"the index holds no dry-field frame of {heavy}"
    ci = config.CLASSES.index("field_furrow")
    carrying = [r for r in field if r["px_rgb"][ci] > 0 or r["px_nir"][ci] > 0]
    assert not carrying, (
        "sec. redundancy says no sampled dry-field frame of site A carries "
        f"the furrow, but {len(carrying)} of {heavy}'s {len(field)} dry-field "
        "frames carry it")


def test_no_fourth_site_separates_the_sequences():
    """Ledger 2026-07-24: separation falls off a cliff past the committed
    site count, so no re-clustering buys env balance without reintroducing
    leakage. Guards the decision to restrict (phase G) rather than re-split."""
    g = _geometry_probe()
    at, past = (g["separation_at_committed_k_m"],
                g["separation_first_past_committed_k_m"])
    assert at > 500 and past < 5, (
        "config ledger 2026-07-24 claims separation collapses from 761 m at "
        f"K=3 to 0.5 m at K=4, but measured {at:.1f} m -> {past:.1f} m")
    # sec. materials states the separation as a claim over every pair of
    # sequences in different sites, so the printed value must be the
    # sweep's minimum at the committed cut, which is that minimum over pairs.
    at_committed = [s for s in g["separation_sweep"]
                    if s["k"] == g["committed_n_sites"]]
    assert len(at_committed) == 1 and at_committed[0]["min_separation_m"] == at, (
        "manuscript sec. materials states that no recording sequence of one "
        f"site passes within {at:.0f} m of a sequence of another, which "
        "is the smallest separation between clusters at the committed cut, "
        f"but the sweep gives {at_committed}")
    for s in g["separation_sweep"]:
        if s["k"] <= g["committed_n_sites"]:
            continue
        big = s["clusters"][0]
        assert big["field_share"] >= 0.45, (
            "config ledger 2026-07-24 claims cutting past the committed "
            "sites makes the big cluster MORE Field-heavy (0.45 -> 0.48), "
            f"but at K={s['k']} its Field share is {big['field_share']:.2f}")
    # sec. materials: 'Any radius from 0.5 m up to, but not including, 761 m
    # returns the same sites, because chains of links no longer than 0.5 m
    # connect each site's sequences' and 'Site A's share of the dry-field
    # frames stays the same for every radius below 761 m, because a radius
    # below 0.5 m splits off only clusters without dry-field frames.'
    # The largest cluster is site A only if its frames are that site's, and a
    # cluster split off is one beyond the committed count, the clusters being
    # listed largest first.
    assert past < config.RADIUS_M, (
        f"sec. materials puts the {past:.1f} m separation past the committed "
        f"cut within the {config.RADIUS_M} m merge radius, and it is not")
    comp = g["env_composition"]
    heavy = max(comp["Field"], key=lambda st: comp["Field"][st]["share_of_env"])
    k0 = g["committed_n_sites"]
    base = next(s for s in g["separation_sweep"] if s["k"] == k0)["clusters"]
    assert base[0]["frames"] == sum(comp[e][heavy]["frames"] for e in comp), (
        "sec. materials reads the largest cluster as site A, the site with most "
        f"dry-field frames, but the largest cluster at the committed cut is not "
        f"{heavy}")
    for s in g["separation_sweep"]:
        if s["k"] <= k0:
            continue
        assert all(c["field_share"] == 0 for c in s["clusters"][k0:]), (
            "sec. materials states that every cluster a radius below the "
            "committed one splits off holds none of the dry-field frames, but "
            f"at K={s['k']} the split-off "
            f"shares are {[c['field_share'] for c in s['clusters'][k0:]]}")
        assert s["clusters"][0]["field_share"] >= base[0]["field_share"], (
            "sec. materials states that site A keeps or "
            f"raises its dry-field share under a finer cut, but at K={s['k']} "
            f"it falls to {s['clusters'][0]['field_share']:.3f}")


def test_the_road_scores_low_at_site_a_without_its_frames():
    """Gate E3 fell as FAIL on fold site000 and the ledger records it as such.
    This asserts the failure, not a pass: if a later change made the control
    clear 0.30, the ledger text and the phase-G rationale are both wrong."""
    runs = _study_runs()
    vals = [m["iou"]["isec"]["common_road"] for m in runs
            if m["protocol"] == "loso000"]
    assert len(vals) == 20, f"expected 20 loso000 runs, found {len(vals)}"
    mean = float(np.mean(vals))
    assert mean < config.PLOSO_CONTROL_FLOOR, (
        "config ledger 2026-07-24 records gate E3 as FAILED on fold site000 "
        f"(common_road {mean:.3f} below the {config.PLOSO_CONTROL_FLOOR} "
        "floor); a passing control would invalidate that ledger entry")
    for proto in ("loso001", "blocked"):
        ok = [m["iou"]["isec"]["common_road"] for m in runs
              if m["protocol"] == proto and m["train_env"] == "all"
              and m["phase"] in ("A", "E")]
        assert np.mean(ok) >= config.PLOSO_CONTROL_FLOOR, (
            f"config ledger 2026-07-24 records the control PASSING on {proto} "
            f"(the failure is site000-specific), but mean is {np.mean(ok):.3f}")


def _index_key2site():
    """index-record key -> GPS site, same mapping src/train.py _key2site uses."""
    return {m: s["site_id"] for s in _splits()["sites"] for m in s["members"]}


def test_index_frame_store_paths_are_relative_and_self_consistent():
    """The index is committed and read by a fresh clone, so its four
    frame-store paths must be relative to the study directory rather than
    absolute to whichever machine built them.

    An absolute path survives every other check here, because every
    assertion in this file reads pixel counts and split labels and none of
    them opens an image. That path is then detectable only by re-running the
    trainer.
    """
    with open(config.INDEX_JSON) as f:
        raw = json.load(f)
    absolute = [r[k] for r in raw for k in config.INDEX_PATH_KEYS
                if os.path.isabs(r[k]) or r[k].startswith("..")]
    assert not absolute, (
        f"{len(absolute)} index paths escape the study directory, e.g. "
        f"{absolute[:2]}; src/phases/build_index.py stores them relative to it")

    # a record's own osplit and env name the subdirectory its frames live in,
    # so a path that disagrees is pointing at another record's store
    for r in raw:
        want = os.path.join("frames", f"{r['osplit']}_{r['env']}")
        for k in config.INDEX_PATH_KEYS:
            assert os.path.dirname(r[k]) == want, (
                f"index record {r['env']}/{r['seq']}/{r['frame']} stores {k} "
                f"under {os.path.dirname(r[k])}, but its own split and "
                f"environment put it under {want}")


def test_pofficial_matches_ploso_in_environment_and_differs_in_split():
    """Gate G2's comparator (config ledger 2026-07-24b) is only a leakage
    measurement if pofficial differs from ploso in the SPLIT ALONE. Verifies
    both halves: identical environment restriction, and pofficial's committed
    split sizes."""
    recs = config.load_index()
    env = [r for r in recs if r["env"] == config.PLOSO_ENV]
    tr = [r for r in env if r["osplit"] == "train"]
    va = [r for r in env if r["osplit"] == "val"]
    assert (len(tr), len(va)) == (8299, 2397), (
        "config ledger 2026-07-24b claims pofficial n_train 8299 / n_val 2397 "
        f"over {len(env)} Paddy frames, but the index gives {len(tr)}/{len(va)}")
    k2s = _index_key2site()
    for fold in config.PLOSO_RUN_FOLDS:
        ploso_val = [r for r in env
                     if k2s[f"{r['osplit']}/{r['env']}/{r['seq']}"] == fold]
        assert ploso_val, f"ploso fold {fold} would evaluate on nothing"
        # the env restriction is identical on both sides; only siteN moves
        assert all(r["env"] == config.PLOSO_ENV for r in ploso_val + tr), (
            "config ledger 2026-07-24b claims pofficial and ploso share the "
            f"environment restriction {config.PLOSO_ENV}, but they do not")


def test_g2b_fixed_sets_are_seen_by_pofficial_and_unseen_by_ploso():
    """Gate G2b (config ledger 2026-07-24b) lifts the frame-fixed seen-vs-unseen
    frame-fixed design
    into the Paddy regime. Its validity rests on three properties of S_N, all
    checkable before any run: neither condition trained on S_N (so it is not
    frame memorization), pofficial trained on OTHER siteN frames (so it is
    'seen'), and plosoN trained on none (so it is 'unseen')."""
    recs = config.load_index()
    k2s = _index_key2site()

    def site(r):
        return k2s[f"{r['osplit']}/{r['env']}/{r['seq']}"]
    env = [r for r in recs if r["env"] == config.PLOSO_ENV]
    po_tr = {id(r) for r in env if r["osplit"] == "train"}
    expected = {"site000": 256, "site001": 0, "site002": 2141}
    for fold in config.PLOSO_RUN_FOLDS:
        S = [r for r in env if r["osplit"] == "val" and site(r) == fold]
        assert len(S) == expected[fold], (
            "config ledger 2026-07-24b specifies |S_N| = "
            f"{expected[fold]} for {fold}, but the index gives {len(S)}")
        ploso_tr = {id(r) for r in env if site(r) != fold}
        assert not (ploso_tr & {id(r) for r in S}), (
            f"G2b requires plosoN to be UNSEEN on S_N, but fold {fold} "
            "trains on some of its own fixed set")
        assert not (po_tr & {id(r) for r in S}), (
            f"G2b requires S_N to be untrained by BOTH conditions "
            f"(the frame-fixed "
            f"design), but pofficial trains on part of fold {fold}'s S_N")
        assert not any(site(r) == fold for r in env if id(r) in ploso_tr), (
            f"ploso{fold[len('site'):]} must never see any {fold} frame")
        seen_ctx = sum(site(r) == fold for r in env if id(r) in po_tr)
        if fold in config.PLOSO_G2B_FOLDS:
            assert len(S) > 0 and seen_ctx > 0, (
                f"{fold} carries gate G2b but has no fixed set or no 'seen' "
                "context frames")
        else:
            assert len(S) == 0, (
                "config ledger 2026-07-24b excludes only folds with an EMPTY "
                f"fixed set from G2b, but {fold} has {len(S)} frames")


def test_pofficial_is_committed_and_matrix_is_intact():
    """The run matrix IS the protocol. Guards the
    2026-07-24b amendment against a silent edit: 20 pofficial runs, mit_b0,
    the committed seed set, and no protocol outside the ledger."""
    matrix = config.RUN_MATRIX
    assert len(matrix) == 245 and len({r["run_id"] for r in matrix}) == 245, (
        f"config ledger 2026-11-16 leaves a 245-run matrix (260 committed "
        f"through phase H, less phase B's 20, plus phase I's 5), found "
        f"{len(matrix)} rows / {len({r['run_id'] for r in matrix})} ids")
    diag = [r for r in matrix if r["phase"] == "I"]
    assert ({r["arm"] for r in diag} == {config.DIAG_ARM}
            and {r["seed"] for r in diag} == set(config.SEEDS)
            and {r["protocol"] for r in diag} == {"official"}
            and all(r["backbone"] == "mit_b0" for r in diag)), (
        "config ledger 2026-11-16 commits phase I as the diagnostic arm on "
        "the whole seed set under the official split, mit_b0")
    po = [r for r in matrix if r["protocol"] == "pofficial"]
    assert len(po) == 20, (
        f"config ledger 2026-07-24b commits 20 pofficial runs, found {len(po)}")
    assert {r["arm"] for r in po} == {"RGB", "NIR", "LUM", "RGBNIR"}
    assert {r["seed"] for r in po} == set(config.SEEDS), (
        "pofficial must run the whole committed seed set, not "
        "a subset")
    assert all(r["backbone"] == "mit_b0" and r["phase"] == "G" for r in po)
    known = {"blocked", "official", "pofficial"} | {
        p + f[len("site"):] for p, folds in
        (("loso", config.LOSO_RUN_FOLDS), ("ploso", config.PLOSO_RUN_FOLDS))
        for f in folds}
    unknown = {r["protocol"] for r in matrix} - known
    assert not unknown, (
        f"run matrix contains protocols absent from the ledger: {unknown}")


def test_phase_h_is_committed():
    """Guards the 2026-08-05 amendment the same way: phase H is 40 runs of
    the second architecture family over blocked+official, mirroring phase F's
    scope exactly so H1/H2 are F1/F2 with the model swapped. Also pins that
    phase H changes NO hyperparameter. The objection H answers is that the
    negative redundancy claim rests on one architecture, and that answer
    holds only if the architecture is the sole difference."""
    matrix = config.RUN_MATRIX
    h = [r for r in matrix if r["phase"] == "H"]
    assert len(h) == 40, (
        f"config ledger 2026-08-05 commits 40 phase-H runs, found {len(h)}")
    assert all(r["arch"] == config.ARCH_H
               and r["backbone"] == config.BACKBONE_H for r in h)
    assert {r["arm"] for r in h} == {"RGB", "NIR", "LUM", "RGBNIR"}
    assert {r["seed"] for r in h} == set(config.SEEDS), (
        "phase H must run the whole committed seed set, not a "
        "subset")
    assert {r["protocol"] for r in h} == {"blocked", "official"}, (
        "H1 is the official-vs-blocked contrast; phase H must carry both "
        "protocols or the gate cannot be computed")
    # Scope parity with phase F: same protocols, arms and seeds, so the only
    # difference between the F and H gates is the model.
    f = [r for r in matrix if r["phase"] == "F"]
    for k in ("protocol", "arm", "seed"):
        assert {r[k] for r in h} == {r[k] for r in f}, (
            f"phase H must mirror phase F's scope on '{k}' (config ledger "
            "2026-08-05), else H1/H2 are not F1/F2 with the model swapped")
    # Every other run keeps the headline architecture.
    assert all(r["arch"] == "segformer" for r in matrix if r["phase"] != "H")


# ------------- phases E/F/G: fold and backbone gates (analyze_folds.py) -----------

_folds = importlib.import_module("analyze_folds")


def _pooled_over_folds(runs, protos, arm_x, arm_y, cls, px):
    """Keys and per-frame deltas concatenated over a fold set - what analyze_folds.py
    pooled_arm_gate tests. Keys travel with deltas because the CI is a
    sequence-cluster bootstrap."""
    kd = [_folds._pooled_deltas(runs, p, arm_x, arm_y, cls, px)
          for p in protos]
    return ([k for ks, _ in kd for k in ks],
            np.concatenate([d for _, d in kd]))


def test_phase_e2_passes_pooled_and_site_a_contributes_most_of_it():
    """Manuscript sec. redundancy: pooled over the leave-one-site-out
    cross-validation's folds, near-infrared is better than the luminance
    control on the tilled paddy surface, and 'On both, site A has the largest
    mean margin (Table~\\ref{tab:folds}) and the most frames that carry the
    tilled surface, so it contributes most of the pooled margin.' Both halves
    are asserted: dropping the second would overstate the pooled result, and
    dropping the first would re-scope the criterion after the fact. The site
    means, frame counts and shares are read off the frames that each pooled
    test concatenates, on both cross-validations."""
    runs = _folds.load_runs("mit_b0")
    px = _folds.A.gt_px()
    cls = "paddy_after_driving"
    per_fold = {}
    for fold, proto in _folds.E_FOLD_PROTO.items():
        si = _folds.A.paired_family(runs, proto, "NIR", "LUM", "isec", px)
        so = _folds.A.paired_family(runs, proto, "NIR", "LUM",
                                    _folds.A.own_proto("NIR"), px)
        per_fold[fold] = _folds.sec7(si, so, cls)
    keys, d = _pooled_over_folds(runs, _folds.E_FOLD_PROTO.values(),
                                 "NIR", "LUM", cls, px)
    lo, hi = _folds.A.cluster_ci(d, _folds.A.seq_units(keys))
    p = float(stats.wilcoxon(d).pvalue)
    assert p < config.ALPHA and lo > 0, (
        "manuscript sec. redundancy reports near-infrared better than the "
        "luminance control pooled over the leave-one-site-out cross-validation's folds, "
        f"but the pooled delta is {d.mean():+.4f} CI [{lo:+.4f},{hi:+.4f}] "
        f"p={p:.2e}")
    assert not per_fold["site001"].startswith("DIRECTIONAL"), (
        "manuscript sec. redundancy reports that no site-level test detects "
        f"the pooled margin, but fold site001 alone is '{per_fold['site001']}'")
    for name, fold_proto in (("leave-one-site-out", _folds.E_FOLD_PROTO),
                             ("paddy-restricted", _folds.G_FOLD_PROTO)):
        parts = {f: _folds._pooled_deltas(runs, pr, "NIR", "LUM", cls, px)[1]
                 for f, pr in fold_proto.items()}
        total = sum(float(dd.sum()) for dd in parts.values())
        means = {f: float(dd.mean()) for f, dd in parts.items()}
        counts = {f: len(dd) for f, dd in parts.items()}
        share = float(parts["site000"].sum()) / total
        assert max(means, key=means.get) == "site000", (
            "manuscript sec. redundancy: 'site A has the largest fold mean', "
            f"but on the {name} cross-validation the fold means are {means}")
        assert max(counts, key=counts.get) == "site000", (
            "manuscript sec. redundancy: site A has 'the most frames that carry "
            f"the tilled paddy surface', but on the {name} cross-validation the "
            f"counts are {counts}")
        assert share > 0.5, (
            "manuscript sec. redundancy: site A 'contributes most of the pooled "
            f"margin', but on the {name} cross-validation its share is {share:.2f}")


def test_no_site_holds_half_of_the_rice_paddy_frames():
    """Manuscript sec. protocols: 'The rice-paddy cross-validation confines training and evaluation to
    rice-paddy frames. No site holds half of those frames, so each fold holds out a new site
    within one environment and still trains on most of that environment's
    frames.' Read off the index and the split's own
    site table, so a re-split that concentrates rice paddy in one site fails
    here."""
    k2s = _index_key2site()
    per_site = defaultdict(int)
    paddy = [r for r in config.load_index() if r["env"] == config.PLOSO_ENV]
    for r in paddy:
        per_site[k2s[f"{r['osplit']}/{r['env']}/{r['seq']}"]] += 1
    shares = {s: n / len(paddy) for s, n in per_site.items()}
    assert max(shares.values()) < 0.5, (
        "sec. protocols states that no site holds half of the sampled "
        f"rice-paddy frames, but the shares are "
        f"{ {s: round(v, 3) for s, v in shares.items()} }")


def test_ploso_fold_split_sizes():
    """config.py specifies the Paddy-restricted LOSO train sizes as
    6630 / 8152 / 6610 against pofficial's 8299 ("at most 1.26x"), the
    disclosed residual asymmetry of gate G2. Each fold trains on the two
    other sites' Paddy frames and evaluates on its own, so a fold's val size
    is that site's Paddy count and train + val is the same Paddy total on
    every fold."""
    recs = config.load_index()
    k2s = _index_key2site()
    env = [r for r in recs if r["env"] == config.PLOSO_ENV]
    per_site = defaultdict(int)
    for r in env:
        per_site[k2s[f"{r['osplit']}/{r['env']}/{r['seq']}"]] += 1
    total = len(env)
    expect = {"site000": (6630, 4066), "site001": (8152, 2544),
              "site002": (6610, 4086)}
    for fold in config.PLOSO_RUN_FOLDS:
        val = per_site[fold]
        assert (total - val, val) == expect[fold], (
            f"config.py specifies ploso {fold} train/val "
            f"{expect[fold]}, but the index gives ({total - val}, {val})")


def _repeat_inflation_is_not_detected(backbone, arch, name):
    """The shared body of gates F1 and H1: the paddy-edge inflation of NIR on a
    repeat model, verdict and interval, and the cell tab:robustness prints."""
    runs = _folds.load_runs(backbone, arch)
    px = _folds.A.gt_px()
    si = _folds.paired_protocols(runs, "NIR", "official", "blocked", "isec", px)
    so = _folds.paired_protocols(runs, "NIR", "official", "blocked", "nir", px)
    v, s = _folds.sec7(si, so, "paddy_edge"), si["paddy_edge"]
    assert not v.startswith("DIRECTIONAL") and s["lo"] < 0 < s["hi"], (
        "manuscript sec. concentration: 'On both repeat models, which hold out "
        "site C alone, the inflation of NIR has an interval spanning zero', but "
        f"under {name} the three conditions return '{v}' with "
        f"[{s['lo']:+.3f}, {s['hi']:+.3f}]")
    assert f"{s['mean']:+.3f}" in _read_table("table_robustness.tex"), (
        f"gate {name} is asserted here but the manuscript never prints its "
        "inflation. It is a cell of tab:robustness(a), not a macro in the body: "
        "the sentence states the verdict and the float carries the value")


def test_phase_f1_inflation_of_nir_is_not_detected_on_mit_b2():
    """Gate F1 (config.py), reported as it fell. The official-vs-blocked
    paddy_edge inflation of NIR was asked to reproduce on the mit_b2 backbone
    (~7x the mit_b0 encoder), and its interval over the four sequences of
    site C spans zero. Manuscript sec. concentration: 'At site C, NIR's
    paddy-edge inflation is positive but not detected on any of the three
    models, because its interval spans zero'."""
    _repeat_inflation_is_not_detected("mit_b2", "segformer", "F1")


def test_phase_f2_does_not_clear_under_clustering():
    """Gate F2 (config.py), reported as it fell. The NIR-minus-luminance
    effect was asked to reproduce on the ~7x larger encoder; under the
    sequence-clustered interval it does not clear on either backbone, so the
    backbone was never the reason. F2 is the single protocol blocked
    fold - one site - which is why its interval is the widest in the
    study at the largest frame count."""
    runs = _folds.load_runs("mit_b2")
    px = _folds.A.gt_px()
    keys, d = _folds._pooled_deltas(runs, "blocked", "NIR", "LUM",
                                    "paddy_after_driving", px)
    lo, hi = _folds.A.cluster_ci(d, _folds.A.seq_units(keys))
    assert lo < 0 < hi, (
        "manuscript sec. discussion reports gate F2 as not cleared once the "
        f"interval respects sequence clustering, but delta {d.mean():+.4f} "
        f"CI [{lo:+.4f},{hi:+.4f}] excludes 0")
    assert f"[{lo:+.3f},\\,{hi:+.3f}]" in _read_table("table_robustness.tex"), (
        "gate F2 is asserted here but the manuscript never prints its "
        "interval. It is a cell of tab:robustness(b), not a macro in the "
        "body: the sentence states the verdict and the float carries it")


def test_phase_h1_inflation_of_nir_is_not_detected_on_convnext():
    """Gate H1 (config.py), reported as it fell. The official-vs-blocked
    paddy_edge inflation of NIR was asked to reproduce off the SegFormer
    family, on a convolutional backbone with a pyramid-pooling decoder, and
    its interval spans zero, as on mit_b2
    (test_phase_f1_inflation_of_nir_is_not_detected_on_mit_b2)."""
    _repeat_inflation_is_not_detected(config.BACKBONE_H, config.ARCH_H, "H1")


def test_phase_h2_does_not_reproduce_off_segformer():
    """Gate H2 (config.py), reported as it fell. The NIR-minus-luminance
    effect was asked to reproduce on a second architecture family and does
    not: the interval spans zero and the point estimate is NEGATIVE, the
    opposite sign to the SegFormer runs. Asserted on the sign as well as the
    interval, because a null that also flips sign is what rules out the
    architecture as the reason the effect was never established."""
    runs = _folds.load_runs(config.BACKBONE_H, config.ARCH_H)
    px = _folds.A.gt_px()
    keys, d = _folds._pooled_deltas(runs, "blocked", "NIR", "LUM",
                                    "paddy_after_driving", px)
    lo, hi = _folds.A.cluster_ci(d, _folds.A.seq_units(keys))
    assert lo < 0 < hi, (
        "manuscript sec. results reports gate H2 as not cleared on the second "
        f"architecture, but delta {d.mean():+.4f} CI [{lo:+.4f},{hi:+.4f}] "
        "excludes 0")
    assert d.mean() < 0, (
        "tab:robustness(b) prints the second-architecture margin as "
        f"negative (NIR below the luminance control), but it is "
        f"{d.mean():+.4f}")
    # tab:robustness(b) prints the margin positive on the primary model and
    # negative under the second architecture, so both signs are read
    _, d0 = _folds._pooled_deltas(_folds.load_runs("mit_b0"), "blocked",
                                  "NIR", "LUM", "paddy_after_driving", px)
    assert d0.mean() > 0, (
        "tab:robustness(b) prints the margin as positive on the primary "
        f"model, but it is {d0.mean():+.4f}")
    assert f"[{lo:+.3f},\\,{hi:+.3f}]" in _read_table("table_robustness.tex"), (
        "gate H2 is asserted here but the manuscript never prints its "
        "interval. It is a cell of tab:robustness(b), not a macro in the "
        "body: the sentence states the verdict and the float carries it")


def test_the_tilled_surface_clears_the_control_on_every_rice_paddy_fold():
    """Gate G1 (config.py). The control E3 failed - the fold-mean
    intersection IoU floor - must clear on all three Paddy-restricted folds.
    G1 passing licenses phase G's cross-site tests. It does not make the E3
    failure an environment effect: the road fails on the paddy-restricted
    site000 fold too
    (test_site_a_fails_on_the_road_with_the_environment_held_fixed)."""
    runs = _folds.load_runs("mit_b0")
    for fold, proto in _folds.G_FOLD_PROTO.items():
        allv = []
        for arm in _folds.ARMS:
            allv += _folds.seed_iou(runs, proto, arm, "paddy_after_driving")
        fm = float(np.mean(allv))
        assert fm >= config.PLOSO_CONTROL_FLOOR, (
            "manuscript sec. results reports gate G1 clears on all three folds "
            f"({config.PLOSO_CONTROL_FLOOR} floor), but fold {fold} mean "
            f"is {fm:.3f}")


def test_site_a_fails_on_the_road_with_the_environment_held_fixed():
    """Manuscript sec. concentration: 'Averaged over the four input configurations, their pooled IoU on the
    road at site
    A is 0.248 on the leave-one-site-out fold and 0.056 on the rice-paddy
    fold, against 0.806 and 0.801 at sites B and C on the leave-one-site-out
    cross-validation. On the tilled surface, averaged in the same way, they
    score 0.362 at site A on the leave-one-site-out fold and 0.392 on the
    rice-paddy fold, the lowest of the three rice-paddy folds. Site A thus
    differs from the other sites on other classes too for networks that have
    not seen it'. Each
    value is the
    quantity floor_gate reads (analyze_folds.py), a pooled IoU per
    configuration and seed, averaged. The road is the class whose failure is
    specific to site A: the paddy edge and the untilled surface are low on
    every fold, so the sentence names neither."""
    runs = _folds.load_runs("mit_b0")
    floor = config.PLOSO_CONTROL_FLOOR
    site_a = {"leave-one-site-out": _folds.E_FOLD_PROTO["site000"],
              "paddy-restricted": _folds.G_FOLD_PROTO["site000"]}

    def per_arm(proto, cls):
        return {arm: float(np.mean(_folds.seed_iou(runs, proto, arm, cls)))
                for arm in _folds.ARMS}

    def averaged(proto, cls):
        return float(np.mean([x for arm in _folds.ARMS
                              for x in _folds.seed_iou(runs, proto, arm, cls)]))

    for cv, proto in site_a.items():
        got = per_arm(proto, "common_road")
        assert max(got.values()) < floor, (
            "manuscript sec. concentration states that on both folds at site A "
            f"every configuration's pooled IoU on the road is below the {floor} "
            f"floor, but on the {cv} fold it is "
            f"{ {k: round(v, 3) for k, v in got.items()} }")
        tilled = averaged(proto, "paddy_after_driving")
        assert tilled >= floor, (
            "manuscript sec. concentration states that the averaged pooled IoU of "
            f"the tilled paddy surface is above the floor on both folds at "
            f"site A, but on the {cv} fold it is {tilled:.3f}")
    for cv, fold_proto in (("leave-one-site-out", _folds.E_FOLD_PROTO),
                           ("paddy-restricted", _folds.G_FOLD_PROTO)):
        flagged = sorted(f for f, proto in fold_proto.items()
                         if averaged(proto, "common_road") < floor)
        assert flagged == ["site000"], (
            "manuscript sec. concentration states that a floor on the farm access "
            "road flags the fold at site A, where the floor's class flags none on "
            "the paddy-restricted cross-validation, but on the "
            f"{cv} cross-validation it flags {flagged}")
    params = _read_params()
    for macro, cv, cls in (
            ("roadRiceSiteA", "paddy-restricted", "common_road"),
            ("tilledLeaveOutSiteA", "leave-one-site-out",
             "paddy_after_driving")):
        printed = re.search(r"\\newcommand\{\\" + macro
                            + r"\}\{(\d\.\d+)\\xspace\}", params).group(1)
        value = averaged(site_a[cv], cls)
        assert f"{value:.3f}" == printed, (
            f"sec. concentration prints \\{macro} as the averaged pooled IoU of "
            f"{cls} on the {cv} fold at site A, but it is {value:.4f}")
    tilled_rp = {f: averaged(p, "paddy_after_driving")
                 for f, p in _folds.G_FOLD_PROTO.items()}
    printed = re.search(r"\\newcommand\{\\tilledRiceSiteA\}\{(\d\.\d+)\\xspace\}",
                        params).group(1)
    assert (min(tilled_rp, key=tilled_rp.get) == "site000"
            and f"{tilled_rp['site000']:.3f}" == printed), (
        "sec. concentration prints \\tilledRiceSiteA as the tilled surface's "
        "averaged pooled IoU at site A on the rice-paddy fold, the lowest of the "
        f"three rice-paddy folds, but the folds give {tilled_rp}")


def test_nir_inflation_at_site_a_on_the_rice_paddy_cross_validation():
    """Gate G2 (config ledger 2026-07-24b). Against the dataset's split
    restricted to rice paddy (which differs from the rice-paddy fold in the
    split alone), NIR's paddy_edge inflation was asked to be directional and
    the largest of the four arms on both evaluable folds, with the frame-fixed
    reading G2b - on frames NEITHER condition trained on - concurring (G2b
    governs). site001 carries no evaluable paddy_edge frame and is protocol
    out (PLOSO_G2B_FOLDS). Manuscript sec. concentration: training with other
    frames of site A 'raises the paddy-edge IoU of each of the four input configurations at
    site A',
    and 'At site C, NIR's paddy-edge inflation is positive but not
    detected'."""
    runs = _folds.load_runs("mit_b0")
    px = _folds.A.gt_px()
    fixed = _folds.fixed_sets()
    verdicts = {}
    for fold in config.PLOSO_G2B_FOLDS:
        proto = _folds.G_FOLD_PROTO[fold]
        got = {}
        for arm in _folds.ARMS:
            si = _folds.paired_protocols(runs, arm, "pofficial", proto,
                                         "isec", px)
            so = _folds.paired_protocols(runs, arm, "pofficial", proto,
                                         "nir" if arm == "NIR" else "rgb", px)
            got[arm] = (_folds.sec7(si, so, "paddy_edge"),
                        si["paddy_edge"]["mean"])
        assert got["NIR"][1] == max(m for _, m in got.values()), (
            "manuscript sec. results reports the G2 leakage is "
            f"NIR-concentrated (NIR the largest inflation) on fold {fold}, "
            f"arms { {a: round(m, 3) for a, (_, m) in got.items()} }")
        sib = _folds.paired_protocols(runs, "NIR", "pofficial", proto, "isec",
                                      px, fixed[fold])
        sob = _folds.paired_protocols(runs, "NIR", "pofficial", proto, "nir",
                                      px, fixed[fold])
        verdicts[fold] = (got["NIR"][0], _folds.sec7(sib, sob, "paddy_edge"))
    # Significance, as opposed to the ordering above, is carried by site000.
    # site002's 322 paddy_edge frames come from four sequences and no arm
    # clears there - stated in the manuscript rather than absorbed into a
    # fold-pooled number that would hide it.
    assert verdicts["site000"] == ("DIRECTIONAL (+)", "DIRECTIONAL (+)"), (
        "manuscript sec. results reports the paddy_edge inflation as "
        "directional in NIR against the matched comparator on site000 under "
        "both readings G2a and G2b (not memorization, not a difficulty gap), "
        f"but they are {verdicts['site000']}")
    assert not verdicts["site002"][0].startswith("DIRECTIONAL"), (
        "manuscript sec. results states the site002 fold is too few sequences "
        "to clear the gate once the interval is clustered, but G2a returns "
        f"'{verdicts['site002'][0]}' there")
    others = {arm: _folds.paired_protocols(
                  runs, arm, "pofficial", _folds.G_FOLD_PROTO["site000"], "isec",
                  px)["paddy_edge"] for arm in _folds.ARMS if arm != "NIR"}
    low = {arm: round(s_["lo"], 4) for arm, s_ in others.items()
           if not (s_["mean"] > 0 and s_["lo"] > 0)}
    assert not low, (
        "manuscript sec. concentration states that at site A the inflation of "
        "every other configuration is also positive with an interval excluding "
        f"zero, but the lower ends are {low}")


def test_the_pooled_rice_paddy_margin_is_carried_by_site_a():
    """Gate G3 (config.py), phase G's load-bearing deliverable. The
    pooled positive (NIR - LUM on paddy_after_driving) must clear pooled
    over all three control-verified folds. Both halves are asserted, as for
    E2: the pooled pass (dropping it would be post-hoc re-scoping) and the
    per-fold decomposition (dropping it would overstate a positive carried by
    site000 while site001 runs directionally negative)."""
    runs = _folds.load_runs("mit_b0")
    px = _folds.A.gt_px()
    cls = "paddy_after_driving"
    per_fold = {}
    for fold, proto in _folds.G_FOLD_PROTO.items():
        si = _folds.A.paired_family(runs, proto, "NIR", "LUM", "isec", px)
        so = _folds.A.paired_family(runs, proto, "NIR", "LUM",
                                    _folds.A.own_proto("NIR"), px)
        per_fold[fold] = _folds.sec7(si, so, cls)
    keys, d = _pooled_over_folds(runs, _folds.G_FOLD_PROTO.values(),
                                 "NIR", "LUM", cls, px)
    lo, hi = _folds.A.cluster_ci(d, _folds.A.seq_units(keys))
    p = float(stats.wilcoxon(d).pvalue)
    assert p < config.ALPHA and lo > 0, (
        "manuscript sec. results reports gate G3 PASSES pooled over all three "
        f"folds, but delta {d.mean():+.4f} CI [{lo:+.4f},{hi:+.4f}] p={p:.2e}")
    assert per_fold["site000"] == "DIRECTIONAL (+)", (
        "manuscript sec. results states the G3 pooled pass is carried by site000, "
        f"but that fold is '{per_fold['site000']}'")
    assert per_fold["site001"] != "DIRECTIONAL (+)", (
        "manuscript sec. results states G3 does not clear the positive on site001 "
        f"individually (it runs negative), but it is '{per_fold['site001']}'")


def test_incomplete_phases_are_not_given_a_verdict():
    """A gate is read on the whole committed seed set or not at all. Guards
    analyze_folds.py against silently scoring a partial phase - the failure mode that
    would let phase F or G 'pass' on one seed."""
    md, ok = _folds.completeness([], _folds.load_runs("mit_b2"), "t",
                                 [("blocked", "mit_b2", 20)])
    n = len([m for m in _folds.load_runs("mit_b2")
             if m["protocol"] == "blocked"])
    assert ok == (n >= 20), (
        f"analyze_folds.py reports phase completeness as {ok} with {n}/20 runs on disk")



# Where each committed criterion's outcome reaches the reader: the macro
# printing its verdict where the verdict is a number, and the clause carrying
# it where the body states the verdict without printing one -- as a direction,
# or as a verdict whose values tab:robustness carries instead. A clause
# entry fails loud here if the sentence
# is reworded, which is what separates these from the manuscript wording
# quoted inside the assertion messages elsewhere in this file. Where the value
# moved into a float, the phase test above binds the float's own cell, so the
# number is still checked; what this map then asks is only that the body still
# says it.
CRITERION_IN_THE_BODY = {
    "E1": r"at site A is also positive on the leave-one-site-out fold, at \nirInflationLeaveOutSiteA{}",
    "E2": "NIR is better than the luminance control on the tilled surface on both cross-validations",
    "E3": r"\roadLeaveOutSiteA",
    "F1": r"but not detected on any of the \nModels{} models",
    "F2": r"the comparison detects no difference on any of the \nModels{} models",
    "G1": r"\tilledRiceSiteA",
    "G2a": r"\nirInflationRiceSiteA",
    "G2b": "raises the paddy-edge IoU of each of the four input configurations at site A",
    "G3": "NIR is better than the luminance control on the tilled surface on both cross-validations",
    "H1": r"but not detected on any of the \nModels{} models",
    "H2": r"the comparison detects no difference on any of the \nModels{} models",
    "F3 under phase G (POST-HOC)": r"\fusionRgbTilledRiceMean",
}


def test_folds_md_counts_exactly_the_committed_runs_of_each_phase():
    """results/folds.md opens each phase with the runs it found against the
    runs the matrix requires. A count above the requirement means the
    analyzer read runs the matrix does not hold, such as a run of another
    configuration left on disk, and a committed report written that way
    differs from what a clean checkout regenerates."""
    md = open(os.path.join(RES, "folds.md")).read()
    rows = re.findall(r"^\| `(\w+)` \| ([\w.-]+) \| (\d+) \| (\d+) \|", md, re.M)
    assert rows, "results/folds.md carries no completeness rows"
    off = [r for r in rows if r[2] != r[3]]
    assert not off, (
        "results/folds.md counts runs that differ from what the matrix "
        f"requires: {off}")


def test_every_committed_criterion_reaches_the_reader():
    """A criterion the paper commits and then never
    mentions again is the quietest way a gate stops binding, and no other
    assertion here can see it: a verdict nobody printed has no number to bind.

    The criterion set is DERIVED from results/folds.md, into which analyze_folds.py
    writes one section per criterion, rather than listed beside the map. A
    hand-written list would pass the first time a phase adds a criterion,
    which is the failure this exists to catch one phase later."""
    md = open(os.path.join(RES, "folds.md")).read()
    committed = re.findall(r"^### (.+)$", md, re.M)
    assert sorted(committed) == sorted(CRITERION_IN_THE_BODY), (
        f"analyze_folds.py computes criteria {sorted(committed)} and the manuscript "
        f"reports {sorted(CRITERION_IN_THE_BODY)}; a criterion the protocol "
        "commits and the paper never states the outcome of is a gate that "
        "has stopped binding")
    body = " ".join(_read_tex().split())     # the source is hard-wrapped
    for name, carrier in CRITERION_IN_THE_BODY.items():
        assert carrier in body, (
            f"results/folds.md returns a verdict for criterion {name}, which "
            f"reaches the reader through {carrier!r}; the manuscript no "
            "longer carries it, so that criterion is committed and never "
            "reported")


def test_every_verdict_holds_under_a_five_class_holm_family():
    """Manuscript sec. criteria: one configuration is better than another when
    the paired Wilcoxon signed-rank test 'rejects a median difference of zero at the 0.05 level after a Holm
    correction ..., which adjusts the tests of the target classes'. On a
    fold where fewer target classes are evaluable, the analyzer corrects over
    the classes that are. Entering each absent class at p = 1 restores a
    family of five, and this must leave every verdict unchanged, so the
    sentence holds for the verdicts that the paper prints."""
    runs = _folds.load_runs("mit_b0")
    px = _folds.A.gt_px()
    fixed = _folds.fixed_sets()
    gates = {"E1": ({f: ("official", p)
                     for f, p in _folds.E_FOLD_PROTO.items()}, None),
             "G2a": ({f: ("pofficial", p)
                      for f, p in _folds.G_FOLD_PROTO.items()}, None),
             "G2b": ({f: ("pofficial", p)
                      for f, p in _folds.G_FOLD_PROTO.items()}, fixed)}
    for gate, (pairs, restrict_by_fold) in gates.items():
        for fold, (leaked, blocked) in pairs.items():
            restrict = (None if restrict_by_fold is None
                        else restrict_by_fold.get(fold))
            if restrict is not None and not restrict:
                continue          # G2b's empty fixed set: no verdict is given
            for arm in _folds.ARMS:
                si = _folds.paired_protocols(runs, arm, leaked, blocked,
                                             "isec", px, restrict)
                so = _folds.paired_protocols(
                    runs, arm, leaked, blocked,
                    "nir" if arm == "NIR" else "rgb", px, restrict)
                live = {c: v["p"] for c, v in si.items()
                        if v is not None and c in TARGETS}
                if not live or si.get("paddy_edge") is None:
                    continue
                absent = [c for c in TARGETS if c not in live]
                got = _folds.sec7(si, so, "paddy_edge")
                forced = _folds.A.verdict(
                    "paddy_edge", si["paddy_edge"], so.get("paddy_edge"),
                    _folds.A.holm({**live, **{c: 1.0 for c in absent}})
                    .get("paddy_edge"))
                assert got == forced, (
                    "manuscript sec. criteria states a correction over the "
                    "five target classes, and forcing the family back to five "
                    "must leave every verdict unchanged, but "
                    f"{gate}/{fold}/{arm} reads '{got}' as run and "
                    f"'{forced}' at the protocol family size")


def test_sampled_frames_are_those_whose_number_is_a_multiple_of_the_stride():
    """Manuscript sec. materials: 'We keep the frames whose number within
    their sequence is a multiple of a fixed stride', which gives the sampled
    frames that every split divides."""
    recs = config.load_index()
    off = [(r["seq"], r["frame"]) for r in recs
           if r["frame"] % config.FRAME_STRIDE]
    assert not off, (
        "manuscript sec. materials states each sampled frame's number is a "
        f"multiple of the stride {config.FRAME_STRIDE}, but {len(off)} are "
        f"not, e.g. {off[:3]}")
    assert len(recs) == _splits()["stats"]["n_frames_total"], (
        "the sampled frames that the index holds are not the frames that the "
        "splits divide")


def test_the_inflation_on_the_fold_below_the_floor_excludes_zero():
    """Manuscript sec. concentration: 'NIR's paddy-edge inflation at site A
    is also positive on the leave-one-site-out fold, at +0.211, and its
    interval lies above zero.' The criterion is
    evaluated at every held-out site where the paddy edge is evaluable, so
    its outcome on that fold reaches the reader too."""
    s = _folds.paired_protocols(_folds.load_runs("mit_b0"), "NIR", "official",
                                _folds.E_FOLD_PROTO["site000"], "isec",
                                _folds.A.gt_px())["paddy_edge"]
    assert s["lo"] > 0, (
        "manuscript sec. concentration reports near-infrared's paddy-edge "
        "inflation on the leave-one-site-out fold at the site with most dry-field frames "
        f"whose interval excludes zero, but it is "
        f"[{s['lo']:+.4f},{s['hi']:+.4f}]")


def test_the_pooled_arm_gate_applies_no_correction_and_none_is_needed():
    """Manuscript sec. redundancy: 'the comparison detects no difference on
    any of the three models', and 'Pooled over the folds, NIR is better than
    the luminance control on the tilled surface on both cross-validations',
    which by the three conditions means that the Wilcoxon test rejects and the
    margin is also positive on the NIR annotation. The criterion runs single
    comparisons, so its Holm correction is its own p value.

    Asserted on the pooled cells analyze_folds.py actually gates, so the
    sentences track the code rather than a memory of it."""
    px = _folds.A.gt_px()
    b0 = _folds.load_runs("mit_b0")
    cells = {
        "E2": (b0, _folds.E_FOLD_PROTO, "NIR", "LUM"),
        "G3": (b0, _folds.G_FOLD_PROTO, "NIR", "LUM"),
        "F3/G": (b0, _folds.G_FOLD_PROTO, "RGBNIR", "RGB"),
        "F2": (_folds.load_runs("mit_b2"), {"site002": "blocked"},
               "NIR", "LUM"),
        "H2": (_folds.load_runs(config.BACKBONE_H, config.ARCH_H),
               {"site002": "blocked"}, "NIR", "LUM"),
    }
    failing = []
    for name, (runs, fold_proto, ax, ay) in cells.items():
        kd = [_folds._pooled_deltas(runs, p, ax, ay, "paddy_after_driving",
                                    px) for p in fold_proto.values()]
        keys = [k for ks, _ in kd for k in ks]
        d = np.concatenate([dd for _, dd in kd])
        p = float(stats.wilcoxon(d).pvalue)
        lo, hi = _folds.A.cluster_ci(d, _folds.A.seq_units(keys))
        ci_excl = lo > 0 or hi < 0
        if not (p < config.ALPHA and ci_excl):
            failing.append(name)
            assert not ci_excl, (
                f"manuscript sec. results states that the failing cells "
                f"detect no difference, but {name} has p={p:.3g} and interval "
                f"[{lo:+.4f},{hi:+.4f}]")
    assert set(failing) == {"F2", "H2"}, (
        "manuscript sec. results places the cells that detect no difference "
        f"on both repeat models, but they are {sorted(failing)}")
    assert (r"detects no difference on any of the \nModels{} models"
            in " ".join(_read_tex().split())), (
        "manuscript sec. results says 'detects no difference on any of the "
        "three models', and the two cells that detect none beside the primary "
        f"model are {sorted(failing)}")
    for fold_proto in (_folds.E_FOLD_PROTO, _folds.G_FOLD_PROTO):
        n_tot, s_tot = 0, 0.0
        for proto in fold_proto.values():
            st = _folds.A.paired_family(b0, proto, "NIR", "LUM", "nir",
                                        px)["paddy_after_driving"]
            n_tot += st["n"]
            s_tot += st["mean"] * st["n"]
        assert s_tot / n_tot > 0, (
            "manuscript sec. results states both pooled margins are positive "
            "on the near-infrared annotation, but one pools to "
            f"{s_tot / n_tot:+.4f}")


def test_the_excess_of_nir_over_every_other_inflation_is_detected_under_mit_b2_and_at_site_a():
    """Manuscript sec. concentration: 'NIR's inflation at site A, ..., exceeds that
    of each other configuration', and 'Under the MiT-B2 encoder, NIR's inflation there nonetheless exceeds
    that of each other configuration, whereas on the primary model and under
    UPerNet/ConvNeXt-V2-T it exceeds that of none.', where sec. criteria
    says that 'NIR's inflation exceeds another configuration's when the 95% interval
    of the mean difference between their per-frame gains excludes zero'; and 'but not
    detected on any of the three models', which the gate tests of each model
    assert.

    A ranking between configurations is a directional claim, so each excess is
    the paired per-frame difference of two inflations with its interval over
    sequences (analyze_folds.py paired_arm_inflation), never the larger of two
    point estimates. The rows are the five tab:robustness(a) carries, read
    through build_tables.py's own robustness(), and the other configurations
    are the arms the runs hold."""
    px = _folds.A.gt_px()
    G = _gen_tables()
    rows_a, _ = G.robustness(px)
    labels = [lbl for lbl, _ in rows_a]
    assert len(rows_a) == 5, f"tab:robustness(a) has {len(rows_a)} rows, not five"
    assert sorted(G.CONTRAST_ARMS) == sorted(a for a in _folds.ARMS if a != "NIR"), (
        "the manuscript compares NIR's inflation with that of every other "
        f"configuration, but tab:robustness(a) pairs it with {G.CONTRAST_ARMS}")
    want = {G.MODEL_LABEL[("segformer", "mit_b2")], G.SITE_ROWS[0][1]}
    assert want <= set(labels), (
        f"tab:robustness(a) has no MiT-B2 row or no site-A row among {labels}")
    for arm in G.CONTRAST_ARMS:
        cells = {lbl: infl["NIR-" + arm] for lbl, infl in rows_a}
        held = {lbl for lbl, st in cells.items() if st["lo"] > 0}
        assert held == want, (
            f"the manuscript detects NIR's excess over {arm} under MiT-B2 and "
            f"at site A alone, but its interval excludes zero on {sorted(held)}")
        below = {lbl for lbl, st in cells.items() if st["hi"] < 0}
        assert not below, (
            f"NIR's excess over {arm} has an interval below zero on "
            f"{sorted(below)}, which the manuscript reports as spanning zero")


def test_no_inflation_is_detected_at_site_c_on_the_primary_model():
    """Manuscript sec. concentration: 'At site C, NIR's paddy-edge inflation
    is positive but not detected on any of the three models, because its
    interval spans zero' and 'Under the MiT-B2 encoder, NIR's inflation there nonetheless exceeds
    that of each other configuration, whereas on the primary model and under
    UPerNet/ConvNeXt-V2-T it exceeds that of none.'. This is gate E1 at site C, the leave-one-site-out fold on
    which the repeat models also run. The differences with NIR are the
    tab:robustness(a) cells."""
    runs, px = _folds.load_runs("mit_b0"), _folds.A.gt_px()
    si = _folds.paired_protocols(runs, "NIR", "official", "blocked", "isec", px)
    so = _folds.paired_protocols(runs, "NIR", "official", "blocked", "nir", px)
    s, v = si["paddy_edge"], _folds.sec7(si, so, "paddy_edge")
    assert (s["mean"] > 0 and s["lo"] < 0 < s["hi"]
            and not v.startswith("DIRECTIONAL")), (
        "sec. concentration reports NIR's paddy-edge inflation at site C as "
        "positive with an interval spanning zero, "
        f"but it is {s['mean']:+.3f}, p={s['p']:.2e}, "
        f"[{s['lo']:+.3f}, {s['hi']:+.3f}], verdict '{v}'")
    rows_a, _ = _gen_tables().robustness(px)
    primary = rows_a[0][1]
    for arm in _gen_tables().CONTRAST_ARMS:
        st = primary["NIR-" + arm]
        assert st["lo"] < 0 < st["hi"], (
            f"sec. concentration reports NIR's excess over {arm} at site C as "
            f"spanning zero, but it is [{st['lo']:+.3f}, {st['hi']:+.3f}]")


# ------------- the body table's digest of the per-class results ------------
# tab:perclass prints every configuration's accuracy under the dataset's split
# and at site C, so two things need checking: the table carries no test that
# the paper does not state, and every configuration reaches the reader under
# both partitions.

def _gen_tables():
    return importlib.import_module("build_tables")


def test_the_body_table_reports_no_test_it_did_not_commit():
    """The body table tab:perclass prints every configuration's pooled IoU on
    every class under two partitions, so it must carry no per-cell test: a
    significance marker or an interval across a class-by-configuration grid
    would be a multiple comparison that sec. criteria does not state. The
    paired comparisons and their intervals are tab:margins', and the
    inflation criterion's interval is printed with its sequence count in
    tab:robustness."""
    t = _read_table("table_perclass_iou.tex")
    body = t.split(r"\midrule", 1)[1]
    # The tells are cell markers, so they are read off the rows alone. The
    # float is a table* - it spans both columns of the two-column layout -
    # and its closing tag would otherwise answer the "*" tell on every run.
    rows = body.split(r"\bottomrule", 1)[0]
    for tell in ("*", "$p", "p =", r"\dagger", "^{"):
        assert tell not in rows, (
            f"the body table carries {tell!r} in a cell beneath its rule, "
            "which reads as a per-cell test; sec. methods commits a criterion "
            "on one cell only")
    assert "[" not in body, (
        "the body table prints an interval, but its cells are pooled IoUs, "
        "whose paired comparisons tab:margins prints")
    folds = importlib.import_module("analyze_folds")
    A = importlib.import_module("analyze")
    gate = folds.paired_protocols(folds.load_runs("mit_b0"), folds.GATE_ARM,
                                  "official", "blocked", "isec",
                                  A.gt_px())["paddy_edge"]
    note = _read_table("table_robustness.tex").split(
        r"\begin{minipage}[t]{0.56\linewidth}", 1)[1]
    assert re.search(rf"\b{gate['k']}\b", note), (
        f"tab:robustness prints the criterion's interval over {gate['k']} "
        "sequences without stating that count")


def test_every_configuration_of_both_partitions_reaches_the_reader():
    """tab:perclass prints each configuration's pooled IoU under the dataset's
    split, the diagnostic network included, and at site C, with the spread
    that sec. criteria promises for every IoU.

    Derived from the runs rather than from the columns the table carries: a
    configuration the runs hold under either partition and no cell prints is
    a condition the paper stopped reporting."""
    G = _gen_tables()
    runs = _folds.A.load_runs()
    held = {(m["arm"], m["protocol"]) for m in runs
            if m["train_env"] == "all" and m["protocol"] in ("blocked", "official")}
    body = _read_table("table_perclass_iou.tex")
    for arm, proto in sorted(held):
        for row in G.BODY_ROWS:
            if row is None:
                continue
            cls, label = row
            c = G.cell(runs, arm, proto, cls)
            assert c in body and c != r"\textemdash", (
                f"the body table's {label} row prints no {proto} accuracy for "
                f"{arm}, so that configuration's runs go unreported")
    params = _read_params()
    for arm, macro in (("NIR", "nirEdgeSplit"), ("RGB", "rgbEdgeSplit"),
                       ("RGBNIR", "fusionEdgeSplit")):
        v = [m["iou"]["isec"]["paddy_edge"] for m in runs
             if m["arm"] == arm and m["protocol"] == "official"
             and m["train_env"] == "all"]
        want = f"{np.mean(v):.3f} \\pm {np.std(v):.3f}"
        got = re.search(rf"\\newcommand\{{\\{macro}\}}\{{\\ensuremath\{{([^}}]+)\}}",
                        params)
        assert got and got.group(1) == want, (
            "the paper prints the dataset split's paddy-edge "
            f"accuracy for {arm} as a mean with its spread over the "
            f"{len(v)} seeds, {want}, but params.tex prints "
            f"{got.group(1) if got else 'no such macro'}")
    assert "standing rice" in _read_table("table_config.tex"), (
        "the models segment standing rice and no other table prints it, so "
        "tab:config must list it among the classes the networks segment")


def test_the_paper_recomputes_without_the_imagery():
    """Availability statement: 'Every comparison between trained networks in
    this letter can be recomputed from the released artifacts alone, whereas
    retraining the networks, measuring the offset and measuring the agreement
    between the annotations need the imagery and its annotations.'

    That is a claim about which files the paper's chain opens, and a clone
    with no data/ directory in it is what tests it. Checked as a static
    property of the four modules every reported
    number comes through - the two generators and the two analyzers they call
    - because a trace only says what today's run happened to touch. The model
    passes are not in this set and do read imagery - through the per-frame
    paths the index carries, which is why both ways in are checked; what they
    leave behind is committed, and that is what makes the claim true."""
    dataset = ("IMAGES_ROOT", "LABELS_ROOT", "STUDY_DATA")
    for mod in ("analyze.py", "analyze_folds.py", "build_parameters.py",
                "build_tables.py"):
        with open(os.path.join(ROOT, "src", mod)) as f:
            src = f.read()
        # Two ways in, and the second is how src/train.py and
        # camera_offset.py actually
        # reach the imagery: the roots in config, and the per-frame image and
        # mask paths the index carries. A check on the roots alone would pass
        # a module that opened every frame through the index.
        hits = [k for k in dataset if re.search(rf"\bC\.{k}\b", src)]
        hits += [k for k in config.INDEX_PATH_KEYS if re.search(rf"\b{k}\b", src)]
        assert not hits, (
            "the availability statement says every comparison recomputes from "
            f"the released artifacts alone, but src/{mod} reads {hits} from "
            "the extracted dataset, which is not released")
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, check=True,
                             capture_output=True, text=True).stdout.split()
    for k in dataset:
        root = os.path.relpath(getattr(config, k), ROOT)
        under = [p for p in tracked if p == root or p.startswith(root + "/")]
        assert not under, (
            f"config.{k} is the extracted dataset the availability statement "
            f"says is not redistributed, and {len(under)} of its files are "
            f"under version control (e.g. {under[0]})")


def test_the_directional_rule_the_paper_states_is_the_one_the_code_applies():
    """Manuscript sec. methods states the three conditions under which one
    configuration is better than another,
    names the annotation the third is measured on. The paper no longer says
    that no target class satisfies the other two conditions and fails the
    third; the check below keeps that fact true of the code.

    analyze.py verdict() applies three; the criterion committed in config.py
    states two. The extra condition is strictly conservative, so the gap is
    silent by construction - it can only turn a ranking into no ranking, and
    every other assertion here reads the verdict the code returned, so none of
    them can see it act. Three things are asserted: that the manuscript still
    states three conditions, that the annotation it names is the one analyze.py
    scores the third on, and that the third separates no cell, which is what
    makes the difference between the two statements harmless rather than a
    criterion applied and never disclosed.

    The manuscript names that annotation by which configurations the family
    holds and analyze.py own_proto by which one leads the subtraction, so the
    agreement is a claim and is checked here over the families analyze.py
    defines. Derived over every family and protocol analyze.py computes rather
    than over the cells the manuscript names, and the rule itself is read from
    analyze.py rather than restated."""
    flat = " ".join(_read_tex().split())
    assert ("One configuration is better than another when three conditions "
            "hold") in flat and ("Third, the margin keeps its sign when the "
            "comparison is scored on a single annotation instead of the "
            "intersection ground truth") in flat, (
        "analyze.py verdict() applies three conditions to every verdict the "
        "paper prints; sec. methods no longer states three, so the paper "
        "reports a rule it did not run")
    assert ("the NIR annotation, on which NIR trains, when NIR is compared "
            "and the RGB annotation otherwise") in flat, (
        "sec. methods states the three conditions but no longer names the "
        "single-annotation one, which is the condition analyze.py adds to the "
        "committed criterion")
    A = importlib.import_module("analyze")
    for _fam, ax, ay in A.FAMILIES:
        stated = "nir" if "NIR" in (ax, ay) else "rgb"
        assert A.own_proto(ax) == stated, (
            "sec. methods says the third condition is measured on 'the NIR "
            "annotation, on which NIR trains, when NIR is compared and the RGB "
            f"annotation otherwise', which gives the "
            f"{stated} annotation for {ax} - {ay}, but analyze.py own_proto "
            f"scores that family on the {A.own_proto(ax)} annotation")
    runs, px = A.load_runs(), A.gt_px()
    split, seen = [], 0
    for _fam, ax, ay in A.FAMILIES:
        for protocol in ("blocked", "official"):
            si = A.paired_family(runs, protocol, ax, ay, "isec", px)
            so = A.paired_family(runs, protocol, ax, ay, A.own_proto(ax), px)
            if not si or not so:
                continue
            ph = A.holm({c: (si[c]["p"] if si.get(c) else None)
                         for c in TARGETS})
            for cls in TARGETS:
                a, b, p = si.get(cls), so.get(cls), ph.get(cls)
                if a is None or b is None or p is None:
                    continue
                seen += 1
                if p < config.ALPHA and (a["lo"] > 0 or a["hi"] < 0) \
                        and np.sign(a["mean"]) != np.sign(b["mean"]):
                    split.append((ax, ay, protocol, cls,
                                  round(a["mean"], 3), round(b["mean"], 3)))
    assert seen, "no family cell was scorable, so the claim has no base"
    assert not split, (
        "manuscript sec. methods says of the single-annotation condition "
        "that 'no target class satisfies the other two and fails it', but "
        f"{split} do, so a comparison in which the paper finds neither "
        "configuration better is ranked by the conditions the paper states")


# ---- the runs that train on one environment at a time -------------------
# An arm the committed results contain under a name no section mentions is
# the one that reads as suppression.

def test_every_training_restriction_the_runs_used_reaches_the_reader():
    """Nothing else in this file derives the RUN set. The criterion-coverage
    test above reads results/folds.md, so a phase that committed no criterion
    is invisible to it, and a sweep over the parameter file cannot see a phase
    that produced no macro; a phase with neither is reported by nothing and
    fails nothing.

    The manuscript reports no run trained on one environment alone, so no run
    on disk may be one. The restrictions are derived from what each run
    recorded, never from a directory suffix or a phase list written beside
    this test: a hand-written list passes the first time a restriction is
    added, which is the failure it would exist to catch."""
    used = sorted({m["train_env"] for m in _study_runs()})
    assert used == ["all"], (
        f"runs on disk train under restrictions {[e for e in used if e != 'all']}, "
        "and the manuscript reports no run trained on one environment alone, "
        "so those runs reach the reader as no number")
    matrix = sorted({r["train_env"] for r in config.RUN_MATRIX})
    assert matrix == ["all"], (
        f"src/config.py RUN_MATRIX commits runs under {matrix}, and the "
        "manuscript reports none trained on one environment alone")


def test_no_configuration_outperforms_color_at_every_held_out_super_site():
    """Manuscript sec. intro: 'every gain of a configuration over RGB on one of the five classes at a
    held-out site is not detected at another held-out site that has at least 50 frames of
    the class', and the abstract states the same.

    The set is every configuration against RGB on every target class, at
    every fold of both cross-validations; a class is scored at a fold only
    with the frame minimum of sec. protocols behind it. A configuration gains
    over RGB at a fold when its paired interval there lies above zero, the
    condition every directional reading requires, so the check can only grant
    more gains than the three conditions would. The claim fails where a gain
    has no other scored fold of the same cross-validation at which it is
    absent, including a class that only one fold scores."""
    A = importlib.import_module("analyze")
    runs, px = A.load_runs(), A.gt_px()
    unanswered = []
    for rot, folds in (("leave-one-site-out", _folds.E_FOLD_PROTO),
                       ("rice-paddy", _folds.G_FOLD_PROTO)):
        fams = {(arm, f): A.paired_family(runs, p, arm, "RGB", "isec", px)
                for arm in ("NIR", "LUM", "RGBNIR") for f, p in folds.items()}
        for arm in ("NIR", "LUM", "RGBNIR"):
            for cls in TARGETS:
                scored = {f: fams[arm, f][cls] for f in folds
                          if fams[arm, f].get(cls) is not None
                          and fams[arm, f][cls]["n"] >= config.MIN_VAL_FRAMES}
                for f, c in scored.items():
                    if c["lo"] > 0 and not any(o["lo"] <= 0 for g, o in
                                               scored.items() if g != f):
                        unanswered.append((rot, arm, cls, f))
    assert not unanswered, (
        "sec. intro states that every gain over RGB at a held-out site is "
        "absent at another held-out site that has at least 50 frames of the "
        "class, but these "
        f"gains have no such site: {unanswered}")


def test_only_evaluable_classes_are_scored():
    """Manuscript sec. protocols: 'A class is scored at a held-out site when at
    least 50 of the site's frames carry it.'

    The analyzers score a class from the frames that enter its comparison,
    so the minimum has to hold there: a cell below it must come back
    unscored, or it joins a Holm family and folds.md prints a verdict the
    paper says is never computed. The rice-paddy fold at site B is where a
    class falls between the two thresholds the code could apply: its
    intersection ground truth marks the field ridge on fewer frames than the
    minimum, and on more than none."""
    runs, px = _folds.A.load_runs(), _folds.A.gt_px()
    proto = _folds.G_FOLD_PROTO["site001"]
    arm = _folds.A.paired_family(runs, proto, "NIR", "RGB", "isec", px)
    infl = _folds.paired_protocols(runs, "NIR", "pofficial", proto, "isec", px)
    ci = config.CLASSES.index("field_levee")
    seq2site = {tuple(m.split("/")[1:]): s
                for m, s in _index_key2site().items()}
    marked = [k for k, v in px.items() if v[0][ci] > 0 and v[1][ci] > 0
              and seq2site.get((k[0], k[1])) == "site001"]
    assert marked, ("no frame of site B carries the field ridge on the "
                    "intersection ground truth, so this check resolves nothing")
    for name, fam in (("NIR - RGB", arm), ("NIR inflation", infl)):
        assert fam, f"the {name} family at site B returned nothing"
        small = {c: v["n"] for c, v in fam.items()
                 if v is not None and v["n"] < config.MIN_VAL_FRAMES}
        assert not small, (
            "sec. protocols says only classes with at least "
            f"{config.MIN_VAL_FRAMES} validation frames are scored, but the "
            f"{name} family of the paddy-restricted fold at site B scores "
            f"{small}")


def test_the_site_blocked_super_site_is_the_only_one_meeting_the_frame_minima():
    """Manuscript sec. protocols: 'Only site C holds at least 50 frames of every target class, with at
    least 200 training frames of each at the other two sites.' Derived over every site from the
    split's own per-site class counts."""
    sp = json.load(open(config.SPLITS_JSON))
    sites = sp["sites"]
    ok = []
    for hold in sites:
        good = True
        for cls in config.CLASSES[1:]:
            if cls in config.FLOOR_EXEMPT:
                continue
            nv = hold["class_frames"].get(cls, 0)
            nt = sum(x["class_frames"].get(cls, 0) for x in sites
                     if x is not hold)
            if nv < config.MIN_VAL_FRAMES or nt < config.MIN_TRAIN_FRAMES:
                good = False
        if good:
            ok.append(hold["site_id"])
    held = sorted(k for k, v in sp["assignment"].items() if v == "val")
    assert ok == held == ["site002"], (
        "manuscript sec. methods states the site-blocked protocol holds out "
        "the only site meeting the frame minima, but the sites "
        f"meeting them are {ok} and the protocol holds out {held}")
    assert config.FLOOR_EXEMPT == ["common_tree"], (
        "manuscript sec. methods exempts only the in-field trees from the "
        f"frame minima, but config.py exempts {config.FLOOR_EXEMPT}")


def test_every_macro_the_parameter_file_defines_reaches_the_page():
    """The mirror of the check below: that one follows a path the
    document names out to the tree, this one follows a macro back to the page.

    A macro defined and never printed is a number the pipeline still computes
    and no reader ever sees, so it drifts with every regeneration unwatched -
    and where its last reader was a sentence someone deleted, the deletion left
    a value behind that no other check in this file can see.

    Consumers are the recursive \\input closure of main.tex, computed here
    rather than listed, so a table added tomorrow joins it by being loaded."""
    paper = os.path.join(ROOT, "paper")
    seen, stack = [], [os.path.join(paper, "main.tex")]
    while stack:
        path = stack.pop()
        if path in seen:
            continue
        assert os.path.exists(path), (
            f"main.tex loads paper/{os.path.relpath(path, paper)} transitively "
            "but no such file exists, so the document typesets against nothing")
        seen.append(path)
        body = re.sub(r"(?<!\\)%.*", "", open(path).read())
        stack += [os.path.join(paper, n if n.endswith(".tex") else n + ".tex")
                  for n in re.findall(r"\\input\{([^}]+)\}", body)]

    defined, used = {}, set()
    for path in seen:
        body = re.sub(r"(?<!\\)%.*", "", open(path).read())
        for name in re.findall(r"\\newcommand\{\\(\w+)\}", body):
            defined[name] = os.path.relpath(path, paper)
        # Drop the \newcommand heads so defining a macro is not using it; what
        # survives includes the macro bodies, where one may print another.
        used |= set(re.findall(r"\\(\w+)",
                               re.sub(r"\\newcommand\{\\(\w+)\}", " ", body)))

    orphans = sorted(n for n in defined if n not in used)
    assert not orphans, (
        f"{len(orphans)} macros are defined and never printed, so the "
        "manuscript carries values no reader sees and a regeneration can move "
        "them unwatched. Delete each, or print it:\n    "
        + "\n    ".join(f"\\{n}  (paper/{defined[n]})" for n in orphans))


def _population_declarations():
    """params.tex's population declarations: the registry (population name ->
    the artifact holding its members) and, per macro, the population it is
    measured over and whether it counts that population's members."""
    registry, macros, block = {}, {}, None
    for line in _read_params().splitlines():
        m = re.match(r"% population (.+?) = (\S+) :", line)
        if m:
            registry[m.group(1)] = m.group(2)
            continue
        m = re.match(r"%.*----\s*\[population: ([^\]]+)\]\s*$", line)
        if m:
            block = m.group(1)
            continue
        m = re.match(r"\\newcommand\{\\(\w+)\}", line)
        if m:
            own = re.search(r"\[population: ([^\]]+)\]", line)
            macros[m.group(1)] = (own.group(1) if own else block,
                                  "[count]" in line)
    return registry, macros


def test_every_macro_declares_the_population_it_is_measured_over():
    """A binding says which artifact a number came from, not what the number
    counts, and this dataset has more than one set a count can be of: the
    scene-matched frames the dataset holds, the strided index every protocol
    partitions, the recording sequences, and the sites they cluster
    into. A sentence printing counts of two of them reads as one.

    So every macro declares its population in params.tex, and this reads the
    declarations rather than the comments around them. It fails a macro with
    none, a population no artifact holds, and two population names resolving
    to one artifact, which is one population under a stale second name. It
    then requires every count to be printed beside its own population's name,
    the unit noun, and fails one printed beside another population's: that is
    the sentence that names the wrong thing beside a correct number. Two
    populations coinciding in size are not an error; the registry naming both
    artifacts is what discharges the coincidence."""
    registry, macros = _population_declarations()
    assert len(macros) >= 50, (
        f"only {len(macros)} macros parsed from params.tex; the declaration "
        "reader is stale, so this check is passing on too few")

    undeclared = sorted(n for n, (pop, _) in macros.items() if not pop)
    assert not undeclared, (
        f"{undeclared} declare no population, so nothing says what set each "
        "number is measured over")
    unknown = sorted({pop for pop, _ in macros.values()}
                     - set(registry) - {"design"})
    assert not unknown, (
        f"params.tex declares populations {unknown} that no '% population' "
        "line resolves to an artifact")

    resolved = {}
    for name, artifact in registry.items():
        path, _, key = artifact.partition("#")
        full = os.path.join(ROOT, path)
        assert os.path.exists(full), (
            f"population '{name}' resolves to {path}, which does not exist")
        if key:
            body = open(full).read()
            present = (key in json.loads(body) if path.endswith(".json")
                       else key in body)
            assert present, (
                f"population '{name}' resolves to {path}#{key}, and {path} "
                f"holds no {key}")
        resolved.setdefault(artifact, []).append(name)
    aliased = {a: n for a, n in resolved.items() if len(n) > 1}
    assert not aliased, (
        "two population names resolve to one artifact, so one is a stale name "
        f"for the other: {aliased}")

    # Every use of a count, in every file the document loads.
    paper = os.path.join(ROOT, "paper")
    text = " ".join(re.sub(r"(?<!\\)%.*", "", open(os.path.join(paper, f)).read())
                    for f in os.listdir(paper)
                    if f.endswith(".tex") and f != "params.tex")
    nouns = {n: n.lower().split() for n in registry}

    def at(words, noun):
        """Position of noun in the first three words, singular or plural."""
        for i in range(min(3, len(words) - len(noun) + 1)):
            seg = words[i:i + len(noun)]
            if seg[:-1] == noun[:-1] and seg[-1].rstrip("s") == noun[-1].rstrip("s"):
                return i
        return None

    wrong = []
    for name, (pop, count) in sorted(macros.items()):
        if not count:
            continue
        for m in re.finditer(r"\\%s(?![A-Za-z])" % name, text):
            after = re.sub(r"\\[A-Za-z]+|[{}~]", " ", text[m.end():m.end() + 80])
            words = re.findall(r"[A-Za-z][A-Za-z-]*", after.lower())
            own = at(words, nouns[pop])
            other = min((i for n in registry if n != pop
                         and len(nouns[n]) > len(nouns[pop])
                         for i in [at(words, nouns[n])] if i is not None),
                        default=None)
            if own is None or (other is not None and other <= own):
                wrong.append(f"\\{name} ({pop}): ...{' '.join(words[:5])}")
    assert not wrong, (
        "each count below is printed beside a noun other than its "
        "population's name, so the sentence names a different set than the "
        "number counts:\n    " + "\n    ".join(wrong))


def test_every_path_the_document_reads_is_generator_output_and_fresh():
    """Every file main.tex pulls in must resolve inside paper/,
    must name the generator that writes it, and must be what that generator
    writes today.

    Enumerated from main.tex rather than from a list of the artifacts we
    remember going stale, because a check written around remembered filenames
    passes the first time a table or figure is added, which is the failure it
    exists to prevent one file later.

    Freshness is byte-identity, and which artifacts it covers is derived
    from both ends - what the document includes and what build_tables.py publishes -
    so a table added to one side and not the other fails here."""
    G = _gen_tables()
    tex = _read_tex()
    paper = os.path.join(ROOT, "paper")
    reads = ([(m, ".tex") for m in re.findall(r"\\input\{([^}]+)\}", tex)]
             + [(m, "") for m in
                re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)])
    assert len(reads) >= 4, (
        f"main.tex reads only {len(reads)} files; the regex that enumerates "
        "them is stale, so this check is passing on an empty list")

    for name, ext in reads:
        path = os.path.join(paper, name + ext)
        assert os.path.exists(path), (
            f"main.tex reads {name}{ext}, which does not exist in paper/")
        if ext == "":                      # a figure carries no comment header
            continue
        head = open(path).read(400)
        assert re.search(r"src/\w+\.py", head), (
            f"paper/{name}{ext} is included by the manuscript but names no "
            "generator in its header, so nothing says where its numbers came "
            "from or how to refresh them")

    published = {os.path.basename(p)[:-len(".tex")] for p in G.OUTPUTS}
    included = {n for n, e in reads if e == ".tex"} - {"params"}
    assert published == included, (
        f"build_tables.py publishes {sorted(published)} and main.tex includes "
        f"{sorted(included)}; a generated table the document does not read is "
        "an artifact no reader sees, and one it reads that no generator "
        "writes is a copy nothing refreshes")

    before = {p: open(p).read() for p in G.OUTPUTS}
    G.write_all()
    stale = [os.path.basename(p) for p, b in before.items()
             if open(p).read() != b]
    for path, b in before.items():         # leave the tree as we found it
        with open(path, "w") as f:
            f.write(b)
    assert not stale, (
        f"paper/{stale} is not what build_tables.py writes today, so the manuscript "
        "is publishing an older run than results/ holds")


def test_every_manuscript_label_the_repo_names_still_resolves():
    """The path rule above, one namespace over. LaTeX resolves \\ref
    inside the document and reports an undefined reference; nothing resolves a
    label named anywhere else - a docstring, an inline comment, a failure
    message - and exactly those are required to cite the manuscript by its
    \\label slug rather than by a rendered number.

    That requirement is what hides this. The slug is chosen because it survives
    renumbering, so it reads as the durable form, while folding or merging a
    section deletes it outright and every other check in this file stays green.

    There is no exemption, because a record that has to describe an older tree
    can always name the section in words instead of naming a slug nothing
    defines. Both ends are derived: the labels from the \\input closure of
    main.tex, the references from git, so a file added tomorrow is scanned by
    being tracked and a label added tomorrow resolves by being defined."""
    paper = os.path.join(ROOT, "paper")
    seen, stack = [], [os.path.join(paper, "main.tex")]
    while stack:
        path = stack.pop()
        if path in seen:
            continue
        assert os.path.exists(path), (
            f"main.tex loads paper/{os.path.relpath(path, paper)} transitively "
            "but no such file exists, so no label it defines can resolve")
        seen.append(path)
        body = re.sub(r"(?<!\\)%.*", "", open(path).read())
        stack += [os.path.join(paper, n if n.endswith(".tex") else n + ".tex")
                  for n in re.findall(r"\\input\{([^}]+)\}", body)]

    defined = set()
    for path in seen:
        body = re.sub(r"(?<!\\)%.*", "", open(path).read())
        defined |= set(re.findall(r"\\label\{([^}]+)\}", body))
    prefixes = sorted({d.split(":")[0] for d in defined if ":" in d})
    assert len(defined) >= 10 and len(prefixes) >= 3, (
        f"only {len(defined)} labels under {prefixes} were found in main.tex "
        "and what it inputs; the regex that enumerates them is stale, so this "
        "check is scanning the repository for nothing")

    pattern = re.compile(r"\b(?:%s):[A-Za-z][\w-]*" % "|".join(prefixes))
    tracked = subprocess.run(["git", "-C", ROOT, "ls-files"],
                             capture_output=True, text=True, check=True)
    dangling = []
    for name in tracked.stdout.split():
        try:
            with open(os.path.join(ROOT, name), encoding="utf-8") as f:
                body = f.read()
        except (UnicodeDecodeError, OSError):
            continue                       # imagery and archives name nothing
        for n, line in enumerate(body.splitlines(), 1):
            for slug in sorted(set(pattern.findall(line)) - defined):
                dangling.append(f"{name}:{n}  {slug}")
    assert not dangling, (
        f"{len(dangling)} references name a manuscript label the document no "
        "longer defines, so each points a reader at nothing:\n    "
        + "\n    ".join(dangling)
        + "\n  Repoint each at the label carrying that material now. A record "
          "describing an older tree names that section in words instead; it "
          "never keeps a slug the manuscript has dropped.")


def test_every_document_section_the_repo_cites_still_resolves():
    """The same namespace rule, one document over. A comment citing a
    numbered section of a sibling markdown file points at nothing once that
    file is deleted, renamed or renumbered, because nothing dereferences it
    and every other check in this repository stays green.

    Scanning nothing is allowed only where nothing could be scanned: no
    tracked document defines a numbered section for a citation to name. The
    day one does, a tree that cites none of them is a stale pattern.

    Both ends are derived. The sections come from the headings of the tracked
    documents and the citations from git, so a file added tomorrow is scanned
    by being tracked. What separates a citation of a deleted document from a
    reference to the citing file's own sections is resolved against every path
    this repository has ever held: the first names a path git knows and the
    tree no longer carries, the second names no path at all.

    A record describing an older tree is not exempt. It can always name the
    section in words, exactly as the label check above requires.

    What is exempt is the typesetter's own support code. The class and font
    descriptions under paper/ are upstream template files, redistributed so
    the manuscript builds without a site-wide install; this repository authors
    none of them, and their prose is about TeX counters rather than about a
    document here."""
    def paths(cmd):
        return subprocess.run(["git", "-C", ROOT] + cmd, capture_output=True,
                              text=True, check=True).stdout.split()

    tracked = paths(["ls-files"])
    ever = paths(["log", "--all", "--pretty=format:", "--name-only"])

    def names_of(path):
        base = os.path.basename(path).lower()
        return {base} | (set(re.split(r"[._-]", base)) - {""})

    def named_by(word, pool):
        return sorted({p for p in pool if word.lower() in names_of(p)})

    def sections(path):
        try:
            with open(os.path.join(ROOT, path), encoding="utf-8") as f:
                return set(re.findall(r"^#+\s*(\d+[a-z]?)\.", f.read(), re.M))
        except (UnicodeDecodeError, OSError):
            return set()

    cite = re.compile(r"([\w./-]+)\s*(?:sec\.\s*|section\s+|§\s*)(\d+[a-z]?)",
                      re.I)
    # The extensions are the typesetter's, not this repository's, so a font
    # committed beside the class tomorrow is exempt by being one of them.
    upstream = (".cls", ".fd", ".map")
    dangling, resolved = [], 0
    for name in tracked:
        if name.endswith(upstream):
            continue                       # upstream template, authored away
        try:
            with open(os.path.join(ROOT, name), encoding="utf-8") as f:
                body = f.read()
        except (UnicodeDecodeError, OSError):
            continue                       # imagery and archives cite nothing
        for n, line in enumerate(body.splitlines(), 1):
            for word, num in cite.findall(line):
                here = named_by(word, tracked)
                if len(here) > 1:
                    dangling.append(f"{name}:{n}  '{word}' section {num} "
                                    f"names {here}, so it is ambiguous")
                elif here:
                    if num in sections(here[0]):
                        resolved += 1
                    else:
                        dangling.append(f"{name}:{n}  '{word}' section {num} "
                                        f"names {here[0]}, which has no "
                                        f"section {num}")
                elif named_by(word, ever):
                    dangling.append(
                        f"{name}:{n}  '{word}' section {num} names "
                        f"{named_by(word, ever)[0]}, which git has and the "
                        "tree no longer tracks")
                elif num not in sections(name):
                    dangling.append(f"{name}:{n}  section {num} names no "
                                    f"document, and {name} has no section "
                                    f"{num} of its own")
    assert resolved or not [p for p in tracked if sections(p)], (
        "no citation resolved, and the tree holds a document with numbered "
        "sections for one to name, so the pattern enumerating citations is "
        "stale and this check scanned nothing")
    assert not dangling, (
        f"{len(dangling)} references name a document section nothing "
        "defines, so each points a reader at nothing:\n    "
        + "\n    ".join(dangling)
        + "\n  Repoint each at the document carrying that material now. A "
          "record describing an older tree names that section in words "
          "instead; it never keeps a number nothing defines.")


def test_every_repo_path_a_file_names_still_resolves():
    """The path rule of the two checks above, turned on the repository's own
    tree. A README command, a docstring, a comment and a failure message all
    name sibling files by path, no program dereferences any of them, and a
    renamed script leaves every one pointing at nothing with the suite green.

    Both ends are derived. The citations come from the tracked files, and what
    one may resolve to comes from git: a path some tracked file lives under,
    or one the repository declares it does not track. A shorthand naming a
    file by its stem resolves the same way a full path does, because what
    makes either dangle is the rename and not the extension.

    A citation carrying no directory is the same reference and dangles the
    same way, so a bare file name is read against the tracked base names.
    What separates one from a file name that is data - a dataset record listed
    in a results file - is the same question the section check asks: a
    citation names something this repository has held, so only a name git
    knows and the tree no longer carries is reported."""
    def git(*cmd):
        return subprocess.run(["git", "-C", ROOT, *cmd], check=True,
                              capture_output=True, text=True).stdout.split()

    tracked = git("ls-files")
    # The first segment of a citation has to name something this tree holds,
    # or every "RGB/NIR" and every URL path is scanned as a repo path. The
    # untracked-but-declared directories are the ones the dataset lands in, so
    # they come from the ignore file rather than from a list here.
    tops = {p.split("/")[0] for p in tracked if "/" in p}
    with open(os.path.join(ROOT, ".gitignore")) as f:
        tops |= {ln.strip().lstrip("!/").split("/")[0] for ln in f
                 if ln.strip() and not ln.startswith("#") and "*" not in ln}
    pat = re.compile(r"(?<![\w/.-])((?:\.\./)*(?:"
                     + "|".join(sorted(re.escape(t) for t in tops if t))
                     + r")(?:/[\w.*-]+)+)")
    bases = {os.path.basename(p) for p in tracked}
    ever = {os.path.basename(p) for p in
            git("log", "--all", "--pretty=format:", "--name-only")}
    exts = sorted({os.path.splitext(p)[1].lstrip(".") for p in tracked} - {""})
    bare = re.compile(r"(?<![\w/.-])([\w-]+\.(?:"
                      + "|".join(re.escape(e) for e in exts) + r"))(?![\w.-])")

    def under(path):
        """The path with every globbed segment and what follows dropped."""
        keep = []
        for seg in path.split("/"):
            if "*" in seg:
                break
            keep.append(seg)
        return "/".join(keep)

    def holds(rel):
        return any(t.startswith(rel) for t in tracked)

    cited, resolved = {}, 0
    for name in tracked:
        try:
            with open(os.path.join(ROOT, name), encoding="utf-8") as f:
                body = f.read()
        except (UnicodeDecodeError, OSError):
            continue                       # imagery and archives cite nothing
        for n, line in enumerate(body.splitlines(), 1):
            for tok in pat.findall(line):
                tok = under(tok.rstrip("._,;:"))
                # A file names a sibling either from the root or from its own
                # directory, and a reader follows both, so a citation dangles
                # only where neither reading reaches anything.
                here = os.path.normpath(
                    os.path.join(os.path.dirname(name), tok))
                if holds(tok) or holds(here):
                    resolved += 1
                else:
                    cited.setdefault(tok, []).append(f"{name}:{n}")
            for tok in bare.findall(line):
                if tok in bases:
                    resolved += 1
                elif tok in ever:
                    cited.setdefault(tok, []).append(f"{name}:{n}")

    ignored = set()
    if cited:
        # A rule naming a directory matches only with the trailing slash the
        # rule carries, and a cited directory is written without one.
        probe = [f"{p}{sep}" for p in cited for sep in ("", "/")]
        out = subprocess.run(["git", "-C", ROOT, "check-ignore", "--stdin"],
                             input="\n".join(probe), capture_output=True,
                             text=True).stdout.split()
        ignored = {p.rstrip("/") for p in out}
    dangling = [f"{p}  ({', '.join(w)})"
                for p, w in sorted(cited.items()) if p not in ignored]
    assert resolved, (
        "no path citation in the tree resolved to a tracked file, so the "
        "pattern enumerating them is stale and this check scanned nothing")
    assert not dangling, (
        f"{len(dangling)} paths are named by a repo file and exist nowhere, "
        "so each points a reader at nothing:\n    "
        + "\n    ".join(dangling)
        + "\n  Repoint each at the path carrying that material now, or drop "
          "the sentence with the file it described.")


def test_every_path_a_script_builds_from_its_own_location_lands_in_the_tree():
    """The check above reads the paths a file names in words. This one reads
    the paths a file computes, which no reader sees and no message quotes: a
    script walks up from __file__ to the repository root and joins a tracked
    directory onto it. Move that script one level deeper and the walk stops
    inside src/, every join lands in a directory that does not exist, and the
    suite stays green because the committed artifact the tests read is still
    sitting where it was written. The generator then stops writing what the
    paper publishes, which is the one state a re-run cannot reveal.

    Both ends are derived. The bases come from evaluating each module-level
    assignment against the file's real location, in order, so a name built
    from an earlier name resolves the way the module resolves it. What counts
    as a repository directory comes from git, the same set the check above
    reads. A script that reaches the tree through the protocol record imports
    a resolved path instead of computing one and defines no base to check."""
    tracked = subprocess.run(["git", "-C", ROOT, "ls-files"], check=True,
                             capture_output=True, text=True).stdout.split()
    tops = {p.split("/")[0] for p in tracked if "/" in p}
    with open(os.path.join(ROOT, ".gitignore")) as f:
        tops |= {ln.strip().lstrip("!/").split("/")[0] for ln in f
                 if ln.strip() and not ln.startswith("#") and "*" not in ln}
    bad = []
    for rel in [p for p in tracked if p.endswith(".py")]:
        path = os.path.join(ROOT, rel)
        with open(path) as f:
            tree = ast.parse(f.read(), rel)
        # Evaluate the module-level names in source order, so a base defined
        # from an earlier base resolves as the module resolves it. Anything
        # needing more than os and __file__ is not a location walk.
        env = {"os": os, "__file__": path}
        bases = {}
        for node in tree.body:
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)):
                continue
            try:
                val = eval(compile(ast.Expression(node.value), rel, "eval"),
                           dict(env))
            except Exception:
                continue
            if isinstance(val, str):
                env[node.targets[0].id] = val
                bases[node.targets[0].id] = val
        # A join of a computed base onto a directory this repository has is a
        # path into the tree, so the base is the root the tree hangs from.
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and node.args
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "join"):
                continue
            head, *rest = node.args
            if not (isinstance(head, ast.Name) and head.id in bases and rest
                    and isinstance(rest[0], ast.Constant)
                    and rest[0].value in tops):
                continue
            landing = os.path.join(bases[head.id], rest[0].value)
            if os.path.dirname(landing) != ROOT:
                bad.append(f"{rel}:{node.lineno}: {head.id} = "
                           f"{bases[head.id]}, so joining "
                           f"{rest[0].value!r} lands at {landing}")
    assert not bad, (
        f"{len(bad)} joins reach outside the repository, so the script writes "
        "or reads where the tree has nothing and the committed artifact it "
        "owns can no longer be regenerated:\n    " + "\n    ".join(bad)
        + "\n  Walk up from __file__ as far as the file's own depth requires, "
          "or take the path from the protocol record.")


def test_every_direction_check_reads_the_supervising_annotation():
    """Manuscript sec. criteria: the third condition is that 'the margin keeps its sign when the
    comparison is scored on a single annotation instead of the intersection
    ground truth, so that the pixels on which the two annotations disagree
    are scored too. That annotation is the NIR
    annotation, on which NIR trains, when NIR is compared and the RGB
    annotation otherwise'. Which
    annotation supervises a configuration is one fact, src/train.py SUP_MASK, so
    every reader of that rule returns what SUP_MASK assigns rather than a
    copy of it.

    Two halves, because a wrong copy leaves no trace in the artifact. The
    third condition can only withhold a ranking the first two grant, so a
    gate reading it off the other annotation leaves every cell those two
    already refused exactly as it was, and the delta it reads can carry the
    opposite sign. The first half checks own_proto against SUP_MASK over
    every arm the dict defines. The second records which annotation each gate
    in analyze_folds.py asks the paired statistic for, over the comparisons
    results/folds.md shows it gating.
    """
    train = importlib.import_module("train")
    A = importlib.import_module("analyze")
    folds = importlib.import_module("analyze_folds")

    for arm, mask in train.SUP_MASK.items():
        want = mask[len("mask_"):]
        for mod, fn in (("analyze.py", A.own_proto),):
            assert fn(arm) == want, (
                f"src/train.py supervises {arm} with {mask}, so sec. methods "
                f"measures a {arm} comparison on the {want!r} annotation; "
                f"{mod} own_proto returns {fn(arm)!r} for it")

    asked = []
    keep_family, keep_protocols = A.paired_family, folds.paired_protocols

    def spy_family(runs, protocol, arm_x, arm_y, proto, px):
        asked.append((arm_x, proto))
        return keep_family(runs, protocol, arm_x, arm_y, proto, px)

    def spy_protocols(runs, arm, proto_x, proto_y, label_proto, px,
                      restrict=None):
        asked.append((arm, label_proto))
        return keep_protocols(runs, arm, proto_x, proto_y, label_proto, px,
                              restrict)

    with open(os.path.join(RES, "folds.md")) as f:
        pairs = sorted(set(re.findall(r"`(\w+)` minus `(\w+)` on `", f.read())))
    assert pairs, ("results/folds.md names no pooled configuration "
                   "comparison, so this test ranges over nothing")

    # No runs are passed: both gates ask for their two annotations before
    # they discover there is nothing to score, and the request is what is
    # under test.
    A.paired_family, folds.paired_protocols = spy_family, spy_protocols
    try:
        for arm_x, arm_y in pairs:
            folds.pooled_arm_gate([], [], {"f": "blocked"}, arm_x, arm_y,
                                  "paddy_after_driving", {}, "spy", "")
        folds.leakage_gate([], [], {"f": ("official", "blocked")}, {},
                           "spy", "")
    finally:
        A.paired_family, folds.paired_protocols = keep_family, keep_protocols

    for arm in sorted({a for a, _ in asked}):
        got = sorted({p for a, p in asked if a == arm})
        assert got == sorted({"isec", A.own_proto(arm)}), (
            f"sec. methods measures the first two directional conditions on "
            f"the intersection ground truth and the third on the "
            f"{A.own_proto(arm)!r} annotation for {arm}; analyze_folds.py asks for "
            f"{got}")


def test_the_manuscript_names_the_rule_and_the_criteria_one_way_each():
    """Manuscript sec. methods separates two objects. The decision procedure
    that every comparison between configurations uses is stated as three
    conditions and carries no name of its own, by the global rule against
    invented terms. A criterion is one committed gate, named by the ordinal
    Methods assigns it, and 'two criteria' names the set. Two names for one object make a reader check
    whether a second object entered, and 'the protocol rule' and 'the protocol
    criteria' differ by one noun while naming different things.

    Derived over the manuscript rather than over the sentences that carried
    the drift, so a further name fails on the day someone writes it."""
    named = set(re.findall(r"the (\w+) (rule|criterion|criteria)",
                           _read_tex()))
    allowed = {("first", "criterion"), ("second", "criterion"),
               ("third", "criterion"), ("three", "criteria")}
    assert named <= allowed, (
        "sec. methods states the decision procedure as three conditions with "
        "no name of its own, and gives each committed gate its ordinal, 'the "
        f"second criterion'; the manuscript also writes "
        f"{sorted(named - allowed)}")


def test_the_manuscript_names_every_class_it_drops_for_want_of_training_data():
    """Table tab:config lists the classes that the networks segment, 'and
    background, which also covers standing water and dry-field crop, the two
    classes that only the validation set annotates'.

    The set that sentence ranges over is derived from the inventory rather
    than read off the sentence. A class annotated only in the validation
    material is untrainable on the same ground as the one the sentence used
    to name alone, and an unnamed one disappears into the background with
    every count in the paper still correct. The inventory counts the whole
    walk, so a class with no training instance there has none in the
    scene-matched subset either."""
    inv = _load("inventory.json")["instance_counts"]
    train, val = defaultdict(int), defaultdict(int)
    for key, n in inv.items():
        split, env, _modality, cls = key.split("|")
        if env in config.ENVS:
            (train if split == "train" else val)[cls] += n
    gloss = {"paddy_water": "standing water", "field_corps": "dry-field crop"}
    tex = " ".join((_read_tex() + _read_table("table_config.tex")).split())
    for cls in sorted(c for c, n in val.items() if n and train[c] == 0):
        assert cls in gloss, (
            f"{cls} is annotated in the studied environments and has no "
            "training instance there, so sec. materials excludes it for the "
            "reason it excludes standing water; this test carries no name "
            "to look for it under")
        assert gloss[cls] in tex, (
            f"the paper excludes {cls} for want of training instances "
            "without naming it, so a reader reads it into the background")


def test_the_paper_prints_every_floor_class_at_site_a():
    """Manuscript sec. concentration prints, for the networks trained without
    site A, their pooled IoU at site A on the farm access road and on the
    tilled paddy surface, on both cross-validations: 'is 0.248 on the leave-one-site-out fold and 0.056 on the rice-paddy
    fold' and 'On the tilled surface, averaged in the same way, they score 0.362 at
    site A on the leave-one-site-out fold and 0.392 on the rice-paddy fold,
    the lowest of the three rice-paddy folds.' analyze_folds.py reads a floor on
    one class per cross-validation, and the two classes give opposite readings
    at site A, so the paper prints both classes on both cross-validations and
    rests no claim on either reading.

    The classes come from results/folds.md, which is what analyze_folds.py
    writes when it applies a floor."""
    with open(os.path.join(RES, "folds.md")) as f:
        classes = set(re.findall(r"Floor: `(\w+)` intersection IoU", f.read()))
    want = {"common_road": {"roadLeaveOutSiteA", "roadRiceSiteA"},
            "paddy_after_driving": {"tilledLeaveOutSiteA", "tilledRiceSiteA"}}
    assert classes == set(want), (
        f"analyze_folds.py reads floors on {sorted(classes)}, and this test "
        f"carries the site-A macros of {sorted(want)}")
    para = next(p for p in _read_tex().split("\n\n") if r"\roadRiceSiteA" in p)
    for cls, macros in want.items():
        for m in macros:
            assert "\\" + m in para, (
                f"sec. concentration prints site A's pooled IoU on {cls} for one "
                f"cross-validation only; \\{m} is missing")
    b0 = _folds.load_runs("mit_b0")
    tilled = {f: _fold_mean(b0, p, "paddy_after_driving")
              for f, p in _folds.G_FOLD_PROTO.items()}
    assert min(tilled, key=tilled.get) == "site000", (
        "sec. concentration prints 'at least' the site-A value for every "
        f"rice-paddy fold, but the lowest is {min(tilled, key=tilled.get)}: "
        f"{tilled}")


def test_every_encoder_starts_from_the_initialization_the_paper_names():
    """Manuscript sec. discussion: 'the encoders' initialization may,
    because every encoder starts from weights pretrained on the color
    photographs of ImageNet. The RGB input matches
    those photographs in band and in color, the luminance control matches them
    in band alone, and NIR matches them in neither.' The scope
    paragraph rests on that sentence, and it ranges over the models
    config.py runs rather than over the ones the sentence names.

    The initialization is resolved from the libraries src/train.py asks rather
    than from a name in a comment: pretrained_settings for an smp encoder,
    the default pretrained tag for a timm one. sec. methods called the second
    architecture's pretraining self-supervised, which names the first stage
    of the checkpoint timm resolves and not the checkpoint."""
    import segmentation_models_pytorch as smp
    import timm.models._registry as timm_registry
    resolved = {}
    for _arch, backbone in config.MODEL_TAG:
        if backbone.startswith("tu-"):
            tag, cfg = timm_registry._model_default_cfgs[
                backbone[len("tu-"):]].default_with_tag
            labeled = cfg.num_classes == 1000 and "_in1k" in tag
        else:
            tag = sorted(smp.encoders.encoders[backbone]["pretrained_settings"])
            labeled = tag == ["imagenet"]
        resolved[backbone] = tag
        assert labeled, (
            "sec. discussion states every encoder starts from weights "
            "pretrained on visible-band natural images, but src/train.py asks "
            f"imagenet weights for {backbone} and the library resolves {tag}")
    calls_it_self_supervised = "self-supervised" in _read_tex()
    assert not calls_it_self_supervised, (
        "the manuscript calls an encoder's pretraining self-supervised; "
        f"src/train.py asks imagenet weights for {config.BACKBONE_H} and timm "
        f"resolves {resolved[config.BACKBONE_H]!r}, masked-autoencoder "
        "weights fine-tuned on labeled ImageNet-22k and ImageNet-1k")


def test_a_site_level_test_sits_beside_the_criteria_that_pool_a_rotation():
    """Manuscript sec. criteria: 'The site-level test, a one-sample $t$ test on the three held-out ...
    mean margins of one cross-validation, asks whether a margin holds across
    those sites.' Every criterion of folds.md that
    pools a cross-validation's folds carries that test beside it.

    Both sets are derived from results/folds.md, and the fold count a
    between-site spread needs comes from analyze_folds.py rather than from a
    number restated here."""
    with open(os.path.join(RES, "folds.md")) as f:
        blocks = [b for b in re.split(r"\n(?=#)", f.read())
                  if b.startswith("### ")]
    rotation, pools, sited = set(), set(), set()
    for sec in blocks:
        name = sec.split("\n", 1)[0][len("### "):].strip()
        folds = {ln.split("|")[1].strip() for ln in sec.splitlines()
                 if ln.startswith("| ") and not ln.startswith("| fold")}
        if len(folds) > 1:
            rotation.add(name)
        if "Pooled: n=" in sec and _folds.site_level_stats(
                range(len(folds))) is not None:
            pools.add(name)
        if "Site-level (" in sec:
            sited.add(name)
    assert pools == sited, (
        "sec. methods reports a site-level test beside a criterion that pools "
        f"a cross-validation's folds, but results/folds.md pools {sorted(pools)} and "
        f"reports one for {sorted(sited)}")
    assert rotation - sited, (
        "every criterion read across a cross-validation now carries a site-level "
        "test, so sec. methods may state the cross-validation as the condition and "
        "this test no longer distinguishes it from pooling")
    flat = " ".join(_read_tex().split())
    assert ("The site-level test, a one-sample $t$ test on the three held-out "
            "sites' mean margins") in flat, (
        f"results/folds.md reports a site-level test for {sorted(sited)}, and "
        "sec. criteria no longer defines the test")


def test_one_blocked_comparison_favors_the_second_camera_and_the_paper_names_it():
    """Manuscript sec. redundancy: 'Of the other configurations, fusion is
    better than RGB on the tilled paddy surface', and that margin changes sign across the held-out sites:
    'Fusion's margin over RGB on the tilled surface is negative at site A on
    both cross-validations'.

    The set is every family against every target class at site C, read with
    the three conditions, and the comparison the paper names is derived from
    it, so a second comparison arriving in the second camera's favor fails
    rather than sits unmentioned."""
    A = importlib.import_module("analyze")
    runs, px = A.load_runs(), A.gt_px()
    favoring = set()
    for _fam, ax, ay in A.FAMILIES:
        si = A.paired_family(runs, "blocked", ax, ay, "isec", px)
        so = A.paired_family(runs, "blocked", ax, ay, A.own_proto(ax), px)
        p_holm = A.holm({c: v["p"] for c, v in si.items()
                         if v is not None and c in TARGETS})
        for cls in TARGETS:
            if A.verdict(cls, si.get(cls), so.get(cls),
                         p_holm.get(cls)).startswith("DIRECTIONAL (+)"):
                favoring.add((ax, ay, cls))
    assert favoring == {("RGBNIR", "RGB", "paddy_after_driving")}, (
        "sec. results names fusion over RGB on the tilled paddy surface as the "
        "one comparison satisfying the three conditions in the second camera's "
        f"favor under the site-blocked protocol, but the runs give "
        f"{sorted(favoring)}")
    tex = " ".join(_read_tex().split())
    assert "fusion is better than RGB on the tilled paddy surface" in tex, (
        "a comparison favors the second camera under the site-blocked "
        "protocol, and sec. redundancy no longer names it")
    folds = dict(_folds.E_FOLD_PROTO)
    folds.update({"p" + f: p for f, p in _folds.G_FOLD_PROTO.items()})
    for ax, ay, cls in favoring:
        signs = [A.paired_family(runs, p, ax, ay, "isec", px)[cls]["mean"]
                 for p in folds.values()
                 if A.paired_family(runs, p, ax, ay, "isec", px).get(cls)]
        assert min(signs) < 0, (
            f"sec. redundancy states {ax} over {ay} on {cls} reversed sign at "
            f"another site, but every fold keeps it positive: {signs}")


def test_fusion_and_rgb_rank_no_other_target_class_under_site_blocking():
    """Manuscript sec. redundancy: 'Neither is better than the other on the
    other three classes.' The sentence ranges over the target classes of the
    family fusion minus RGB under the site-blocked protocol, read with the
    three conditions. The two classes the paper ranks are derived from the
    verdicts rather than named in the sentence, so a third ranked class fails
    here instead of joining 'the other three' unseen."""
    A = importlib.import_module("analyze")
    runs, px = A.load_runs(), A.gt_px()
    si = A.paired_family(runs, "blocked", "RGBNIR", "RGB", "isec", px)
    so = A.paired_family(runs, "blocked", "RGBNIR", "RGB",
                         A.own_proto("RGBNIR"), px)
    p_holm = A.holm({c: (si[c]["p"] if si.get(c) else None) for c in TARGETS})
    ranked = {cls for cls in TARGETS
              if A.verdict(cls, si.get(cls), so.get(cls),
                           p_holm.get(cls)).startswith("DIRECTIONAL")}
    assert ranked == {"paddy_after_driving", "field_furrow"}, (
        "sec. redundancy ranks fusion and RGB on the tilled paddy surface and "
        "the field furrow only, and calls the other three target classes "
        f"unranked, but the three conditions rank {sorted(ranked)}")


def test_every_test_a_repo_file_names_exists():
    """Macro comments in params.tex, comments across src/, the README and
    this file's own docstrings name the test that binds a number. A renamed
    or deleted test leaves those names pointing at nothing while every
    assertion still passes, so the set of names is derived from the files
    and checked against the tests defined here."""
    defined = set(re.findall(r"^def (test_\w+)", open(os.path.abspath(__file__)).read(),
                             re.M))
    cited = {}
    paths = [os.path.join(ROOT, "paper", "params.tex"),
             os.path.join(ROOT, "README.md"), os.path.abspath(__file__)] + sorted(
        glob.glob(os.path.join(ROOT, "src", "**", "*.py"), recursive=True))
    for path in paths:
        text = open(path).read()
        if path == os.path.abspath(__file__):
            # This file defines test_* identifiers of its own, so only its
            # prose cites a test: docstrings and comments.
            tree = ast.parse(text)
            docs = [ast.get_docstring(n) or "" for n in ast.walk(tree)
                    if isinstance(n, (ast.Module, ast.FunctionDef))]
            notes = [ln.split("#", 1)[1] for ln in text.splitlines()
                     if "#" in ln]
            text = "\n".join(docs + notes)
        for name in re.findall(r"\b(test_[a-z0-9_]+)", text):
            if name != "test_assertions":
                cited.setdefault(name, os.path.relpath(path, ROOT))
    missing = {n: f for n, f in cited.items() if n not in defined}
    assert not missing, (
        "these files name tests that test_assertions.py does not define: "
        f"{missing}")


def _quoted_manuscript_fragments(source):
    """Every single-quoted span of a test docstring, possessives left alone.

    An apostrophe delimits a quotation only where it does not sit between two
    letters, which is what separates a closing quote from the possessive in
    near-infrared's margin."""
    out = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.FunctionDef)
                and node.name.startswith("test_")):
            continue
        doc = ast.get_docstring(node) or ""
        n, i = len(doc), 0
        while i < n:
            opens = ((i == 0 or doc[i - 1] in " (\n")
                     and i + 1 < n and doc[i + 1].isalpha())
            if doc[i] == "'" and opens:
                j = i + 1
                while j < n and not (doc[j] == "'" and doc[j - 1] not in " (\n"
                                     and (j + 1 == n
                                          or doc[j + 1] in " ,.;:)\n")):
                    j += 1
                if j < n:
                    out.append((node.name, node.lineno, doc[i + 1:j]))
                    i = j
            i += 1
    return out


def _tex_body(text):
    """One .tex source with its LaTeX comments removed. Separate from the
    tokenizer because a percent sign inside a quoted sentence is a percent
    sign, and stripping from it would truncate the quote at 95%."""
    return re.sub(r"(?<!\\)%.*", "", text)


def _tex_tokens(text):
    """A stretch of LaTeX as lowercase words, every macro collapsed to a
    wildcard. A docstring quotes the paper as a reader receives it, so
    \\nTargetClasses{} has to match the word five that it renders as."""
    text = text.replace(r"\%", "%")
    # A command whose argument is prose the reader receives keeps that
    # prose: collapsed with the rest, a caption with no macro inside it would
    # vanish into one wildcard, and every quote of it would read as stale.
    text = re.sub(r"\\(?:caption|emph|textbf|textit)\{", " ", text)
    text = re.sub(r"\\[a-zA-Z]+(\{[^{}]*\})*", "\x01", text)
    text = re.sub(r"[{}$~&\\]", " ", text)
    # a macro fused to a following symbol (\ciLevel\% renders as 95%) is
    # still one rendered word, so the whole token is the wildcard
    return ["\x01" if w.startswith("\x01") else w
            for w in (t.strip(".,;:()").lower() for t in text.split()) if w]


def test_every_manuscript_sentence_a_test_quotes_is_still_in_the_manuscript():
    """A docstring quoting the paper is a string literal, so no comparison in
    this file notices when the paper stops saying it. The quote then names a
    sentence that does not exist, and the test underneath it goes on passing.

    The quotes are derived from this file rather than listed, and the document
    from what main.tex inputs, so a table caption counts as manuscript text
    and a test added tomorrow is covered the day it is written."""
    body = _read_tex().split(r"\begin{document}", 1)[1]
    doc = _tex_tokens(_tex_body(body))
    for name in re.findall(r"\\input\{(\w+)\}", _read_tex()):
        if name != "params":
            doc += _tex_tokens(_tex_body(_read_table(name + ".tex")))

    # A macro renders as some word, so it matches any quoted word; a window
    # matched mostly by macros matches every quote of its length, so a run of
    # preamble macros would pass quotes the paper does not hold.
    def present(words):
        for s in range(len(doc) - len(words) + 1):
            window = doc[s:s + len(words)]
            if (all(h == "\x01" or h == w for h, w in zip(window, words))
                    and 2 * sum(h == w for h, w in zip(window, words))
                    >= len(words)):
                return True
        return False

    stale = []
    for test_name, lineno, quote in _quoted_manuscript_fragments(
            open(os.path.abspath(__file__)).read()):
        parts = [_tex_tokens(p) for p in quote.split("...")]
        if sum(len(p) for p in parts) < 6:
            continue
        if not all(present(p) for p in parts):
            stale.append(f"{test_name} (line {lineno}) quotes "
                         f'"{" ".join(quote.split())}"')
    assert not stale, (
        "these test docstrings quote manuscript prose the paper no longer "
        "contains, so each names a sentence a reader cannot find:\n  "
        + "\n  ".join(stale))


def test_every_accuracy_the_paper_prints_carries_its_spread_over_all_seeds():
    """Manuscript sec. criteria: 'Every IoU is a mean over five seeds, and a
    ... beside it gives its standard deviation over the seeds.'

    The quantifier is the claim. A win printed without the loser's spread is
    unfalsifiable, so the sentence has to hold over every cell of every table,
    and a comparison of build_tables.py's cells against its own output agrees
    with itself whether or not a spread is in it.

    Read off the typeset tables instead, and the seed count with it. A table
    regenerated over three seeds prints narrower spreads that no macro and no
    cell comparison would catch, while the sentence goes on saying five."""
    printed, bare = [], []
    for name in re.findall(r"\\input\{(\w+)\}", _read_tex()):
        if name == "params":
            continue
        cells = re.findall(r"\$([^$]+)\$", _read_table(name + ".tex"))
        spread = [re.fullmatch(r"(\d\.\d+)_\{\\pm(\d\.\d+)\}", c)
                  for c in cells]
        if not any(spread):
            continue          # a table of signed differences or of probes
        printed += [(name,) + m.groups() for m in spread if m]
        bare += [(name, c) for c in cells if re.fullmatch(r"\d\.\d+", c)]
    assert not bare, (
        "sec. methods reports a standard deviation for every configuration "
        "and not only for the one a comparison favors, but these tables "
        f"print an accuracy beside spreads without one: {bare[:6]}")

    runs = _folds.A.load_runs()
    groups = defaultdict(list)
    for m in runs:
        key = (m["arm"], m["protocol"], m["train_env"])
        for cls, v in m["iou"]["isec"].items():
            if v is not None:
                groups[key + (cls,)].append(v)
    admissible = {(f"{np.mean(v):.3f}", f"{np.std(v):.3f}")
                  for v in groups.values() if len(v) == len(config.SEEDS)}

    orphan = [p for p in printed if (p[1], p[2]) not in admissible]
    assert printed and not orphan, (
        f"the paper prints {len(printed)} accuracies, and {len(orphan)} of "
        f"them match no cell's mean and standard deviation over all "
        f"{len(config.SEEDS)} committed seeds: {orphan[:6]}")


def test_every_site_level_test_the_paper_reports_runs_on_all_three_folds():
    """Manuscript sec. criteria: the site-level test is 'a one-sample $t$ test
    on the three held-out ... mean margins'. The claim ranges over the
    site-level p values the paper prints, so both ends are derived: the
    printed set from the parameter file, and the reproducible set from the
    fold maps, over every comparison of a cross-validation's folds the
    analyzers can form. A site-level p that only a two-fold map reproduces
    fails."""
    printed = set()
    for m in re.finditer(r"\\newcommand\{\\[A-Za-z]+\}\{([^}]*)\}\s*%([^\n]*)",
                         _read_params()):
        value, comment = m.group(1), m.group(2)
        # A p value is what the site-level test returns; the critical value
        # its comment also names is a t statistic, which exceeds 1.
        if "site-level p" in comment and re.fullmatch(r"[01]\.\d+", value):
            printed.add(value)
    assert printed, ("sec. methods states what every reported site-level test "
                     "runs on, but params.tex prints no site-level p")
    px = _folds.A.gt_px()
    runs = _folds.load_runs("mit_b0")
    by_folds = defaultdict(set)
    for proto_map in (_folds.E_FOLD_PROTO, _folds.G_FOLD_PROTO):
        for _fam, ax, ay in _folds.A.FAMILIES:
            # paired_family returns every class of one fold, so it is called
            # once per fold and read per class.
            folds_ = [_folds.A.paired_family(runs, proto, ax, ay, "isec", px)
                      for proto in proto_map.values()]
            for cls in TARGETS + config.CONTROLS:
                per = [f[cls]["mean"] for f in folds_
                       if f.get(cls) is not None]
                stats_ = _folds.site_level_stats(per) if per else None
                if stats_ is not None:
                    by_folds[len(per)].add(f"{stats_['p']:.2f}")
    n_sites = len(_folds.E_FOLD_PROTO)
    assert _folds.site_level_stats([0.1, 0.2]) is None, (
        "sec. methods states every reported site-level test runs on all "
        f"{n_sites} folds, and analyze_folds.py must be what makes that so "
        "by returning nothing below three fold means")
    short = sorted(printed - by_folds[n_sites])
    assert not short, (
        f"sec. methods states every reported site-level test runs on all "
        f"{n_sites} folds, but these printed values are reproduced by no "
        f"{n_sites}-fold map: {short}")


# ------------- the band or the annotation (annotation_diagnosis.py) ---------
# NIR is the only configuration that trains on the NIR annotation, so sec.
# split reads how the two annotations mark the paddy edge, scores NIR's lead
# where they agree and on the RGB annotation, and decomposes the lead with the
# diagnostic network. Each value is recomputed here from the committed
# agreement file, the index and the runs' per-frame counts.

def _macro_value(name):
    m = re.search(r"\\newcommand\{\\" + name + r"\}\{(?:\\ensuremath\{)?([^}\\]+)",
                  _read_params())
    assert m, f"params.tex defines no \\{name}"
    return m.group(1).strip()


def _macro_ci(name):
    m = re.search(r"\\newcommand\{\\" + name
                  + r"\}\{\\ensuremath\{\[([^,]+),\\,([^\]]+)\]\}\}",
                  _read_params())
    assert m, f"params.tex defines no interval \\{name}"
    return m.group(1), m.group(2)


def test_the_two_annotations_mark_the_paddy_edge_alike_overall():
    """Manuscript sec. split: 'Over all sampled frames, the NIR annotation
    marks the class in 1.06 times as many frames as the RGB annotation, with
    0.96 times the area per frame, so the two annotations differ in where they mark the edge rather than in
    how much of it they mark.' "About" is read as within a tenth of one, for each ratio
    and for their product, the ratio of the marked areas."""
    recs = _index()
    c = config.CLASSES.index("paddy_edge")
    r = np.array([x["px_rgb"][c] for x in recs])
    n = np.array([x["px_nir"][c] for x in recs])
    count = (n > 0).sum() / (r > 0).sum()
    area = n[n > 0].mean() / r[r > 0].mean()
    for v in (count, area, count * area):
        assert abs(v - 1) < 0.1, (
            "sec. split says the two annotations mark about as much of the "
            f"paddy edge, but the ratios are {count:.3f}, {area:.3f} and "
            f"{count * area:.3f}")
    for macro, v in (("annCountRatio", count), ("annAreaRatio", area)):
        assert _macro_value(macro) == f"{v:.2f}", (
            f"sec. split prints \\{macro} as {_macro_value(macro)}, and the "
            f"committed annotations give {v:.4f}")


def test_the_two_annotations_agree_less_on_the_validation_frames():
    """Manuscript sec. split: 'The IoU between the two annotations, the pixels
    that both mark divided by the pixels that either marks, is 0.18 on the
    validation frames, against 0.54 on the training frames'."""
    D = importlib.import_module("phases.annotation_diagnosis")
    z = np.load(D.AGREE_NPZ)
    side = {(x["env"], x["seq"], x["frame"]): x["osplit"] for x in _index()}
    sides = np.array([side[(str(e), str(q), int(f))]
                      for e, q, f in zip(z["env"], z["seq"], z["frame"])])
    c = config.CLASSES.index("paddy_edge")
    share = {sp: z["inter_raw"][sides == sp, c].sum()
             / z["union_raw"][sides == sp, c].sum() for sp in ("train", "val")}
    assert (_macro_value("annAgreeEdgeVal") == f"{share['val']:.2f}"
            and _macro_value("annAgreeEdgeTrain") == f"{share['train']:.2f}"
            and share["val"] < share["train"]), (
        "sec. split prints the paddy edge's shared pixels as "
        f"{_macro_value('annAgreeEdgeVal')} on the validation frames and "
        f"{_macro_value('annAgreeEdgeTrain')} on the training frames, and the "
        f"committed annotations give {share}")


def _diag_check(macro, ci_macro, units_d, sign):
    """A margin of sec. split against its recomputation; sign is +1 for an
    interval above zero and 0 for one spanning zero."""
    A = importlib.import_module("analyze")
    units, d = units_d
    lo, hi = A.cluster_ci(d, units)
    assert _macro_value(macro) == f"{d.mean():+.3f}" and _macro_ci(ci_macro) \
        == (f"{lo:+.3f}", f"{hi:+.3f}"), (
        f"\\{macro} and \\{ci_macro} print {_macro_value(macro)} "
        f"{_macro_ci(ci_macro)}, and the runs give {d.mean():+.4f} "
        f"[{lo:+.4f}, {hi:+.4f}]")
    assert (lo > 0) if sign > 0 else (lo < 0 < hi), (
        f"sec. split reads \\{macro}'s interval as "
        f"{'above zero' if sign > 0 else 'spanning zero'}, but it is "
        f"[{lo:+.4f}, {hi:+.4f}]")


def test_the_lead_holds_where_the_two_annotations_agree():
    """Manuscript sec. split: 'NIR's lead does not rest on the validation
    frames whose annotations agree least after the shift. On the half of the
    paddy-edge validation frames whose two annotations have the highest IoU
    after the shift, the margin of NIR over RGB and its interval lie above
    zero' tab:margins prints both halves."""
    pairs = _diag_pairs()
    _diag_check("annUpperMargin", "annUpperMarginCI",
                pairs["annUpperMarginCI"], +1)
    units, d = pairs["annLowerMarginCI"]
    lo, hi = importlib.import_module("analyze").cluster_ci(d, units)
    assert _macro_value("annLowerMargin") == f"{d.mean():+.3f}" and \
        _macro_ci("annLowerMarginCI") == (f"{lo:+.3f}", f"{hi:+.3f}"), (
        "tab:margins prints the lower half's margin differently from the runs")


def test_the_lead_is_not_detected_on_the_rgb_annotation():
    """Manuscript sec. split: 'Scored on the RGB annotation alone, which RGB trains on, the comparison
    of NIR with RGB, however, detects no difference'."""
    _diag_check("nirRgbEdgeOnRgb", "nirRgbEdgeOnRgbCI",
                _diag_pairs()["nirRgbEdgeOnRgbCI"], 0)


def test_no_configuration_learns_the_paddy_edge_at_a_held_out_site():
    """Manuscript sec. redundancy: 'On the paddy edge, each input
    configuration's pooled IoU at site C on the leave-one-site-out
    cross-validation lies between 0.011 and 0.059'. The range is taken
    over the four configurations' seed-mean pooled IoUs at site C, the cells
    of tab:perclass."""
    runs = _study_runs()
    v = [_fs_iou(runs, a, "blocked", "isec", "paddy_edge")
         for a in ("RGB", "LUM", "NIR", "RGBNIR")]
    assert (_macro_value("edgeSiteCLow") == f"{min(v):.3f}"
            and _macro_value("edgeSiteCHigh") == f"{max(v):.3f}"), (
        f"sec. redundancy prints the range {_macro_value('edgeSiteCLow')} to "
        f"{_macro_value('edgeSiteCHigh')}, and the runs give {v}")
    # 'On the rice-paddy cross-validation, NIR scores 0.064 at site C and RGB
    # 0.138, although the per-frame comparison detects no difference between
    # the two.'
    proto = _folds.G_FOLD_PROTO["site002"]
    rice = {a: _fs_iou(runs, a, proto, "isec", "paddy_edge") for a in ("NIR", "RGB")}
    assert (_macro_value("edgeSiteCRiceNir") == f"{rice['NIR']:.3f}"
            and _macro_value("edgeSiteCRiceRgb") == f"{rice['RGB']:.3f}"), (
        f"sec. redundancy prints NIR {_macro_value('edgeSiteCRiceNir')} and RGB "
        f"{_macro_value('edgeSiteCRiceRgb')} at site C on the rice-paddy "
        f"cross-validation, and the runs give {rice}")
    A = importlib.import_module("analyze")
    px = A.gt_px()
    allruns = A.load_runs()
    fam = A.paired_family(allruns, proto, "NIR", "RGB", "isec", px)
    own = A.paired_family(allruns, proto, "NIR", "RGB", A.own_proto("NIR"), px)
    p_holm = A.holm({c: (fam[c]["p"] if fam.get(c) else None) for c in TARGETS})
    v_edge = A.verdict("paddy_edge", fam.get("paddy_edge"),
                       own.get("paddy_edge"), p_holm.get("paddy_edge"))
    assert not v_edge.startswith("DIRECTIONAL"), (
        "sec. redundancy says the per-frame comparison of NIR with RGB on the "
        "paddy edge at site C on the rice-paddy cross-validation detects no "
        f"difference, but the three conditions return {v_edge}")


def test_the_registration_bound_leaves_the_upper_half_margin_above_zero():
    """Manuscript sec. split: 'Part of that contrast is the camera offset,
    which is larger on the validation frames. We shift the NIR annotation by the offset of up to 16 px that best
    aligns it with the RGB annotation over every class but the background.
    The shift raises the IoU from 0.18 to 0.28 on the validation frames and
    from 0.54 to 0.71 on the training frames, but it reaches its bound on 45% of the validation frames that mark
    the paddy edge, against 12% of such training frames. With a bound of 48
    px, the IoU is 0.39 and 0.75, although 31% and 5% of those frames still
    reach that bound.' and 'With bounds of 16 px, 32 px and 48 px, that margin lies
    between +0.184 and +0.188, its interval above zero at each.' Recomputed
    from the committed sensitivity file and the runs' per-frame counts, with
    S1's committed readings (src/phases/annotation_diagnosis.py)."""
    A = importlib.import_module("analyze")
    D = importlib.import_module("phases.annotation_diagnosis")
    recs = _index()
    zs = np.load(D.SENS_NPZ)
    z = np.load(D.AGREE_NPZ)
    assert (np.array_equal(zs["shift_16"], z["shift"])
            and np.array_equal(zs["inter_edge_16"], z["inter_reg"][:, D.EDGE])
            and np.array_equal(zs["union_edge_16"], z["union_reg"][:, D.EDGE])), (
        "the sensitivity file's 16 px registration differs from the committed "
        "agreement file, so the two bounds are not one search widened")
    row = {(str(e), str(q), int(f)): i
           for i, (e, q, f) in enumerate(zip(zs["env"], zs["seq"], zs["frame"]))}
    side = np.array([x["osplit"] for x in recs])
    idx = np.array([row[(x["env"], x["seq"], x["frame"])] for x in recs])
    both = np.array([x["px_rgb"][D.EDGE] > 0 and x["px_nir"][D.EDGE] > 0
                     for x in recs])
    px = A.gt_px()
    runs = [m for m in A.load_runs() if m["protocol"] == "official"]

    def fi(arm):
        return A.frame_iou_by_arm([m for m in runs if m["arm"] == arm], "isec",
                                  D.EDGE, A.inc_own("isec"), px)

    nir, rgb = fi("NIR"), fi("RGB")
    got, uppers = {}, []
    for b in D.REG_BOUNDS:
        sat = (np.abs(zs[f"shift_{b}"][idx]) >= b).any(1)
        inter, union = zs[f"inter_edge_{b}"][idx], zs[f"union_edge_{b}"][idx]
        got[b] = {n: (float(sat[(side == n) & both].mean()),
                      float(inter[side == n].sum() / union[side == n].sum()))
                  for n in ("train", "val")}
        agree = {k: zs[f"inter_edge_{b}"][row[k]] / zs[f"union_edge_{b}"][row[k]]
                 for k in nir}
        med = float(np.median(list(agree.values())))
        ks = sorted(k for k, a in agree.items() if a >= med and k in rgb)
        d = np.array([nir[k] - rgb[k] for k in ks])
        lo, hi = A.cluster_ci(d, A.seq_units(ks))
        assert lo > 0, (
            f"sec. split says the upper-half margin's interval lies above zero "
            f"with a {b} px bound, but it is [{lo:+.4f}, {hi:+.4f}]")
        uppers.append(float(d.mean()))
    lo16, wide = got[D.SHIFT_PX], got[max(D.REG_BOUNDS)]
    assert all(lo16[n][1] > float(_macro_value(f"annAgreeEdge{n.capitalize()}"))
               for n in ("train", "val")), (
        "sec. split says the shift raises the IoU of both sets, but the "
        f"registered values are {lo16}")
    assert lo16["val"][0] > lo16["train"][0] + D.SAT_TOL, (
        "sec. split says the camera offset is larger on the validation frames, "
        f"but the saturation shares are {lo16}")
    printed = {
        "regAgreeVal": f"{lo16['val'][1]:.2f}",
        "regAgreeTrain": f"{lo16['train'][1]:.2f}",
        "regSatValPct": f"{100 * lo16['val'][0]:.0f}",
        "regSatTrainPct": f"{100 * lo16['train'][0]:.0f}",
        "regAgreeValWide": f"{wide['val'][1]:.2f}",
        "regAgreeTrainWide": f"{wide['train'][1]:.2f}",
        "regSatValWidePct": f"{100 * wide['val'][0]:.0f}",
        "regSatTrainWidePct": f"{100 * wide['train'][0]:.0f}",
        "regUpperLow": f"{min(uppers):+.3f}",
        "regUpperHigh": f"{max(uppers):+.3f}",
        "regBoundPx": f"{D.SHIFT_PX}",
        "regBoundMidPx": f"{sorted(D.REG_BOUNDS)[1]}",
        "regBoundWidePx": f"{max(D.REG_BOUNDS)}",
    }
    wrong = {m: (_macro_value(m), v) for m, v in printed.items()
             if _macro_value(m) != v}
    assert not wrong, f"sec. split prints these values, and S1 gives: {wrong}"
    assert len(D.REG_BOUNDS) == 3 and D.SHIFT_PX == min(D.REG_BOUNDS), (
        "sec. split names three bounds, the smallest the one D1 and D2 use")


def _d5_reading():
    """The committed reading of D5 (src/phases/annotation_diagnosis.py),
    recomputed here: 'band', 'annotation' or 'both or neither'."""
    D = importlib.import_module("phases.annotation_diagnosis")
    A = importlib.import_module("analyze")
    runs = _study_runs()
    pool = {a: _fs_iou(runs, a, "official", "isec", "paddy_edge")
            for a in ("RGB", "NIR", config.DIAG_ARM)}
    pairs = _diag_pairs()
    band = A.cluster_ci(pairs["diagRgbEdgeCI"][1], pairs["diagRgbEdgeCI"][0])
    ann = A.cluster_ci(pairs["nirDiagEdgeCI"][1], pairs["nirDiagEdgeCI"][0])
    point = pool["RGB"] + D.HALFWAY * (pool["NIR"] - pool["RGB"])
    hi_side = pool[config.DIAG_ARM] >= point
    if band[0] > 0 and hi_side:
        return "band"
    if ann[0] > 0 and not hi_side:
        return "annotation"
    return "both or neither"


def test_the_band_or_the_annotation():
    """Manuscript sec. split: 'The diagnostic configuration, which receives
    the NIR image and trains on the RGB annotation, scores ... on the paddy
    edge'. The reading that the paper states is the one that
    src/phases/annotation_diagnosis.py D5 commits, recomputed here, and each
    printed margin of the decomposition is recomputed from the runs."""
    runs = _study_runs()
    A = importlib.import_module("analyze")
    pairs = _diag_pairs()
    reading = _d5_reading()
    flat = " ".join(_read_tex().split())
    stated = {"band": "so the image carries most of NIR's lead",
              "both or neither": "The comparisons therefore credit neither the "
                                 "image nor the annotation alone with NIR's "
                                 "lead"}
    assert reading in stated and stated[reading] in flat, (
        f"the committed D5 reading recomputes as '{reading}', and sec. split "
        "does not state that reading")
    for macro, arm in (("diagEdgeSplit", config.DIAG_ARM),
                       ("fusionEdgeSplit", "RGBNIR")):
        v = [m["iou"]["isec"]["paddy_edge"] for m in runs
             if m["arm"] == arm and m["protocol"] == "official"]
        assert len(v) == len(config.SEEDS), (
            f"{arm} has {len(v)} runs under the dataset's split, not one per seed")
        want = f"{np.mean(v):.3f} \\pm {np.std(v):.3f}"
        got = re.search(rf"\\newcommand\{{\\{macro}\}}\{{\\ensuremath\{{([^}}]+)\}}",
                        _read_params())
        assert got and got.group(1) == want, (
            f"\\{macro} prints {got.group(1) if got else 'nothing'}, and the "
            f"runs give {want}")
    band_sign = +1 if reading == "band" else 0
    for macro, key, sign in (("diagRgbEdge", "diagRgbEdgeCI", band_sign),
                             ("nirDiagEdge", "nirDiagEdgeCI", +1),
                             ("fusionRgbEdgeSplit", "fusionRgbEdgeSplitCI", +1)):
        _diag_check(macro, key, pairs[key], sign)
