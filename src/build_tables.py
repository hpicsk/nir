"""Emit generated LaTeX table bodies for the manuscript from committed results.

Same one-source-of-truth contract as build_parameters.py: every
number in a manuscript table is recomputed HERE from the committed run
artifacts through the SAME analyzer functions the gates and test_assertions.py
use (analyze.py's per-seed IoU aggregation, analyze_folds.py's protocol
pairing), never by
parsing a rendered .md. The cells therefore cannot drift from summary.md or
folds.md: all three call the same two modules.

Tables:

  tab:perclass       body. Site-blocked IoU beside the LEAKAGE INFLATION,
                     for the classes that carry a claim.
  tab:robustness     body. Both cross-model contrasts on every input
                     configuration, so the body can state the two the criteria
                     are declared on and leave the rest here.
  tab:config         body. Every setting Methods merely fixes.

The inflation column is NOT the difference of the two IoU blocks. A
site-blocked IoU is scored on the held-out site and a sequence-level-split
IoU on the sequence-level split's own validation partition, so subtracting them
crosses two frame sets and mixes site difficulty into the leak (sec:methods
defines the estimator; the two populations are why sec:results reads the
ranking off the paired statistic instead). The inflation is analyze_folds.py
`paired_protocols`: a per-frame paired difference over the frames BOTH
protocols scored, which is the same estimator behind every inflation figure the
body prints.

The IoU cells are RNG-free (means and standard deviations over the 5
committed seeds, exactly analyze.py `mean_sd`). The inflation cells are means of
paired per-frame deltas and carry no interval: the one cell the inflation
criterion is declared on takes its interval in tab:robustness's first row, and
the prose macros of build_parameters.py own the determinism contract.

Captions are titles and the notes beneath the tables define the cells; a
verdict read off the cells is the body's.

Usage: python src/build_tables.py   (reads results/; writes paper/*.tex)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C
import importlib
A = importlib.import_module("analyze")   # per-seed IoU aggregation (mit_b0)
folds = importlib.import_module("analyze_folds")   # the gate implementations

OUT = os.path.join(C.ROOT, "paper", "table_perclass_iou.tex")
OUT_ROBUST = os.path.join(C.ROOT, "paper", "table_robustness.tex")
# Every path this module publishes. write_all() writes exactly these, and the
# freshness check reads them against what main.tex includes, so a table added
# here is regenerated and checked without either side being edited.
OUT_CONFIG = os.path.join(C.ROOT, "paper", "table_config.tex")
OUTPUTS = (OUT, OUT_ROBUST, OUT_CONFIG)

# The interval level every caption names, from the analyzer's tail, so a
# change of the tail cannot leave a caption naming the old level.
CI_LEVEL = f"{100 * (1 - 2 * A.CI_TAIL):g}"


# display order: the 5 target classes, then the control the protocols are
# required to segment. Names are the manuscript's, not the repo's. These are
# the classes a claim is made on. In-field trees have no site-blocked cell to
# compare (9 val frames, config.py FLOOR_EXEMPT), and standing rice is neither
# target nor control, so a headline table that ranked them would rank rows the
# paper does not use.
BODY_ROWS = [
    ("paddy_edge", "Paddy edge"),
    ("paddy_before_driving", "Paddy, untilled"),
    ("paddy_after_driving", "Paddy, tilled"),
    ("field_furrow", "Field furrow"),
    ("field_levee", "Field ridge"),
    None,                                   # midrule: targets above, control below
    ("common_road", "Farm access road (control)"),
]
COLS = [("RGB", "RGB"), ("LUM", "Lum."), ("NIR", "NIR"), ("RGBNIR", "Fusion")]


def cell(runs, arm, protocol, cls):
    """Mean$\\pm$sd intersection IoU over the 5 seeds, exactly analyze.py mean_sd.
    common_tree is not evaluable under the site-blocked split (9 val frames,
    config.py FLOOR_EXEMPT), so that one cell is a dash, not a false 0."""
    if cls == "common_tree" and protocol == "blocked":
        return r"\textemdash"
    vals = [m["iou"]["isec"].get(cls) for m in runs
            if m["arm"] == arm and m["protocol"] == protocol
            and m["train_env"] == "all"]
    vals = [v for v in vals if v is not None]
    if not vals:
        return r"\textemdash"
    return f"${np.mean(vals):.3f}_{{\\pm{np.std(vals):.3f}}}$"


# ---------------------------------------------------------------------------
# leakage inflation
# ---------------------------------------------------------------------------
def inflation(px):
    """Provided-split MINUS site-blocked, per arm, over the frames both
    protocols scored. analyze_folds.py's estimator, called rather than re-derived, so
    this table and the E1/F1/G2/H1 gates cannot report different inflations.

    Returns arm -> class -> stats (or None where the shared frame set is too
    small to score, which is every class on the site-blocked in-field trees)."""
    runs = folds.load_runs("mit_b0")
    return {arm: folds.paired_protocols(runs, arm, "official", "blocked",
                                        "isec", px) for arm, _ in COLS}


def infl_cell(infl, arm, cls):
    s = infl[arm].get(cls)
    return r"\textemdash" if s is None else f"${s['mean']:+.3f}$"


def write_body(runs, infl, px):
    ncol = len(COLS)
    lines = [
        r"% Generated by src/build_tables.py from committed results. DO NOT HAND-EDIT.",
        r"% Regenerate:  python src/build_tables.py",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Per-class IoU under the site-blocked protocol, and each "
        r"configuration's leakage inflation}",
        r"\label{tab:perclass}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "c" * (2 * ncol) + "}",
        r"\toprule",
        r" & \multicolumn{" + str(ncol) + r"}{c}{Site-blocked IoU}"
        r" & \multicolumn{" + str(ncol) + r"}{c}{Leakage inflation} \\",
        rf"\cmidrule(lr){{2-{1 + ncol}}} "
        rf"\cmidrule(lr){{{2 + ncol}-{1 + 2 * ncol}}}",
        r"Class & " + " & ".join(name for _ in range(2) for _, name in COLS)
        + r" \\",
        r"\midrule",
    ]
    for row in BODY_ROWS:
        if row is None:
            lines.append(r"\midrule")
            continue
        cls, name = row
        lines.append(name + " & " + " & ".join(
            [cell(runs, arm, "blocked", cls) for arm, _ in COLS]
            + [infl_cell(infl, arm, cls) for arm, _ in COLS]) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"\vspace{2pt}",
        r"\begin{minipage}{\linewidth}\footnotesize NIR, near-infrared; Lum., "
        r"luminance control. An IoU cell is the "
        r"pooled IoU on the intersection ground truth over every sampled frame "
        r"of site C, as a mean over \nSeeds{} seeds with its standard "
        r"deviation. An inflation cell is "
        r"the mean per-frame paired difference between the "
        r"sequence-level-split and site-blocked networks over the frames that "
        r"both scored and whose two annotations both mark the class."
        r"\end{minipage}",
        r"\end{table*}",
    ]
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT)


# ---------------------------------------------------------------------------
# encoder size and architecture family
# ---------------------------------------------------------------------------

# Row labels, keyed by the model registry of config.py so a model added
# to the run matrix either reaches this table or raises here. The labels are
# the manuscript's own macros rather than repo backbone strings, so the table
# cannot name a model differently from the body that cites it.
MODEL_LABEL = {
    ("segformer", "mit_b0"): r"\backboneSmall{} (primary)",
    ("segformer", "mit_b2"): r"\backboneLarge{}",
    (C.ARCH_H, C.BACKBONE_H): r"\archSecond{}",
}
# Panel (b): the one IoU margin the protocol asked to reproduce across
# models (config.py F2 and H2), as (arm_x, arm_y, class).
MARGIN = ("NIR", "LUM", "paddy_after_driving")
MARGIN_ROW = "the tilled paddy surface"
# Panel (a)'s contrasts: every other arm, because the third criterion compares
# near-infrared's inflation with the inflation of every other configuration.
CONTRAST_ARMS = ("RGB", "LUM", "RGBNIR")
# Panel (a)'s last row: the primary model at the second site the paddy
# edge is evaluable at (config.py PLOSO_G2B_FOLDS).
SITE_ROWS = [("site000", r"Paddy-restricted, site A"),
             ("site002", r"Paddy-restricted, site C")]


def robustness(px):
    """Both cross-model contrasts per model, through analyze_folds.py's estimators.

    Panel (a) is the same paired inflation as the body table, recomputed on
    each model over the frames both protocols scored. Panel (b) is the
    site-blocked MARGIN, pooled per-frame with analyze.py's sequence-clustered
    interval, which is the quantity F2 and H2 threshold. Calling the gate implementations
    rather than re-deriving them is what stops this table and the gates
    reporting different numbers for one contrast."""
    def inflations(runs, leaked, held):
        infl = {arm: folds.paired_protocols(runs, arm, leaked, held, "isec",
                                            px)["paddy_edge"]
                for arm, _ in COLS}
        # near-infrared's excess over each arm the argument ranks it against,
        # paired per frame, so the ordering is tested and not read off two
        # point estimates
        for arm in CONTRAST_ARMS:
            infl["NIR-" + arm] = folds.paired_arm_inflation(
                runs, "NIR", arm, leaked, held, px)
        return infl

    rows_a, rows_b = [], []
    for arch, backbone in C.MODEL_TAG:
        runs = folds.load_runs(backbone, arch)
        rows_a.append((MODEL_LABEL[(arch, backbone)],
                       inflations(runs, "official", "blocked")))
        keys, d = folds._pooled_deltas(runs, "blocked", MARGIN[0], MARGIN[1],
                                       MARGIN[2], px)
        lo, hi = A.cluster_ci(d, A.seq_units(keys))
        rows_b.append((MODEL_LABEL[(arch, backbone)],
                       {"mean": float(d.mean()), "lo": lo, "hi": hi,
                        "k": len(set(A.seq_units(keys)))}))
    # The same inflation at the other site the paddy edge is evaluable
    # at, on the paddy-restricted cross-validation against its matched comparator:
    # the cross-validation whose control holds on every fold, which the
    # leave-one-site-out cross-validation's fold there does not.
    b0 = folds.load_runs("mit_b0")
    for site, label in SITE_ROWS:
        rows_a.append((label, inflations(b0, "pofficial",
                                         folds.G_FOLD_PROTO[site])))
    return rows_a, rows_b


def write_robust(rob):
    """The model table. Two panels rather than one wide block: the
    inflation is read across configurations and the margin is one contrast,
    so they share no column but the model."""
    rows_a, rows_b = rob
    models, sites = rows_a[:-len(SITE_ROWS)], rows_a[-len(SITE_ROWS):]
    shared = {infl[folds.GATE_ARM]["n"] for _, infl in models}
    assert len(shared) == 1, (
        "the models are compared on different frame sets "
        f"({sorted(shared)}), so one frame count in the note would be wrong")
    k_models = {infl[folds.GATE_ARM]["k"] for _, infl in models}
    assert len(k_models) == 1, (
        f"the model rows resample different sequence counts {k_models}, so one "
        "count in the note would be wrong")
    k_models = k_models.pop()
    k_sites = [infl[folds.GATE_ARM]["k"] for _, infl in sites]
    k_b = {m["k"] for _, m in rows_b}
    assert len(k_b) == 1, (
        f"the panel (b) rows resample different sequence counts {k_b}, so one "
        "count in the note would be wrong")
    k_panel_b = k_b.pop()
    # The note names every interval t-based, which holds only while each rests
    # on fewer sequences than analyze.py bootstraps over.
    k_all = [k_models, *k_sites, k_panel_b]
    assert max(k_all) < A.MIN_BOOT_UNITS, (
        f"the note calls every interval t-based, but rows rest on {k_all} "
        f"sequences and analyze.py bootstraps over {A.MIN_BOOT_UNITS} or more")
    n_sites = [infl[folds.GATE_ARM]["n"] for _, infl in sites]
    assert len(CONTRAST_ARMS) == 3, (
        "the note names three contrast columns, but the table now has "
        f"{len(CONTRAST_ARMS)}")
    # A contrast column prints no mean because the note reads it off the cells
    # to its left, which holds only where the paired difference and the two
    # inflations are taken over the same frames.
    for lbl, infl in rows_a:
        for a in CONTRAST_ARMS:
            assert (infl["NIR-" + a]["n"] == infl[folds.GATE_ARM]["n"]
                    == infl[a]["n"]), (
                f"{lbl}: NIR - {a} is paired over {infl['NIR-' + a]['n']} "
                f"frames, the two inflations over {infl[folds.GATE_ARM]['n']} "
                f"and {infl[a]['n']}, so its mean is not the difference of "
                "the two cells")
    lines = [
        r"% Generated by src/build_tables.py from committed results. DO NOT HAND-EDIT.",
        r"% Regenerate:  python src/build_tables.py",
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Paddy-edge leakage inflation on each model and on the "
        r"paddy-restricted cross-validation, and the margin NIR $-$ luminance control "
        r"on " + MARGIN_ROW + r"}",
        r"\label{tab:robustness}",
        r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
        r"\emph{(a) Paddy-edge leakage inflation}\par\vspace{2pt}",
        r"\begin{tabular}{l" + "c" * (len(COLS) + 1 + len(CONTRAST_ARMS))
        + "}", r"\toprule",
        " & " * len(COLS) + r" & \multicolumn{"
        + str(1 + len(CONTRAST_ARMS)) + r"}{c}{" + CI_LEVEL + r"\% CI} \\",
        r"\cmidrule(l){" + str(len(COLS) + 2) + "-"
        + str(len(COLS) + 2 + len(CONTRAST_ARMS)) + "}",
        "Model & " + " & ".join(n for _, n in COLS) + r" & NIR & "
        + " & ".join(f"NIR $-$ {dict(COLS)[a]}" for a in CONTRAST_ARMS)
        + r" \\",
        r"\midrule",
    ]
    for k, (lbl, infl) in enumerate(rows_a):
        if k == len(models):
            lines.append(r"\midrule")
        lines.append(lbl + " & " + " & ".join(
            f"${infl[arm]['mean']:+.3f}$" for arm, _ in COLS)
            + f" & ${_ci(infl[folds.GATE_ARM])}$ & " + " & ".join(
                f"${_ci(infl['NIR-' + a])}$" for a in CONTRAST_ARMS)
            + " \\\\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\par\vspace{6pt}",
        # Panel (b) beside the note rather than above it: the two share no
        # column, and stacking them costs the page a band of whitespace.
        r"\begin{minipage}[t]{0.42\linewidth}\centering",
        r"\emph{(b) NIR $-$ luminance control on " + MARGIN_ROW
        + r"}\par\vspace{2pt}",
        r"\begin{tabular}[t]{lcc}", r"\toprule",
        r"Model & NIR $-$ Lum. & " + CI_LEVEL + r"\% CI \\", r"\midrule",
    ]
    for lbl, m in rows_b:
        lines.append(f"{lbl} & ${m['mean']:+.3f}$ & ${_ci(m)}$ \\\\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\end{minipage}\hfill",
        r"\begin{minipage}[t]{0.56\linewidth}\footnotesize (a) An inflation is the "
        r"mean per-frame paired difference between the sequence-level-split "
        r"and site-blocked networks over the \num{" + str(shared.pop()) + r"} "
        r"frames that both scored and whose two annotations both mark "
        r"the paddy edge. The last two rows are the primary model on the "
        r"paddy-restricted cross-validation against its matched comparator, at "
        r"sites A and C, over the \num{" + str(n_sites[0]) + r"} and \num{"
        + str(n_sites[1]) + r"} such frames. The last three columns pair "
        r"NIR's inflation with each other configuration's on the same frames, "
        r"so each pair's mean is the difference of two inflation cells. "
        r"(b) The mean per-frame IoU difference under the site-blocked "
        r"protocol. Intervals are "
        r"$t$-based " + CI_LEVEL + r"\% intervals over the "
        r"\num{" + str(k_models) + r"} recording sequences behind each model row "
        r"of (a), over the \num{" + str(k_sites[0]) + r"} and the \num{"
        + str(k_sites[1]) + r"} behind its last two rows, and over the \num{"
        + str(k_panel_b) + r"} behind (b)."
        + r"\end{minipage}",
        r"\end{table*}"]
    with open(OUT_ROBUST, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT_ROBUST)


def _ci(s):
    """Interval in the format build_parameters.py gives the prose macros, so a value the
    body prints and the same value in a cell are the same string."""
    return f"[{s['lo']:+.3f},\\,{s['hi']:+.3f}]"


def _sci(x):
    """3e-4 as $3\\times10^{-4}$ for a table cell."""
    e = int(np.floor(np.log10(x)))
    m = round(x / 10 ** e, 6)
    return (f"${m:g}\\times10^{{{e}}}$" if m != 1 else f"$10^{{{e}}}$")


def write_config():
    """The configuration table: every training setting
    Methods merely fixes, read from the module that fixes it, so the prose
    keeps the reasons and a change to the code changes the table. The GPS
    merge radius is not a training setting, and sec:materials prints it
    beside its reason."""
    rows = [
        ("Input resolution", rf"${C.W}\times{C.H}$ pixels"),
        ("Frame stride within a sequence", f"{C.FRAME_STRIDE} frames"),
        ("Training crop", f"{C.CROP} pixels, square, random position"),
        ("Horizontal flip", f"probability {C.FLIP_P:g}, training crops only"),
        ("Epochs", f"{C.EPOCHS}"),
        ("Batch size", f"{C.BATCH}"),
        ("Optimizer", "AdamW"),
        ("Learning rate",
         _sci(C.LR) + ", falling to zero along a cosine curve"),
        ("Weight decay", _sci(C.WEIGHT_DECAY)),
        ("Rare-class sampling", r"frame weight: the square root of $m/r$, "
         rf"clipped to $[1, {C.W_CAP:g}]$. A class's share is the fraction of training "
         r"frames in which the union of the two annotations contains the "
         r"class. $m$ is the median share over the target classes, and $r$ "
         r"is the share that the frame's rarest target class has, or $m$ for a "
         r"frame with no target class"),
        ("Random seeds", ", ".join(str(x) for x in C.SEEDS)),
    ]
    lines = [
        r"% Generated by src/build_tables.py from committed results. DO NOT HAND-EDIT.",
        r"% Regenerate:  python src/build_tables.py",
        r"\begin{table}[t]", r"\centering",
        r"\caption{Training settings. Every setting is the same for every "
        r"input configuration and every network.}",
        r"\label{tab:config}",
        r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lp{0.51\linewidth}}", r"\toprule",
        r"Setting & Value \\", r"\midrule",
    ] + [f"{a} & {b} \\\\" for a, b in rows] + [
        r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(OUT_CONFIG, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT_CONFIG)


def write_all():
    """Every table this module publishes, from the committed runs."""
    runs = A.load_runs()
    px = A.gt_px()
    write_body(runs, inflation(px), px)
    write_robust(robustness(px))
    write_config()


def main():
    write_all()


if __name__ == "__main__":
    main()
