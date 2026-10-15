"""Full-study locked configuration and committed run matrix.

This file IS the criterion record: each phase's
criteria are committed to git BEFORE that phase runs, amendments included,
and no criterion changes after the results it governs are on disk. Git history is the
audit trail.

Parameter ledger:
  SEEDS = 0..4          CHOSEN  5 seeds - a margin read off one seed is
                                an anecdote; committed before any run.
  EPOCHS = 30           CHOSEN  the study's compute line; 2x the preliminary epochs
                                for 3x the preliminary data. Fixed before runs, not
                                tuned per arm.
  BATCH/CROP/LR/WD      CHOSEN  carried unchanged from the preliminary recipe
                                (results/preliminary_runs/*): stable convergence
                                on all three arms; identical across arms
                                by design so the arm is the only factor.
  FRAME_STRIDE = 5      CHOSEN  ~3x the preliminary data. Video
                                near-duplicates within a sequence are
                                handled by SITE-blocked splits, not stride.
  W, H = 960, 544       CHOSEN  half res, /32-divisible (preliminary/build_index.py).
  VAL_TARGET = 0.20     CHOSEN  matches the de-facto usable official ratio
                                on the scene-matched set (preliminary: 1032/5256
                                = 0.196); tolerance VAL_TOL below.
  VAL_TOL = 0.08        AMENDED (was 0.03) 2026-07-12, before any training
                                run. Track-distance clustering shows the
                                dataset has exactly THREE GPS super-sites
                                (47.2/27.9/24.9% of strided frames), so
                                achievable val shares are quantized to
                                that set; 0.20+/-0.03 is infeasible. The
                                unique allocation satisfying the class
                                floors is val={site002} at 0.279. Data
                                geometry, not observed outcomes, forced
                                this; git history records the sequence.
  FLOOR_EXEMPT          AMENDED with VAL_TOL: common_tree lives in ONE
                                super-site (6453/5/9 strided frames), so
                                no site-blocked split can retain it as an
                                evaluable control. On the blocked protocol
                                it is reported descriptively only (val
                                n=9 disclosed); it remains a full held-out
                                control on the official protocol.
  PAD_DEG = 1e-4        MEASURED ~11 m at lat 35-37 deg; the exact pad the
                                dataset integrity check used to find the 276
                                official-split overlaps (integrity_checks.py). Kept
                                for that bbox-level regression test and
                                for descriptive cross-split bbox contact
                                counts only.
  RADIUS_M = 11.0       CHOSEN  a ~11 m merge radius: two
                                sequences are the same site iff their GPS
                                TRACKS come within RADIUS_M (exact point
                                distance, KD-tree). Bbox intersection is
                                NOT the site criterion - long trajectories
                                give bboxes whose intersection chains 68
                                of 94 sequences into one site (single-
                                linkage pathology; observed 2026-07-12
                                before any training run).
  M_PER_DEG_LAT/LON     CHOSEN  standard equirectangular scale (110540 /
                                111320*cos(lat)); error <1% here, nothing
                                at 11 m scale.
  MIN_VAL_FRAMES = 50   CHOSEN  eval-stability floor: every class must
  MIN_TRAIN_FRAMES = 200        appear (RGB-or-NIR label) in at least this
                                many blocked-val / blocked-train frames.
  SPLIT_RNG_SEED = 0    CHOSEN  deterministic site assignment; committed
                                before the assignment ran; the seed is a
                                researcher degree of freedom.
  W_CAP = 8, sqrt law   CHOSEN  class-balanced frame oversampling (the
                                sampler row of tab:config). Weights are
                                computed once from the RGB-union-NIR
                                label presence, so they are
                                IDENTICAL for every arm - sampling can not
                                become an arm confound. sqrt tempering is
                                standard practice; the cap bounds worst-
                                case duplication of rare-class frames.
  ALPHA = 0.05, Holm    CHOSEN  See GATES below.

Committed gates:
  - Headline claims are made on the site-BLOCKED protocol; the official
    protocol is reported to quantify leakage inflation.
  - A directional claim ("arm X beats arm Y on class c") requires, on the
    per-frame paired IoU deltas pooled over the 5 committed seeds:
    Holm-corrected (over the 5 target classes) paired Wilcoxon p < ALPHA
    AND a bootstrap 95% CI excluding 0. Otherwise the manuscript word is
    "comparable".
  - Held-out non-targets: common_road and common_tree are
    geometry/texture controls; no directional claim is committed on
    them, and a headline effect that also appears on the controls is
    reported as suspect, not as a win.
  - Successor-falsifier: the full study must reproduce the
    falsified premise of the preliminary work (RGB > NIR boundary separability) via
    test_assertions.py; a full-study result contradicting it halts the
    pipeline for audit rather than silently overriding it.

2026-07-23 AMENDMENT (committed before phases E and F run; git is the
record). Phases A-C are complete and frozen (60 runs); nothing below alters
a computed A-C number. One unrun phase is REMOVED and two are ADDED.

  D REMOVED. The LiDAR arms (RGBL, RGBNIRL) needed a projective depth-feature
    build (a week-3 item) that was never implemented; no phase-D run exists
    on disk, so removal changes no result. LiDAR<->camera association was
    already disclosed as weak/unverified in the preliminary work; rather than ship an
    under-built modality as a comparison arm - deleting an arbitrary,
    non-load-bearing choice rather than defending it - the compute
    is reallocated to E and F.

  E ADDED - leave-one-site-out (LOSO). The headline leakage result rests on a
    single held-out site (site002); LOSO is its committed self-falsifier
    (a claim must survive a re-split it did not choose). All
    THREE folds are reported regardless of outcome. Fold site002 is split-
    identical to the blocked protocol (verified: blocked-val == {site002}
    exactly, 4407 frames; blocked-train == {site000,site001}, 11383), so the
    phase-A blocked runs ARE the site002 fold; only site000 and site001 are
    run here (40 runs = 2 folds x 4 arms x 5 seeds). Locked hyperparameters
    and the committed seed set are unchanged.
    Gates (per-frame paired deltas, seeds pooled; the sec:methods directional gate
    = Holm p<ALPHA over the 5 targets AND bootstrap 95% CI excl. 0 on the
    intersection protocol, see GATES above):
      E1 leakage-generalizes: on every fold where paddy_edge is evaluable
         (site000 n=3627, site002 n=754; site001 n=1, descriptive only) the
         official-protocol paddy_edge IoU exceeds the LOSO paddy_edge IoU by
         a margin clearing the sec:methods gate. Refutes "the inflation is an
         artifact of the single site002 holdout."
      E2 positive-generalizes: NIR-LUM on paddy_after_driving, pooled over
         all 3 folds (evaluable on every site: n=3430/2374/2522), clears the
         sec:methods gate. Else the one real positive is demoted to site-specific
         ("comparable").
      E3 control-does-not-collapse: common_road (generic, well-represented:
         n=4411/3935/3742) retains LOSO IoU >= 0.30 on all 3 folds. CHOSEN
         floor - a loose collapse-detector far below typical road IoU (~0.7);
         its only job is to catch "LOSO is so hard everything dies," which
         would confound reading paddy_edge's collapse as leakage-specific.
         Not a headline threshold.

  F ADDED - mit_b2 second backbone. The mit_b0 headline rests on one small
    backbone, the obvious alternative being a larger one in the same family.
    mit_b2 (same SegFormer family, ~7x encoder params)
    re-runs the blocked and official protocols (40 runs = 2 protocols x 4
    arms x 5 seeds). The mit_b0 comparison is untouched; mit_b2 is analyzed
    separately (analyze_folds.py; the headline analyzer analyze.py loads mit_b0 only, so
    backbones are never pooled).
    Gates:
      F1 leakage-reproduces: the official-vs-blocked paddy_edge inflation
         reproduces under mit_b2 (official >> blocked, sec:methods gate). If it
         vanishes, the leakage finding is backbone-specific - reported as
         such, not hidden.
      F2 positive-reproduces: NIR-LUM on paddy_after_driving (blocked) clears
         the sec:methods gate under mit_b2 too, else the positive is mit_b0-only.
    A backbone swap must reproduce both as emergent properties, never tune
    them (the successor-falsifier gate above).

2026-07-24 AMENDMENT (committed before phase G runs; git is the record).
Phases A-C stay frozen. Phase E is NOT withdrawn, re-scoped, or re-run: its
gates were committed and its verdict ships as it fell (below).

  E3 FELL AS "FAIL" on fold site000 (final: all 20 loso000 runs, 4 arms x
    5 seeds, verdict identical on all three eval protocols). Measured
    common_road LOSO IoU (isec, seed-mean): site000 0.248 (per-arm
    0.236-0.258) against the committed >= 0.30 floor; site001 0.806
    PASS; site002 0.801 PASS. The collapse is FOLD-WIDE, not road-specific:
    on site000 field_levee 0.000, paddy_rice 0.000, field_furrow 0.000,
    paddy_before_driving 0.011, background down to 0.802 (vs 0.945/0.960);
    only paddy_after_driving survives (0.362 vs 0.467/0.499).
    CAUSE (measured, geometry_probe.py): Field frames are physically
    concentrated - 3381/1392/321 across site000/001/002, so site000 holds
    66.4% of all Field data. Holding it out trains on 33.6% of the Field
    distribution and evaluates on 66.4% of it; fold site000 is near-out-of-
    domain transfer, not a same-domain held-out site. This is exactly the
    confound E3 was written to detect.
    CONSEQUENCE, committed with this amendment: the site000 fold cannot be read as
    leakage-SPECIFIC evidence for paddy_edge. site002 == blocked == the
    existing headline and site001 has paddy_edge n=1 (descriptive only), so
    phase E supplies a DISCLOSURE, not an independent confirmation. E2 is
    still computed pooled over all 3 folds exactly as written - re-scoping it
    to drop site000 after seeing E3 fail would be post-hoc recalibration
    and is refused. The headline is untouched: it rests
    on phases A/C plus the independent leakage_isolation.py leakage isolation, and fold
    site002 passes E3 cleanly.

  ENV-STRATIFIED RE-SPLIT CONSIDERED AND REJECTED AS INFEASIBLE (measured,
    geometry_probe.py, before any phase-G decision). Single-linkage
    over exact min-track distance between the 94 sequences is a cliff:
    K=3 clusters -> 761 m minimum inter-cluster separation; K=4 -> 0.5 m.
    Cutting past 3 peels off only 58-305-frame slivers (site000 stays 7447
    -> 7032 even at K=14) and every sliver has Field share 0.00, so the big
    cluster gets MORE Field-heavy (0.45 -> 0.48), not less. Rebalancing by
    re-clustering therefore buys no balance and would drop train/val
    separation from 761 m to sub-meter - reintroducing the very leakage this
    study measures. The imbalance is a property of where the farms are, not
    of the split algorithm, and is disclosed rather than engineered away.

  G ADDED - Paddy-restricted LOSO. Balance is reached by RESTRICTION, not by
    re-shuffling: the Field concentration is what breaks LOSO, while Paddy is
    already balanced across sites (4066/2544/4086 strided frames = 38/24/38%).
    Training AND evaluation are restricted to Paddy; the site clustering, the
    761 m separation, the locked hyperparameters and the committed seed set
    are all unchanged - no re-split, no re-tune, no re-seed. Protocol
    ploso<NNN> holds out siteNNN with both sides Paddy-only. All THREE folds
    are run and reported regardless of outcome (60 runs = 3 folds x 4 arms x
    5 seeds); no fold is inherited, because Paddy-only training makes every
    fold a new training regime.
    Fold sizes (VERIFIED against the built index): n_train/n_val =
    6630/4066 (hold site000), 8152/2544 (site001), 6610/4086 (site002).
    Class-presence val frames per fold, same OR definition (present in the
    RGB or the NIR polygon labels) the phase-E amendment used:
    paddy_after_driving 3430/1948/2374 - present on ALL folds, so it can
    carry the control; paddy_edge 3409/0/748 - site001 has NONE, so paddy_edge
    keeps only 2 evaluable folds. That class-by-site concentration
    (paddy_rice absent from site000, common_tree effectively site000-only) is
    intrinsic to the dataset; it is disclosed as a limitation, and no gate
    below depends on a fold where its class is absent.
    Phase G is motivated by E3's failure. That is a legitimate successor
    experiment, not a re-scoping, because (a) phase E ships unchanged with its
    failed gate, and (b) the gates below are committed before phase G runs.
    Gates (per-frame paired deltas, seeds pooled; "sec:methods gate" = Holm
    p<ALPHA over the 5 targets AND bootstrap 95% CI excluding 0 on the
    intersection protocol, as in GATES above):
      G1 control-holds - the E3 re-test: paddy_after_driving retains LOSO IoU
         >= 0.30 on all 3 folds. Same CHOSEN floor and same job as E3, now on
         a fold set whose train/eval env composition matches by construction.
         If G1 also fails, cross-site generalization is not supportable on
         this dataset at all and the paper says so instead of reporting E2/G3.
      G2 collapse-is-class-specific: on each fold where paddy_edge is
         evaluable (site000 n=3409, site002 n=748) paddy_edge IoU stays far
         below the G1 control while G1 holds. Descriptive within phase G; the
         comparator that makes it a leakage MEASUREMENT is committed in
         the 2026-07-24b amendment below, not read in from phase C.
      G3 positive-generalizes - the E2 re-test: NIR-LUM on
         paddy_after_driving, pooled over all 3 folds, clears the sec:methods gate.
         Fully internal to G (both arms trained identically), so it carries no
         G2 confound. This is phase G's load-bearing deliverable: whether the
         study's one real positive survives across sites on a fold set whose
         control is verified. If it fails, the positive is demoted to
         site-specific ("comparable") and the manuscript sentence changes.

2026-07-24b AMENDMENT - pofficial, the matched comparator that closes G2.
Committed with ZERO phase-G runs on disk (no ploso* and no pofficial* run
directory existed; phase F was mid-queue), so this STRENGTHENS a gate before
any of its data exists rather than re-reading one after. Nothing already run
is altered: phases A-C stay frozen, E ships with its failed gate, and G1/G3
are untouched.

  WHY. G2 as first written could only say "paddy_edge sits far below the
  control", because the only leaked comparator on record (phase C, official)
  trains on BOTH environments - so a G2-vs-C gap mixes the env restriction
  with the split and cannot be read as a leakage measurement. The clean fix
  is a comparator that differs from ploso in the SPLIT ALONE.

  pofficial - the official train/val split with both sides restricted to
  PLOSO_ENV, i.e. ploso's environment restriction with ploso's site blocking
  removed. 20 runs = 4 arms x 5 seeds, mit_b0, locked hyperparameters and the
  committed seed set (~9.2 GPU-h, estimated from the measured mit_b0 blocked
  4-arm cost of 2.51 GPU-h/seed scaled by 8299/11383 train frames).
  Split sizes (VERIFIED against the built index): n_train 8299, n_val 2397,
  over 10696 Paddy frames.

  G2 (RESTATED, committed). Two readings from the same 20 runs; both are
  reported, and the second is the one the manuscript leans on.
    G2a MATCHED POOLED CONTRAST. paddy_edge IoU under pofficial vs under
        ploso, seeds pooled, on the folds where paddy_edge is evaluable. Both
        conditions are Paddy-only, so the env restriction cancels and the
        remaining difference is site blocking. Reported per arm, since a gap
        present in every arm is generic memorization while an NIR-concentrated
        gap is the study's finding (same logic as leakage_isolation.py).
    G2b FRAME-FIXED PAIRED TEST - leakage_isolation.py's design lifted into the Paddy
        regime, and the stronger statement because evaluation frames are held
        IDENTICAL. For fold N the fixed set is
            S_N = { frame : env==PLOSO_ENV, site==siteN, osplit==val }
        which NEITHER condition trained on, so this is not frame memorization
        and not a difficulty difference:
          seen   = pofficial-trained (trained on OTHER siteN Paddy frames);
          unseen = plosoN-trained (never saw any siteN frame).
        Gate: per-frame seen-minus-unseen on paddy_edge clears the sec:methods gate
        (bootstrap 95% CI excluding 0, paired Wilcoxon), NIR-concentrated.
        EVALUABLE ON 2 OF 3 FOLDS, committed as such: |S_N| = 256
        (site000, 3 sequences, paddy_edge present in all 256 under the
        intersection definition), 0 (site001 - it has NO official-val Paddy
        frame at all), 2141 (site002, 8 sequences, 322 with paddy_edge);
        256+0+2141 = 2397 = pofficial n_val exactly. site001 contributes
        nothing to G2b and is reported as contributing nothing; that it also
        has zero paddy_edge under ploso (above) is a second, independent
        reason and neither is engineered away.
        DISCLOSED ASYMMETRY, as in leakage_isolation.py: pofficial trains on
        8299 frames vs
        ploso's 6630/8152/6610, so seen/unseen differ in data QUANTITY by up
        to 1.26x as well as in site coverage. Phase A-C measured an 86x split
        effect against an 11% quantity gap; the ratio is disclosed with the
        result rather than hidden, and G2a's per-arm reading is what separates
        quantity (all arms) from leakage (NIR-concentrated).
  If G2a and G2b disagree, G2b governs - it is the one holding frames fixed.

2026-08-05 AMENDMENT (committed before phase H runs; git is the record).
Phases A-C stay frozen, E ships with its failed gate, F and G are untouched.
Zero phase-H runs exist on disk at this commit.

  H ADDED - UPerNet / ConvNeXt-V2 second ARCHITECTURE FAMILY. Phase F varied
    encoder SIZE within one family (mit_b0 -> mit_b2, ~7x) and both findings
    reproduced, so capacity is excluded as the cause. It leaves the adjacent
    one-line objection standing - "SegFormer is a 2021 design; a current
    model would not show this" - and the redundancy claim is NEGATIVE, so it
    carries the burden of excluding our own implementation as its cause -
    the obligation scales with the claim's shape, not the calendar.
    Disclosing the boundary is not discharging it, and at this
    commit the manuscript only discloses it (sec:discussion, "We did not vary
    the architecture family").
    Scope: blocked + official x 4 arms x 5 seeds = 40 runs, mirroring phase F
    exactly so H1/H2 are the F1/F2 gates with the model swapped and nothing
    else. Locked hyperparameters, the committed seed set, the site clustering
    and the splits are all unchanged - no re-tune, no re-split, no re-seed.
    Analyzed separately (analyze_folds.py); analyze.py stays
    mit_b0/segformer-only, so the
    architectures are never silently pooled.

    MODEL: UPerNet decoder over a ConvNeXt-V2-Tiny encoder, FCMAE
    self-supervised pretraining (Woo et al., CVPR 2023). Chosen to differ
    from SegFormer on all three axes the objection names at once - decoder
    family (UPerNet vs all-MLP), backbone family (convolutional vs
    hierarchical transformer), and pretraining paradigm (masked autoencoder
    vs supervised ImageNet-1k) - while remaining LARGER than the phase-F
    model (37.1 M vs 24.7 M total parameters), so it cannot be dismissed as
    a weaker substitute. Assembly mirrors smp's stock UPerNet exactly, and
    the fusion arm reuses the phase-A/F dual-encoder pattern with the
    UPerNet decoder over the doubled channel list.
    2026-09-07: timm resolves BACKBONE_H to tag fcmae_ft_in22k_in1k, so
    the masked-autoencoder weights arrive fine-tuned on labeled
    ImageNet-22k and ImageNet-1k. The paradigm axis above names the
    checkpoint's origin, not the supervision its weights carry.

    WHY TINY AND NOT BASE (measured, before the choice was committed; a memory and
    throughput benchmark at the locked BATCH=16 / CROP=512 on one
    RTX 5000 Ada 32 GB):
    ConvNeXt-V2-Base peaks at 24.2 GB on a single-encoder arm and OOMs
    outright on the dual-encoder fusion arm; it fits only with gradient
    checkpointing, at 31.0 GB of 32.8 GB and 1.677 s/step (2.5x Tiny's
    0.683 s/step). A 40-run queue whose fusion arm sits at 95% of device
    memory for ~12 h/run is a queue that dies on a fragmentation spike, and
    the alternative - cutting BATCH for phase H alone - would break
    hyperparameter matching with every other phase and introduce exactly the
    confound the locked config exists to prevent. Tiny fits the locked
    config unchanged. Size is not the axis under test here; phase F already
    varied it.
    Estimated cost 149 GPU-h (~3.1 d on 2 GPUs), from measured step times
    scaled by the measured mit_b2 blocked+official run cost (6309 s
    single-encoder, 9736 s fusion): 30 single-encoder runs at ~11.8 ks and
    10 fusion runs at ~18.2 ks.

    Parameter ledger for this phase:
      ARCH_H / BACKBONE_H   CHOSEN  structural, per MODEL above.
        BASELINE: SegFormer MiT-B0/B2 (2021 design, supervised ImageNet) |
        obvious alternative: "a current architecture would not show this" |
        discharge: phase H (this amendment)
      everything else       unchanged from the locked config above; phase H
                            introduces no new tunable.

    Gates (per-frame paired deltas, seeds pooled; "sec:methods gate" = Holm
    p<ALPHA over the 5 targets AND bootstrap 95% CI excluding 0 on the
    intersection protocol, as in GATES above). Both are phase F's gates with
    the model swapped, so a divergence is attributable to the architecture
    and to nothing else:
      H1 leakage-reproduces: the official-vs-blocked paddy_edge inflation
         reproduces under UPerNet/ConvNeXt-V2 (sec:methods gate), NIR-concentrated
         as in F1. If it vanishes, the leakage finding is specific to the
         SegFormer family and the manuscript says so - that is a publishable
         outcome, not a failure to hide.
      H2 positive-reproduces: NIR-LUM on paddy_after_driving (blocked)
         clears the sec:methods gate under this architecture too, else the
         surviving positive is demoted to SegFormer-only.
    Both must be reproduced as emergent properties, never tuned (the
    successor-falsifier gate above). Whichever way they fall, the manuscript
    sentence bounding the redundancy claim to "efficient transformer
    segmentation" is rewritten in the same commit as the phase-H result.

CLOSED. Phase H ran, and both sentences quoted above were rewritten with its
result, so neither is greppable in the manuscript now: they record what
sec:discussion said before phase H, not what it says. The H1 and H2 verdicts
are reported there and asserted in test_assertions.py.

2026-09-22 AMENDMENT (after every phase ran; no criterion changes).
Phase B LEAVES the run matrix, which now holds 240 runs. Phase B committed no
criterion. Its dry-field comparison rests on the one driving sequence that the
held-out super-site holds in dry field, so it has no sequence-clustered
interval and settles nothing. The manuscript is an eight-page letter that
admits no supplementary text, so it does not carry the comparison, and the 20
runs are kept outside this repository. Every run below trains on both
environments of its training partition.

2026-09-25 AMENDMENT (after every phase ran; no criterion changes).
The successor-falsifier above, the boundary measurement, leaves this
repository. It reproduced: near-infrared minus color in
boundary separability has a per-class mean between -0.135 and -0.036, and
each class's sequence-clustered interval excludes zero. No finding rests on
it, and the letter's eight pages went to the definitions that a reader from
another field needs, so test_assertions.py no longer carries it.

2026-10-13 AMENDMENT (after every phase ran; the interval of every gate
changes). The gates above name a bootstrap 95% CI. Most of them rest on 3 to
13 sequences, over which a percentile bootstrap covers below its stated level,
so analyze.py cluster_ci now gives every interval over fewer than
MIN_BOOT_UNITS sequences as a cluster-robust t interval with k-1 degrees of
freedom, and keeps the sequence-cluster bootstrap for the two comparisons that
pool all folds (76 and 72 sequences). The change was made after every result
was known and runs against the claims: it widens every interval it touches and
turns no comparison into a pass. It turned nine verdicts from directional into
comparable: NIR minus RGB on the paddy edge under the blocked protocol; NIR
minus the luminance control on the untilled paddy surface under the blocked
protocol, which the bootstrap had left within Monte Carlo error of zero; the
paddy-edge inflation of NIR at site002 (E1) and on both larger models (F1,
H1); that of the luminance control at site000 of E1 and on both larger models
(F1, H1); and that of RGB under H1. The farm access road's control flag under
both protocols cleared too. NIR's paired excess over each other
configuration's inflation at site002 now spans zero on mit_b0 and under H, and
over RGB's at site002 of G2 as well.
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
LABELS_ROOT = os.path.join(ROOT, "data", "labels")
IMAGES_ROOT = os.path.join(ROOT, "data", "images")
STUDY_DATA = os.path.join(ROOT, "data", "study")
PROCESSED = os.path.join(ROOT, "data", "processed")
INDEX_JSON = os.path.join(PROCESSED, "index.json")
SPLITS_JSON = os.path.join(PROCESSED, "splits_blocked.json")
INDEX_STATS = os.path.join(PROCESSED, "index_stats.json")
RUNS_ROOT = os.path.join(ROOT, "results", "runs")

INDEX_PATH_KEYS = ("img_rgb", "mask_rgb", "img_nir", "mask_nir")


def load_index():
    """Per-frame index records, with the four image and mask paths resolved.

    src/phases/build_index.py stores those four relative to STUDY_DATA
    rather than as
    absolute paths, so the index travels with the repository and a move of the
    study directory cannot leave it pointing somewhere that no longer exists.
    """
    with open(INDEX_JSON) as f:
        recs = json.load(f)
    for r in recs:
        for k in INDEX_PATH_KEYS:
            r[k] = os.path.join(STUDY_DATA, r[k])
    return recs

# ---- taxonomy (8 shared classes; paddy_water dropped -
# 0 train instances on disk, see test_assertions.py) ----
CLASSES = [
    "background",
    "common_road", "common_tree",
    "field_furrow", "field_levee",
    "paddy_before_driving", "paddy_after_driving",
    "paddy_edge", "paddy_rice",
]
N_CLASSES = len(CLASSES)
TARGETS = ["paddy_edge", "field_furrow", "field_levee",
           "paddy_before_driving", "paddy_after_driving"]
CONTROLS = ["common_road", "common_tree"]  # held-out non-targets
ENVS = ["Paddy", "Field"]

# ---- data build ----
FRAME_STRIDE = 5
W, H = 960, 544

# ---- blocked-split construction ----
PAD_DEG = 1e-4
RADIUS_M = 11.0
M_PER_DEG_LAT = 110_540.0
M_PER_DEG_LON = 111_320.0  # x cos(latitude); CHOSEN with M_PER_DEG_LAT (ledger)
VAL_TARGET = 0.20
VAL_TOL = 0.08         # CHOSEN, amended before any run (ledger)
MIN_VAL_FRAMES = 50
MIN_TRAIN_FRAMES = 200  # CHOSEN with MIN_VAL_FRAMES (ledger)
FLOOR_EXEMPT = ["common_tree"]  # see ledger above
SPLIT_RNG_SEED = 0
N_SPLIT_TRIES = 1000   # CHOSEN: site permutations the split search tries

# ---- training (locked, identical for every arm) ----
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 30
BATCH = 16
CROP = 512
LR = 3e-4              # AdamW, cosine to 0
WEIGHT_DECAY = 1e-4    # CHOSEN with BATCH/CROP/LR (ledger: WD)
W_CAP = 8.0            # sampler weight cap
FLIP_P = 0.5           # CHOSEN with BATCH/CROP/LR: horizontal-flip probability
                       # of a training crop, carried from the preliminary recipe

# ---- statistics ----
ALPHA = 0.05           # Holm-corrected over the 5 TARGETS

# ---- leave-one-site-out (phase E; 2026-07-23 committed amendment) ----
LOSO_FOLDS = ["site000", "site001", "site002"]  # the 3 GPS sites
LOSO_RUN_FOLDS = ["site000", "site001"]         # site002 fold == blocked (A)

# ---- Paddy-restricted LOSO (phase G; 2026-07-24 committed amendment).
# Every fold is run: Paddy-only training makes each one a new training
# regime, so none is inherited from an earlier phase. See ledger amendment
# for why re-clustering for env balance was rejected as infeasible. ----
PLOSO_ENV = "Paddy"                             # the balanced environment
PLOSO_RUN_FOLDS = ["site000", "site001", "site002"]
PLOSO_CONTROL_FLOOR = 0.30   # gate G1; same CHOSEN floor and job as E3
# Folds carrying gate G2b (2026-07-24b amendment): the fixed set S_N is
# empty for site001, which has no official-val Paddy frame at all. Named
# here so the analyzer cannot silently drop or silently add a fold.
PLOSO_G2B_FOLDS = ["site000", "site002"]

# ---- second architecture family (phase H; 2026-08-05 amendment) ----
ARCH_H = "upernet"
BACKBONE_H = "tu-convnextv2_tiny"
# (arch, backbone) -> run_id suffix. Explicit registration, never derived:
# a KeyError on an unregistered model is a loud failure, and the empty tag
# keeps every phase A-C/E/G run_id byte-identical to what is on disk.
MODEL_TAG = {("segformer", "mit_b0"): "",             # headline
             ("segformer", "mit_b2"): "b2",           # phase F
             (ARCH_H, BACKBONE_H): "cnx"}             # phase H

# ---- committed run matrix (execution order; seed-major so paired
# seeds complete together). Phases:
#   A  headline: blocked protocol, joint-env, 4 spectral arms x 5 seeds
#   C  official-protocol arms: quantifies split-leakage inflation
#   E  leave-one-site-out: 2 run folds x 4 arms x 5 seeds (site002 fold
#      reuses phase A). LOSO self-falsifier for the leakage claim.
#   F  mit_b2 second backbone: blocked+official x 4 arms x 5 seeds. Backbone
#      robustness of the mit_b0 headline against the baseline objection.
#   G  Paddy-restricted LOSO: 3 folds x 4 arms x 5 seeds. Re-test of E3's
#      failed control on the environment that IS balanced across sites.
#   H  second ARCHITECTURE family (UPerNet/ConvNeXt-V2): blocked+official
#      x 4 arms x 5 seeds. Phase F varied encoder size within one family;
#      H varies the family itself, discharging the baseline objection
#      that the negative redundancy claim rests on a 2021 design.
#   (B  single-environment training - LEFT the matrix 2026-09-22; see ledger
#       amendment.)
#   (D  LiDAR arms - REMOVED 2026-07-23, never ran; see ledger amendment.)
# ----


def _run(arm, seed, protocol, phase, backbone="mit_b0", arch="segformer"):
    rid = f"{arm}_s{seed}_{protocol}"
    if MODEL_TAG[(arch, backbone)]:
        rid += f"_{MODEL_TAG[(arch, backbone)]}"   # phase F -> _b2, H -> _cnx
    return {"run_id": rid, "arm": arm, "seed": seed, "protocol": protocol,
            "train_env": "all", "phase": phase, "backbone": backbone,
            "arch": arch}


RUN_MATRIX = []
for _seed in SEEDS:
    for _arm in ("RGB", "NIR", "LUM", "RGBNIR"):
        RUN_MATRIX.append(_run(_arm, _seed, "blocked", "A"))
for _seed in SEEDS:
    for _arm in ("RGB", "NIR", "LUM", "RGBNIR"):
        RUN_MATRIX.append(_run(_arm, _seed, "official", "C"))
# Phase E - LOSO. loso<NNN> protocol holds out siteNNN; site002 is omitted
# (split-identical to blocked, supplied by phase A). See ledger amendment.
for _seed in SEEDS:
    for _fold in LOSO_RUN_FOLDS:
        for _arm in ("RGB", "NIR", "LUM", "RGBNIR"):
            RUN_MATRIX.append(
                _run(_arm, _seed, "loso" + _fold[len("site"):], "E"))
# Phase F - mit_b2 second backbone over blocked + official (mit_b0 headline
# untouched; analyzed separately by analyze_folds.py).
for _seed in SEEDS:
    for _proto in ("blocked", "official"):
        for _arm in ("RGB", "NIR", "LUM", "RGBNIR"):
            RUN_MATRIX.append(
                _run(_arm, _seed, _proto, "F", backbone="mit_b2"))
# Phase G - Paddy-restricted LOSO. ploso<NNN> holds out siteNNN with BOTH
# train and val restricted to PLOSO_ENV; the protocol itself carries the
# restriction.
for _seed in SEEDS:
    for _fold in PLOSO_RUN_FOLDS:
        for _arm in ("RGB", "NIR", "LUM", "RGBNIR"):
            RUN_MATRIX.append(
                _run(_arm, _seed, "ploso" + _fold[len("site"):], "G"))
# Phase G - pofficial: the official split with both sides restricted to
# PLOSO_ENV. Differs from ploso in the SPLIT ALONE, which is what makes G2 a
# leakage measurement rather than a description (2026-07-24b amendment).
for _seed in SEEDS:
    for _arm in ("RGB", "NIR", "LUM", "RGBNIR"):
        RUN_MATRIX.append(_run(_arm, _seed, "pofficial", "G"))
# Phase H - second ARCHITECTURE family over blocked + official, mirroring
# phase F's scope exactly so H1/H2 are F1/F2 with the model swapped and
# nothing else (mit_b0 headline untouched; analyzed separately by analyze_folds.py).
for _seed in SEEDS:
    for _proto in ("blocked", "official"):
        for _arm in ("RGB", "NIR", "LUM", "RGBNIR"):
            RUN_MATRIX.append(
                _run(_arm, _seed, _proto, "H",
                     backbone=BACKBONE_H, arch=ARCH_H))

assert len(RUN_MATRIX) == 240, len(RUN_MATRIX)
assert len({r["run_id"] for r in RUN_MATRIX}) == 240
