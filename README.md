# Near-infrared's apparent advantage over RGB in paddy and dry-field traversability segmentation comes from site leakage

Does forward-facing near-infrared (NIR) add traversability signal to RGB for
farm-machinery navigation, or does its apparent advantage come from site
leakage? This repository holds the code, splits, per-frame metadata, results, and
manuscript for a leakage-aware answer on the AI Hub *Precision Agriculture Farm
Machinery Autonomous Driving* dataset (rice-paddy and dry-field agricultural
driving).

The study compares four input configurations under protocols that either admit or
control cross-site leakage:

- **RGB**, **NIR**, **LUM** (the luminance control: the Rec. 601 luma of the
  RGB image, a single grayscale channel in NIR's single-channel format and
  with no NIR information), and **RGB+NIR** fusion.
- **Site-blocked** and **leave-one-site-out** protocols (train and evaluate on
  disjoint GPS sites) against the dataset's sequence-level split, plus a
  paddy-restricted leave-one-site-out cross-validation and its matched
  comparator.
- Three evaluation ground truths (RGB annotations, NIR annotations, and their
  intersection, the pixels on which the two annotations agree, where a
  configuration cannot gain by matching its own annotation).
- Five random seeds behind every variance estimate; a second SegFormer backbone
  (MiT-B2 alongside the primary MiT-B0) and a second architecture family
  (UPerNet/ConvNeXt-V2) for robustness.

The headline finding is that, for daylight paddy and dry-field driving, no
configuration outperforms RGB at every held-out site. Fusion's gain over RGB on
the tilled paddy surface at the site-blocked protocol's held-out site reverses
sign at another site, and NIR's apparent advantage on the paddy edge under the
dataset's sequence-level split comes from site leakage, which raises NIR's
IoU relative to RGB's. The quantified claims live in the manuscript; every
one is re-derived from on-disk data by `test_assertions.py`.

## Repository structure

```
nlr/
├── README.md                     # This file
├── LICENSE                       # MIT for the code; its first section names
│                                 #   what it does not cover
├── requirements.txt              # Pinned dependencies
├── run_full_pipeline.py          # Regenerates every published number, table and figure
├── test_assertions.py            # Manuscript-claim assertions (pytest target)
│
├── src/
│   ├── config.py                 # Locked protocol record: seeds 0-4, the 240-run
│   │                             #   matrix, the protocols, the numeric gates, and
│   │                             #   the amendment ledger
│   ├── train.py                  # One run: arm x seed x protocol (x backbone x arch);
│   │                             #   own-, cross- and intersection-annotation eval
│   ├── analyze.py                # Headline analyzer, phases A and C: per-seed spread,
│   │                             #   paired Wilcoxon, clustered CI, Holm over targets
│   ├── analyze_folds.py          # Fold, second-backbone and second-architecture
│   │                             #   analyzer: gates E1-E3, F1-F2, G1-G3, H1-H2
│   ├── build_parameters.py       # -> paper/params.tex (generated region)
│   ├── build_tables.py           # -> the manuscript's three generated tables
│   ├── build_figures.py          # -> paper/fig_leakage_isolation.pdf
│   │
│   ├── phases/                   # The study's stages, in run order
│   │   ├── inventory.py          #   Dataset walk: class x env x modality x split
│   │   ├── integrity_checks.py   #   Official-split GPS leakage; RGB-NIR agreement
│   │   ├── splits_blocked.py     #   GPS track-distance clustering into sites
│   │   ├── build_index.py        #   Stride-5 scene-matched frames, both annotations
│   │   ├── run_queue.py          #   Resumable sequential queue over the matrix
│   │   ├── leakage_isolation.py  #   Frame-fixed seen-vs-unseen test, per arm
│   │   ├── geometry_probe.py     #   GPS and environment composition, read-only
│   │   └── camera_offset.py      #   Offset between the two cameras, read-only
│   │
│   └── preliminary/              # The 2026-07-11 feasibility experiment
│       ├── build_index.py        #   Shared-index frames, stride 15
│       └── train.py              #   One preliminary arm, 2 seeds
│
├── data/
│   └── processed/                # Released artifacts, all tracked
│       ├── splits_blocked.json   #   Site-blocked and leave-one-site-out splits
│       ├── index.json            #   Per-frame metadata: site, environment, both
│       │                         #     annotation paths, per-class GT pixel counts
│       ├── index_stats.json      #   Per-split class totals
│       └── geometry_probe.json   #   Site separation and environment composition
│
├── results/
│   ├── runs/<run_id>/            # metrics.json + perframe.npz per run (240); model.pt
│   │                             #   is not tracked
│   ├── preliminary_runs/<run>/   # metrics.json per preliminary arm
│   ├── summary.md                # Phase A and C comparison tables
│   ├── folds.md                  # Phase E/F/G/H criterion tables
│   ├── inventory.json  integrity_checks.json
│   └── leakage_isolation.json  camera_offset.json
│
└── paper/                        # main.tex, params.tex, references.bib, the three
                                  #   generated tables and the figure, and
                                  #   ieeeconf.cls, the RA-L submission class
                                  #   that TeX Live does not ship
```

## Installation

```bash
conda create -n nlr python=3.12 && conda activate nlr
pip install -r requirements.txt
```

The `+cu126` torch wheels come from <https://download.pytorch.org/whl/cu126>.

## Reproducing the manuscript

Every number, table and figure the manuscript prints recomputes from the committed
artifacts alone:

```bash
python run_full_pipeline.py
```

It runs the geometry probe, both analyzers and all three generators in process,
then the claim tests. It opens no imagery, and it stops naming what is missing
rather than regenerating a paper from part of a run. Its outputs reproduce the
committed ones byte for byte. The training stages below do not: `src/train.py`
seeds every generator from the committed seed but runs mixed-precision GPU
kernels without deterministic mode, so a retrained network reproduces the
committed per-frame IoUs within run-to-run variation, not exactly.

The stages upstream of it do need the dataset, and the training queue a GPU:

```bash
python src/phases/inventory.py                          # -> results/inventory.json
python src/phases/integrity_checks.py                   # -> results/integrity_checks.json
python src/phases/splits_blocked.py                     # -> data/processed/splits_blocked.json
python src/phases/build_index.py                        # -> data/study/frames/, data/processed/index.json
CUDA_VISIBLE_DEVICES=<gpu> python src/phases/run_queue.py 0 1 ACEFGH
python src/phases/leakage_isolation.py                  # -> results/leakage_isolation.json
python src/phases/camera_offset.py                      # -> results/camera_offset.json
```

To shard the queue across two GPUs, run two shells with queue index 0 and 1 of 2
on separate devices (`... run_queue.py 0 2 ACEFGH` and `... run_queue.py 1 2
ACEFGH`). Runs that already have a `metrics.json` are skipped, so the queue is
resumable after an interruption.

## Running a single stage

Every module under `src/` is runnable on its own:

```bash
python src/analyze_folds.py
```

## Verifying manuscript claims

```bash
python -m pytest test_assertions.py -q
```

Every quantitative and directional claim that reaches the manuscript is asserted
here against the committed artifacts, in the same commit as the claim. The same
suite is the last stage of `run_full_pipeline.py`.

## The run matrix

**A** blocked headline (4 arms × 5 seeds, 20 runs) · **C** sequence-level-split arms, protocol `official` (leakage-inflation
quantification, 4 × 5, 20) · **E** leave-one-site-out cross-validation (2 run folds ×
4 × 5, 40; the site002 fold is split-identical to blocked and supplied by phase A) ·
**F** MiT-B2 second backbone over the site-blocked protocol and the sequence-level split (2 × 4 × 5, 40) ·
**G** paddy-restricted leave-one-site-out cross-validation (3 folds × 4 × 5, 60) plus the
`pofficial` matched comparator (4 × 5, 20) · **H** UPerNet/ConvNeXt-V2 second
*architecture family* over the same two protocols (2 × 4 × 5, 40): F varies encoder
size within one family, H varies the family itself. 240 runs total.

## The preliminary experiment

`src/preliminary/` holds the feasibility training of 2026-07-11, this project's own
earlier work on the same dataset: two seeds per arm on a stride-15 shared-scene
index, where the study specifies at least five. Its validation split carries the
site leakage `integrity_checks.py` quantifies, and each arm was evaluated against
its own modality's annotations. Its numbers are not published as results. The
locked configuration carries its recipe: `src/config.py` justifies BATCH, CROP, LR
and WEIGHT_DECAY against `results/preliminary_runs/`.

The dataset measurements of that same day sit under `src/phases/` instead. They read
the released labels and imagery and nothing this study produced, so they are the
study's first stages rather than preliminary to it.

## Evaluation protocol and integrity

`src/config.py` is the locked protocol record: the seeds, the complete run matrix,
the protocols, and the numeric criteria. `test_assertions.py` re-computes every
published claim from the committed data, so a claim that drifts fails a test rather
than shipping silently. Numerical values in the manuscript body are LaTeX macros in
`paper/params.tex` (no hardcoded numerals in prose); its generated region is emitted
from the committed `results/` by `src/build_parameters.py`.

## Dataset

**Source:** AI Hub *Precision Agriculture Farm Machinery Autonomous Driving*,
<https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=526>. Extract the label and
image zips under `data/`.

## Data and code availability

Under the dataset's license, this repository releases the evaluation protocol, the
analysis code, the site-blocked and leave-one-site-out splits, the per-frame
metadata, and the per-frame intersection and union pixel counts of every run. It
does not redistribute the imagery, which retraining the networks and measuring the
camera offset need.

## Licensing

`LICENSE` covers the code — `src/`, `run_full_pipeline.py`, `test_assertions.py`
and `requirements.txt` — under MIT. It covers nothing else, and the file says so
in its own first section, because parts of this repository are not the authors'
to license. `data/` and `results/` derive from the AI Hub dataset and stay under
the AI Hub usage policy; they are published so the paper's results can be
checked, and reuse states that the data are a result of a project of the
National Information Society Agency, as that policy requires.
`paper/ieeeconf.cls` is the IEEE conference class under the Perl Artistic
License that its header states. The rest of `paper/` and this README are the
authors' writing, all rights reserved.
