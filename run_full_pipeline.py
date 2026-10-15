"""Regenerate every number, table and figure the manuscript prints.

The stages here recompute from the committed artifacts alone, which is what
the availability statement promises a reader: the analyzers read the per-run
metrics and per-frame accuracies under results/runs/, the generators write
paper/, and the claim tests re-derive every manuscript number from the same
files. No stage here opens the imagery.

The stages that do need it - the site-blocked split, the frame index, the
240 training runs, and the two probes that score models - are upstream, and
the README carries their sequence. Their outputs are committed, so this
script checks that all of them are present and stops naming what is missing
rather than regenerating a paper from part of a run.

Usage: python run_full_pipeline.py
"""
import os
import runpy
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import config as C  # noqa: E402

# Each stage with the artifact it writes, in dependency order: the geometry
# probe and the two analyzers, then the three generators that read them.
STAGES = [
    ("phases.geometry_probe", "data/processed/geometry_probe.json"),
    ("analyze", "results/summary.md"),
    ("analyze_folds", "results/folds.md"),
    ("build_parameters", "paper/params.tex"),
    ("build_tables", "paper/table_perclass_iou.tex"),
    ("build_figures", "paper/fig_leakage_isolation.pdf"),
]

# What the dataset stages leave behind. Derived from the run matrix rather
# than counted, so a queue that finished the wrong 240 runs is missing runs.
UPSTREAM = ["data/processed/index.json",
            "data/processed/splits_blocked.json",
            "results/leakage_isolation.json",
            "results/camera_offset.json"]


def missing():
    absent = [os.path.join("results", "runs", r["run_id"], "metrics.json")
              for r in C.RUN_MATRIX]
    return ([p for p in absent if not os.path.exists(os.path.join(ROOT, p))]
            + [p for p in UPSTREAM
               if not os.path.exists(os.path.join(ROOT, p))])


def main():
    absent = missing()
    if absent:
        raise SystemExit(
            f"{len(absent)} of the artifacts this chain reads are absent, "
            f"starting with {absent[0]}. Each comes from a stage that opens "
            "the imagery, which this repository does not redistribute. Build "
            "the split and the index, run the training queue, then the two "
            "probes; the README carries the sequence.")
    for module, writes in STAGES:
        print(f"== {module} -> {writes}", flush=True)
        runpy.run_module(module, run_name="__main__")
    print("== test_assertions.py", flush=True)
    sys.exit(subprocess.run(
        [sys.executable, "-m", "pytest", "test_assertions.py", "-q"],
        cwd=ROOT).returncode)


if __name__ == "__main__":
    main()
