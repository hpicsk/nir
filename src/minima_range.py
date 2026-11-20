"""The range of each frame minimum that leaves the study unchanged.

MIN_VAL_FRAMES and MIN_TRAIN_FRAMES are CHOSEN (config.py ledger), so
sec:protocols justifies them by a sensitivity check rather than a source:
each minimum is varied with the other held at its committed value, and the
range kept is the one that selects the same held-out site and scores the same
classes.

  - Site selection. splits_blocked.py assign() is rerun on the committed sites
    of data/processed/splits_blocked.json for every value of one minimum.
  - Classes scored. analyze.py and analyze_folds.py are rerun with their
    reports written to a temporary directory, and every frame count that
    evaluable() compares with MIN_VAL_FRAMES is collected. A validation
    minimum scores the same classes as the committed one when no collected
    count lies between the two.

MIN_TRAIN_FRAMES enters only the site selection. Writes
results/minima_range.json, which build_parameters.py reads.
"""
import contextlib
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from phases import splits_blocked as S  # noqa: E402

OUT_JSON = os.path.join(C.ROOT, "results", "minima_range.json")


def selected(sites, min_val, min_train):
    """The site assignment that assign() returns under these minima, or None
    where no permutation satisfies them."""
    saved = C.MIN_VAL_FRAMES, C.MIN_TRAIN_FRAMES
    C.MIN_VAL_FRAMES, C.MIN_TRAIN_FRAMES = min_val, min_train
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return S.assign(sites)
    except AssertionError:      # assign() asserts that some permutation passed
        return None
    finally:
        C.MIN_VAL_FRAMES, C.MIN_TRAIN_FRAMES = saved


def same_range(committed, keep, top):
    """The run of integers around `committed`, within [0, top], on which
    keep() holds."""
    lo = committed
    while lo > 0 and keep(lo - 1):
        lo -= 1
    hi = committed
    while hi < top and keep(hi + 1):
        hi += 1
    return lo, hi


def evaluation_counts():
    """Every frame count compared with MIN_VAL_FRAMES by the two analyzers."""
    import analyze as A
    import analyze_folds as F
    with tempfile.TemporaryDirectory() as tmp:
        A.OUT_MD = os.path.join(tmp, "summary.md")
        F.OUT_MD = os.path.join(tmp, "folds.md")
        with contextlib.redirect_stdout(io.StringIO()):
            A.main()
            F.main()
    return A.EVAL_COUNTS


def main():
    with open(C.SPLITS_JSON) as f:
        sites = json.load(f)["sites"]
    committed = selected(sites, C.MIN_VAL_FRAMES, C.MIN_TRAIN_FRAMES)
    assert committed is not None, (
        "the committed minima select no site assignment on the committed sites")
    top = max(n for s in sites for n in s["class_frames"].values())
    site_val = same_range(
        C.MIN_VAL_FRAMES,
        lambda v: selected(sites, v, C.MIN_TRAIN_FRAMES) == committed, top)
    site_train = same_range(
        C.MIN_TRAIN_FRAMES,
        lambda t: selected(sites, C.MIN_VAL_FRAMES, t) == committed, top)

    counts = evaluation_counts()
    below = [n for n in counts if n < C.MIN_VAL_FRAMES]
    above = [n for n in counts if n >= C.MIN_VAL_FRAMES]
    scored_val = (max(below) + 1 if below else 0,
                  min(above) if above else top)

    out = {
        "min_val_frames": C.MIN_VAL_FRAMES,
        "min_train_frames": C.MIN_TRAIN_FRAMES,
        "site_selection_val": list(site_val),
        "site_selection_train": list(site_train),
        "classes_scored_val": list(scored_val),
        "n_counts_compared": len(counts),
        "val": [max(site_val[0], scored_val[0]),
                min(site_val[1], scored_val[1])],
        "train": list(site_train),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(json.dumps(out, indent=1))
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
