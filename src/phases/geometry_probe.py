"""Why the phase-E control failed, and why an env-stratified re-split cannot
fix it (config.py ledger, 2026-07-24 amendment).

Read-only: measures the GPS geometry and the environment composition of the
committed split. Writes no split, trains nothing, and touches no result.

Two questions, both answered from composition alone - never from a trained
model's output, so the answers may inform a criterion without
spending the statistics a later gate is read against:

  Q1 WHERE IS THE ENVIRONMENT IMBALANCE?  Phase-E gate E3 failed on fold
     site000 (control class collapsed fold-wide). Per-site Field/Paddy
     frame counts locate the cause: if one site holds most of one
     environment, holding it out is out-of-domain transfer, not a
     same-domain held-out site.

  Q2 COULD RE-CLUSTERING REBALANCE THE FOLDS?  Sites are the connected
     components of "min track distance <= RADIUS_M" (splits_blocked.py). Cutting the
     single-linkage tree at K clusters gives the exact minimum inter-cluster
     separation a K-fold split would have. If separation collapses as soon
     as K exceeds the committed site count, no re-clustering can buy
     balance without reintroducing the leakage this study measures.

Method:
  1. Sequence tracks: the STRIDED per-sequence GPS tracks stored in
     data/processed/splits_blocked.json - the frames training actually uses, and the
     same points splits_blocked.py asserts its separation invariant on.
  2. Exact min point-to-point distance between every sequence pair (KD-tree,
     equirectangular meters via config M_PER_DEG_*), giving a 94x94
     dissimilarity whose single-linkage components at height RADIUS_M are
     the committed sites - the property step 4 checks.
  3. Env/frame composition per site from the built index (src/phases/build_index.py).
  4. Assert the reconstruction reproduces the committed sites, so the
     dendrogram is verified against the split on disk rather than trusted.

Output: data/processed/geometry_probe.json (numbers quoted by the config ledger and
re-checked in test_assertions.py).
"""
import json
import os
import sys
from collections import Counter

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import cKDTree
from scipy.spatial.distance import squareform

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

# Cut the tree from 2 up to this many clusters. Purely descriptive: the
# question is where separation collapses, and it collapses immediately past
# the committed site count, so a wide sweep only documents that it stays
# collapsed. No result depends on the upper end.
MAX_K = 14   # CHOSEN: descriptive sweep bound (above); no result depends on it


def to_meters(latlon, lat0):
    """Equirectangular projection - identical to splits_blocked.py to_meters, so the
    distances here are the ones the split invariant was asserted on."""
    a = np.asarray(latlon, float)
    xy = np.empty_like(a)
    xy[:, 0] = a[:, 1] * C.M_PER_DEG_LON * np.cos(np.radians(lat0))
    xy[:, 1] = a[:, 0] * C.M_PER_DEG_LAT
    return xy


def main():
    with open(C.SPLITS_JSON) as f:
        sp = json.load(f)
    tracks = sp["strided_tracks"]
    lat0 = sp["params"]["lat0"]
    keys = sorted(tracks)
    xy = {k: to_meters(tracks[k], lat0) for k in keys}
    trees = {k: cKDTree(xy[k]) for k in keys}

    n = len(keys)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = keys[i], keys[j]
            # query the smaller track against the larger tree; exact either way
            if len(xy[a]) <= len(xy[b]):
                m = trees[b].query(xy[a], k=1)[0].min()
            else:
                m = trees[a].query(xy[b], k=1)[0].min()
            d[i, j] = d[j, i] = m

    recs = C.load_index()
    key2site = {m: s["site_id"] for s in sp["sites"] for m in s["members"]}
    frames, env_frames = Counter(), Counter()
    for r in recs:
        k = f"{r['osplit']}/{r['env']}/{r['seq']}"
        frames[k] += 1
        env_frames[(key2site[k], r["env"])] += 1

    # --- Q1: environment composition per committed site ---
    sites = sorted({s["site_id"] for s in sp["sites"]})
    env_total = {e: sum(env_frames[(s, e)] for s in sites) for e in C.ENVS}
    composition = {
        e: {s: {"frames": env_frames[(s, e)],
                "share_of_env": env_frames[(s, e)] / env_total[e]}
            for s in sites}
        for e in C.ENVS}

    # --- Q2: separation vs number of clusters ---
    z = linkage(squareform(d, checks=False), method="single")
    nseq = np.array([frames[k] for k in keys])
    is_field = np.array([k.split("/")[1] == "Field" for k in keys])
    sweep = []
    for k in range(2, MAX_K + 1):
        lab = fcluster(z, k, criterion="maxclust")
        if lab.max() != k:            # ties can collapse a requested cut
            continue
        sep = min(d[np.ix_(lab == a, lab == b)].min()
                  for a in range(1, k + 1) for b in range(a + 1, k + 1))
        cl = []
        for c in range(1, k + 1):
            sel = lab == c
            f = int(nseq[sel].sum())
            cl.append({"frames": f,
                       "field_share": float(nseq[sel & is_field].sum() / f)})
        sweep.append({"k": k, "min_separation_m": float(sep),
                      "clusters": sorted(cl, key=lambda c: -c["frames"])})

    # --- step 4: the reconstruction must reproduce the committed sites ---
    lab_r = fcluster(z, C.RADIUS_M, criterion="distance")
    rebuilt = {}
    for k, c in zip(keys, lab_r):
        rebuilt.setdefault(int(c), set()).add(k)
    committed = {s["site_id"]: set(s["members"]) for s in sp["sites"]}
    assert sorted(map(sorted, rebuilt.values())) == \
        sorted(map(sorted, committed.values())), (
            "single-linkage at RADIUS_M does not reproduce the committed "
            "sites; the dendrogram below would not describe splits_blocked.py's split")

    committed_k = len(committed)
    at_committed = next(s for s in sweep if s["k"] == committed_k)
    first_past = next(s for s in sweep if s["k"] == committed_k + 1)

    out = {
        "n_sequences": n,
        "radius_m": C.RADIUS_M,
        "committed_n_sites": committed_k,
        "env_composition": composition,
        "separation_sweep": sweep,
        "separation_at_committed_k_m": at_committed["min_separation_m"],
        "separation_first_past_committed_k_m": first_past["min_separation_m"],
    }
    dst = os.path.join(C.PROCESSED, "geometry_probe.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)

    print(f"{n} sequences, {int(nseq.sum())} strided frames")
    print("\nQ1  environment concentration per site")
    for e in C.ENVS:
        row = "  ".join(f"{s}={composition[e][s]['frames']:5d} "
                        f"({composition[e][s]['share_of_env']:.1%})"
                        for s in sites)
        print(f"  {e:6s} {row}")
    print(f"\nQ2  separation vs cluster count (committed K={committed_k})")
    print(f"  {'K':>3} {'min_sep_m':>11}  field share per cluster")
    for s in sweep:
        mark = "  <- committed split" if s["k"] == committed_k else ""
        print(f"  {s['k']:>3} {s['min_separation_m']:>11.1f}  "
              + " ".join(f"{c['field_share']:.2f}" for c in s["clusters"])
              + mark)
    print(f"\nseparation collapses {at_committed['min_separation_m']:.1f} m "
          f"(K={committed_k}) -> {first_past['min_separation_m']:.1f} m "
          f"(K={committed_k + 1})")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
