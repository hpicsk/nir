"""Build the GPS-clustered site-blocked train/val split.

The dataset integrity check (integrity_checks.py) found 276 padded-bbox overlaps
between official train and val sequences - the same physical plot is
driven in both splits, so the official protocol leaks sites. This script
re-splits at the SITE level: sequences whose GPS tracks come within
RADIUS_M (~11 m, the merge radius) form one site, and a site
lands wholly in train or wholly in val.

Site criterion - exact TRACK distance, not bbox intersection: two
sequences belong to one site iff the minimum point-to-point distance
between their full GPS tracks is <= RADIUS_M (KD-tree; padded-bbox
prefilter only prunes pairs that provably cannot be that close). Bbox
intersection is NOT usable as the criterion: drive trajectories are long,
their axis-aligned bboxes intersect without the tracks ever approaching,
and single-linkage chaining then merged 68 of 94 sequences into one
"site" (observed 2026-07-12, before any training run; see git history).
Bbox overlap at PAD_DEG remains (a) the official-split leak
detector - a deliberate over-approximation, fine for flagging - and
(b) a descriptive cross-split contact count disclosed in the output.

Method:
  1. Universe = every (official_split, env, seq) in Paddy/Field with BOTH
     RGB and NIR polygon labels (the scene-matched universe; sequences
     missing a modality cannot serve any arm).
  2. Sequence GPS track = all (lat, lon) over ALL frames of BOTH modality
     JSONs (full track: more conservative than the strided subset).
  3. Union-find over min-track-distance <= RADIUS_M (global across envs
     and official splits) -> connected components = sites.
  4. Deterministic assignment (RandomState(SPLIT_RNG_SEED), committed
     before this ran): shuffle site order, greedily fill val to
     VAL_TARGET of stride-FRAME_STRIDE shared frames; among the
     N_SPLIT_TRIES permutations keep those satisfying
       - val share within VAL_TARGET +/- VAL_TOL,
       - every class present (RGB-or-NIR polygon labels) in
         >= MIN_VAL_FRAMES val frames and >= MIN_TRAIN_FRAMES train
         frames (strided),
     and pick the one closest to VAL_TARGET (ties: lowest try index).
  5. Assert the operative invariant - no STRIDED (i.e. actually used)
     val frame's GPS within RADIUS_M of any strided train frame's - and
     write data/processed/splits_blocked.json, including the strided per-sequence
     tracks so test_assertions.py re-verifies the invariant from inputs
     rather than ratifying a stored verdict.

All parameters: config.py (the criterion record).
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

SPLIT_LABEL_ROOTS = {"train": "Tel_Polygon", "val": "VL_Polygon"}
# CHOSEN: the latitude and longitude bounds of the Republic of Korea, a guard
# that rejects a malformed GPS fix before it enters a track distance.
KOREA_LAT = (33.0, 39.0)
KOREA_LON = (124.0, 132.0)   # CHOSEN with KOREA_LAT, above


def bbox_overlap(a, b, pad=C.PAD_DEG):
    """Padded bbox overlap, identical to integrity_checks.py (bbox = minlat, maxlat,
    minlon, maxlon). Used for descriptive contact counts only."""
    return not (a[1] + pad < b[0] or b[1] + pad < a[0]
                or a[3] + pad < b[2] or b[3] + pad < a[2])


def to_meters(latlon, lat0):
    """Equirectangular local projection (config ledger)."""
    xy = np.empty_like(latlon)
    xy[:, 0] = (latlon[:, 1]) * C.M_PER_DEG_LON * np.cos(np.radians(lat0))
    xy[:, 1] = (latlon[:, 0]) * C.M_PER_DEG_LAT
    return xy


def min_track_dist_le(xy_a, xy_b, radius):
    """True iff min point distance between tracks <= radius."""
    d, _ = cKDTree(xy_a).query(xy_b, k=1, distance_upper_bound=radius * 1.001)
    return bool((d <= radius).any())


def load_universe():
    """One node per (official_split, env, seq) with both modalities:
    full GPS track, merged bbox, strided shared frames + their GPS,
    per-class strided frame-presence counts (RGB-or-NIR labels)."""
    nodes = []
    for osplit, root in SPLIT_LABEL_ROOTS.items():
        for env in C.ENVS:
            for seq_dir in sorted(glob.glob(
                    os.path.join(C.LABELS_ROOT, root, env, "sequence*"))):
                seq = os.path.basename(seq_dir)
                per_mod = {}
                track = []
                frame_gps = {}
                for mod in ("RGB", "NIR"):
                    js = glob.glob(os.path.join(seq_dir, mod, "*.json"))
                    if not js:
                        continue
                    d = json.load(open(js[0]))
                    fr = {}
                    for img in d["images"]:
                        idx = int(img["name"].rsplit("_", 1)[1].split(".")[0])
                        fr[idx] = {o["label"] for o in img["objects"]}
                        if img.get("latitude"):
                            track.append((img["latitude"], img["longitude"]))
                            frame_gps[idx] = (img["latitude"],
                                              img["longitude"])
                    per_mod[mod] = fr
                if len(per_mod) < 2:
                    print(f"[skip] {osplit}/{env}/{seq}: missing a modality")
                    continue
                track = np.array(track)
                assert len(track), f"{osplit}/{env}/{seq}: no GPS at all"
                bb = (track[:, 0].min(), track[:, 0].max(),
                      track[:, 1].min(), track[:, 1].max())
                assert KOREA_LAT[0] < bb[0] and bb[1] < KOREA_LAT[1] \
                    and KOREA_LON[0] < bb[2] and bb[3] < KOREA_LON[1], \
                    f"{osplit}/{env}/{seq}: GPS outside Korea range: {bb}"
                shared = sorted(set(per_mod["RGB"]) & set(per_mod["NIR"]))
                picked = [i for i in shared if i % C.FRAME_STRIDE == 0]
                class_frames = defaultdict(int)
                for i in picked:
                    for cls in (per_mod["RGB"][i] | per_mod["NIR"][i]):
                        if cls in C.CLASSES:
                            class_frames[cls] += 1
                strided_track = [
                    [round(frame_gps[i][0], 7), round(frame_gps[i][1], 7)]
                    for i in picked if i in frame_gps]
                nodes.append({"osplit": osplit, "env": env, "seq": seq,
                              "bbox": bb, "track": track,
                              "strided_track": strided_track,
                              "n_frames": len(picked),
                              "class_frames": dict(class_frames)})
    return nodes


def cluster(nodes):
    lat0 = float(np.mean([n["track"][:, 0].mean() for n in nodes]))
    xys = [to_meters(n["track"], lat0) for n in nodes]
    parent = list(range(len(nodes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # prefilter: bbox gap along either axis > RADIUS_M already implies
    # min point distance > RADIUS_M (sound pruning, no false negatives)
    mins = [xy.min(0) for xy in xys]
    maxs = [xy.max(0) for xy in xys]
    n_exact = 0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            gap = np.maximum(np.maximum(mins[i] - maxs[j],
                                        mins[j] - maxs[i]), 0.0)
            if gap.max() > C.RADIUS_M:
                continue
            n_exact += 1
            if min_track_dist_le(xys[i], xys[j], C.RADIUS_M):
                parent[find(i)] = find(j)
    print(f"cluster: {n_exact} exact KD-tree pair checks")

    comps = defaultdict(list)
    for i in range(len(nodes)):
        comps[find(i)].append(i)
    ordered = sorted(comps.values(),
                     key=lambda m: min((nodes[i]["osplit"], nodes[i]["env"],
                                        nodes[i]["seq"]) for i in m))
    sites = []
    for k, members in enumerate(ordered):
        members = sorted(members, key=lambda i: (
            nodes[i]["osplit"], nodes[i]["env"], nodes[i]["seq"]))
        bbs = [nodes[i]["bbox"] for i in members]
        cf = defaultdict(int)
        for i in members:
            for cls, n in nodes[i]["class_frames"].items():
                cf[cls] += n
        sites.append({
            "site_id": f"site{k:03d}",
            "members": [f"{nodes[i]['osplit']}/{nodes[i]['env']}/"
                        f"{nodes[i]['seq']}" for i in members],
            "member_idx": members,
            "bbox": [float(min(b[0] for b in bbs)),
                     float(max(b[1] for b in bbs)),
                     float(min(b[2] for b in bbs)),
                     float(max(b[3] for b in bbs))],
            "n_frames": sum(nodes[i]["n_frames"] for i in members),
            "class_frames": dict(cf),
        })
    return sites, lat0


def assign(sites):
    total = sum(s["n_frames"] for s in sites)
    rng = np.random.RandomState(C.SPLIT_RNG_SEED)
    best = None
    for t in range(C.N_SPLIT_TRIES):
        order = rng.permutation(len(sites))
        val_ids, val_frames = set(), 0
        for si in order:
            if val_frames < C.VAL_TARGET * total:
                val_ids.add(sites[si]["site_id"])
                val_frames += sites[si]["n_frames"]
        share = val_frames / total
        if abs(share - C.VAL_TARGET) > C.VAL_TOL:
            continue
        ok = True
        for cls in C.CLASSES[1:]:
            if cls in C.FLOOR_EXEMPT:
                continue
            nv = sum(s["class_frames"].get(cls, 0) for s in sites
                     if s["site_id"] in val_ids)
            nt = sum(s["class_frames"].get(cls, 0) for s in sites
                     if s["site_id"] not in val_ids)
            if nv < C.MIN_VAL_FRAMES or nt < C.MIN_TRAIN_FRAMES:
                ok = False
                break
        if ok and (best is None or abs(share - C.VAL_TARGET) < best[0]):
            best = (abs(share - C.VAL_TARGET), t, val_ids, share)
    assert best is not None, (
        f"No site permutation in {C.N_SPLIT_TRIES} tries satisfies the "
        f"committed constraints (share {C.VAL_TARGET}+/-{C.VAL_TOL}, "
        f"floors {C.MIN_VAL_FRAMES}/{C.MIN_TRAIN_FRAMES}) - the "
        "constraints, not the data, must be re-examined and re-committed.")
    _, t, val_ids, share = best
    print(f"assignment: try {t}, val share {share:.4f}")
    return {s["site_id"]: ("val" if s["site_id"] in val_ids else "train")
            for s in sites}


def main():
    nodes = load_universe()
    print(f"universe: {len(nodes)} sequences, "
          f"{sum(n['n_frames'] for n in nodes)} strided shared frames")
    sites, lat0 = cluster(nodes)
    sizes = sorted((s["n_frames"] for s in sites), reverse=True)
    print(f"sites: {len(sites)} | largest {sizes[:5]} frames | "
          f"largest membership {max(len(s['members']) for s in sites)} seqs")
    assignment = assign(sites)

    # operative invariant: no strided val frame within RADIUS_M of a
    # strided train frame (the frames the study actually uses)
    def strided_xy(split):
        pts = [p for s in sites if assignment[s["site_id"]] == split
               for i in s["member_idx"] for p in nodes[i]["strided_track"]]
        return to_meters(np.array(pts), lat0)

    tr_xy, va_xy = strided_xy("train"), strided_xy("val")
    d, _ = cKDTree(tr_xy).query(va_xy, k=1)
    min_cross = float(d.min())
    assert min_cross > C.RADIUS_M, (
        f"blocked split leaks: min cross-split track distance "
        f"{min_cross:.2f} m <= {C.RADIUS_M} m")
    # descriptive: residual bbox-level contacts (disclosed, not asserted)
    bbox_contacts = sum(
        1 for a in sites for b in sites
        if assignment[a["site_id"]] == "train"
        and assignment[b["site_id"]] == "val"
        and bbox_overlap(a["bbox"], b["bbox"]))

    per_sequence = {}
    tracks = {}
    env_frames = {sp: defaultdict(int) for sp in ("train", "val")}
    for s in sites:
        sp = assignment[s["site_id"]]
        for i, m in zip(s["member_idx"], s["members"]):
            per_sequence[m] = sp
            tracks[m] = nodes[i]["strided_track"]
            env_frames[sp][nodes[i]["env"]] += nodes[i]["n_frames"]
        del s["member_idx"]  # internal only

    stats = {"n_sites": len(sites),
             "n_sequences": len(nodes),
             "n_frames_total": sum(s["n_frames"] for s in sites),
             "n_frames_val": sum(s["n_frames"] for s in sites
                                 if assignment[s["site_id"]] == "val"),
             "env_frames": {sp: dict(d) for sp, d in env_frames.items()},
             "min_cross_split_track_m": min_cross,
             "bbox_contacts_across_split": bbox_contacts,
             "class_frames": {}}
    stats["val_share"] = stats["n_frames_val"] / stats["n_frames_total"]
    for cls in C.CLASSES[1:]:
        stats["class_frames"][cls] = {
            sp: sum(s["class_frames"].get(cls, 0) for s in sites
                    if assignment[s["site_id"]] == sp)
            for sp in ("train", "val")}

    out = {"params": {"radius_m": C.RADIUS_M, "pad_deg": C.PAD_DEG,
                      "lat0": lat0,
                      "m_per_deg_lat": C.M_PER_DEG_LAT,
                      "m_per_deg_lon": C.M_PER_DEG_LON,
                      "val_target": C.VAL_TARGET, "val_tol": C.VAL_TOL,
                      "min_val_frames": C.MIN_VAL_FRAMES,
                      "min_train_frames": C.MIN_TRAIN_FRAMES,
                      "split_rng_seed": C.SPLIT_RNG_SEED,
                      "frame_stride": C.FRAME_STRIDE},
           "sites": sites, "assignment": assignment,
           "per_sequence": per_sequence,
           "strided_tracks": tracks, "stats": stats}
    with open(C.SPLITS_JSON, "w") as f:
        json.dump(out, f, indent=1)

    print(f"\nblocked split: {stats['n_frames_val']}/"
          f"{stats['n_frames_total']} strided frames in val "
          f"({stats['val_share']:.3f})")
    print(f"min cross-split track distance: {min_cross:.1f} m | "
          f"residual bbox contacts: {bbox_contacts}")
    print(f"{'class':24s} {'train':>7s} {'val':>6s}")
    for cls, dd in stats["class_frames"].items():
        print(f"{cls:24s} {dd['train']:7d} {dd['val']:6d}")
    print(f"\nwrote {C.SPLITS_JSON}")


if __name__ == "__main__":
    main()
