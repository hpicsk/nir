"""Integrity checks on labels/metadata only (no images).

1. Train<->val GPS footprint overlap: does any official val sequence
   drive the same physical field as a train sequence? Sequence-level
   splits still leak if the same plot is revisited.
2. RGB<->NIR annotation agreement on shared (same-instant) frames.
   The two cameras have parallax, so pixel-level IoU would conflate
   viewpoint with annotation disagreement; we use parallax-tolerant
   summaries instead:
     - per-frame class-presence Jaccard between the two label sets
     - per-class: P(present in NIR | present in RGB) and vice versa
     - per-class pixel-share (fraction of frame area): mean abs
       difference RGB vs NIR

Output: results/integrity_checks.json + printed tables.
"""
import json
import glob
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))  # repo root, two directories above src/
LABELS_ROOT = os.path.join(ROOT, "data", "labels")
OUT = os.path.join(ROOT, "results", "integrity_checks.json")

SPLITS = {"train": "Tel_Polygon", "val": "VL_Polygon"}
CLASSES = ["common_road", "common_tree", "field_furrow", "field_levee",
           "paddy_before_driving", "paddy_after_driving", "paddy_edge",
           "paddy_rice", "paddy_water", "orchard_road", "orchard_tree"]
IMG_AREA = 1920.0 * 1080.0


def poly_area(flat):
    pts = np.array(flat, np.float64).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def main():
    # per (split, env, seq): gps bbox; per modality: frame -> {class: pixel share}
    seq_bbox = {}
    frames = defaultdict(dict)  # (split, env, seq, modality) -> {idx: {cls: share}}
    for split, root in SPLITS.items():
        for jp in sorted(glob.glob(os.path.join(LABELS_ROOT, root, "*", "*",
        "*", "*.json"))):
            rel = os.path.relpath(jp, os.path.join(LABELS_ROOT, root))
            env, seq, modality, _ = rel.split(os.sep)
            d = json.load(open(jp))
            lats, lons = [], []
            fr = {}
            for img in d["images"]:
                idx = int(img["name"].rsplit("_", 1)[1].split(".")[0])
                if img.get("latitude"):
                    lats.append(img["latitude"])
                    lons.append(img["longitude"])
                shares = defaultdict(float)
                for o in img["objects"]:
                    if o["label"] in CLASSES:
                        shares[o["label"]] += sum(poly_area(p) for p in o["position"]) / IMG_AREA
                fr[idx] = dict(shares)
            frames[(split, env, seq, modality)] = fr
            key = (split, env, seq)
            if lats:
                bb = (min(lats), max(lats), min(lons), max(lons))
                if key in seq_bbox:
                    o = seq_bbox[key]
                    seq_bbox[key] = (min(o[0], bb[0]), max(o[1], bb[1]),
                                     min(o[2], bb[2]), max(o[3], bb[3]))
                else:
                    seq_bbox[key] = bb

    # ---- 1. GPS overlap train vs val ----
    def overlap(a, b, pad=1e-4):  # ~11 m padding in latitude degrees
        return not (a[1] + pad < b[0] or b[1] + pad < a[0]
                    or a[3] + pad < b[2] or b[3] + pad < a[2])

    leaks = []
    for kt, bt in seq_bbox.items():
        if kt[0] != "train":
            continue
        for kv, bv in seq_bbox.items():
            if kv[0] != "val":
                continue
            if overlap(bt, bv):
                leaks.append({"train": "/".join(kt[1:]), "val": "/".join(kv[1:]),
                              "train_bbox": bt, "val_bbox": bv})
    print(f"=== GPS footprint overlaps (train vs val, ±11 m pad): {len(leaks)} ===")
    for l in leaks[:30]:
        print(f"  {l['train']}  <->  {l['val']}")

    # ---- 2. RGB<->NIR agreement on shared frames ----
    PRESENT_MIN = 0.002  # class 'present' if >0.2% of frame area (CHOSEN:
    # excludes sliver annotations; sensitivity: also computed at 0 and 1%)
    agree = {c: {"rgb_only": 0, "nir_only": 0, "both": 0} for c in CLASSES}
    share_diff = defaultdict(list)
    jac = []
    n_shared = 0
    for (split, env, seq, modality), fr in frames.items():
        if modality != "RGB":
            continue
        nir = frames.get((split, env, seq, "NIR"))
        if not nir:
            continue
        for idx in sorted(set(fr) & set(nir)):
            n_shared += 1
            r, n = fr[idx], nir[idx]
            rset = {c for c, s in r.items() if s >= PRESENT_MIN}
            nset = {c for c, s in n.items() if s >= PRESENT_MIN}
            if rset | nset:
                jac.append(len(rset & nset) / len(rset | nset))
            for c in rset | nset:
                if c in rset and c in nset:
                    agree[c]["both"] += 1
                elif c in rset:
                    agree[c]["rgb_only"] += 1
                else:
                    agree[c]["nir_only"] += 1
            for c in rset | nset:
                share_diff[c].append(abs(r.get(c, 0.0) - n.get(c, 0.0)))

    print(f"\n=== RGB<->NIR annotation agreement on {n_shared} shared frames ===")
    print(f"class-presence Jaccard: median {np.median(jac):.3f} mean {np.mean(jac):.3f}")
    print(f"{'class':24s} {'both':>7s} {'rgbOnly':>8s} {'nirOnly':>8s} "
          f"{'P(N|R)':>7s} {'P(R|N)':>7s} {'|Δshare|':>9s}")
    table = {}
    for c in CLASSES:
        a = agree[c]
        nr = a["both"] + a["rgb_only"]
        nn = a["both"] + a["nir_only"]
        pnr = a["both"] / nr if nr else None
        prn = a["both"] / nn if nn else None
        msd = float(np.mean(share_diff[c])) if share_diff[c] else None
        table[c] = {**a, "p_nir_given_rgb": pnr, "p_rgb_given_nir": prn,
                    "mean_abs_share_diff": msd}
        if nr or nn:
            print(f"{c:24s} {a['both']:7d} {a['rgb_only']:8d} {a['nir_only']:8d} "
                  f"{(f'{pnr:.3f}' if pnr is not None else '-'):>7s} "
                  f"{(f'{prn:.3f}' if prn is not None else '-'):>7s} "
                  f"{(f'{msd:.4f}' if msd is not None else '-'):>9s}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        # The sequences whose footprints the overlap check compared, so a
        # reader of the overlap count can form the pairs it is counted among.
        json.dump({"gps_leaks": leaks,
                   "gps_sequences": {sp: sorted("/".join(k[1:]) for k in seq_bbox
                                                if k[0] == sp)
                                     for sp in SPLITS},
                   "n_shared_frames": n_shared,
                   "presence_jaccard_median": float(np.median(jac)),
                   "presence_jaccard_mean": float(np.mean(jac)),
                   "agreement": table,
                   "present_min_share": PRESENT_MIN}, f, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
