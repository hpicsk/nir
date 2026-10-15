"""Build the full-study scene-matched dataset.

Differences from the preliminary build (src/preliminary/build_index.py):
  - FRAME_STRIDE 5 (preliminary stride 15): ~3x frames.
  - EVERY frame carries BOTH annotation sets: mask_rgb (RGB annotators)
    and mask_nir (NIR annotators), enabling own-label, cross-label and
    label-intersection evaluation (endogeneity bounds).
  - Per-frame per-class GT pixel counts for both label sets are stored in
    the index. This fixes the preliminary build's disclosed paired-test caveat:
    frame inclusion can condition on GT presence (pre-treatment) instead
    of prediction-dependent union>0 (mildly post-treatment).
  - Each record carries BOTH split labels: the official split it came
    from and the site-blocked split from data/processed/splits_blocked.json; training
    selects by protocol at run time, so the frame store exists once.

Output:
  data/study/frames/{osplit}_{env}/{stem}_{RGB,NIR}.jpg      (960x544)
  data/study/frames/{osplit}_{env}/{stem}_{mRGB,mNIR}.png    (class-index)
  data/processed/index.json         tracked: one record per frame (splits, px counts,
                        and the four frame-store paths, held relative to
                        data/study so the index travels with the repository)
  data/processed/index_stats.json   tracked summary: per protocol/split/class frame
                        counts and pixel shares (floor assertions +
                        sampler documentation read this)

All parameters: config.py. Taxonomy: 8 shared classes + background
(paddy_water dropped - 0 train instances).
"""
import glob
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import cv2

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

CLASS_ID = {c: i for i, c in enumerate(C.CLASSES)}
SPLIT_LABEL_ROOTS = {"train": "Tel_Polygon", "val": "VL_Polygon"}


def rasterize(objects, w0, h0):
    """Class-index mask at (H, W); ascending z_order so foreground
    overwrites background (identical to src/preliminary/build_index.py)."""
    mask = np.zeros((C.H, C.W), np.uint8)
    sx, sy = C.W / w0, C.H / h0
    objs = [o for o in objects if o["label"] in CLASS_ID]
    objs.sort(key=lambda o: int(o.get("z_order", 0) or 0))
    for o in objs:
        cid = CLASS_ID[o["label"]]
        for poly in o["position"]:
            pts = np.array(poly, np.float64).reshape(-1, 2)
            pts[:, 0] *= sx
            pts[:, 1] *= sy
            cv2.fillPoly(mask, [pts.astype(np.int32)], cid)
    return mask


def process_one(job):
    (osplit, bsplit, env, seq, frame, meta_rgb, meta_nir) = job
    img_root = os.path.join(C.IMAGES_ROOT, osplit, env, seq)
    d = os.path.join(C.STUDY_DATA, "frames", f"{osplit}_{env}")
    stem = f"{seq}_f{frame:06d}"
    rec = {"env": env, "seq": seq, "frame": frame,
           "osplit": osplit, "bsplit": bsplit}
    for mod, meta, img_key, mask_key, msuf in (
            ("RGB", meta_rgb, "img_rgb", "mask_rgb", "mRGB"),
            ("NIR", meta_nir, "img_nir", "mask_nir", "mNIR")):
        img = cv2.imread(os.path.join(img_root, mod, meta["name"]),
                         cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        img = cv2.resize(img, (C.W, C.H), interpolation=cv2.INTER_AREA)
        ip = os.path.join(d, f"{stem}_{mod}.jpg")
        cv2.imwrite(ip, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        mask = rasterize(meta["objects"], meta["width"], meta["height"])
        mp = os.path.join(d, f"{stem}_{msuf}.png")
        cv2.imwrite(mp, mask)
        # stored relative to STUDY_DATA, resolved by config.load_index(): the
        # index is committed, so an absolute path would be one machine's only
        rec[img_key] = os.path.relpath(ip, C.STUDY_DATA)
        rec[mask_key] = os.path.relpath(mp, C.STUDY_DATA)
        rec["px_" + mod.lower()] = np.bincount(
            mask.ravel(), minlength=C.N_CLASSES).tolist()
    return rec


def main():
    with open(C.SPLITS_JSON) as f:
        per_sequence = json.load(f)["per_sequence"]

    jobs = []
    n_seq = 0
    for osplit, root in SPLIT_LABEL_ROOTS.items():
        for env in C.ENVS:
            for seq_dir in sorted(glob.glob(
                    os.path.join(C.LABELS_ROOT, root, env, "sequence*"))):
                seq = os.path.basename(seq_dir)
                key = f"{osplit}/{env}/{seq}"
                if key not in per_sequence:  # missing-modality sequences
                    continue
                bsplit = per_sequence[key]
                per_mod = {}
                for mod in ("RGB", "NIR"):
                    js = glob.glob(os.path.join(seq_dir, mod, "*.json"))
                    d = json.load(open(js[0]))
                    per_mod[mod] = {
                        int(i["name"].rsplit("_", 1)[1].split(".")[0]): i
                        for i in d["images"]}
                shared = sorted(set(per_mod["RGB"]) & set(per_mod["NIR"]))
                picked = [i for i in shared if i % C.FRAME_STRIDE == 0]
                os.makedirs(os.path.join(C.STUDY_DATA, "frames",
                                         f"{osplit}_{env}"), exist_ok=True)
                for i in picked:
                    jobs.append((osplit, bsplit, env, seq, i,
                                 per_mod["RGB"][i], per_mod["NIR"][i]))
                n_seq += 1

    print(f"{n_seq} sequences, {len(jobs)} frame jobs", flush=True)
    with ProcessPoolExecutor(max_workers=16) as ex:
        recs = [r for r in ex.map(process_one, jobs, chunksize=8) if r]
    n_bad = len(jobs) - len(recs)
    print(f"built {len(recs)} frames ({n_bad} dropped: unreadable image)")

    with open(C.INDEX_JSON, "w") as f:
        json.dump(recs, f)

    # tracked stats: frame counts + pixel shares per protocol/split/class
    stats = {"n_frames": len(recs), "n_dropped_unreadable": n_bad,
             "frame_stride": C.FRAME_STRIDE, "protocols": {}}
    for proto, field in (("official", "osplit"), ("blocked", "bsplit")):
        pstats = {}
        for split in ("train", "val"):
            sel = [r for r in recs if r[field] == split]
            total_px = C.W * C.H * len(sel)
            cls = {}
            for ci, cname in enumerate(C.CLASSES):
                if ci == 0:
                    continue
                cls[cname] = {
                    "frames_rgb": sum(1 for r in sel if r["px_rgb"][ci]),
                    "frames_nir": sum(1 for r in sel if r["px_nir"][ci]),
                    "frames_union": sum(1 for r in sel
                                        if r["px_rgb"][ci] or r["px_nir"][ci]),
                    "px_share_rgb": sum(r["px_rgb"][ci] for r in sel) / total_px,
                    "px_share_nir": sum(r["px_nir"][ci] for r in sel) / total_px,
                }
            pstats[split] = {"n_frames": len(sel),
                             "n_paddy": sum(1 for r in sel if r["env"] == "Paddy"),
                             "n_field": sum(1 for r in sel if r["env"] == "Field"),
                             "classes": cls}
        stats["protocols"][proto] = pstats
    with open(C.INDEX_STATS, "w") as f:
        json.dump(stats, f, indent=1)

    for proto in ("blocked", "official"):
        p = stats["protocols"][proto]
        print(f"\n[{proto}] train {p['train']['n_frames']} "
              f"(P {p['train']['n_paddy']} / F {p['train']['n_field']}) | "
              f"val {p['val']['n_frames']} "
              f"(P {p['val']['n_paddy']} / F {p['val']['n_field']})")
        print(f"{'class':24s} {'tr-union':>9s} {'va-union':>9s} {'tr-px%':>8s}")
        for cname, d in p["train"]["classes"].items():
            va = p["val"]["classes"][cname]
            print(f"{cname:24s} {d['frames_union']:9d} {va['frames_union']:9d} "
                  f"{100 * d['px_share_rgb']:8.3f}")
    print(f"\nwrote {C.INDEX_JSON}\nwrote {C.INDEX_STATS}")


if __name__ == "__main__":
    main()
