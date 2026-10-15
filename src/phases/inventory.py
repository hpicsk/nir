"""Dataset inventory for the NIR-traversability study (sec. materials verification).

Walks every polygon label JSON (Training + Validation), tabulates
instances per class x environment x modality x split, image counts,
records each sequence's label file name (which carries its capture
conditions), and tests RGB<->NIR frame correspondence within each
sequence (same frame index -> same GPS/time?).

Output: results/inventory.json + printed summary tables.
"""
import json
import glob
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))  # repo root, two directories above src/
LABELS_ROOT = os.path.join(ROOT, "data", "labels")
RESULTS_DIR = os.path.join(ROOT, "results")

SPLITS = {
    "train": os.path.join(LABELS_ROOT, "Tel_Polygon"),
    "val": os.path.join(LABELS_ROOT, "VL_Polygon"),
}


def walk_polygon_jsons():
    for split, root in SPLITS.items():
        for path in sorted(glob.glob(os.path.join(root, "*", "*", "*", "*.json"))):
            rel = os.path.relpath(path, root)
            env, seq, modality, fname = rel.split(os.sep)
            yield split, env, seq, modality, path


def main():
    inst_counts = defaultdict(int)   # (split, env, modality, label) -> n instances
    img_counts = defaultdict(int)    # (split, env, modality) -> n images
    seq_index = defaultdict(dict)    # (split, env, seq) -> {modality: {frame_idx: (lat, lon, time)}}
    label_types = defaultdict(set)   # label -> annotation types seen
    seq_meta = {}

    for split, env, seq, modality, path in walk_polygon_jsons():
        with open(path) as f:
            d = json.load(f)
        frames = {}
        for img in d["images"]:
            img_counts[(split, env, modality)] += 1
            # frame index from file name suffix: ..._000022.jpg
            idx = int(img["name"].rsplit("_", 1)[1].split(".")[0])
            frames[idx] = (img.get("latitude"), img.get("longitude"), img.get("time"))
            for obj in img["objects"]:
                inst_counts[(split, env, modality, obj["label"])] += 1
                label_types[obj["label"]].add(obj["type"])
        seq_index[(split, env, seq)][modality] = frames
        seq_meta[(split, env, seq, modality)] = os.path.basename(path)

    # ---- RGB <-> NIR frame correspondence check ----
    corr = []
    for (split, env, seq), mods in sorted(seq_index.items()):
        has_rgb, has_nir = "RGB" in mods, "NIR" in mods
        row = {"split": split, "env": env, "seq": seq,
               "rgb_frames": len(mods.get("RGB", {})), "nir_frames": len(mods.get("NIR", {}))}
        if has_rgb and has_nir:
            shared = sorted(set(mods["RGB"]) & set(mods["NIR"]))
            row["shared_frame_indices"] = len(shared)
            # GPS agreement on shared indices: max abs lat/lon delta
            if shared:
                dlat = max(abs((mods["RGB"][i][0] or 0) - (mods["NIR"][i][0] or 0)) for i in shared)
                dlon = max(abs((mods["RGB"][i][1] or 0) - (mods["NIR"][i][1] or 0)) for i in shared)
                same_time = sum(1 for i in shared if mods["RGB"][i][2] == mods["NIR"][i][2])
                row["max_dlat"] = dlat
                row["max_dlon"] = dlon
                row["same_time_frac"] = round(same_time / len(shared), 4)
        corr.append(row)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {
        "instance_counts": {"|".join(k): v for k, v in sorted(inst_counts.items())},
        "image_counts": {"|".join(k): v for k, v in sorted(img_counts.items())},
        "label_types": {k: sorted(v) for k, v in sorted(label_types.items())},
        "rgb_nir_correspondence": corr,
        # the label file names carry the capture conditions the frames were
        # recorded under (region, time, daylight, weather), which is the only
        # record of them outside the gitignored data tree
        "sequence_files": {"|".join(k): v for k, v in sorted(seq_meta.items())},
    }
    with open(os.path.join(RESULTS_DIR, "inventory.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    # ---- printed summary ----
    print("=" * 70)
    print("IMAGE COUNTS (split, env, modality)")
    for k, v in sorted(img_counts.items()):
        print(f"  {k[0]:5s} {k[1]:8s} {k[2]:4s} : {v:7d}")

    print("=" * 70)
    print("CLASS INVENTORY (instances)  [train/val, RGB vs NIR]")
    labels = sorted({k[3] for k in inst_counts})
    envs = sorted({k[1] for k in inst_counts})
    print(f"{'label':28s} {'env':8s} {'tr-RGB':>8s} {'tr-NIR':>8s} {'va-RGB':>8s} {'va-NIR':>8s}")
    for lab in labels:
        for env in envs:
            r = [inst_counts.get((s, env, m, lab), 0)
                 for s in ("train", "val") for m in ("RGB", "NIR")]
            if any(r):
                print(f"{lab:28s} {env:8s} {r[0]:8d} {r[1]:8d} {r[2]:8d} {r[3]:8d}")

    print("=" * 70)
    print("RGB<->NIR CORRESPONDENCE (per sequence)")
    n_both = sum(1 for r in corr if r["rgb_frames"] and r["nir_frames"])
    print(f"sequences: {len(corr)} | with both modalities: {n_both}")
    for r in corr[:12]:
        print(" ", r)
    print("  ...")
    # aggregate
    shared_tot = sum(r.get("shared_frame_indices", 0) for r in corr)
    rgb_tot = sum(r["rgb_frames"] for r in corr)
    nir_tot = sum(r["nir_frames"] for r in corr)
    print(f"TOTAL frames  RGB {rgb_tot} | NIR {nir_tot} | shared indices {shared_tot}")


if __name__ == "__main__":
    main()
