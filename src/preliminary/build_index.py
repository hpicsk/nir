"""Build the scene-matched preliminary dataset: resized images + rasterized masks.

Design: modality-level RGB vs NIR
comparison on a SHARED-SCENE set — only frame indices labeled in BOTH
modalities are used (train and val), so both arms train/eval on the
same instants and scene content. Removes the unequal-data confound;
remaining differences are the modality itself + its annotation set.

Parameter ledger:
  ENVS = Paddy, Field   CHOSEN - all five target classes live here
                        (inventory.json); Orchard lacks them and has
                        51 val NIR images total.
  FRAME_STRIDE = 15     CHOSEN - breaks video near-duplicates; yields
                        ~4.2k train frames/arm, enough for the preliminary runs.
  SIZE = 960 x 544      CHOSEN - half res, /32-divisible for SegFormer.
  CLASSES               fixed 9-class shared taxonomy (below);
                        paddy_water EXCLUDED: 0 train instances on disk.

Output:
  data/preliminary/{split}/{modality}/img/*.jpg   (960x544)
  data/preliminary/{split}/{modality}/mask/*.png  (class-index, same size)
  data/preliminary/index_{split}_{modality}.json  (per-frame records)
"""
import json
import glob
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))  # repo root, two directories above src/
LABELS_ROOT = os.path.join(ROOT, "data", "labels")
IMAGES_ROOT = os.path.join(ROOT, "data", "images")
PRELIMINARY_ROOT = os.path.join(ROOT, "data", "preliminary")

ENVS = ["Paddy", "Field"]
FRAME_STRIDE = 15
W, H = 960, 544  # resize target (aspect 1.006x distortion vs 1920x1080)

CLASSES = [  # index 0 = background (unlabeled area)
    "background",
    "common_road", "common_tree",
    "field_furrow", "field_levee",
    "paddy_before_driving", "paddy_after_driving",
    "paddy_edge", "paddy_rice",
]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}

SPLIT_DIRS = {
    "train": (os.path.join(LABELS_ROOT, "Tel_Polygon"),
    os.path.join(IMAGES_ROOT, "train")),
    "val": (os.path.join(LABELS_ROOT, "VL_Polygon"), os.path.join(IMAGES_ROOT, "val")),
}


def rasterize(objects, w0, h0):
    """Class-index mask at (H, W). Objects painted in ascending z_order,
    so higher z_order (foreground) overwrites background classes."""
    mask = np.zeros((H, W), np.uint8)
    sx, sy = W / w0, H / h0
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
    split, env, seq, modality, name, objects, w0, h0 = job
    src = os.path.join(SPLIT_DIRS[split][1], env, seq, modality, name)
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    stem = f"{env}_{seq}_{name.rsplit('.',1)[0]}"
    img_out = os.path.join(PRELIMINARY_ROOT, split, modality, "img", stem + ".jpg")
    mask_out = os.path.join(PRELIMINARY_ROOT, split, modality, "mask", stem + ".png")
    cv2.imwrite(img_out, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    mask = rasterize(objects, w0, h0)
    cv2.imwrite(mask_out, mask)
    present = sorted(int(c) for c in np.unique(mask))
    return {"split": split, "env": env, "seq": seq, "modality": modality,
            "frame": int(name.rsplit("_", 1)[1].split(".")[0]),
            "img": img_out, "mask": mask_out, "classes_present": present}


def main():
    for split in SPLIT_DIRS:
        for m in ("RGB", "NIR"):
            os.makedirs(os.path.join(PRELIMINARY_ROOT, split, m, "img"), exist_ok=True)
            os.makedirs(os.path.join(PRELIMINARY_ROOT, split, m, "mask"), exist_ok=True)

    jobs = defaultdict(list)   # (split, modality) -> job list
    for split, (label_root, _) in SPLIT_DIRS.items():
        for env in ENVS:
            for seq_dir in sorted(glob.glob(os.path.join(label_root, env,
            "sequence*"))):
                seq = os.path.basename(seq_dir)
                per_mod = {}
                for modality in ("RGB", "NIR"):
                    js = glob.glob(os.path.join(seq_dir, modality, "*.json"))
                    if not js:
                        continue
                    d = json.load(open(js[0]))
                    per_mod[modality] = {
                        int(i["name"].rsplit("_", 1)[1].split(".")[0]): i
                        for i in d["images"]
                    }
                if len(per_mod) < 2:
                    print(f"[skip] {split}/{env}/{seq}: missing a modality")
                    continue
                shared = sorted(set(per_mod["RGB"]) & set(per_mod["NIR"]))
                picked = [i for i in shared if i % FRAME_STRIDE == 0]
                for modality in ("RGB", "NIR"):
                    for i in picked:
                        meta = per_mod[modality][i]
                        jobs[(split, modality)].append(
                            (split, env, seq, modality, meta["name"],
                             meta["objects"], meta["width"], meta["height"]))

    for (split, modality), job_list in sorted(jobs.items()):
        with ProcessPoolExecutor(max_workers=16) as ex:
            recs = [r for r in ex.map(process_one, job_list, chunksize=8) if r]
        out = os.path.join(PRELIMINARY_ROOT, f"index_{split}_{modality}.json")
        with open(out, "w") as f:
            json.dump(recs, f)
        # class pixel stats
        n_with = defaultdict(int)
        for r in recs:
            for c in r["classes_present"]:
                n_with[CLASSES[c]] += 1
        print(f"[{split}/{modality}] {len(recs)} frames "
              f"({len(job_list) - len(recs)} unreadable) -> {out}")
        for c in CLASSES:
            print(f"    {c:24s} in {n_with[c]:5d} frames")


if __name__ == "__main__":
    main()
