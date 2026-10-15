"""Preliminary trainer: one modality arm (RGB, NIR, or LUM), one seed.

LUM is the luminance control: RGB images collapsed to grayscale
(BT.601 via cv2) and trained against the RGB annotation set. It
separates "mono texture suffices" from "the NIR spectral band itself
carries signal": NIR ~ LUM means the band is not the cause of any
NIR-vs-RGB difference; NIR > LUM isolates a genuine spectral effect.

Usage: python src/preliminary/train.py {RGB|NIR|LUM} {seed} [train_env]
  train_env: all (default) | Paddy | Field - a transfer arm trains on
  a single environment and is evaluated on both.
The modality, seed, and train_env are the experimental factors;
everything below is the locked preliminary configuration.
PRELIMINARY scope: 2 seeds/arm. The full study commits >=5 seeds and a
paired significance test - 2 seeds only gauges runtime
and effect direction, and is reported as preliminary, not publishable.

Both arms use the identical architecture, schedule, and scene-matched
frame set (src/preliminary/build_index.py); NIR's single channel is
replicated to
3 channels so the ImageNet-pretrained stem is reusable (CHOSEN -
standard practice for single-channel transfer).

Parameter ledger: EPOCHS, BATCH, CROP, LR, WEIGHT_DECAY CHOSEN - the
feasibility recipe, one configuration for every arm, not tuned.

Per-run output: results/preliminary_runs/{modality}_s{seed}/metrics.json
  - final per-class IoU (overall + per env)
  - per-frame intersection/union counts (for paired frame-level stats)
"""
import json
import os
import random
import sys
import time

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))  # repo root, two directories above src/
PRELIMINARY_ROOT = os.path.join(ROOT, "data", "preliminary")
RUNS_ROOT = os.path.join(ROOT, "results", "preliminary_runs")

CLASSES = [
    "background",
    "common_road", "common_tree",
    "field_furrow", "field_levee",
    "paddy_before_driving", "paddy_after_driving",
    "paddy_edge", "paddy_rice",
]
N_CLASSES = len(CLASSES)

# ---- locked preliminary config ----
EPOCHS = 15
BATCH = 16
CROP = 512
LR = 3e-4            # AdamW, cosine to 0
WEIGHT_DECAY = 1e-4
IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class PreliminarySeg(Dataset):
    def __init__(self, split, modality, train, env="all"):
        self.lum = modality == "LUM"  # luminance control: RGB set, gray input
        index_mod = "RGB" if self.lum else modality
        with open(os.path.join(PRELIMINARY_ROOT, f"index_{split}_{index_mod}.json")) as f:
            self.recs = json.load(f)
        if env != "all":
            self.recs = [r for r in self.recs if r["env"] == env]
        self.train = train

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        img = cv2.imread(r["img"], cv2.IMREAD_UNCHANGED)
        if self.lum and img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.ndim == 2:  # NIR/LUM single channel -> replicate
            img = np.repeat(img[..., None], 3, axis=2)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(r["mask"], cv2.IMREAD_GRAYSCALE)
        if self.train:
            y = random.randint(0, img.shape[0] - CROP)
            x = random.randint(0, img.shape[1] - CROP)
            img = img[y:y + CROP, x:x + CROP]
            mask = mask[y:y + CROP, x:x + CROP]
            if random.random() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
        img = (img.astype(np.float32) / 255.0 - IMNET_MEAN) / IMNET_STD
        return (torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(mask)).long(), i)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    inter = np.zeros(N_CLASSES, np.int64)
    union = np.zeros(N_CLASSES, np.int64)
    per_frame = []
    for img, mask, idx in loader:
        img = img.to(device, non_blocking=True)
        with torch.autocast("cuda", torch.float16):
            logits = model(img)
        pred = logits.argmax(1).cpu().numpy()
        gt = mask.numpy()
        for b in range(pred.shape[0]):
            fi, fu = [], []
            for c in range(N_CLASSES):
                p, g = pred[b] == c, gt[b] == c
                i_ = int((p & g).sum())
                u_ = int((p | g).sum())
                fi.append(i_)
                fu.append(u_)
                inter[c] += i_
                union[c] += u_
            per_frame.append({"rec": int(idx[b]), "inter": fi, "union": fu})
    iou = {CLASSES[c]: (inter[c] / union[c] if union[c] else None)
           for c in range(N_CLASSES)}
    return iou, per_frame


def main():
    modality, seed = sys.argv[1], int(sys.argv[2])
    train_env = sys.argv[3] if len(sys.argv) > 3 else "all"
    assert modality in ("RGB", "NIR", "LUM") and train_env in ("all", "Paddy", "Field")
    run_id = f"{modality}_s{seed}" if train_env == "all" \
        else f"{modality}_s{seed}_{train_env.lower()}only"
    out_dir = os.path.join(RUNS_ROOT, run_id)
    os.makedirs(out_dir, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = "cuda"
    model = smp.create_model("segformer", encoder_name="mit_b0",
                             encoder_weights="imagenet",
                             in_channels=3, classes=N_CLASSES).to(device)

    tr = PreliminarySeg("train", modality, train=True, env=train_env)
    va = PreliminarySeg("val", modality, train=False)
    tl = DataLoader(tr, batch_size=BATCH, shuffle=True, num_workers=8,
                    pin_memory=True, drop_last=True, persistent_workers=True)
    vl = DataLoader(va, batch_size=8, shuffle=False, num_workers=8,
                    pin_memory=True)
    print(f"[{run_id}] train {len(tr)} frames | val {len(va)} frames", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps = EPOCHS * len(tl)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    scaler = torch.amp.GradScaler()
    lossf = nn.CrossEntropyLoss()

    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        tot, n = 0.0, 0
        for img, mask, _ in tl:
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.autocast("cuda", torch.float16):
                loss = lossf(model(img), mask)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * img.size(0)
            n += img.size(0)
        print(f"[{run_id}] epoch {ep+1}/{EPOCHS} loss {tot/n:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    iou, per_frame = evaluate(model, vl, device)
    # split IoU by env using index records
    env_iou = {}
    for env in ("Paddy", "Field"):
        sel = [pf for pf in per_frame if va.recs[pf["rec"]]["env"] == env]
        ii = np.zeros(N_CLASSES, np.int64)
        uu = np.zeros(N_CLASSES, np.int64)
        for pf in sel:
            ii += np.array(pf["inter"])
            uu += np.array(pf["union"])
        env_iou[env] = {CLASSES[c]: (float(ii[c] / uu[c]) if uu[c] else None)
                        for c in range(N_CLASSES)}

    metrics = {
        "run_id": run_id, "modality": modality, "seed": seed,
        "train_env": train_env,
        "epochs": EPOCHS, "batch": BATCH, "crop": CROP, "lr": LR,
        "n_train": len(tr), "n_val": len(va),
        "iou": iou, "env_iou": env_iou,
        "per_frame": per_frame,
        "val_index": [{"env": r["env"], "seq": r["seq"], "frame": r["frame"]}
                      for r in va.recs],
        "train_seconds": time.time() - t0,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    print(f"[{run_id}] done. per-class IoU:", flush=True)
    for c in CLASSES:
        v = iou[c]
        print(f"    {c:24s} {v:.4f}" if v is not None else f"    {c:24s} n/a",
              flush=True)


if __name__ == "__main__":
    main()
