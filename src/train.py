"""Full-study trainer: one arm x one seed x one protocol.

Usage: python src/train.py {RGB|NIR|LUM|RGBNIR|NIRonRGB} {seed}
       {blocked|official|losoNNN|plosoNNN|pofficial} all
       [backbone] [arch]
The first four arguments are explicit; the 5th and 6th default to the
headline model, mit_b0 and segformer. The combination must be a row of
config.py RUN_MATRIX - the criterion record - or the script
refuses to run. Every hyperparameter is locked in config.py; this file
contains no run-specific tuning.

Fixed choices (research conventions, baselines). Each BASELINE tag below
carries the four fields that answer an objection to the choice it names:
the choice, the obvious alternative, the tuning that configuration
received, and where the objection is discharged.
  BASELINE: SegFormer (all-MLP decoder) on MiT-B0, repeated on MiT-B2 and on
  UPerNet/ConvNeXt-V2-T | obvious alternative: "a larger encoder or a current
  architecture family would not show this" | tuning: none - one
  configuration (config.py EPOCHS, BATCH, CROP, LR, WEIGHT_DECAY), recorded
  identically in every run's metrics.json | discharge: sec:results -
  tab:robustness repeats both contrasts on the two further models.
  BASELINE: AdamW at LR on a cosine schedule to zero, EPOCHS epochs | obvious
  alternative: "near-infrared needed a different learning rate or a longer
  schedule" | tuning: none - one configuration for every configuration and
  model, so the budget is matched rather than favoring RGB | discharge:
  sec:discussion - the comparison is stated at the budget tried, not as a
  bound on the band.

Arms:
  RGB     3ch RGB image; supervised on RGB annotations.
  NIR     NIR replicated to 3ch (reuses the ImageNet stem, as in the preliminary runs);
          supervised on NIR annotations.
          BASELINE: visible-band (ImageNet) pretraining for a
          near-infrared input | obvious alternative: "NIR-pretrained or
          from-scratch encoders would close the gap" | tuning: none - one
          configuration, as for every fixed choice here | discharge:
          sec:discussion - bounded, not excluded. LUM takes the same
          repeat-to-3ch path (see chans below), so NIR>LUM is read at
          matched handicap and a per-arm handicap cancels in an
          official-vs-blocked difference; the NIR-vs-RGB redundancy
          comparison is NOT matched on pretraining domain, and the
          manuscript bounds it to an ImageNet-initialized encoder rather
          than claiming it for the band.
  LUM     grayscale-of-RGB (BT.601) replicated to 3ch; RGB annotations.
          The format control: LUM has NIR's single-channel input, so
          NIR>LUM cannot come from the format. The two still differ in
          band and in the annotation each trains on.
  NIRonRGB  the NIR image exactly as the NIR arm receives it, supervised on
          the RGB annotation (config.py DIAG_ARM, phase I). It crosses the
          two factors that the NIR arm varies together, the input band and
          the annotation; annotation_diagnosis.py reads it.
  RGBNIR  mid-fusion: two ImageNet-initialized encoders (RGB stream /
          NIR-3ch stream; model per config.py - segformer/mit_b0 default,
          mit_b2 in phase F, upernet/ConvNeXt-V2 in phase H), stage-wise
          channel concatenation, the arch's stock decoder over the
          doubled channel list. Supervised on
          RGB annotations (CHOSEN: the deployment question is "does
          adding NIR to an RGB stack help?" - holding supervision fixed
          vs the RGB arm isolates the added input; cross-label and
          intersection scores are reported like every arm).
          BASELINE: two-stream mid-fusion, stage-wise concat | obvious
          alternative: "a learned or late-fusion design would recover the
          gain" | tuning: none - one fusion design, one configuration |
          discharge: sec:discussion - the fusion claim is bounded to
          stage-wise concatenation; sec:methods answers the capacity half of
          the objection: the two streams carry >=1.8x a
          single-stream arm's parameters on every model in config.py, so
          the fusion nulls are not a capacity handicap.

Evaluation (endogeneity bounds) - every run is scored
against three ground truths on every val frame:
  rgb   RGB annotation set
  nir   NIR annotation set
  isec  pixels where the two sets agree (parallax- and annotator-
        consistent regions; the endogeneity-free bound)
Cross-label pixel scores include RGB<->NIR parallax; own-label scores
include annotator endogeneity; isec bounds both. Per-frame inter/union
for all three go to perframe.npz for the paired seed-pooled statistics;
per-frame GT pixel counts live in the index (pre-treatment
frame inclusion for the paired tests - fixes the preliminary caveat).

Class-balanced sampling: WeightedRandomSampler over
training frames, w(f) = clip(sqrt(share_ref / share(c_min)), 1, W_CAP)
where c_min is the rarest (by union-label frame share) target class
present in f and share_ref is the median target share. Weights depend
only on the RGB-union-NIR labels of the training set - identical for
every arm in a comparison cell, so sampling cannot confound the arm.
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.base import initialization as smp_init
from segmentation_models_pytorch.decoders.segformer.decoder import SegformerDecoder
from segmentation_models_pytorch.decoders.upernet.decoder import UPerNetDecoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)
SUP_MASK = {"RGB": "mask_rgb", "LUM": "mask_rgb",
            "NIR": "mask_nir", "RGBNIR": "mask_rgb", C.DIAG_ARM: "mask_rgb"}


# Decoder class and its channel-width keyword per architecture. Both smp
# decoders take the same (encoder_channels, encoder_depth) contract and feed
# a 256-channel SegmentationHead at upsampling=4, so one dual-encoder
# assembly mirrors either stock model exactly.
DECODERS = {"segformer": (SegformerDecoder, "segmentation_channels"),
            "upernet": (UPerNetDecoder, "decoder_channels")}


class DualEncoderNet(nn.Module):
    """Mid-fusion segmentation net: two encoders (RGB stream / NIR-3ch
    stream), stage-wise concat, stock decoder and head over the doubled
    channel list (mirrors smp's own assembly for `arch`, incl. weight init).
    Architecture and backbone are parameterised - segformer/mit_b0 is the
    headline, segformer/mit_b2 is phase F, upernet/ConvNeXt-V2 is phase H."""

    def __init__(self, n_classes, arch="segformer", backbone="mit_b0"):
        super().__init__()
        self.enc_a = smp.encoders.get_encoder(
            backbone, in_channels=3, depth=5, weights="imagenet")
        self.enc_b = smp.encoders.get_encoder(
            backbone, in_channels=3, depth=5, weights="imagenet")
        ch = [2 * c for c in self.enc_a.out_channels]
        decoder_cls, width_kw = DECODERS[arch]
        self.decoder = decoder_cls(encoder_channels=ch, encoder_depth=5,
                                   **{width_kw: 256})
        self.head = SegmentationHead(
            in_channels=256, out_channels=n_classes,
            kernel_size=1, upsampling=4)
        smp_init.initialize_decoder(self.decoder)
        smp_init.initialize_head(self.head)

    def forward(self, x):
        fa = self.enc_a(x[:, :3])
        fb = self.enc_b(x[:, 3:])
        feats = [torch.cat([a, b], 1) for a, b in zip(fa, fb)]
        return self.head(self.decoder(feats))


def build_model(arm, arch="segformer", backbone="mit_b0"):
    if arm == "RGBNIR":
        return DualEncoderNet(C.N_CLASSES, arch, backbone)
    return smp.create_model(arch, encoder_name=backbone,
                            encoder_weights="imagenet",
                            in_channels=3, classes=C.N_CLASSES)


def _key2site():
    """index-record key '{osplit}/{env}/{seq}' -> GPS site id, from the
    blocked-split assignment (data/processed/splits_blocked.json).
    LOSO fold selection."""
    with open(C.SPLITS_JSON) as f:
        sites = json.load(f)["sites"]
    return {m: s["site_id"] for s in sites for m in s["members"]}


class FSSeg(Dataset):
    def __init__(self, recs, arm, train):
        self.recs, self.arm, self.train = recs, arm, train

    def __len__(self):
        return len(self.recs)

    def _input(self, r):
        chans = []
        if self.arm in ("RGB", "RGBNIR"):
            chans.append(cv2.cvtColor(cv2.imread(r["img_rgb"]),
                                      cv2.COLOR_BGR2RGB))
        if self.arm == "LUM":
            g = cv2.cvtColor(cv2.imread(r["img_rgb"]), cv2.COLOR_BGR2GRAY)
            chans.append(np.repeat(g[..., None], 3, axis=2))
        if self.arm in ("NIR", "RGBNIR", C.DIAG_ARM):
            g = cv2.imread(r["img_nir"], cv2.IMREAD_GRAYSCALE)
            chans.append(np.repeat(g[..., None], 3, axis=2))
        return np.concatenate(chans, axis=2)  # 3 or 6 channels

    @staticmethod
    def _norm(img):
        img = img.astype(np.float32) / 255.0
        k = img.shape[2] // 3
        img = (img - np.tile(IMNET_MEAN, k)) / np.tile(IMNET_STD, k)
        return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))

    def __getitem__(self, i):
        r = self.recs[i]
        img = self._input(r)
        if self.train:
            mask = cv2.imread(r[SUP_MASK[self.arm]], cv2.IMREAD_GRAYSCALE)
            y = random.randint(0, img.shape[0] - C.CROP)
            x = random.randint(0, img.shape[1] - C.CROP)
            img = img[y:y + C.CROP, x:x + C.CROP]
            mask = mask[y:y + C.CROP, x:x + C.CROP]
            if random.random() < C.FLIP_P:
                img, mask = img[:, ::-1], mask[:, ::-1]
            return self._norm(img), \
                torch.from_numpy(np.ascontiguousarray(mask)).long()
        m_rgb = cv2.imread(r["mask_rgb"], cv2.IMREAD_GRAYSCALE)
        m_nir = cv2.imread(r["mask_nir"], cv2.IMREAD_GRAYSCALE)
        return self._norm(img), torch.from_numpy(m_rgb).long(), \
            torch.from_numpy(m_nir).long(), i


def frame_weights(recs):
    """Arm-independent oversampling weights from union-label presence."""
    tid = [C.CLASSES.index(c) for c in C.TARGETS]
    present = np.array([[bool(r["px_rgb"][c]) or bool(r["px_nir"][c])
                         for c in tid] for r in recs])
    share = present.mean(0)                       # per-target frame share
    share_ref = float(np.median(share))
    w = np.ones(len(recs))
    for f in range(len(recs)):
        pres = np.where(present[f])[0]
        if len(pres):
            s_min = share[pres].min()
            w[f] = min(max(np.sqrt(share_ref / s_min), 1.0), C.W_CAP)
    return w, {"share": {c: float(s) for c, s in zip(C.TARGETS, share)},
               "share_ref": share_ref,
               "n_upweighted": int((w > 1).sum()),
               "w_mean": float(w.mean()), "w_max": float(w.max())}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    protos = ("rgb", "nir", "isec")
    inter = {p: [] for p in protos}
    union = {p: [] for p in protos}
    order = []
    for img, m_rgb, m_nir, idx in loader:
        img = img.to(device, non_blocking=True)
        with torch.autocast("cuda", torch.float16):
            logits = model(img)
        pred = logits.argmax(1).cpu().numpy()
        g_rgb, g_nir = m_rgb.numpy(), m_nir.numpy()
        for b in range(pred.shape[0]):
            order.append(int(idx[b]))
            agree = g_rgb[b] == g_nir[b]
            row_i = {p: [] for p in protos}
            row_u = {p: [] for p in protos}
            for c in range(C.N_CLASSES):
                p_ = pred[b] == c
                r_, n_ = g_rgb[b] == c, g_nir[b] == c
                row_i["rgb"].append(int((p_ & r_).sum()))
                row_u["rgb"].append(int((p_ | r_).sum()))
                row_i["nir"].append(int((p_ & n_).sum()))
                row_u["nir"].append(int((p_ | n_).sum()))
                row_i["isec"].append(int((p_ & r_ & agree).sum()))
                row_u["isec"].append(int(((p_ | r_) & agree).sum()))
            for p in protos:
                inter[p].append(row_i[p])
                union[p].append(row_u[p])
    return ({p: np.array(inter[p], np.int64) for p in protos},
            {p: np.array(union[p], np.int64) for p in protos}, order)


def iou_table(inter, union, sel=None):
    if sel is not None:
        inter, union = inter[sel], union[sel]
    ii, uu = inter.sum(0), union.sum(0)
    return {C.CLASSES[c]: (float(ii[c] / uu[c]) if uu[c] else None)
            for c in range(C.N_CLASSES)}


def main():
    arm, seed, protocol, train_env = (
        sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4])
    backbone = sys.argv[5] if len(sys.argv) > 5 else "mit_b0"
    arch = sys.argv[6] if len(sys.argv) > 6 else "segformer"
    spec = {"arm": arm, "seed": seed, "protocol": protocol,
            "train_env": train_env, "backbone": backbone, "arch": arch}
    match = [r for r in C.RUN_MATRIX
             if all(r[k] == v for k, v in spec.items())]
    assert match, f"not a committed run (config.py): {spec}"
    run_id = match[0]["run_id"]
    out_dir = os.path.join(C.RUNS_ROOT, run_id)
    os.makedirs(out_dir, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    recs = C.load_index()
    if protocol == "pofficial":     # phase G: Paddy-restricted official split
        # ploso's env restriction with ploso's site blocking REMOVED, so the
        # pair differs in the split alone - the comparator gate G2 is read
        # against (config ledger, 2026-07-24b). Same reason as ploso for
        # leaving train_env at "all": the protocol carries the restriction.
        env_recs = [r for r in recs if r["env"] == C.PLOSO_ENV]
        tr_recs = [r for r in env_recs if r["osplit"] == "train"]
        va_recs = [r for r in env_recs if r["osplit"] == "val"]
    elif protocol.startswith("ploso"):  # phase G: Paddy-restricted LOSO
        # BOTH sides restricted to PLOSO_ENV - phase E's control failed
        # because Field is 66% concentrated in site000, while Paddy is
        # balanced across sites (config ledger, 2026-07-24).
        hold = "site" + protocol[len("ploso"):]        # ploso000 -> site000
        key2site = _key2site()

        def _site(r):
            return key2site[f"{r['osplit']}/{r['env']}/{r['seq']}"]
        env_recs = [r for r in recs if r["env"] == C.PLOSO_ENV]
        tr_recs = [r for r in env_recs if _site(r) != hold]
        va_recs = [r for r in env_recs if _site(r) == hold]
    elif protocol.startswith("loso"):        # phase E: hold out one GPS site
        hold = "site" + protocol[len("loso"):]         # loso000 -> site000
        key2site = _key2site()

        def _site(r):
            return key2site[f"{r['osplit']}/{r['env']}/{r['seq']}"]
        tr_recs = [r for r in recs if _site(r) != hold]
        va_recs = [r for r in recs if _site(r) == hold]
    else:
        # Guard the fallthrough: without this, any new protocol string silently
        # trains on the unrestricted official split and the run looks
        # successful with the wrong data.
        assert protocol in ("blocked", "official"), protocol
        field = "bsplit" if protocol == "blocked" else "osplit"
        tr_recs = [r for r in recs if r[field] == "train"]
        va_recs = [r for r in recs if r[field] == "val"]

    w, w_stats = frame_weights(tr_recs)
    gen = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                    num_samples=len(tr_recs),
                                    replacement=True, generator=gen)

    device = "cuda"
    model = build_model(arm, arch, backbone).to(device)
    tr = FSSeg(tr_recs, arm, train=True)
    va = FSSeg(va_recs, arm, train=False)
    tl = DataLoader(tr, batch_size=C.BATCH, sampler=sampler, num_workers=8,
                    pin_memory=True, drop_last=True, persistent_workers=True)
    vl = DataLoader(va, batch_size=8, shuffle=False, num_workers=8,
                    pin_memory=True)
    print(f"[{run_id}] train {len(tr)} frames | val {len(va)} frames | "
          f"upweighted {w_stats['n_upweighted']} (w_max {w_stats['w_max']:.1f})",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=C.LR,
                            weight_decay=C.WEIGHT_DECAY)
    steps = C.EPOCHS * len(tl)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    scaler = torch.amp.GradScaler()
    lossf = nn.CrossEntropyLoss()

    t0 = time.time()
    for ep in range(C.EPOCHS):
        model.train()
        tot, n = 0.0, 0
        for img, mask in tl:
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
        print(f"[{run_id}] epoch {ep + 1}/{C.EPOCHS} loss {tot / n:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    inter, union, order = evaluate(model, vl, device)
    envs = np.array([va_recs[i]["env"] for i in order])
    iou = {p: iou_table(inter[p], union[p]) for p in inter}
    env_iou = {p: {e: iou_table(inter[p], union[p], envs == e)
                   for e in C.ENVS} for p in inter}

    metrics = {
        "run_id": run_id, **spec, "phase": match[0]["phase"],
        "epochs": C.EPOCHS, "batch": C.BATCH, "crop": C.CROP, "lr": C.LR,
        "weight_decay": C.WEIGHT_DECAY, "frame_stride": C.FRAME_STRIDE,
        "n_train": len(tr), "n_val": len(va), "sampler": w_stats,
        "iou": iou, "env_iou": env_iou,
        "train_seconds": time.time() - t0,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)
    np.savez_compressed(
        os.path.join(out_dir, "perframe.npz"),
        env=envs, seq=np.array([va_recs[i]["seq"] for i in order]),
        frame=np.array([va_recs[i]["frame"] for i in order], np.int32),
        **{f"inter_{p}": inter[p] for p in inter},
        **{f"union_{p}": union[p] for p in union})
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    print(f"[{run_id}] done in {metrics['train_seconds']:.0f}s. "
          f"IoU (own={SUP_MASK[arm][5:]}, isec):", flush=True)
    own = "rgb" if SUP_MASK[arm] == "mask_rgb" else "nir"
    for c in C.CLASSES[1:]:
        vo, vi = iou[own][c], iou["isec"][c]
        fo = f"{vo:.4f}" if vo is not None else "  n/a "
        fi = f"{vi:.4f}" if vi is not None else "  n/a "
        print(f"    {c:24s} {fo} {fi}", flush=True)


if __name__ == "__main__":
    main()
