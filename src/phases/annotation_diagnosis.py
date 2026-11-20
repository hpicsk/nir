"""Does near-infrared's paddy-edge lead under the dataset's split come from the
band or from the annotation that the NIR frames carry? (sec:results)

Under the dataset's own split, NIR scores 0.339 paddy-edge IoU on the
intersection ground truth, RGB 0.009 and fusion 0.055. NIR is the only
configuration that trains on the NIR annotation, so in the four configurations
the NIR band and the NIR annotation never vary apart. This module reads the
two annotations and the per-frame scores of the committed runs, and it reads
one more configuration, NIRonRGB (the NIR image trained on the RGB
annotation, config.py DIAG_ARM), which crosses the two.

Comparisons, each with what its outcome means for the paper. Every margin is
the per-frame paired difference of seed-averaged IoUs on the validation frames
of the sequence-level split, with analyze.cluster_ci's interval over the
recording sequences (a t interval below MIN_BOOT_UNITS sequences).

D1  Count, area and placement of each class in the two annotations, over every
    sampled frame and separately over the dataset's training and validation
    frames: frames that mark the class in each annotation; mean pixels per
    marking frame; pooled agreement |R & N| / |R | N| as the two masks are
    distributed (raw), and after the NIR mask is shifted by the integer offset
    within SHIFT_PX that best aligns the frame's two label maps (registered).
    Meaning: the count and area ratios NIR / RGB of the paddy edge inside
    [1 - RATIO_TOL, 1 + RATIO_TOL] read as "marked in as many frames and with
    as much area"; outside it the paper prints the ratio. The paddy edge's
    registered agreement at or above the lower of the furrow's and the ridge's
    (the two other strip-shaped target classes, whose agreement the same
    boundary jitter lowers) reads as "placed as alike as the other strips";
    below both, the paddy edge is the strip class that the annotations place
    least alike. Registered agreement well above raw agreement means that the
    camera offset, not the annotators, separates the two masks.
D2  NIR - RGB on the intersection ground truth over the validation frames that
    mark the paddy edge in both annotations, split at the median of the
    frames' registered paddy-edge agreement. Meaning: a positive margin whose
    interval excludes zero on the upper half says that NIR's lead does not
    depend on frames where the two annotations mark the class differently; an
    interval spanning zero on the upper half, with the lead on the lower half,
    says it does.
D3  NIR - RGB scored on the RGB annotation, the one that RGB trains on, over
    the validation frames whose RGB annotation marks the paddy edge. Meaning:
    an interval above zero says that the NIR network finds what the RGB
    annotation marks better than a network trained on that annotation, so the
    lead does not come from scoring NIR on its own annotation. The NIR network
    pays the camera offset on this ground truth, so the comparison is biased
    against NIR.
D4  Fusion - RGB on the intersection ground truth (both train on the RGB
    annotation, fusion adds the NIR image), set beside NIR - RGB. Meaning:
    the share (fusion - RGB) / (NIR - RGB) of pooled IoU is the part of NIR's
    lead that adding the NIR image reproduces with the annotation held fixed.
D5  NIRonRGB, five seeds under the sequence-level split. Pooled paddy-edge IoU
    on the intersection ground truth as mean and SD over seeds, and the
    margins NIRonRGB - RGB (the band, annotation held) and NIR - NIRonRGB (the
    annotation, image held). Meaning, read on the pooled seed means:
      - the band carries the lead if NIRonRGB - RGB's interval excludes zero
        and NIRonRGB's pooled IoU lies at least HALFWAY from RGB's to NIR's;
        the paper may then credit the band with the lead;
      - the annotation carries the lead if NIR - NIRonRGB's interval excludes
        zero and NIRonRGB's pooled IoU lies below that point; the paper then
        states the lead as the NIR configuration's, image and annotation
        together, and credits neither the band nor the camera's input format;
      - otherwise the paper reports both components and credits neither alone.
    The RGB annotation is drawn on the RGB image, which the NIR image is offset
    from (camera_offset.py), so under the second outcome D1's registered
    agreement says whether the annotators' convention or the offset of the
    labels from the NIR image is what NIRonRGB lacks.
S1  Sensitivity of SHIFT_PX (2026-11-17, committed before it runs). 45% of
    the validation frames that mark the paddy edge in both annotations, and
    12% of such training frames, reach the 16 px bound, which contradicts the
    argument for SHIFT_PX below. S1 repeats the registration at every bound
    of REG_BOUNDS and reports, for the training and the validation frames
    separately: the share of the frames marking the paddy edge in both
    annotations whose shift reaches the bound in either axis (saturation);
    the pooled registered paddy-edge agreement over all sampled frames, as
    in D1; and D2's two margins, split at the median of the registered
    agreement at that bound. Meaning:
      (a) D2 holds if the upper-half margin's interval lies above zero at
          every bound. The paper then keeps D2's sentence and prints, in the
          sentence that defines the registration, the saturation share of
          the validation frames at SHIFT_PX and the range of the upper-half
          margin over the bounds. If the interval spans zero at some bound,
          the paper prints the margin at that bound and D2's sentence states
          that the lead's resting on the frames with high agreement depends
          on the bound.
      (b) The validation frames agree less than the training frames, as
          registered, if their registered agreement stays below the training
          frames' at every bound. A rise of more than AGREE_TOL from its
          value at SHIFT_PX, on either set, is reported as the share of that
          set's disagreement that offsets beyond SHIFT_PX produce.
      (c) If the validation saturation at SHIFT_PX exceeds the training
          saturation by more than SAT_TOL, the paper states beside the
          printed (unregistered) contrast of the two sets that the
          validation labels are offset further, so the contrast includes the
          offset as well as the annotators.
      (d) If more than SAT_TOL of either set's frames still reach the widest
          bound, the registration finds no optimum inside any tested window
          for them, and the paper does not read their registered agreement
          as the annotators'.
    A wider window can align a label map to a neighboring strip of a
    repeating field pattern, so the agreement at a wide bound is an upper
    bound on the annotators' agreement. results/registration_sensitivity.npz
    holds the shifts and the paddy-edge counts at each bound; the default
    run adds S1 to results/annotation_diagnosis.json when that file exists.

Parameters:
  SHIFT_PX = 16      CHOSEN  half-width of the registration search, a little
                             over twice the median camera offset (7 px, 6 px),
                             so that it seldom reaches a neighboring strip of a
                             repeating field pattern. The search reaches it on
                             45% of the validation frames that mark the paddy
                             edge, so it does not contain every offset; S1
                             shows that D2's reading holds at 32 and 48 px.
  RATIO_TOL = 0.10   CHOSEN  tolerance of the count and area ratios in D1.
  REG_BOUNDS = (16, 32, 48)  CHOSEN  S1's three bounds: SHIFT_PX, twice and
                             three times it.
  AGREE_TOL = 0.05   CHOSEN  S1 (b): a change of the pooled agreement that is
                             reported as caused by offsets beyond SHIFT_PX.
  SAT_TOL = 0.10     CHOSEN  S1 (c) and (d): a saturation gap between the two
                             sets, or a saturation at the widest bound, that
                             the paper reports.
  HALFWAY = 0.5      CHOSEN  the point of a two-component split, read in D5.

Usage: python src/phases/annotation_diagnosis.py measure  (no GPU; reads the
       masks under data/study/, writes results/annotation_agreement.npz)
       python src/phases/annotation_diagnosis.py sensitivity  (no GPU; reads
       the masks, writes results/registration_sensitivity.npz)
       python src/phases/annotation_diagnosis.py          (reads the committed
       artifacts only, writes results/annotation_diagnosis.json)
"""
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import config as C  # noqa: E402
import analyze as A  # noqa: E402

AGREE_NPZ = os.path.join(C.ROOT, "results", "annotation_agreement.npz")
OUT = os.path.join(C.ROOT, "results", "annotation_diagnosis.json")

SHIFT_PX = 16
RATIO_TOL = 0.10
HALFWAY = 0.5
REG_BOUNDS = (16, 32, 48)
AGREE_TOL = 0.05
SAT_TOL = 0.10
SENS_NPZ = os.path.join(C.ROOT, "results", "registration_sensitivity.npz")

EDGE = C.CLASSES.index("paddy_edge")
STRIPS = ("field_furrow", "field_levee")


def _register(m_rgb, m_nir, bound=SHIFT_PX):
    """Integer shift (dy, dx), |dy|, |dx| <= bound, maximizing the number of
    non-background pixels p with m_rgb[p] == m_nir[p + (dy, dx)]."""
    h, w = m_rgb.shape
    shape = (h + bound, w + bound)   # no wrap-around within the window
    corr = np.zeros(shape)
    for c in range(1, C.N_CLASSES):
        r, n = m_rgb == c, m_nir == c
        if not r.any() or not n.any():
            continue
        corr += np.fft.irfft2(np.conj(np.fft.rfft2(r, shape))
                              * np.fft.rfft2(n, shape), shape)
    offs = np.arange(-bound, bound + 1)
    win = corr[np.ix_(offs % shape[0], offs % shape[1])]
    iy, ix = np.unravel_index(int(np.argmax(np.round(win, 6))), win.shape)
    return int(offs[iy]), int(offs[ix])


def _registered_counts(m_rgb, m_nir, dy, dx):
    """Intersection and union of each class between m_rgb and the NIR mask
    shifted by (dy, dx), over the pixels where both are defined."""
    h, w = m_rgb.shape
    ys, xs = slice(max(0, -dy), min(h, h - dy)), slice(max(0, -dx), min(w, w - dx))
    r_v = m_rgb[ys, xs]
    n_v = m_nir[ys.start + dy:ys.stop + dy, xs.start + dx:xs.stop + dx]
    out = np.zeros((2, C.N_CLASSES), np.int64)
    for c in range(C.N_CLASSES):
        rv, nv = r_v == c, n_v == c
        out[0, c], out[1, c] = (rv & nv).sum(), (rv | nv).sum()
    return out


def _sensitivity_one(rec):
    import cv2
    m_rgb = cv2.imread(rec["mask_rgb"], cv2.IMREAD_GRAYSCALE)
    m_nir = cv2.imread(rec["mask_nir"], cv2.IMREAD_GRAYSCALE)
    res = []
    for b in REG_BOUNDS:
        dy, dx = _register(m_rgb, m_nir, b)
        cnt = _registered_counts(m_rgb, m_nir, dy, dx)
        res.append((dy, dx, cnt[0, EDGE], cnt[1, EDGE]))
    return res


def sensitivity():
    recs = C.load_index()
    with ProcessPoolExecutor(max_workers=min(64, os.cpu_count())) as ex:
        res = list(ex.map(_sensitivity_one, recs, chunksize=32))
    arr = {}
    for i, b in enumerate(REG_BOUNDS):
        arr[f"shift_{b}"] = np.array([[r[i][0], r[i][1]] for r in res], np.int32)
        arr[f"inter_edge_{b}"] = np.array([r[i][2] for r in res], np.int64)
        arr[f"union_edge_{b}"] = np.array([r[i][3] for r in res], np.int64)
    np.savez_compressed(
        SENS_NPZ, env=np.array([r["env"] for r in recs]),
        seq=np.array([r["seq"] for r in recs]),
        frame=np.array([r["frame"] for r in recs], np.int32), **arr)
    print(f"wrote {SENS_NPZ} ({len(recs)} frames, bounds {REG_BOUNDS})")


def _measure_one(rec):
    import cv2
    m_rgb = cv2.imread(rec["mask_rgb"], cv2.IMREAD_GRAYSCALE)
    m_nir = cv2.imread(rec["mask_nir"], cv2.IMREAD_GRAYSCALE)
    dy, dx = _register(m_rgb, m_nir)
    h, w = m_rgb.shape
    # registered NIR mask: n_reg[p] = m_nir[p + (dy, dx)], valid where inside
    ys, xs = slice(max(0, -dy), min(h, h - dy)), slice(max(0, -dx), min(w, w - dx))
    r_v = m_rgb[ys, xs]
    n_v = m_nir[ys.start + dy:ys.stop + dy, xs.start + dx:xs.stop + dx]
    out = np.zeros((4, C.N_CLASSES), np.int64)
    for c in range(C.N_CLASSES):
        r, n = m_rgb == c, m_nir == c
        out[0, c], out[1, c] = (r & n).sum(), (r | n).sum()
        rv, nv = r_v == c, n_v == c
        out[2, c], out[3, c] = (rv & nv).sum(), (rv | nv).sum()
    return dy, dx, out


def measure():
    recs = C.load_index()
    with ProcessPoolExecutor(max_workers=min(64, os.cpu_count())) as ex:
        res = list(ex.map(_measure_one, recs, chunksize=32))
    cnt = np.stack([r[2] for r in res])
    np.savez_compressed(
        AGREE_NPZ,
        env=np.array([r["env"] for r in recs]),
        seq=np.array([r["seq"] for r in recs]),
        frame=np.array([r["frame"] for r in recs], np.int32),
        shift=np.array([[r[0], r[1]] for r in res], np.int32),
        inter_raw=cnt[:, 0], union_raw=cnt[:, 1],
        inter_reg=cnt[:, 2], union_reg=cnt[:, 3])
    print(f"wrote {AGREE_NPZ} ({len(recs)} frames)")


def _d1(z, recs):
    side = np.array([r["osplit"] for r in recs])
    px_r = np.array([r["px_rgb"] for r in recs])
    px_n = np.array([r["px_nir"] for r in recs])
    out = {}
    for name, sel in (("all", np.ones(len(recs), bool)),
                      ("train", side == "train"), ("val", side == "val")):
        per = {}
        for cname in C.TARGETS + ["common_road"]:
            c = C.CLASSES.index(cname)
            fr, fn = px_r[sel, c] > 0, px_n[sel, c] > 0
            per[cname] = {
                "frames_rgb": int(fr.sum()), "frames_nir": int(fn.sum()),
                "px_per_frame_rgb": float(px_r[sel, c][fr].mean()),
                "px_per_frame_nir": float(px_n[sel, c][fn].mean()),
                "agree_raw": float(z["inter_raw"][sel, c].sum()
                                   / z["union_raw"][sel, c].sum()),
                "agree_reg": float(z["inter_reg"][sel, c].sum()
                                   / z["union_reg"][sel, c].sum()),
            }
            p = per[cname]
            p["count_ratio"] = p["frames_nir"] / p["frames_rgb"]
            p["area_ratio"] = p["px_per_frame_nir"] / p["px_per_frame_rgb"]
        out[name] = per
    sh = z["shift"]
    out["shift_median_px"] = {"dy": float(np.median(sh[:, 0])),
                              "dx": float(np.median(sh[:, 1]))}
    e = out["all"]["paddy_edge"]
    out["reading"] = {
        "count_alike": abs(e["count_ratio"] - 1) <= RATIO_TOL,
        "area_alike": abs(e["area_ratio"] - 1) <= RATIO_TOL,
        "placed_as_alike_as_other_strips":
            e["agree_reg"] >= min(out["all"][s]["agree_reg"] for s in STRIPS),
    }
    return out


def _margin(fx, fy, keys=None):
    keys = sorted(set(fx) & set(fy) if keys is None
                  else set(keys) & set(fx) & set(fy))
    d = [fx[k] - fy[k] for k in keys]
    lo, hi = A.cluster_ci(d, A.seq_units(keys))
    return {"mean": float(np.mean(d)), "ci": [lo, hi], "n_frames": len(keys),
            "n_sequences": len(set(A.seq_units(keys)))}


def _pooled(runs, cname):
    v = [m["iou"]["isec"][cname] for m in runs]
    return {"mean": float(np.mean(v)), "sd": float(np.std(v))}


def _s1(recs, nir, rgb):
    """S1: saturation, registered paddy-edge agreement and D2 at each bound."""
    zs = np.load(SENS_NPZ)
    keyed = {(str(e), str(q), int(f)): i for i, (e, q, f) in
             enumerate(zip(zs["env"], zs["seq"], zs["frame"]))}
    assert len(keyed) == len(recs)
    idx = np.array([keyed[(r["env"], r["seq"], r["frame"])] for r in recs])
    side = np.array([r["osplit"] for r in recs])
    both = np.array([r["px_rgb"][EDGE] > 0 and r["px_nir"][EDGE] > 0
                     for r in recs])
    out = {}
    for b in REG_BOUNDS:
        sh = zs[f"shift_{b}"][idx]
        inter, union = zs[f"inter_edge_{b}"][idx], zs[f"union_edge_{b}"][idx]
        sat = (np.abs(sh) >= b).any(1)
        row = {}
        for name in ("train", "val"):
            sel = side == name
            row[name] = {
                "saturation": float(sat[sel & both].mean()),
                "n_edge_frames": int((sel & both).sum()),
                "agree_reg": float(inter[sel].sum() / union[sel].sum()),
            }
        agree = {k: zs[f"inter_edge_{b}"][keyed[k]] / zs[f"union_edge_{b}"][keyed[k]]
                 for k in nir}
        med = float(np.median(list(agree.values())))
        row["D2"] = {"median_agree_reg": med,
                     "upper": _margin(nir, rgb, [k for k, a in agree.items()
                                                 if a >= med]),
                     "lower": _margin(nir, rgb, [k for k, a in agree.items()
                                                 if a < med])}
        out[str(b)] = row
    ref = out[str(SHIFT_PX)]
    out["reading"] = {
        "a_D2_holds": all(out[str(b)]["D2"]["upper"]["ci"][0] > 0
                          for b in REG_BOUNDS),
        "b_val_below_train": all(out[str(b)]["val"]["agree_reg"]
                                 < out[str(b)]["train"]["agree_reg"]
                                 for b in REG_BOUNDS),
        "b_rise_beyond_tol": {name: max(out[str(b)][name]["agree_reg"]
                                        for b in REG_BOUNDS)
                              - ref[name]["agree_reg"] > AGREE_TOL
                              for name in ("train", "val")},
        "c_val_saturates_more": ref["val"]["saturation"]
        - ref["train"]["saturation"] > SAT_TOL,
        "d_saturated_at_widest": {
            name: out[str(max(REG_BOUNDS))][name]["saturation"] > SAT_TOL
            for name in ("train", "val")},
    }
    return out


def analyze():
    recs = C.load_index()
    z = np.load(AGREE_NPZ)
    keyed = {(str(e), str(s), int(f)): i for i, (e, s, f) in
             enumerate(zip(z["env"], z["seq"], z["frame"]))}
    assert len(keyed) == len(recs)
    out = {"D1": _d1(z, recs)}

    px = A.gt_px()
    runs = [m for m in A.load_runs() if m["protocol"] == "official"]
    by_arm = defaultdict(list)
    for m in runs:
        by_arm[m["arm"]].append(m)
    assert all(len(by_arm[a]) == len(C.SEEDS)
               for a in ("RGB", "NIR", "LUM", "RGBNIR", C.DIAG_ARM)), \
        {a: len(v) for a, v in by_arm.items()}

    def fi(arm, proto):
        return A.frame_iou_by_arm(by_arm[arm], proto, EDGE, A.inc_own(proto),
                                  px)

    nir, rgb = fi("NIR", "isec"), fi("RGB", "isec")
    agree = {k: z["inter_reg"][keyed[k], EDGE] / z["union_reg"][keyed[k], EDGE]
             for k in nir}
    med = float(np.median(list(agree.values())))
    upper = [k for k, a in agree.items() if a >= med]
    lower = [k for k, a in agree.items() if a < med]
    out["D2"] = {"median_agree_reg": med,
                 "upper": _margin(nir, rgb, upper),
                 "lower": _margin(nir, rgb, lower),
                 "all": _margin(nir, rgb)}

    out["D3"] = {"NIR-RGB_on_rgb": _margin(fi("NIR", "rgb"), fi("RGB", "rgb"))}

    fus = fi("RGBNIR", "isec")
    pool = {a: _pooled(by_arm[a], "paddy_edge")
            for a in ("RGB", "LUM", "NIR", "RGBNIR", C.DIAG_ARM)}
    out["pooled_paddy_edge"] = pool
    out["D4"] = {"RGBNIR-RGB": _margin(fus, rgb), "NIR-RGB": _margin(nir, rgb),
                 "share": (pool["RGBNIR"]["mean"] - pool["RGB"]["mean"])
                 / (pool["NIR"]["mean"] - pool["RGB"]["mean"])}

    cross = fi(C.DIAG_ARM, "isec")
    band, ann = _margin(cross, rgb), _margin(nir, cross)
    point = pool["RGB"]["mean"] + HALFWAY * (pool["NIR"]["mean"]
                                             - pool["RGB"]["mean"])
    hi_side = pool[C.DIAG_ARM]["mean"] >= point
    if band["ci"][0] > 0 and hi_side:
        verdict = "band"
    elif ann["ci"][0] > 0 and not hi_side:
        verdict = "annotation"
    else:
        verdict = "both or neither"
    out["D5"] = {"band": band, "annotation": ann, "halfway_point": point,
                 "reading": verdict,
                 "pooled_by_class": {
                     c: {a: _pooled(by_arm[a], c)
                         for a in ("RGB", "NIR", C.DIAG_ARM)}
                     for c in C.TARGETS}}
    if os.path.exists(SENS_NPZ):
        out["S1"] = _s1(recs, nir, rgb)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    {"measure": measure, "sensitivity": sensitivity}.get(
        (sys.argv[1:] or [None])[0], analyze)()
