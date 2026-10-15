"""The offset between the RGB and the NIR camera (sec:materials).

RGB and NIR come from two cameras, so their frames need not be
co-registered. An offset between them misaligns the two annotations, and where
the misaligned annotations disagree the intersection ground truth loses pixels
(sec:methods). It also decides whether a per-pixel vegetation index, which
combines the near-infrared and red values of one pixel, can be computed on
these frames (sec:discussion). Measured as the rigid shift that best aligns the
two frames' gradient magnitude. Phase correlation returns an argmax whether or
not the two images share a scene, so the offset carries a null: the same
estimator on frame pairs drawn from different driving sequences.

Parameters: SHIFT_SEARCH_PX, REG_SAMPLE_PER_SITE and PROBE_SEED are CHOSEN
(structural, see their comments); nothing here is fitted.

Usage: python src/phases/camera_offset.py    (no GPU; reads data/study/)
Output: results/camera_offset.json (+ summary to stdout)
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import config as C

OUT = os.path.join(C.ROOT, "results", "camera_offset.json")

# Half-width of the rigid-shift search, in pixels of the 960x544 working
# resolution. CHOSEN: wide enough to contain any offset that would materially
# erode an annotation (a shift this large moves a paddy edge off itself), and
# small enough that the search cannot lock onto a repeating field texture.
SHIFT_SEARCH_PX = 24

# Frames per site. CHOSEN: the shift estimate is per frame and its
# median is what is reported, so this only has to make the median stable.
REG_SAMPLE_PER_SITE = 400
PROBE_SEED = 0          # draws the frame sample and the null's pairing;
                        # every reported value is a median over the whole
                        # draw, so none of them turns on it


def key2site():
    with open(C.SPLITS_JSON) as f:
        sites = json.load(f)["sites"]
    return {m: s["site_id"] for s in sites for m in s["members"]}


def load_grey(path):
    """Frame as float32 grayscale. RGB is reduced by the Rec. 601 luma the
    luminance-control configuration uses, so the two modalities are compared
    on the same definition of brightness that sec:methods gives that arm."""
    im = Image.open(path).convert("L")
    return np.asarray(im, dtype=np.float32)


def grad_mag(a):
    gy, gx = np.gradient(a)
    return np.hypot(gx, gy)


def best_shift(a, b, half):
    """Integer (dy, dx) maximising phase correlation of a against b.

    Phase correlation rather than raw cross-correlation because the two
    modalities have unrelated intensity scales; normalizing the cross-power
    spectrum keeps the estimate about structure rather than about which image
    is brighter."""
    A, B = np.fft.rfft2(a - a.mean()), np.fft.rfft2(b - b.mean())
    R = A * np.conj(B)
    R /= np.maximum(np.abs(R), 1e-12)
    c = np.fft.fftshift(np.fft.irfft2(R, s=a.shape))
    cy, cx = a.shape[0] // 2, a.shape[1] // 2
    win = c[cy - half:cy + half + 1, cx - half:cx + half + 1]
    dy, dx = np.unravel_index(np.argmax(win), win.shape)
    return int(dy - half), int(dx - half), float(win.max())


def cross_pairs(sample, rng):
    """Partner index for each sampled frame, from a different driving sequence.

    The null for a displacement this estimator reports. Pairing a frame's
    RGB image with another sequence's NIR image holds the
    estimator, the search width and the sample fixed and removes only the
    thing being measured, so whatever the estimator returns here it returns
    from no alignment at all. A different SEQUENCE rather than a different
    frame, because consecutive frames of one drive share their scene.

    Rotating a shuffled order gives every frame a partner in one pass; a
    rotation landing inside the frame's own sequence walks on to the next."""
    seq = [f"{r['env']}/{r['seq']}" for r in sample]
    order = rng.permutation(len(sample))
    partner = {}
    for a, i in enumerate(order):
        for b in range(1, len(order)):
            j = order[(a + b) % len(order)]
            if seq[j] != seq[i]:
                partner[int(i)] = int(j)
                break
    return partner


def camera_offset(recs, k2s):
    rng = np.random.RandomState(PROBE_SEED)
    by_site = defaultdict(list)
    for r in recs:
        by_site[k2s[f"{r['osplit']}/{r['env']}/{r['seq']}"]].append(r)
    sample = []
    for site in sorted(by_site):
        pool = by_site[site]
        take = min(REG_SAMPLE_PER_SITE, len(pool))
        sample += [pool[i] for i in
                   rng.choice(len(pool), take, replace=False)]

    partner = cross_pairs(sample, rng)
    shifts, peaks = [], []
    ctrl_shifts, ctrl_peaks = [], []   # the same estimator, mismatched pairs
    for i, r in enumerate(sample):
        g_rgb = grad_mag(load_grey(r["img_rgb"]))
        g_nir = grad_mag(load_grey(r["img_nir"]))
        dy, dx, peak = best_shift(g_rgb, g_nir, SHIFT_SEARCH_PX)
        ctrl_dy, ctrl_dx, ctrl_peak = best_shift(
            g_rgb, grad_mag(load_grey(sample[partner[i]]["img_nir"])),
            SHIFT_SEARCH_PX)
        shifts.append((dy, dx))
        peaks.append(peak)
        ctrl_shifts.append((ctrl_dy, ctrl_dx))
        ctrl_peaks.append(ctrl_peak)

    s = np.array(shifts)
    cs = np.array(ctrl_shifts)
    return {
        "n_frames": len(sample),
        "search_half_width_px": SHIFT_SEARCH_PX,
        "median_dy": float(np.median(s[:, 0])),
        "median_dx": float(np.median(s[:, 1])),
        "median_peak": float(np.median(peaks)),
        "cross_pair": {
            "n_pairs": len(ctrl_shifts),
            "median_dy": float(np.median(cs[:, 0])),
            "median_dx": float(np.median(cs[:, 1])),
            "median_peak": float(np.median(ctrl_peaks)),
        },
    }


def main():
    reg = camera_offset(C.load_index(), key2site())
    with open(OUT, "w") as f:
        json.dump(reg, f, indent=2)
    ctrl = reg["cross_pair"]
    print(f"Camera offset ({reg['n_frames']} frames, phase correlation on "
          f"gradient magnitude, +/-{reg['search_half_width_px']} px)")
    print(f"  median shift (dy, dx) = ({reg['median_dy']:+.0f}, "
          f"{reg['median_dx']:+.0f}) px, peak {reg['median_peak']:.5f}")
    print(f"  null, {ctrl['n_pairs']} pairs across driving sequences: "
          f"median ({ctrl['median_dy']:+.0f}, {ctrl['median_dx']:+.0f}) px, "
          f"peak {ctrl['median_peak']:.5f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
