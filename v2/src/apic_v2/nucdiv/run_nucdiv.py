"""Run the nucdiv shape-diversity port over a slide's per-patch binary masks → 3264-d vector.

Orchestration (faithful to step1_CDF_shape_ccf_best.m):
  for each tile mask: binarize (ch1>0), skimage.label(connectivity=2) + regionprops; SKIP if ≤5
  regions. Build the CCG once per tile from centroids (alpha=0.44, r=0.2); keep networks (connected
  components) with ≥2 members. For each of the 8 shape features: discretize; per kept network, if its
  members span ≥3 distinct bins, form the 7×7 co-occurrence (diagonal bug) → 24 GLCM. Accumulate one
  24-vector per qualifying network (per shape) across all tiles. Then per shape: 24 columns ×
  getStats_all(17) → 408 (haralick-outer, stat-inner); concat 8 shapes → 3264. Empty shape → 408 NaN.

Reads regionprops in v1's order (built-in MATLAB props on the binarized mask). Perimeter/Circularity
are the known skimage-vs-MATLAB divergence points — validate via the MATLAB .m oracle.

Run in env apic_v2_feat (numpy 1.26.4, scikit-image 0.19.3).

Usage:
    python -m apic_v2.nucdiv.run_nucdiv --masks_dir <bridge_out>/<slide>/nucdiv_masks \
        --out <final_dir> --slide_name <name>
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from skimage.io import imread
from skimage.measure import label, regionprops

from apic_v2.nucdiv.nucdiv import (
    FEATURE_MAX, FEATURE_MIN, HARALICK_FEATURES, SHAPE_FEATURES, calc_glcm,
    cell_cluster_networks, column_names, cooccurrence, discretize, get_stats_all,
)

N_HAR, N_STAT, N_SHAPE = 24, 17, 8
MIN_REGIONS = 5          # MATLAB: skip tile if regionprops returns <=5 regions


def matlab_axes(coords):
    """MATLAB regionprops MajorAxisLength/MinorAxisLength/Eccentricity from region pixel coords.

    MATLAB normalizes the second central moments and adds the +1/12 pixel-area correction that
    skimage omits (skimage's axis lengths run ~0.1–1% low). coords: (N,2) int (row, col)."""
    ys = coords[:, 0].astype(float)
    xs = coords[:, 1].astype(float)
    xc = xs - xs.mean()
    yc = ys - ys.mean()
    n = len(coords)
    uxx = (xc * xc).sum() / n + 1.0 / 12.0
    uyy = (yc * yc).sum() / n + 1.0 / 12.0
    uxy = (xc * yc).sum() / n
    common = math.sqrt((uxx - uyy) ** 2 + 4.0 * uxy * uxy)
    major = 2.0 * math.sqrt(2.0) * math.sqrt(uxx + uyy + common)
    minor = 2.0 * math.sqrt(2.0) * math.sqrt(max(uxx + uyy - common, 0.0))
    ecc = (2.0 * math.sqrt((major / 2.0) ** 2 - (minor / 2.0) ** 2) / major) if major > 0 else 0.0
    return major, minor, ecc


def shape_props_for_tile(mask_path: Path):
    """Return (centroids_rc (N,2), shape_values (N,8)) for one binary mask, or (None,None) if ≤5 regions."""
    img = imread(str(mask_path))
    if img.ndim == 3:
        img = img[..., 0]            # MATLAB takes channel 1
    binary = img > 0
    lbl = label(binary, connectivity=2)   # MATLAB default 8-connectivity on the binarized mask
    props = regionprops(lbl)
    if len(props) <= MIN_REGIONS:
        return None, None
    cent = np.array([p.centroid for p in props], dtype=float)        # (row, col)
    vals = np.zeros((len(props), N_SHAPE), dtype=float)
    for i, p in enumerate(props):
        major, minor, ecc = matlab_axes(p.coords)   # MATLAB-exact (skimage's differ ~1%)
        per = p.perimeter                            # NOTE: skimage Perimeter/Circularity/Solidity
        circ = (4.0 * math.pi * p.area / (per ** 2)) if per > 0 else 0.0  # still differ from MATLAB
        vals[i] = [p.area, major, minor, ecc,
                   p.equivalent_diameter, p.solidity, per, circ]
    return cent, vals


def run(masks_dir: Path, out_dir: Path, slide_name: str, faithful_matlab: bool = True):
    masks = sorted(masks_dir.glob("*.png"))
    # per shape feature: a list of 24-dim GLCM vectors (one per qualifying network across all tiles)
    feature_tiles = [[] for _ in range(N_SHAPE)]
    n_tiles_used = 0

    for mp in masks:
        cent, vals = shape_props_for_tile(mp)
        if cent is None:
            continue
        n_tiles_used += 1
        labels = cell_cluster_networks(cent)                 # connected-component id per nucleus
        # networks with >=2 members
        nets = [np.where(labels == g)[0] for g in np.unique(labels)]
        nets = [ix for ix in nets if len(ix) >= 2]
        if not nets:
            continue
        for s in range(N_SHAPE):
            disc = discretize(vals[:, s], FEATURE_MIN[s], FEATURE_MAX[s])   # 1..7 bins per nucleus
            for ix in nets:
                bins = disc[ix]
                glcm = cooccurrence(bins, faithful_matlab=faithful_matlab)  # None if <3 distinct bins
                if glcm is not None:
                    feature_tiles[s].append(calc_glcm(glcm))

    # per shape: (n_net, 24) -> per haralick col getStats_all(17) -> 408 (haralick-outer, stat-inner)
    blocks = []
    per_shape_counts = []
    for s in range(N_SHAPE):
        if not feature_tiles[s]:
            blocks.append(np.full(N_HAR * N_STAT, np.nan))
            per_shape_counts.append(0)
            continue
        mat = np.vstack(feature_tiles[s])                    # (n_net, 24)
        per_shape_counts.append(mat.shape[0])
        stats = np.array([get_stats_all(mat[:, h]) for h in range(N_HAR)])  # (24, 17)
        blocks.append(stats.reshape(-1))                     # haralick-outer, stat-inner
    vec = np.concatenate(blocks)                             # 3264

    out_dir.mkdir(parents=True, exist_ok=True)
    names = column_names()
    assert len(names) == len(vec) == 3264, (len(names), len(vec))
    import csv
    with open(out_dir / f"{slide_name}_nucdiv.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(names)
        w.writerow(["" if (isinstance(v, float) and np.isnan(v)) else repr(float(v)) for v in vec])

    finite = int(np.isfinite(vec).sum())
    print(f"[nucdiv] {slide_name}: {n_tiles_used}/{len(masks)} tiles used; "
          f"networks/shape={per_shape_counts}")
    print(f"[nucdiv] vec len={len(vec)} finite={finite} nan={int(np.isnan(vec).sum())}")
    # the 4 Cox-selected columns (0-based indices from the port spec)
    for nm, idx in [("Area.DiffAvg.Prcnt10", 74), ("Area.Energy.var", 171),
                    ("Area.InvDiffMom.Skewness", 257), ("MinorAxisLength.Energy.Prcnt90", 993)]:
        assert names[idx] == nm, (idx, names[idx], nm)
        print(f"[nucdiv]   {nm} (idx {idx}) = {vec[idx]:.6g}")
    return vec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masks_dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--slide_name", required=True)
    ap.add_argument("--corrected", action="store_true",
                    help="Use the bug-corrected co-occurrence diagonal (default: faithful MATLAB).")
    args = ap.parse_args(argv)
    run(args.masks_dir, args.out, args.slide_name, faithful_matlab=not args.corrected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
