"""spaTIL features for APIC v2 — fed by the bridge's per-patch centroids + immune membership.

Replaces v1's PNG-mask path (extract_til_features.py): the 350-d spaTIL vector depends only on
nuclei centroids split into two groups [non-immune, immune] (verified — the RGB image and
epi/stroma masks were dead code). For each Trident patch we build those two centroid groups and
call the **verbatim** v1 ``get_spaTIL_features`` (in ./v1src), then aggregate patch→slide by
``np.nanmean`` exactly as v1 step6_aggregate_features.

Calibration pinned to the validated regime (PLAN.md D7): alpha=[0.56, 0.56], r=0.07 (1024px@40x).
Coordinates are at the 40x reference scale (the bridge already converted), so r stays valid.

v1 parity notes:
- A patch with <3 total cells contributes ``zeros(350)`` (v1 extract_til_features line 54-55), and
  those zeros ARE included in the nanmean (v1 saved them as CSVs). Replicated.
- Per-patch compute is wrapped to fall back to zeros(350) on error (e.g. a degenerate empty group),
  which is strictly more robust than v1; the count of such patches is reported.

Usage (env: histoplus — has numpy/scipy/shapely/cv2):
    python -m apic_v2.spatil.run_spatil --spatil_input <bridge_out>/<slide>/spatil_input.npz \
        --out <final_dir> --slide_name <name>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# v1 spaTIL modules use flat sibling imports (`from construct_nodesCluster_new import ...`);
# put v1src on the path so the validated code runs unchanged.
_V1SRC = os.path.join(os.path.dirname(__file__), "v1src")
if _V1SRC not in sys.path:
    sys.path.insert(0, _V1SRC)

from get_spaTIL_features import get_spaTIL_features  # noqa: E402  (from v1src)

ALPHA = [0.56, 0.56]   # D7 validated regime
R = 0.07
N_FEATURES = 350


def spatil_for_patch(xy: np.ndarray, immune: np.ndarray) -> np.ndarray:
    """350-d spaTIL vector for one patch (>=3 cells). Raises on failure — caller decides policy."""
    coords = [xy[~immune].astype(float), xy[immune].astype(float)]  # [non-immune, immune] (v1 order)
    feats, _ = get_spaTIL_features(coords, alpha=ALPHA, r=R)
    feats = np.asarray(feats, dtype=float).reshape(-1)
    if feats.shape[0] != N_FEATURES:
        raise ValueError(f"got {feats.shape[0]} features, expected {N_FEATURES}")
    return feats


def run(spatil_input: Path, out_dir: Path, slide_name: str):
    z = np.load(spatil_input)
    xy, immune, patch_id = z["xy"], z["immune"], z["patch_id"]
    n_patches = int(z["patch_coords"].shape[0])

    per_patch = np.full((n_patches, N_FEATURES), np.nan)
    n_sparse = n_failed = 0
    for i in range(n_patches):
        sel = patch_id == i
        xy_i, imm_i = xy[sel], immune[sel]
        if len(xy_i) < 3:
            per_patch[i] = 0.0   # v1: <3 cells -> zeros(350), INCLUDED in the nanmean
            n_sparse += 1
            continue
        try:
            per_patch[i] = spatil_for_patch(xy_i, imm_i)
        except Exception as e:
            per_patch[i] = np.nan  # v1 skipped failed patches -> EXCLUDED from nanmean (NaN, not zero)
            n_failed += 1
            if n_failed <= 3:
                print(f"[spatil] WARN patch {i} failed: {type(e).__name__}: {e}")
    if n_failed > 0.02 * n_patches:
        print(f"[spatil] *** {n_failed}/{n_patches} patches FAILED — likely a systematic bug, not noise ***")

    # slide-level aggregation: nanmean across patches (v1 step6)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        agg = np.nanmean(per_patch, axis=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [f"feature_{i}" for i in range(N_FEATURES)]
    import csv
    with open(out_dir / f"{slide_name}_spatil_aggregated.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerow(["" if np.isnan(v) else repr(float(v)) for v in agg])
    # keep the per-patch matrix for debugging / re-aggregation
    np.savez_compressed(out_dir / f"{slide_name}_spatil_per_patch.npz", per_patch=per_patch)

    print(f"[spatil] {slide_name}: {n_patches} patches ({n_sparse} <3 cells, {n_failed} zero-vectors)")
    print(f"[spatil] feature_52={agg[52]:.6g}  feature_342={agg[342]:.6g}  "
          f"(nan cols: {int(np.isnan(agg).sum())}/{N_FEATURES})")
    return agg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spatil_input", required=True, type=Path, help="bridge spatil_input.npz")
    ap.add_argument("--out", required=True, type=Path, help="final-features output dir")
    ap.add_argument("--slide_name", required=True)
    args = ap.parse_args(argv)
    run(args.spatil_input, args.out, args.slide_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
