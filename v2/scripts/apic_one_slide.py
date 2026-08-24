"""APIC for ONE slide, end to end, streaming — WSI in, {risk_score, risk_group} out.

    WSI ─tissue(GrandQC)→ 1024px level-0 tissue tiles ─HoVer-Net(GPU, batched, in-memory)→
        nucleus masks ─→ nucdiv (4 feats) + spaTIL (X341/X51) ─→ frozen v1 Cox ─→ apic.json

Nothing is written to disk except the final JSON: tiles and masks live in memory for the few
milliseconds they are needed, so a slide costs ~0 storage instead of v1's tens of GB of JPEG+PNG.

FIDELITY — every parameter below was read off v1's own code, not guessed:
  * 1024 px tiles at OpenSlide LEVEL 0, stride 1024, origin (0,0), partial edge tiles dropped, and a
    tile is kept when tissue_fraction >= 0.25 (v1 `max_background` 0.75). **1024, not 2048**: v1's
    docker was later changed to 2048 px (with spaTIL r=0.04748), but the PUBLISHED model — the one
    frozen in apic_cox_frozen.json — was fit on 1024-px-regime features (r=0.07). Mixing regimes
    silently produces wrong scores.
  * v1 saved tiles as JPEG quality 75 and HoVer-Net read them back, so that lossy step is part of the
    published pipeline. Replicated as an IN-MEMORY encode/decode round-trip.
  * No stain normalization before HoVer-Net (v1 applies it only inside spaTIL's get_nuclei_features).
  * spaTIL: alpha=[0.56,0.56], r=0.07; centroids are the mean pixel of DILATED (disk(1)) +
    8-connected-labelled components — NOT HoVer-Net instances (dilation merges touching nuclei, so
    the counts legitimately differ); groups are [non-immune, immune] and immune = PanNuke type 2
    (inflam) = fill value 3. (v1's on-disk masks encode lymphocyte as 128, but that is an artifact of
    `plt.imsave(cmap='gray')` autoscaling in the OLD container, not a type id — using type==2 is the
    correct semantic for masks we render ourselves.)
  * A tile whose mask has NO nuclei is dropped from every aggregate (v1 wrote no PNG for it); a tile
    with <3 total nuclei contributes zeros(350) to the spaTIL mean, which IS included.
  * spaTIL slide value = np.nanmean over per-tile vectors; X341 = index 342, X51 = index 52 (no shift).

Deviation from v1 (accepted by the project lead): tissue comes from Trident/GrandQC instead of
HistoQC. It changes WHICH tiles pass the 0.25 rule, not how anything is computed.

Runs in the `trident` env (torch+CUDA, openslide, skimage, cv2) — verified that both the nucdiv shape
primitives and v1's spaTIL code are numerically identical there vs the old numpy-1.26/skimage-0.19 pins.

Usage: python scripts/apic_one_slide.py --slide <wsi> --out <apic.json> [--stem NAME] [--batch 2]
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for p in (os.path.join(_REPO, "src"), _HERE, os.path.join(_REPO, "src/apic_v2/spatil/v1src")):
    if p not in sys.path:
        sys.path.insert(0, p)

TILE = 1024              # v1 patch_extraction.patch_size (published regime)
LEVEL = 0
MIN_TISSUE = 0.25        # keep tile if tissue_fraction >= 0.25  (v1 max_background 0.75)
JPEG_QUALITY = 75        # PIL default that v1 relied on
ALPHA, RADIUS = [0.56, 0.56], 0.07
SPATIL_N = 350
IMMUNE_TYPE = 2          # PanNuke inflam -> mask fill value 3 (= type + 1)
SIG = {"Area.DiffAvg.Prcnt10": 74, "Area.Energy.var": 171,
       "Area.InvDiffMom.Skewness": 257, "MinorAxisLength.Energy.Prcnt90": 993}
SIG_LOC = {n: (i // 408, i % 408) for n, i in SIG.items()}
SHAPES = sorted({s for s, _ in SIG_LOC.values()})


def log(msg):
    print(msg, flush=True)


# ---- tissue mask -----------------------------------------------------------------------------
# MEASURED 2026-07-24: the tissue mask is NOT interchangeable. spaTIL is a per-tile graph averaged
# over tiles and is decided by a HANDFUL of them (19 of 20 tiles contribute exactly 0 to X341), so a
# different tissue mask scrambles it — GrandQC gave X341 r=0.39 vs v1 and cost ~20 points of APIC
# group agreement (72% vs 92%). v1's own HistoQC + config_v3.ini reproduces v1's tile set exactly
# (43/43 of its tiles recovered, 1 extra, Jaccard 0.977) in ~14 s, so we use it and keep GrandQC only
# as a fallback when Docker isn't available.
# MEASURED 2026-07-27 on slide 1080 (v1's stored HistoQC mask = ground truth, tissue fraction 0.0289):
#   v1.0.4                 -> 0.0291  (1.04% px diff)   <- matches, and 82 vs v1's 80 tiles
#   v1.0.2                 -> 0.0230  (1.59% px diff)
#   GIFU-2048-patch-fix    -> 0.0182  (37% less tissue)  <- WRONG ERA, only 49 of v1's 80 tiles
# v1's batch-3 masks are dated 2026-01-11, which is the v1.0.x era, not GIFU (Apr 2026). HistoQC is
# itself NONDETERMINISTIC (same image/slide/config twice = 1.23% px diff), so v1.0.4's 1.04% gap vs
# v1's stored mask is INSIDE that noise — it is the right version and the residual is irreducible.
HISTOQC_IMAGE = os.environ.get("APIC_HISTOQC_IMAGE", "madabhushilabapic/apic:v1.0.4")
# Set inside the apic_v2 image so HistoQC runs as a local subprocess instead of via docker.
HISTOQC_PYTHON = os.environ.get("APIC_HISTOQC_PYTHON", "")
HISTOQC_CONFIG = os.environ.get("APIC_HISTOQC_CONFIG", "/opt/apic/v1/src/config_v3.ini")


def tissue_mask_histoqc(slide_path, workdir, downsample=None, cache_dir=None):
    """v1's HistoQC mask (run from the APIC container so the version + config are v1's own).
    Returns (binary mask, (scale_x, scale_y), (W,H)) or raises if HistoQC produced nothing.

    DETERMINISM: HistoQC is not reproducible — the same image/slide/config run twice differs by
    ~1.2-1.5% of mask pixels (measured; `-n 1` does not fix it, so it is internal randomness, not
    the worker pool). The total tissue is stable, but boundary pixels wobble, and because a tile is
    kept at exactly 25% tissue that noise flips borderline tiles — which moves spaTIL, which is
    decided by a handful of tiles. So the mask is CACHED per slide: computed once, reused forever.
    Re-scoring a slide then gives a bit-identical answer, and re-runs also skip the ~14 s HistoQC.
    (Switching to GrandQC would not fix this: it is a GPU net with its own nondeterminism, and it
    selects a materially different tile set than v1.)"""
    import shutil
    import subprocess
    import openslide
    from PIL import Image
    if cache_dir:
        cached = os.path.join(cache_dir, "tissue_mask.png")
        if os.path.isfile(cached):
            try:
                m = np.array(Image.open(cached))
                if m.ndim == 3:
                    m = m[..., 0]
                W, H = openslide.OpenSlide(os.path.abspath(slide_path)).dimensions
                log("        tissue mask: reusing cached HistoQC mask (deterministic re-run)")
                return (m > 0).astype(np.uint8), (W / m.shape[1], H / m.shape[0]), (W, H)
            except Exception as e:
                # a truncated/corrupt cache must not raise out of here — the caller treats ANY
                # exception as "HistoQC unavailable" and silently falls back to GrandQC forever.
                log("        tissue mask: cached mask unreadable (%s) — recomputing" % type(e).__name__)
                try:
                    os.unlink(cached)
                except OSError:
                    pass
    slide_path = os.path.abspath(slide_path)
    sdir, sname = os.path.dirname(slide_path), os.path.basename(slide_path)
    # Per-slide output dir: a shared one lets the mask glob below pick up a PREVIOUS slide's mask
    # when a caller reuses the workdir across slides (silently scoring slide N with slide N-1's
    # tissue). Keyed by the slide's own name, and the glob is filtered by that name as a second guard.
    key = os.path.splitext(sname)[0]
    out = os.path.join(workdir, "histoqc", key)
    run = os.path.join(workdir, "histoqc_run")     # HistoQC writes its logfile to CWD, and then
    shutil.rmtree(out, ignore_errors=True)          # copies it into -o, so CWD must be writable AND
    os.makedirs(out, exist_ok=True)                 # different from -o (else SameFileError).
    os.makedirs(run, exist_ok=True)
    # Two ways to reach v1's HistoQC, same binary and same flags either way:
    #  * APIC_HISTOQC_PYTHON set (inside the apic_v2 image) -> call the vendored histoqc_env directly.
    #    HistoQC is py3.8/numpy1.23/skimage0.19 and our runtime is py3.10/numpy2.2 — they cannot share
    #    an env, and forcing HistoQC onto modern pins would change the tissue mask, which is the one
    #    artifact measured as fidelity-critical. So it ships as its own interpreter, not as a port.
    #  * otherwise (dev box) -> run it from the v1 image via docker.
    # cwd MUST be `run` and writable: HistoQC copies its pen/models/templates into a TemporaryDirectory
    # under the CWD and rewrites the config at runtime.
    if HISTOQC_PYTHON:
        subprocess.run([HISTOQC_PYTHON, "-m", "histoqc", "-c", HISTOQC_CONFIG,
                        "-n", "3", "--force", "-o", out, slide_path],
                       check=True, capture_output=True, timeout=1800, cwd=run)
    else:
        subprocess.run(
            ["docker", "run", "--rm", "--name", "apic-hqc-%s" % os.environ.get("APIC_JOB_ID", os.getpid()),
             "--user", "%d:%d" % (os.getuid(), os.getgid()), "-w", "/run",
             "-v", "%s:/wsi:ro" % sdir, "-v", "%s:/out" % out, "-v", "%s:/run" % run,
             "--entrypoint", "bash", HISTOQC_IMAGE, "-c",
             "conda run --no-capture-output -n histoqc_env python -m histoqc "
             "-c /app/src/config_v3.ini -n 3 --force -o /out '/wsi/%s'" % sname],
            check=True, capture_output=True, timeout=1800)
    hits = []
    for root, _, files in os.walk(out):
        hits += [os.path.join(root, f) for f in files
                 if "mask_use" in f and f.endswith(".png") and key in f]
    if not hits:
        raise RuntimeError("HistoQC produced no mask_use image for %s" % key)
    if len(hits) > 1:
        raise RuntimeError("ambiguous HistoQC masks for %s: %d found" % (key, len(hits)))
    m = np.array(Image.open(hits[0]))
    if m.ndim == 3:
        m = m[..., 0]
    if cache_dir:                                   # freeze it so every future run is identical
        try:
            os.makedirs(cache_dir, exist_ok=True)
            Image.fromarray(((m > 0).astype(np.uint8) * 255)).save(os.path.join(cache_dir, "tissue_mask.png"))
        except Exception:
            pass
    W, H = openslide.OpenSlide(slide_path).dimensions
    return (m > 0).astype(np.uint8), (W / m.shape[1], H / m.shape[0]), (W, H)


def tissue_mask(slide_path, workdir, device="cuda:0", downsample=32):
    """Binary tissue mask at 1/downsample of level 0, from Trident's GrandQC segmenter (FALLBACK —
    prefer tissue_mask_histoqc; see the note above on why the mask source is not interchangeable)."""
    import cv2
    from trident import load_wsi
    from trident.segmentation_models import segmentation_model_factory
    seg = segmentation_model_factory(model_name="grandqc", confidence_thresh=0.5)
    s = load_wsi(slide_path=slide_path, lazy_init=False)
    s.segment_tissue(segmentation_model=seg, target_mag=seg.target_mag, job_dir=workdir,
                     device=device, holes_are_tissue=True)
    gj = os.path.join(workdir, "contours_geojson", f"{s.name}.geojson")
    W, H = s.dimensions if hasattr(s, "dimensions") else (None, None)
    if W is None:
        import openslide
        W, H = openslide.OpenSlide(slide_path).dimensions
    mh, mw = int(np.ceil(H / downsample)), int(np.ceil(W / downsample))
    mask = np.zeros((mh, mw), np.uint8)
    with open(gj) as f:
        fc = json.load(f)
    for feat in fc.get("features", []):
        g = feat.get("geometry") or {}
        polys = g.get("coordinates", []) if g.get("type") == "Polygon" else \
            [r for poly in g.get("coordinates", []) for r in poly]
        for ring in polys:
            pts = np.asarray(ring, float)
            if pts.ndim != 2 or len(pts) < 3:
                continue
            cv2.fillPoly(mask, [np.round(pts / downsample).astype(np.int32)], 1)
    return mask, downsample, (W, H)


def tile_grid(mask, scale, dims):
    """v1's grid + acceptance rule (tissue_segmentation_patches.py): level-0 grid anchored at (0,0),
    stride == TILE, partial edge tiles dropped, keep when tissue_fraction >= 0.25. `scale` is
    (scale_x, scale_y) = level0 px per mask px — a scalar is accepted for the square case."""
    W, H = dims
    mh, mw = mask.shape
    sx, sy = scale if isinstance(scale, (tuple, list)) else (scale, scale)
    keep = []
    for y in range(0, H - TILE + 1, TILE):
        for x in range(0, W - TILE + 1, TILE):
            x0, x1 = max(0, min(mw, int(x / sx))), max(0, min(mw, int((x + TILE) / sx)))
            y0, y1 = max(0, min(mh, int(y / sy))), max(0, min(mh, int((y + TILE) / sy)))
            if x1 > x0 and y1 > y0 and float((mask[y0:y1, x0:x1] > 0).mean()) >= MIN_TISSUE:
                keep.append((x, y))
    return keep


# ---- feature accumulation --------------------------------------------------------------------
def spatil_centroids(mask):
    """v1 get_nuclei_features geometry: binarize -> dilate(disk(1)) -> label -> mean pixel."""
    from skimage.measure import label
    from skimage.morphology import binary_dilation, disk
    lbl = label(binary_dilation(mask > 0, disk(1)))
    n = int(lbl.max())
    if n == 0:
        return np.zeros((0, 2)), lbl
    cents = np.zeros((n, 2), float)
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        cents[i - 1] = (ys.mean(), xs.mean())
    return cents, lbl


def process_slide(slide_path, stem, workdir, model_path, streamer=None, batch=2, mask_cache=None):
    """Score ONE slide and return the result dict. `streamer` lets a caller (e.g. the concordance
    harness) reuse one GPU-resident HoVer-Net across many slides instead of reloading it each time."""
    import openslide
    from PIL import Image
    from hovernet_stream import HoverNetStreamer, shape_props_from_mask
    from apic_v2.nucdiv.nucdiv import (cooccurrence, calc_glcm, get_stats_all, discretize,
                                       cell_cluster_networks, FEATURE_MIN, FEATURE_MAX)
    from get_spaTIL_features import get_spaTIL_features
    import apic_score

    t0 = time.time()
    args = type("A", (), {"slide": slide_path, "batch": batch, "model": model_path})()

    log("SEGSTAGE:tissue")
    try:                                            # v1-faithful mask (reproduces v1's tile set)
        mask, ds, dims = tissue_mask_histoqc(slide_path, workdir, cache_dir=mask_cache)
        tissue_src = "histoqc"
    except Exception as e:
        log(f"        HistoQC unavailable ({type(e).__name__}) -> GrandQC fallback "
            f"(NOTE: changes tile selection, which shifts spaTIL)")
        mask, ds, dims = tissue_mask(slide_path, workdir)
        tissue_src = "grandqc"
    tiles = tile_grid(mask, ds, dims)
    log(f"        slide {dims[0]}x{dims[1]} | {len(tiles)} tissue tiles @ {TILE}px level0 [{tissue_src}]")
    if not tiles:
        raise SystemExit("no tissue tiles found")

    osl = openslide.OpenSlide(args.slide)

    def tile_iter():
        """Read a tile, replicate v1's JPEG-q75 round-trip, hand it to HoVer-Net — all in memory."""
        for (x, y) in tiles:
            rgb = osl.read_region((x, y), LEVEL, (TILE, TILE)).convert("RGB")
            buf = io.BytesIO()
            rgb.save(buf, "JPEG", quality=JPEG_QUALITY)   # lossy step v1 baked in
            buf.seek(0)
            yield (x, y), np.array(Image.open(buf).convert("RGB"))

    log("SEGSTAGE:nuclei")
    st = streamer or HoverNetStreamer(batch_size=args.batch)
    acc = {s: [] for s in SHAPES}
    per_patch, n_kept, n_nuclei, done = [], 0, 0, 0
    for (x, y), mk in st.masks(tile_iter()):
        done += 1
        if done % 5 == 0 or done == len(tiles):
            log(f"apic_tiles {done}/{len(tiles)}")
        if not mk.any():                     # v1 wrote no PNG -> tile drops out of every aggregate
            continue
        n_kept += 1
        # --- nucdiv (4 signature features) ---
        cent, vals = shape_props_from_mask(mk)
        if cent is not None:
            lab = cell_cluster_networks(cent)
            nets = [ix for ix in (np.where(lab == g)[0] for g in np.unique(lab)) if len(ix) >= 2]
            for s in SHAPES:
                dd = discretize(vals[:, s], FEATURE_MIN[s], FEATURE_MAX[s])
                for ix in nets:
                    gl = cooccurrence(dd[ix])
                    if gl is not None:
                        acc[s].append(calc_glcm(gl))
        # --- spaTIL (350-d, we keep X341/X51) ---
        cents, _ = spatil_centroids(mk)
        n_nuclei += len(cents)
        if len(cents) < 3:
            per_patch.append(np.zeros(SPATIL_N))          # v1: counted in the mean
            continue
        rc = np.round(cents).astype(int)
        rc[:, 0] = np.clip(rc[:, 0], 0, mk.shape[0] - 1); rc[:, 1] = np.clip(rc[:, 1], 0, mk.shape[1] - 1)
        immune = (mk[rc[:, 0], rc[:, 1]] == IMMUNE_TYPE + 1)      # fill == type+1
        try:
            f, _ = get_spaTIL_features([cents[~immune], cents[immune]], alpha=ALPHA, r=RADIUS)
            per_patch.append(np.asarray(f, float).reshape(-1))
        except Exception:
            per_patch.append(np.full(SPATIL_N, np.nan))   # v1: crashed tile excluded from the mean

    log("SEGSTAGE:scoring")

    def block_stats(g):
        if not g:
            return np.full(408, np.nan)
        return np.array([get_stats_all(np.vstack(g)[:, h]) for h in range(24)]).reshape(-1)

    feats = {n: float(block_stats(acc[s])[loc]) for n, (s, loc) in SIG_LOC.items()}
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agg = np.nanmean(np.vstack(per_patch), axis=0) if per_patch else np.full(SPATIL_N, np.nan)
    feats["X341"], feats["X51"] = float(agg[342]), float(agg[52])

    model = apic_score.load_model(args.model)
    bad = [k for k, v in feats.items() if not (isinstance(v, float) and math.isfinite(v))]
    if bad:                                          # sparse slide / failed stage -> FAIL the job,
        raise SystemExit(                            # never emit a confident-looking negative
            "SEGSTAGE:error APIC features non-finite %s (tiles=%d with_nuclei=%d nuclei=%d)"
            % (bad, len(tiles), n_kept, n_nuclei))
    res = apic_score.score(model, feats)

    # ---- magnification provenance + extrapolation check -------------------------------------
    # MEASURED: the published Cox model was fit on CHAARTED, which is 20x (mpp ~0.503). v1 applies
    # it to slides of ANY magnification with NO mpp normalization, and that is the published assay —
    # so we reproduce it rather than "fixing" it. But it has a consequence worth surfacing:
    # Area.DiffAvg.Prcnt10 is magnification-dependent (median 1.045 at 20x vs 0.888889 at 40x), and
    # on 40x slides 99.5% of v1's own values fall BELOW the model's training minimum — i.e. the score
    # extrapolates. We record the slide's magnification and flag every feature that lands outside the
    # range the model was trained on, so a clean-looking number never hides that caveat.
    props = osl.properties
    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    mpp = _f(props.get("openslide.mpp-x"))
    obj = _f(props.get("openslide.objective-power"))
    oor = []
    for k in model["features"]:
        mn, mx = model["feat_min"][k], model["feat_max"][k]
        x = feats[k]
        if x == x and (x < mn or x > mx):
            oor.append({"feature": k, "value": x, "train_min": mn, "train_max": mx,
                        "side": "below" if x < mn else "above"})
    out = {"stem": stem, "slide": args.slide, "model": model.get("model"),
           "risk_score": res["risk_score"], "risk_group": res["risk_group"],
           "apic_status": res["apic_status"], "threshold": res["threshold"],
           "features": feats, "n_tiles_tissue": len(tiles), "n_tiles_with_nuclei": n_kept,
           "n_nuclei": int(n_nuclei), "tile_px": TILE, "spatil_r": RADIUS, "tissue": tissue_src,
           "objective_power": obj, "mpp_x": mpp,
           "model_trained_mag": "20x (CHAARTED, mpp~0.503)",
           "features_out_of_train_range": oor,
           "extrapolating": bool(oor),
           "warning": (("score extrapolates: %d of 6 features fall outside the model's training "
                        "range (%s). Expected on %sx slides — the model was fit on 20x CHAARTED and "
                        "v1 applies it without magnification normalization."
                        % (len(oor), ", ".join(o["feature"] for o in oor),
                           ("%g" % obj) if obj else "?")) if oor else None),
           "seconds": round(time.time() - t0, 1)}
    log(f"        {res['risk_group']} ({res['apic_status']}) risk={res['risk_score']:.6f} "
        f"thr={res['threshold']:.6f} | tiles {n_kept}/{len(tiles)} nuclei {n_nuclei} "
        f"| {out['seconds']}s")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slide", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stem", default=None)
    ap.add_argument("--batch", type=int, default=int(os.environ.get("APIC_BATCH", "2")))
    ap.add_argument("--model", default=os.path.join(_REPO, "models", "apic_cox_frozen.json"))
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()
    stem = args.stem or os.path.splitext(os.path.basename(args.slide))[0]
    workdir = args.workdir or os.path.join(os.path.dirname(os.path.abspath(args.out)), "_trident")
    out = process_slide(args.slide, stem, workdir, args.model, batch=args.batch,
                        mask_cache=os.path.dirname(os.path.abspath(args.out)))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, allow_nan=False)
    log("SEGSTAGE:done")


if __name__ == "__main__":
    main()
