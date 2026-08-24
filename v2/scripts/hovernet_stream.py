"""Streaming HoVer-Net nucleus segmentation for APIC — faithful to v1, no disk.

v1's pipeline extracts EVERY patch to JPEG, runs HoVer-Net writing EVERY mask to PNG, then reads
those PNGs back to compute features. That triple round-trip is both the storage blowup and a large
part of the wall-clock. This module keeps the identical computation but streams it: a tile goes
in memory -> HoVer-Net (batched, GPU) -> v1's own mask renderer -> shape properties -> dropped.

Fidelity: it calls v1's ACTUAL code (`nucleusSegmentationTiles.py` from the APIC container — same
weights, same shift=96 padding, same `process()` post-processing, same `render_pred_info_mask`), and
the shape props replicate `run_nucdiv.shape_props_for_tile` exactly. Writing a PNG and reading it
back is lossless, so removing that round-trip cannot change a value.

Runs entirely in the `trident` env (torch+CUDA+openslide+skimage). VERIFIED 2026-07-24: nucdiv shape
properties are BIT-IDENTICAL under skimage 0.19.3 (the v1-faithful pin) and 0.25.2 (trident), across
12,072 values — so the pinned env is not required for these primitives.
"""
from __future__ import annotations

import math
import os
import sys
import types

import numpy as np

APIC_DOCKER = os.environ.get("APIC_DOCKER_DIR", "/opt/apic/v1")

# The v1 driver imports `geojson` only to json.load the type_info file. It isn't installed in the
# trident env, so stub it rather than mutate a shared environment (behaviour is identical).
if "geojson" not in sys.modules:
    try:
        import geojson  # noqa: F401
    except ImportError:
        import json as _json
        _stub = types.ModuleType("geojson")
        _stub.load = _json.load
        _stub.dump = _json.dump
        sys.modules["geojson"] = _stub

if os.path.join(APIC_DOCKER, "src") not in sys.path:
    sys.path.insert(0, os.path.join(APIC_DOCKER, "src"))

import nucleusSegmentationTiles as HN   # noqa: E402  (v1's HoVer-Net: model + postproc + renderer)

SHIFT = 96                    # v1 main(): pad each tile by 96 px (48 per side) before inference
HALF_SHIFT = SHIFT // 2
NR_TYPES = 6                  # PanNuke: 5 nucleus types + background
DEFAULT_WEIGHTS = os.path.join(APIC_DOCKER, "models", "hovernet_fast_pannuke_type_tf2pytorch.tar")
DEFAULT_TYPES = os.path.join(APIC_DOCKER, "types", "type_info_pannuke.json")


class HoverNetStreamer:
    """HoVer-Net held on the GPU; feed it tiles, get in-memory masks. Model loads once."""

    def __init__(self, weights=DEFAULT_WEIGHTS, type_info=DEFAULT_TYPES, mode="fast",
                 nr_types=NR_TYPES, batch_size=8, deterministic=True):
        import json
        if deterministic:
            # cuDNN picks conv algorithms by benchmarked speed, so the same tile can segment
            # slightly differently run-to-run (measured: ~9-11 px of mask, ~1 nucleus in 11k, which
            # moves the APIC score ~0.03%). Pinning the algorithm makes a re-score bit-identical —
            # worth more here than the small throughput loss, since a biomarker that returns a
            # different number for the same slide is hard to defend.
            import torch
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        with open(type_info) as f:
            self.type_info = json.load(f)
        self.batch_size = int(batch_size)
        # infer_step()/process() read these as module globals in the v1 driver
        HN.nr_types = nr_types
        HN.type_info = self.type_info
        HN.model = HN.load_model(weights, mode, nr_types=nr_types)
        self.model = HN.model

    def masks(self, tiles):
        """tiles: iterable of (key, HxWx3 uint8 RGB array). Yields (key, mask uint8 HxW).

        Mask == v1's on-disk PNG content: polygons filled with (nucleus_type + 1), cropped back to
        the tile. Batched so the GPU stays busy; nothing is written to disk."""
        buf_keys, buf_imgs, buf_sizes = [], [], []

        def flush():
            if not buf_imgs:
                return
            batch = np.stack(buf_imgs, axis=0).astype(np.float32)
            for key, info, size in zip(buf_keys, HN.infer_step(batch), buf_sizes):
                yield key, HN.render_pred_info_mask(info, self.type_info, size, SHIFT, HALF_SHIFT)
            buf_keys.clear(); buf_imgs.clear(); buf_sizes.clear()

        for key, arr in tiles:
            if arr is None:
                continue
            size = int(arr.shape[0])
            padded = np.zeros((size + SHIFT, size + SHIFT, 3), dtype=arr.dtype)   # == v1 build_padded_tile
            padded[HALF_SHIFT:HALF_SHIFT + size, HALF_SHIFT:HALF_SHIFT + arr.shape[1]] = arr
            buf_keys.append(key); buf_imgs.append(padded); buf_sizes.append(size)
            if len(buf_imgs) >= self.batch_size:
                yield from flush()
        yield from flush()


# ---- shape properties straight off an in-memory mask ---------------------------------------
# Mirrors apic_v2.nucdiv.run_nucdiv.shape_props_for_tile, which reads a PNG then does exactly this.
MIN_REGIONS = 5
N_SHAPE = 8


def shape_props_from_mask(mask, matlab_axes=None):
    """(centroids_rc (N,2), shape_values (N,8)) for one mask array, or (None, None) if <=5 regions."""
    from skimage.measure import label, regionprops
    if matlab_axes is None:
        from apic_v2.nucdiv.run_nucdiv import matlab_axes as _ma
        matlab_axes = _ma
    img = mask
    if img.ndim == 3:
        img = img[..., 0]                      # MATLAB takes channel 1
    binary = img > 0                            # NOTE: only occupancy matters — types are irrelevant
    lbl = label(binary, connectivity=2)         # MATLAB default 8-connectivity
    props = regionprops(lbl)
    if len(props) <= MIN_REGIONS:
        return None, None
    cent = np.array([p.centroid for p in props], dtype=float)
    vals = np.zeros((len(props), N_SHAPE), dtype=float)
    for i, p in enumerate(props):
        major, minor, ecc = matlab_axes(p.coords)
        per = p.perimeter
        circ = (4.0 * math.pi * p.area / (per ** 2)) if per > 0 else 0.0
        vals[i] = [p.area, major, minor, ecc, p.equivalent_diameter, p.solidity, per, circ]
    return cent, vals
