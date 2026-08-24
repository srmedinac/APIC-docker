"""Nuclear-diversity (shape) features — Python port of the v1 MATLAB pipeline.

Faithful port of old_matlab/{step1_CDF_shape_ccf_best.m, auxFuns/*} per
reference/nucdiv_port_spec.md. Produces the 3264-d vector (8 shape × 24 Haralick × 17 stat) with
column order = stat-fastest → haralick → shape, matching v1 generate_nucdiv_column_names().

Per patient: for each tile binary mask (ch1>0), regionprops the nuclei (skip tile if ≤5 regions);
build a cell-cluster graph (alpha=0.44, r=0.2) on the centroids; per connected component (network)
with ≥2 members AND ≥3 distinct discretized bins of a shape feature, form a 7×7 co-occurrence matrix
→ 24 GLCM features. Accumulate one 24-vector per qualifying network (per shape feature) across all
tiles; then per GLCM column take the 17 getStats_all stats → 17×24 → 408 per shape; concat 8 → 3264.

PARITY-CRITICAL (validated against MATLAB .m — see the spec):
  * the co-occurrence DIAGONAL BUG is replicated verbatim under faithful_matlab=True;
  * MATLAB prctile (plotting positions, end-clamped) via scipy mquantiles(alphap=.5,betap=.5);
  * population var / biased skew / excess kurtosis;
  * regionprops Perimeter/Circularity are the known divergence points (skimage ≠ MATLAB) — validate.

Run in env apic_v2_feat (numpy 1.26.4, scikit-image 0.19.3).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kurtosis, skew
from scipy.stats.mstats import mquantiles

# ---- constants (nucdiv_port_spec.md) -----------------------------------------------------------
SHAPE_FEATURES = ["Area", "MajorAxisLength", "MinorAxisLength", "Eccentricity",
                  "EquivDiameter", "Solidity", "Perimeter", "Circularity"]
FEATURE_MIN = np.array([50, 5, 2, 0.4, 6, 0.8, 10, 0.85], dtype=float)
FEATURE_MAX = np.array([2000, 80, 15, 1, 60, 1, 180, 1], dtype=float)
NUM_LEVEL = 6
NG = NUM_LEVEL + 1                      # 7 — co-occurrence / GLCM size (clip-to-max → bin 7)

HARALICK_FEATURES = ["MaxProb", "JointAvg", "JointVar", "Entropy", "DiffAvg", "DiffVar", "DiffEnt",
                     "SumAvg", "SumVar", "SumEnt", "Energy", "Contrast", "Dissimilarity", "InvDiff",
                     "InvDiffNorm", "InvDiffMom", "InvDiffMomNorm", "InvVar", "Correlation",
                     "AutoCorr", "ClusterTend", "ClusterShade", "ClusterProm", "InfoCorr1"]
STAT_NAMES = ["Mean", "var", "Skewness", "Kurtosis", "Median", "Min", "Prcnt10", "Prcnt90", "Max",
              "IqntlRange", "Range", "MAD", "RMAD", "MedAD", "CoV", "Energy", "RMS"]

CG_ALPHA = 0.44
CG_R = 0.2
REALMIN = np.finfo(float).tiny        # 2.2250738585072014e-308


def column_names():
    """3264 names: f'{shape}.{haralick}.{stat}', stat innermost (matches v1)."""
    return [f"{s}.{h}.{st}" for s in SHAPE_FEATURES for h in HARALICK_FEATURES for st in STAT_NAMES]


# ---- MATLAB prctile (plotting positions (i-0.5)/n, linear interp, end-clamped) -----------------
def matlab_prctile(x, q):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.full(np.size(q), np.nan)
    return mquantiles(x, prob=np.asarray(q, dtype=float) / 100.0, alphap=0.5, betap=0.5)


# ---- getStats_all: 17 stats in export order ----------------------------------------------------
def get_stats_all(col):
    """col: 1-D array of per-network values for one GLCM feature. Returns 17 stats (spec order)."""
    x = np.asarray(col, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.full(17, np.nan)
    mean = x.mean()
    var_pop = x.var(ddof=0)
    # MATLAB skewness/kurtosis of a single value => NaN (0/0); match that, not scipy's 0.
    sk = float(skew(x, bias=True)) if x.size > 1 else np.nan
    ku = float(kurtosis(x, fisher=True, bias=True)) if x.size > 1 else np.nan   # kurtosis(x,1)-3
    med = np.median(x)
    xmin, xmax = x.min(), x.max()
    p10, p25, p75, p90 = matlab_prctile(x, [10, 25, 75, 90])
    iqr = p75 - p25
    rng = xmax - xmin
    nV = x.size
    mad = np.abs(x - mean).sum() / nV
    robust = x[(x >= p10) & (x <= p90)]
    rmad = (np.abs(robust - robust.mean()).sum() / robust.size) if robust.size else np.nan
    medad = np.abs(x - med).sum() / nV
    cov = np.sqrt(var_pop) / mean if mean != 0 else np.nan
    energy = np.sum(x ** 2)
    rms = np.sqrt(np.sum(x ** 2) / nV)
    return np.array([mean, var_pop, sk, ku, med, xmin, p10, p90, xmax,
                     iqr, rng, mad, rmad, medad, cov, energy, rms], dtype=float)


# ---- CalcGLCM: 24 Haralick features, exact order -----------------------------------------------
def calc_glcm(glcm):
    g = np.asarray(glcm, dtype=float)
    n = g.shape[0]
    I, J = np.meshgrid(np.arange(1, n + 1), np.arange(1, n + 1))  # indexing='xy' (MATLAB meshgrid)
    out = np.zeros(24)

    out[0] = g.max()                                              # MaxProb
    joint_avg = np.sum(I * g)
    out[1] = joint_avg                                            # JointAvg
    out[2] = np.sum(((I - joint_avg) ** 2) * g)                   # JointVar
    out[3] = -np.sum(g * np.log2(g + REALMIN))                    # Entropy

    # p(x-y) (pDiag, length n) and p(x+y) (pCross, length 2n-1)
    pdiag = np.array([np.trace(g, offset=k) + (np.trace(g, offset=-k) if k > 0 else 0.0)
                      for k in range(n)])
    gf = np.fliplr(g)
    pcross = np.array([np.trace(gf, offset=n - 1 - k) for k in range(2 * n - 1)])

    kdiff = np.arange(n)                                          # 0..n-1
    diff_avg = np.sum(kdiff * pdiag)
    out[4] = diff_avg                                             # DiffAvg
    out[5] = np.sum(((kdiff - diff_avg) ** 2) * pdiag)            # DiffVar
    out[6] = -np.sum(pdiag * np.log2(pdiag + REALMIN))           # DiffEnt
    ksum = np.arange(2, 2 * n + 1)                                # 2..2n
    sum_avg = np.sum(ksum * pcross)
    out[7] = sum_avg                                             # SumAvg
    out[8] = np.sum(((ksum - sum_avg) ** 2) * pcross)            # SumVar
    out[9] = -np.sum(pcross * np.log2(pcross + REALMIN))        # SumEnt

    out[10] = np.sum(g ** 2)                                     # Energy
    out[11] = np.sum(((I - J) ** 2) * g)                         # Contrast
    out[12] = np.sum(np.abs(I - J) * g)                          # Dissimilarity
    out[13] = np.sum(g / (1 + np.abs(I - J)))                    # InvDiff
    out[14] = np.sum(g / (1 + np.abs(I - J) / n))               # InvDiffNorm
    out[15] = np.sum(g / (1 + (I - J) ** 2))                     # InvDiffMom
    out[16] = np.sum(g / (1 + (I - J) ** 2 / n ** 2))           # InvDiffMomNorm
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = g / (I - J) ** 2
    out[17] = 2 * np.sum(np.triu(inv, 1))                        # InvVar (strict upper; diag Inf excluded)

    Pi = g.sum(axis=1)                                           # row marginals
    Pj = g.sum(axis=0)                                           # col marginals
    ui = np.sum(np.arange(1, n + 1) * Pi)
    uj = np.sum(np.arange(1, n + 1) * Pj)
    si = np.sqrt(np.sum(((np.arange(1, n + 1) - ui) ** 2) * Pi))
    sj = np.sqrt(np.sum(((np.arange(1, n + 1) - uj) ** 2) * Pj))
    corr = np.sum((I - ui) * (J - uj) * g) / (si * sj) if si > 0 and sj > 0 else np.nan
    out[18] = 0.0 if np.isnan(corr) else corr                    # Correlation (NaN→0)
    out[19] = np.sum(I * J * g)                                  # AutoCorr
    out[20] = np.sum(((I + J - 2 * ui) ** 2) * g)                # ClusterTend
    out[21] = np.sum(((I + J - 2 * ui) ** 3) * g)                # ClusterShade
    out[22] = np.sum(((I + J - 2 * ui) ** 4) * g)                # ClusterProm

    hx = -np.sum(Pi * np.log2(Pi + REALMIN))
    hy = -np.sum(Pj * np.log2(Pj + REALMIN))
    hxy1 = -np.sum(g * np.log2(np.outer(Pi, Pj) + REALMIN))
    denom = max(hx, hy)
    ic1 = (out[3] - hxy1) / denom if denom > 0 else np.nan
    out[23] = 0.0 if np.isnan(ic1) else ic1                      # InfoCorr1 (NaN→0)
    return out


# ---- per-network co-occurrence (faithful diagonal-bug replication) -----------------------------
def cooccurrence(bins_in_network, faithful_matlab=True):
    """bins_in_network: 1-D int array of discretized bin values (1..NG) for cells in one network.
    Returns the normalized NG×NG co-occurrence, or None if <3 distinct bins (network skipped)."""
    cur = np.unique(bins_in_network[~np.isnan(bins_in_network.astype(float))]).astype(int)
    if len(cur) <= 2:                       # MATLAB: length(cur_g) > 2 required
        return None
    p = np.zeros((NG, NG))
    counts = {v: int(np.sum(bins_in_network == v)) for v in cur}
    # off-diagonal: indexed by bin VALUE (1-based)
    from itertools import combinations
    for a, b in combinations(cur.tolist(), 2):
        p[a - 1, b - 1] = counts[a] * counts[b]
    p = p + p.T
    # diagonal loop: MATLAB indexes by LOOP POSITION jj (1..K), not bin value (the bug)
    if faithful_matlab:
        for jj in range(1, len(cur) + 1):
            p[jj - 1, jj - 1] = int(np.sum(bins_in_network == jj))
    else:
        for v in cur:
            p[v - 1, v - 1] = counts[v]
    s = p.sum()
    if s == 0:
        return None
    return p / s


# ---- graph (Lconstruct_ccgs): alpha=0.44, r=0.2 ------------------------------------------------
def cell_cluster_networks(centroids_rc):
    """centroids_rc: (N,2) array [centroid_r, centroid_c]. Returns labels (N,) of connected comps.
    Edge iff r < D^-alpha  ⇔  D < r^(-1/alpha)."""
    n = len(centroids_rc)
    if n < 2:
        return np.zeros(n, dtype=int)
    thr = CG_R ** (-1.0 / CG_ALPHA)         # ≈ 38.78 px
    D = squareform(pdist(centroids_rc, metric="euclidean"))
    adj = (D < thr) & ~np.eye(n, dtype=bool)
    _, labels = connected_components(adj, directed=False)
    return labels


def discretize(values, fmin, fmax):
    """floor((clip(v,min,max)-min)/w)+1, 1-based; clip-to-max → bin NG. NaN stays NaN."""
    w = (fmax - fmin) / NUM_LEVEL
    v = np.clip(np.asarray(values, dtype=float), fmin, fmax)
    return np.floor((v - fmin) / w) + 1
