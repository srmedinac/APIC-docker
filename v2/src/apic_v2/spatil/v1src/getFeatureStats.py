import numpy as np
from scipy.stats import kurtosis, skew


def getFeatureStats(feat_vector):
    # GetFeatureStats Summary of this function goes here
    # Detailed explanation goes here

    stats = [
        np.nansum(feat_vector, axis=0),
        np.nanmean(feat_vector, axis=0),
        np.nanstd(feat_vector, axis=0, ddof=1),
        np.nanmedian(feat_vector, axis=0),
        np.nanmax(feat_vector, axis=0),
        np.nanmin(feat_vector, axis=0),
        kurtosis(feat_vector, nan_policy="omit", axis=0, fisher=False),
        skew(feat_vector, nan_policy="omit", axis=0),
    ]

    return stats
