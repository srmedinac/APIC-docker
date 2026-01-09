import numpy as np
from getFeatureStats import getFeatureStats


def getClusterDensityMeasures_v2(groups):
    features = []
    featureNames = []

    numGroups = len(groups)

    stats = ["Total", "Mean", "Std", "Median", "Max", "Min", "Kurtosis", "Skewness"]
    meas = ["AreaClusters_G", "DensityClusters_G"]

    for i in range(numGroups):
        areas = groups[i]["areas"]
        dens = groups[i]["densities"]

        if len(areas) > 0:
            features += getFeatureStats(areas) + getFeatureStats(dens)
        else:
            features += [0] * (2 * 8)

        for ms in range(len(meas)):
            for st in range(len(stats)):
                featureNames.append(f"{stats[st]}{meas[ms]}{i + 1}")

    return features, featureNames
