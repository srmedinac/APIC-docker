import numpy as np
from scipy.spatial.distance import pdist, squareform


def getSumNodeWeightsThreshold(feature, distance, threshold):
    # getting distances, removing 0
    dist = pdist(feature, distance)
    dist[dist == 0] = 1
    dist = dist**-1

    # normalizing
    # dist = dist / max(dist)

    # applying threshold
    dist[dist < threshold] = 0

    # computing the value of each node
    vect = np.sum(squareform(dist), axis=0)

    return vect
