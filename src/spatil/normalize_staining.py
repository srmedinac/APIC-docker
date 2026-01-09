import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt
import sys


def normalize_staining(I, Io=240, beta=0.15, alpha=1, HERef=None, maxCRef=None):
    if HERef is None:
        HERef = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])

    if maxCRef is None:
        maxCRef = np.array([1.9705, 1.0308])

    h, w, _ = I.shape
    I = I.astype(float).reshape((-1, 3))

    # Calculate optical density
    OD = -np.log((I + 1) / Io)

    # Remove transparent pixels
    ODhat = OD[~np.any(OD < beta, axis=1)]

    if ODhat.size == 0:
        Inorm = np.ones((w, h, 3)) * 255
        plt.imshow(Inorm.astype(np.uint8))
        plt.show()
        sys.exit()

    # Calculate eigenvectors
    _, V = eig(np.cov(ODhat, rowvar=False))

    _, c = V.shape
    if c < 3:
        Inorm = np.ones((w, h, 3)) * 255
        plt.imshow(Inorm.astype(np.uint8))
        plt.show()
        sys.exit()

    # Project on the plane spanned by the eigenvectors corresponding to the two largest eigenvalues
    That = np.dot(ODhat, V[:, 1:3])

    # Find the min and max vectors and project back to OD space
    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = np.dot(V[:, 1:3], [np.cos(minPhi), np.sin(minPhi)])
    vMax = np.dot(V[:, 1:3], [np.cos(maxPhi), np.sin(maxPhi)])

    # A heuristic to make the vector corresponding to hematoxylin first and the one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.column_stack((vMin, vMax))
    else:
        HE = np.column_stack((vMax, vMin))

    # Rows correspond to channels (RGB), columns to OD values
    Y = OD.T

    # Determine concentrations of the individual stains
    C, _, _, _ = np.linalg.lstsq(HE, Y, rcond=None)

    # Normalize stain concentrations
    maxC = np.percentile(C, 95, axis=1)
    C /= maxC[:, None]
    C *= maxCRef[:, None]

    # Recreate the image using reference mixing matrix
    Inorm = Io * np.exp(-np.dot(HERef, C))
    Inorm = Inorm.T.reshape((h, w, 3))
    Inorm = Inorm.astype(np.uint8)

    return Inorm
