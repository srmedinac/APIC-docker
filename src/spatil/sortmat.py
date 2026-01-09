import numpy as np


def sortmat(
    invect,
    inmat1,
    inmat2,
    inmat3=None,
    inmat4=None,
    inmat5=None,
    inmat6=None,
    inmat7=None,
    inmat8=None,
    inmat9=None,
):

    # If invect is 1D
    if invect.ndim == 1:
        r, c = invect.shape[0], 1
    # If invect is 2D
    else:
        r, c = invect.shape

    if c > 1:
        if r > 1:
            raise ValueError("SORTMAT: first input argument must be a column vector")
        else:
            invect = invect.T
            r = c

    outmat1 = np.empty((0, 0), dtype=invect.dtype)
    outmat2 = np.empty((0, 0), dtype=invect.dtype)
    outmat3 = np.empty((0, 0), dtype=invect.dtype)
    outmat4 = np.empty((0, 0), dtype=invect.dtype)
    outmat5 = np.empty((0, 0), dtype=invect.dtype)
    outmat6 = np.empty((0, 0), dtype=invect.dtype)
    outmat7 = np.empty((0, 0), dtype=invect.dtype)
    outmat8 = np.empty((0, 0), dtype=invect.dtype)
    outmat9 = np.empty((0, 0), dtype=invect.dtype)

    seq = np.argsort(invect)
    outvect = np.sort(invect)

    if inmat1 is not None:
        if inmat1.shape[0] != r:
            raise ValueError(
                "SORTMAT: <inmat1> must have the same number of rows as <invect>"
            )
        outmat1 = inmat1[seq, :]

    if inmat2 is not None:
        if inmat2.shape[0] != r:
            raise ValueError(
                "SORTMAT: <inmat2> must have the same number of rows as <invect>"
            )

    # Check the dimensionality of inmat2
    if inmat2.ndim == 1:
        outmat2 = inmat2[seq]
    else:
        outmat2 = inmat2[seq, :]

    if inmat3 is not None:
        if inmat3.shape[0] != r:
            raise ValueError(
                "SORTMAT: <inmat3> must have the same number of rows as <invect>"
            )
        outmat3 = inmat3[seq, :]

    if inmat4 is not None:
        if inmat4.shape[0] != r:
            raise ValueError(
                "SORTMAT: <inmat4> must have the same number of rows as <invect>"
            )
        outmat4 = inmat4[seq, :]

    if inmat5 is not None:
        if inmat5.shape[0] != r:
            raise ValueError(
                "SORTMAT: <inmat5> must have the same number of rows as <invect>"
            )
        outmat5 = inmat5[seq, :]

    if inmat6 is not None:
        if inmat6.shape[0] != r:
            raise ValueError(
                "SORTMAT: <inmat6> must have the same number of rows as <invect>"
            )
        outmat6 = inmat6[seq, :]

    if inmat7 is not None:
        if inmat7.shape[0] != r:
            raise ValueError(
                "SORTMAT: <inmat7> must have the same number of rows as <invect>"
            )
        outmat7 = inmat7[seq, :]

    if inmat8 is not None:
        if inmat8.shape[0] != r:
            raise ValueError(
                "SORTMAT: <inmat8> must have the same number of rows as <invect>"
            )
        outmat8 = inmat8[seq, :]

    if inmat9 is not None:
        if inmat9.shape[0] != r:
            raise ValueError(
                "SORTMAT: <inmat9> must have the same number of rows as <invect>"
            )
        outmat9 = inmat9[seq, :]

    outvect = np.sort(invect, axis=0)

    return (
        outvect,
        outmat1,
        outmat2,
        outmat3,
        outmat4,
        outmat5,
        outmat6,
        outmat7,
        outmat8,
        outmat9,
    )
