import numpy as np
from scipy.ndimage import binary_erosion
import matplotlib.pyplot as plt

def bwperim(b, conn=None):
    if conn is None:
        num_dims = len(b.shape)
        conn = conndef(num_dims, 'minimal')
    elif isinstance(conn, int):
        conn = ScalarToArray(conn)

    if b.dtype != bool:
        b = b.astype(bool)

    if len(b.shape) > 2:
        raise ValueError("Invalid input: bwperim supports only 2D images.")

    # If it's a 2-D problem with 4- or 8-connectivity, use binary_erosion
    if b.shape[0] == 2 and np.array_equal(conn, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])):
        p = binary_erosion(b, structure=np.ones((3, 3)))
    elif b.shape[0] == 2 and np.array_equal(conn, np.ones((3, 3))):
        p = binary_erosion(b, structure=np.ones((3, 3)))
    else:
        # Use a general technique that works for any dimensionality and any connectivity.
        num_dims = max(len(b.shape), len(conn.shape))
        pad_width = [(1, 1)] * num_dims
        b = np.pad(b, pad_width, mode='constant', constant_values=0)
        b_eroded = binary_erosion(b, structure=conn)
        p = np.logical_and(b, ~b_eroded)

        slices = [slice(1, s - 1) for s in b.shape]
        p = p[tuple(slices)]

    return p


def imshow(image):
    plt.imshow(image, cmap='gray')
    plt.show()


def conndef(num_dims, conn_type):
    # Validation
    if not isinstance(num_dims, int) or num_dims < 2:
        raise ValueError("num_dims must be a scalar integer greater than or equal to 2.")
    
    conn_type = conn_type.lower()
    if conn_type not in ['minimal', 'maximal']:
        raise ValueError("Invalid connectivity type. Use 'minimal' or 'maximal'.")

    # Create connectivity array
    if conn_type == 'minimal':
        conn = np.zeros(tuple([3] * num_dims))
        center = tuple((np.array(conn.shape) - 1) // 2)  # Corrected indexing

        for k in range(num_dims):
            conn[tuple(center + np.eye(num_dims, dtype=int)[k])] = 1
            conn[tuple(center - np.eye(num_dims, dtype=int)[k])] = 1

    elif conn_type == 'maximal':
        conn = np.ones(tuple([3] * num_dims))

    return conn



def ScalarToArray(conn):
    if np.isscalar(conn):
        if conn == 1:
            conn_out = 1
        elif conn == 4:
            conn_out = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        elif conn == 8:
            conn_out = np.ones((3, 3))
        elif conn == 6:
            conn_out = conndef(3, 'minimal')
        elif conn == 18:
            conn_out = np.dstack([
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]],
                np.ones((3, 3)),
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0])
            
        elif conn == 26:
            conn_out = conndef(3, 'maximal')
        else:
            raise ValueError("Unexpected conn value")
    else:
        conn_out = conn

    return conn_out
