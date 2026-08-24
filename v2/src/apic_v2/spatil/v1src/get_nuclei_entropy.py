import numpy as np


def get_nuclei_entropy(image):
    J = image.flatten()

    # TODO: Perform this center calculation automatically
    ctrs = np.array(
        [12.75, 38.25, 63.75, 89.25, 114.75, 140.25, 165.75, 191.25, 216.75, 242.25]
    )
    H, _ = np.histogram(J, bins=ctrs)
    P = H / np.sum(H)
    # print(P)
    val = 0
    for i in range(9):
        # print(i)
        if P[i] != 0:
            val += P[i] * np.log2(P[i])

    E = val * -1
    return E


# Example usage:
# Replace 'your_image_array' with the actual image array
# result = get_nuclei_entropy(your_image_array)
# print(result)
