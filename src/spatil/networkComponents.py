import numpy as np


def networkComponents(A):
    # Number of nodes
    N = A.shape[0]
    # Remove diagonals
    np.fill_diagonal(A, 0)
    # Make symmetric, just in case it isn't
    A = np.maximum(A, A.T)
    # Have we visited a particular node yet?
    isDiscovered = np.zeros(N, dtype=bool)
    # Empty members list
    members = []

    # Check every node
    for n in range(N):
        if not isDiscovered[n]:
            # Started a new group so add it to members
            members.append([n])
            # Account for discovering n
            isDiscovered[n] = True
            # Set the pointer to 1
            ptr = 0
            while ptr < len(members[-1]):
                # Find neighbors
                nbrs = np.nonzero(A[:, members[-1][ptr]])[0]
                # Here are the neighbors that are undiscovered
                new_nbrs = nbrs[~isDiscovered[nbrs]]
                # We can now mark them as discovered
                isDiscovered[new_nbrs] = True
                # Add them to member list
                members[-1].extend(new_nbrs)
                # Increment ptr so we check the next member of this component
                ptr += 1

    # Number of components
    nComponents = len(members)

    sizes = np.array([len(component) for component in members])
    # Sort sizes and members in descending order
    idx = np.argsort(sizes)[::-1]
    sizes = sizes[idx]
    members = [members[i] for i in idx]

    return nComponents, sizes, members
