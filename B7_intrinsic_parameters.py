import numpy as np


# Find epipole
def epipoles(F):

    # Compute SVD of F.T
    U_FT, S_FT, Vt_FT = np.linalg.svd(F.T)

    # The left epipole e' is the last column of Vt
    e1 = Vt_FT[-1]

    # Normalize the epipole (make the last coordinate 1)
    e1 = e1 / e1[2]

    # Compute SVD of F
    U_F, S_F, Vt_F = np.linalg.svd(F)

    # Right Epipole --> right null space of F
    e2 = Vt_F[-1]

    # Normalize the epipole (make the last coordinate 1)
    e2 = e2 / e2[2]

    print("Left epipole: ", e1)
    print("Right epipole: ", e2)
    
    # Extend epipoles to matrix form
    e1x = np.array([[0, e1[2], e1[1]],
            [e1[2], 0, -e1[0]],
            [-e1[1], e1[0], 0]])
    e2x = np.array([[0, e2[2], e2[1]],
            [e2[2], 0, -e2[0]],
            [-e2[1], e2[0], 0]])
    
    print("Left epipole (matrix form): ", e1x)
    print("Right epipole (matrix form): ", e2x)

    return e1x, e2x


def compute_Kruppa(F, e1x, e2x):
    pass

def compute_K_param():
    pass
