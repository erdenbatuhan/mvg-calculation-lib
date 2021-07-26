import numpy as np


# ====================================================================================================================================================================
# 1. Linear Independence
# ====================================================================================================================================================================
def is_lin_ind(*vectors):
    A = np.row_stack(vectors)
    U, s, V = np.linalg.svd(A)

    if len(vectors[0]) > s.shape[0]:
        return False

    for i in s:
        if i < 1e-5:
            return False
    
    return True
# ====================================================================================================================================================================


# ====================================================================================================================================================================
# 2. Calculate the pixel-position of the projected point (u, v) in the image and tick the correct answer
# ====================================================================================================================================================================
def pixel_pos_of_proj_point(P=None, R=np.identity(3), C=np.array([[0, 0, 0]]).T, K=None, g_cam_to_world=None, print_steps=False):
    assert(P is not None and K is not None)  # TEST

    # Generic projection matrix
    PI = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])

    # Translator (-C), g = [R, T] in homogeneous coordinates [[R, T], [0, 1]]
    G = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Update translator G
    G[0:3, 0:3] = R  # Rotation
    G[:, 3] = np.append(-C, [1])  # Translation

    # Cam coordinates
    PC = np.array([np.append(P, [1])]).T

    if g_cam_to_world is not None:
        # Transform 3D Point P to camera coordinates
        g_world_to_cam = np.linalg.inv(g_cam_to_world)
        PC = np.dot(g_world_to_cam, PC)

    # Print inputs
    if print_steps:
        print("Inputs:")

        # Print inputs
        print("P:", P)
        print("C:", C)
        print("K:", K)
        print("Generic projection matrix:", PI)
        print("Translator (-C), g:", G)
        print("Cam coordinates:", PC, end="\n\n")

    # ======================================
    # Calculation
    # ======================================
    i1 = np.dot(K, PI)
    i2 = np.dot(i1, G)
    i3 = np.dot(i2, PC)

    final = i3

    # Print final
    if print_steps:
        print("Final: ", final, end="\n\n")

    # Print the solutioon: projected point (u, v)
    lamb, u, v = (final[2])[0], (final[0] / final[2])[0], (final[1] / final[2])[0]
    return lamb, u, v
# ====================================================================================================================================================================


# ====================================================================================================================================================================
# 3. ??
# ====================================================================================================================================================================

