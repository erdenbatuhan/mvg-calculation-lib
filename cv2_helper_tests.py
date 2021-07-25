import numpy as np

import cv2_helper as helper


# ====================================================================================================================================================================
# 1. Linear Independence
# ====================================================================================================================================================================
def is_lin_ind_test_1():
    v1, v2, v3 = [1, 1, 1], [0, 1, 1], [0, 0, 1]
    assert(helper.is_lin_ind([v1, v2, v3]))

def is_lin_ind_test_2():
    v1, v2 = [2, 1, 0], [1, 1, 0]
    assert(helper.is_lin_ind([v1, v2]))

def is_lin_ind_test_3():
    v1, v2, v3, v4 = [2, 1, 0], [3, 1, 0], [0, 0, 1], [1, 0, 1]
    assert(not helper.is_lin_ind([v1, v2, v3, v4]))

def is_lin_ind_tests():
    is_lin_ind_test_1()  # Test 1
    is_lin_ind_test_2()  # Test 2
    is_lin_ind_test_3()  # Test 3
# ====================================================================================================================================================================


# ====================================================================================================================================================================
# 2. Calculate the pixel-position of the projected point (u, v) in the image and tick the correct answer
# ====================================================================================================================================================================
def pixel_pos_of_proj_point_test_1():
    # 3D Point P
    P = np.array([[4, -1, 7]]).T

    # C (Optical Center)
    C = np.array([[-1, -1, 0]]).T

    # Instrinsic parameter matrix
    K = np.array([
        [700,  0,    350],
        [0,    350,  135],
        [0,    0,    1]
    ])

    assert(helper.pixel_pos_of_proj_point(P=P, C=C, K=K) == (7, 850.0, 135.0))

def pixel_pos_of_proj_point_test_2():
    # 3D Point P
    P = np.array([[1, 2, 4]]).T

    # Instrinsic parameter matrix
    K = np.array([
        [400,  0,    250],
        [0,    400,  200],
        [0,    0,    1]
    ])

    assert(helper.pixel_pos_of_proj_point(P=P, K=K) == (4, 350.0, 400.0))

def pixel_pos_of_proj_point_test_3():
    # 3D Point P
    P = np.array([[0, 1, 2]]).T

    # C (Optical Center)
    C = np.array([[1, 0, 0]]).T

    # Instrinsic parameter matrix
    K = np.array([
        [500,  0,    320],
        [0,    400,  240],
        [0,    0,    1]
    ])

    assert(helper.pixel_pos_of_proj_point(P=P, C=C, K=K) == (2, 70.0, 440.0))
    # 3D Point P
    P = np.array([[1, 2, 4]]).T

    # Instrinsic parameter matrix
    K = np.array([
        [400,  0,    250],
        [0,    400,  200],
        [0,    0,    1]
    ])

    assert(helper.pixel_pos_of_proj_point(P=P, K=K) == (4, 350.0, 400.0))

def pixel_pos_of_proj_point_test_4():
    # 3D Point P
    P = np.array([[-1, 1, 8]]).T

    # Instrinsic parameter matrix
    K = np.array([
        [640,  0,    320],
        [0,    480,  240],
        [0,    0,    1]
    ])

    assert(helper.pixel_pos_of_proj_point(P=P, K=K) == (8, 240.0, 300.0))

def pixel_pos_of_proj_point_tests():
    pixel_pos_of_proj_point_test_1()  # Test 1
    pixel_pos_of_proj_point_test_2()  # Test 2
    pixel_pos_of_proj_point_test_3()  # Test 3
    pixel_pos_of_proj_point_test_4()  # Test 4
# ====================================================================================================================================================================


# ====================================================================================================================================================================
# 3. ??
# ====================================================================================================================================================================

