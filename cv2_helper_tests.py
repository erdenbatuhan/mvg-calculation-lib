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

def pixel_pos_of_proj_point_test_5():
    # 3D Point P in world coordinates
    P = np.array([[8, -1, 1]]).T

    # Transform Matrix (CAM TO WORLD)
    g_cam_to_world = np.array([
        [0,   0,  1,  4],
        [-1,  0,  0,  2],
        [0,   -1, 0,  3],
        [0,   0,  0,  1]
    ])

    # Instrinsic parameter matrix
    K = np.array([
        [640,  0,    320],
        [0,    480,  240],
        [0,    0,    1]
    ])

    assert(helper.pixel_pos_of_proj_point(P=P, K=K, g_cam_to_world=g_cam_to_world) == (4, 800.0, 480.0))

def pixel_pos_of_proj_point_tests():
    pixel_pos_of_proj_point_test_1()  # Test 1
    pixel_pos_of_proj_point_test_2()  # Test 2
    pixel_pos_of_proj_point_test_3()  # Test 3
    pixel_pos_of_proj_point_test_4()  # Test 4
    pixel_pos_of_proj_point_test_5()  # Test 5
# ====================================================================================================================================================================


# ====================================================================================================================================================================
# 3. The equation of the epipolar line associated to the pixel
# ====================================================================================================================================================================
def epipolar_line_assoc_pixel_test_1():
    actual = helper.epipolar_line_assoc_pixel(P=np.array([50, 20, 1]).T, T=[5, 4, 1])
    expected = ((16.0, 45.0), (100.0, 45.0))

    assert(actual == expected)

def epipolar_line_assoc_pixel_test_2():
    actual = helper.epipolar_line_assoc_pixel(P=np.array([3, 2, 1]).T, T=[1, 2, 3])
    expected = ((4.0, 8.0), (4.0, 8.0))

    assert(actual == expected)

def epipolar_line_assoc_pixel_tests():
    epipolar_line_assoc_pixel_test_1()  # Test 1
    epipolar_line_assoc_pixel_test_2()  # Test 2


