#date: 2022-05-05T17:09:14Z
#url: https://api.github.com/gists/59e3c5421e440b4353a8802204b53acf
#owner: https://api.github.com/users/flannelhead

import numpy as np


# CAT16 from chromatic_adaptation.h
XYZ_D50_TO_D65 = np.array([
    [9.89466254e-01, -4.00304626e-02, 4.40530317e-02],
    [-5.40518733e-03, 1.00666069e+00, -1.75551955e-03],
    [-4.03920992e-04, 1.50768030e-02, 1.30210211e+00],
])


# D50 adapted Rec.709 to XYZ as used in darktable
RGB_TO_XYZ = np.array([
    [0.43604112, 0.38511333, 0.14304553],
    [0.22248445, 0.71690542, 0.06061015],
    [0.01392029, 0.09706775, 0.71391195],
])


# Kirk colorspace conversion matrices from colorspace_inline_conversions.h

XYZ_D65_TO_LMS = np.array([
    [ 0.257085,  0.859943, -0.031061],
    [-0.394427,  1.175800,  0.106423],
    [ 0.064856, -0.076250,  0.559067],
])

FILMLIGHT_lms_TO_rgb = np.array([
    [ 1.0877193, -0.66666667,  0.02061856],
    [-0.0877193,  1.66666667, -0.05154639],
    [        0.,          0.,  1.03092784],
])


def xyz_to_yrg(XYZ):
    LMS = XYZ_D65_TO_LMS @ XYZ
    Y = 0.68990272 * LMS[0] + 0.34832189 * LMS[1]
    a = LMS[0] + LMS[1] + LMS[2]
    rgb = FILMLIGHT_lms_TO_rgb @ LMS / a
    return np.array([Y, rgb[0], rgb[1]])


print(xyz_to_yrg(XYZ_D50_TO_D65 @ RGB_TO_XYZ @ np.array([1, 1, 1])))
