import pdb

import funcy
import numpy as np

from scipy.spatial.distance import euclidean

from bs_regularization import regularize_Bsplines


def calculate_segment_midpoint_and_angle(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    x_mid = min(x2, x1) + np.abs(x2 - x1) / 2
    y_mid = min(y2, y1) + np.abs(y2 - y1) / 2

    midpoint = x_mid, y_mid

    hip = euclidean(p1, p2)
    sin = (y2 - y1) / hip
    cos = (x2 - x1) / hip

    theta = np.arctan2(cos, sin)
    theta_p90 = np.pi - (theta + np.pi / 2)

    return midpoint, theta


def calculate_points(midpoint, angle, w_int=3, w_ext=3):
    x0, y0 = midpoint

    x_ext = w_ext
    y_ext = 0

    x_int = -w_int
    y_int = 0

    points = np.array([
        [x_ext, y_ext],
        [x_int, y_int]
    ])

    rot_mtx = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    rot_points = np.matmul(points, rot_mtx)
    p_ext, p_int = rot_points

    p_ext[0] += x0
    p_ext[1] += y0

    p_int[0] += x0
    p_int[1] += y0

    return p_ext, p_int


def reconstruct_apex(center, radius, ext_width, int_width, angle_rad):
    n_samples = 20
    samples = np.arange(0, n_samples) / n_samples
    radius_int = int_width + (np.sin(np.pi * (-0.5 + samples)) + 1.) / 2 * (radius - int_width)
    radius_ext = ext_width + (np.sin(np.pi * (-0.5 + samples)) + 1.) / 2 * (radius - ext_width)
    alphas = (np.pi / 2) * samples

    apex_int = np.flip(np.array([-int_width * np.cos(alphas), radius_int * np.sin(alphas)]), axis=1)
    apex_ext = np.array([ext_width * np.cos(alphas), radius_ext * np.sin(alphas)])
    apex = np.concatenate([apex_ext, apex_int], axis=1).transpose(1, 0)

    rot_mtx = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])

    rot_apex = np.matmul(apex, rot_mtx)
    xc, yc = center
    rot_apex[:, 0] += xc
    rot_apex[:, 1] += yc

    return rot_apex


def reconstruct_snail_from_midline(midline_, width_int, width_ext, width_apex_int, width_apex_ext):
    x_start, _ = midline_[0]
    x_end, _ = midline_[-1]

    if x_start > x_end:
        midline = np.flip(midline_.copy(), axis=0)
    else:
        midline = midline_.copy()

    midline_m1 = midline[:-1]
    midline_p1 = midline[1:]

    segments = list(zip(midline_m1, midline_p1))
    midpoints_angles = [calculate_segment_midpoint_and_angle(*segment) for segment in segments]

    decay_int = np.arctan((width_apex_int - width_int) / len(midpoints_angles))
    widths_int = funcy.lmap(lambda x: decay_int * x + width_int, range(len(midpoints_angles)))

    decay_ext = np.arctan((width_apex_ext - width_ext) / len(midpoints_angles))
    widths_ext = funcy.lmap(lambda x: decay_ext * x + width_ext, range(len(midpoints_angles)))

    midpoints = [d[0] for d in midpoints_angles]
    angles = [d[1] for d in midpoints_angles]

    snail_points = [
        calculate_points(midpoint, angle, w_ext=wext, w_int=wint)
        for (midpoint, angle), wint, wext in zip(midpoints_angles, widths_int, widths_ext)
    ]

    int_snail_points = [d[1] for d in snail_points]
    ext_snail_points = [d[0] for d in reversed(snail_points)]

    xc, yc = midpoints[-1]

    int_width = widths_int[-1]
    ext_width = widths_ext[-1]
    radius = max(int_width, ext_width) * 1.2
    apex_angle = angles[-1]
    apex = reconstruct_apex(
        center=(xc, yc),
        radius=radius,
        ext_width=ext_width,
        int_width=int_width,
        angle_rad=apex_angle
    )

    snail = np.array(int_snail_points + list(np.flip(apex, axis=0)) + ext_snail_points)
    resX, resY = regularize_Bsplines(snail, 3)
    reg_snail = np.array([resX, resY]).T

    return reg_snail
