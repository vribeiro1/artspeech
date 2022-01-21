import pdb

import funcy
import numpy as np
import torch

from numba import jit
from shapely.geometry import LineString, Point
from vt_tracker.metrics import euclidean, distance_matrix


def rotate(point, ang_rad):
    """
    Rotate a point by an angle in radians.

    Args:
    p (np.ndarray): (x, y) coordinates of the point to rotate
    ang_rad (float): Angle in radians.
    """

    rot_mtx = np.array([
        [np.cos(ang_rad), np.sin(ang_rad)],
        [-np.sin(ang_rad), np.cos(ang_rad)]
    ])

    res = np.matmul(rot_mtx, point)

    return res


def build_semipolar_grid(center, theta_rad, omega_rad, linear_step, polar_step_rad, grid_res=50):
    """
    Build Maeda's semipolar grid.

    Args:
    center (np.ndarray): (x, y) coordinates of the semipolar's grid center.
    theta_rad (float): Rotation of the mouth cavity grid in radians.
    omega_rad (float): Rotation of the larynx cavity grid in radians.
    linear_step (float): Step size in the linear grids.
    polar_step_rad (float): Step size in the polar grid in radians.
    """
    x0, y0 = center

    # Mouth cavity grid

    xs = np.arange(0., -0.5, -linear_step)  # TODO: Parameterize the max grid size
    ys_int = np.zeros(len(xs))
    ys_ext = -0.4 * np.ones(len(xs))  # TODO: Parameterize the grid width

    grid_mouth_int = np.array([xs, ys_int]).T
    grid_mouth_ext = np.array([xs, ys_ext]).T

    rotate_by_theta = lambda p: rotate(p, theta_rad) + center
    rot_grid_mouth_int = np.array(funcy.lmap(rotate_by_theta, grid_mouth_int))
    rot_grid_mouth_ext = np.array(funcy.lmap(rotate_by_theta, grid_mouth_ext))

    # Larynx cavity grid

    ys = np.arange(0., 0.5, linear_step)  # TODO: Parameterize the max grid size
    xs_int = np.zeros(len(ys))
    xs_ext = 0.4 * np.ones(len(ys))  # TODO: Parameterize grid width

    grid_larynx_int = np.array([xs_int, ys]).T
    grid_larynx_ext = np.array([xs_ext, ys]).T

    rotate_by_omega = lambda p: rotate(p, omega_rad)  + center
    rot_grid_larynx_int = np.array(funcy.lmap(rotate_by_omega, grid_larynx_int))
    rot_grid_larynx_ext = np.array(funcy.lmap(rotate_by_omega, grid_larynx_ext))

    # Polar grid

    angles = np.arange(
        theta_rad - polar_step_rad,
        -(np.pi / 2) + omega_rad,
        -polar_step_rad
    )

    p = np.array([0., -0.4])  # TODO: Parameterize grid width
    grid_polar_ext = np.array([rotate(p, ang) + center for ang in angles])
    grid_polar_int = np.zeros(shape=(len(grid_polar_ext), 2)) + center

    semipolar_grid = []
    for pt_int, pt_ext in reversed(list(zip(rot_grid_larynx_int, rot_grid_larynx_ext))):
        x_int, y_int = pt_int
        x_ext, y_ext = pt_ext

        x_line = np.linspace(x_int, x_ext, grid_res)
        y_line = np.linspace(y_int, y_ext, grid_res)

        semipolar_grid.append(np.array([x_line, y_line]).T)

    for pt_int, pt_ext in reversed(list(zip(grid_polar_int, grid_polar_ext))):
        x_int, y_int = pt_int
        x_ext, y_ext = pt_ext

        x_line = np.linspace(x_int, x_ext, grid_res)
        y_line = np.linspace(y_int, y_ext, grid_res)

        semipolar_grid.append(np.array([x_line, y_line]).T)

    for pt_int, pt_ext in zip(rot_grid_mouth_int, rot_grid_mouth_ext):
        x_int, y_int = pt_int
        x_ext, y_ext = pt_ext

        x_line = np.linspace(x_int, x_ext, grid_res)
        y_line = np.linspace(y_int, y_ext, grid_res)

        semipolar_grid.append(np.array([x_line, y_line]).T)

    return np.array(semipolar_grid)


@jit(nopython=True)
def mid_point(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    x_mid = min(x1, x2) + abs(x1 - x2) / 2
    y_mid = min(y1, y2) + abs(y1 - y2) / 2

    return x_mid, y_mid


def area_function(internal_wall, external_wall, alpha=np.pi, beta=2.):
    assert internal_wall.shape == external_wall.shape

    xs_radius = np.array([
        (*mid_point(p_int, p_ext), euclidean(p_int, p_ext) / 2)
        for p_int, p_ext in zip(internal_wall, external_wall)
    ], dtype=np.float)

    xs = xs_radius[:, 0:2]
    radius = xs_radius[:, 2]
    fx = alpha * radius ** beta

    dists = [0.0] + [euclidean(p1, p2) for p1, p2 in zip(xs[:-1], xs [1:])]
    dists = [d + sum(dists[:i]) for i, d in enumerate(dists)]

    return dists, fx


def evenly_spaced_fx(x, fx, n_samples=200):
    x_min = x[0]
    x_max = x[-1]
    xs = np.linspace(x_min, x_max, n_samples)

    fx_max = max(fx) + 10
    xfx_line_string = LineString(np.array([x, fx]).T)
    xfx = []
    for x in xs:
        x_line_string = LineString([[x, 0], [x, fx_max]])
        xfx_val = x_line_string.intersection(xfx_line_string)
        xfx.append((xfx_val.x, xfx_val.y))
    xfx = torch.tensor(xfx).T

    return xfx


def argmatrix(mtx, minmax):
    assert minmax in ["min", "max"]

    arg_fn = getattr(np, f"arg{minmax}")
    aminmax = arg_fn(mtx)
    _, m = mtx.shape

    i_minmax = aminmax // m
    j_minmax = aminmax % m

    return i_minmax, j_minmax


def intersect_semipolar_grid(internal_wall, external_wall, semipolar_grid):
    internal_line_string = LineString(internal_wall)
    external_line_string = LineString(external_wall)

    internal_intersec = []
    external_intersec = []
    for grid_line in map(lambda coords: LineString(coords), semipolar_grid):
        internal_contact = grid_line.intersects(internal_line_string)
        external_contact = grid_line.intersects(external_line_string)

        if not internal_contact and not external_contact:
            continue

        list_internal = []
        if internal_contact:
            p_int = grid_line.intersection(internal_line_string)
            list_internal = [(p_int.x, p_int.y)] if isinstance(p_int, Point) else [(p.x, p.y) for p in p_int.geoms]
        list_internal = np.array(list_internal)

        list_external = []
        if external_contact:
            p_ext = grid_line.intersection(external_line_string)
            list_external = [(p_ext.x, p_ext.y)] if isinstance(p_ext, Point) else [(p.x, p.y) for p in p_ext.geoms]
        list_external = np.array(list_external)

        default_compare_to = np.array([
            external_line_string.coords[0],
            external_line_string.coords[-1]
        ])

        if internal_contact:
            compare_to = list_external.copy() if external_contact else default_compare_to
            internal_dist_mtx = distance_matrix(list_internal, compare_to)
            i_min, j_min = argmatrix(internal_dist_mtx, "min")

            internal_intersec.append(list_internal[i_min])
            if not external_contact:
                external_intersec.append(external_line_string.coords[int(-1 * j_min)])

        if external_contact:
            compare_to = list_internal.copy() if internal_contact else default_compare_to
            external_dist_mtx = distance_matrix(list_external, compare_to)
            i_min, j_min = argmatrix(external_dist_mtx, "min")

            external_intersec.append(list_external[i_min])
            if not internal_contact:
                internal_intersec.append(internal_line_string.coords[int(-1 * j_min)])

    internal_intersec = torch.from_numpy(np.array(internal_intersec))
    external_intersec = torch.from_numpy(np.array(external_intersec))

    return internal_intersec, external_intersec
