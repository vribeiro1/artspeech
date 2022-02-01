import pdb

import funcy
import numpy as np
import torch
import torch.nn.functional as F

from copy import deepcopy
from vt_tools.metrics import distance_matrix, euclidean
from vt_tools.reconstruct_snail import reconstruct_snail_from_midline

from phoneme_to_articulation.dataset import ArtSpeechDataset
from settings import DatasetConfig

mm_to_percent = lambda v_mm: v_mm / (DatasetConfig.RES * DatasetConfig.PIXEL_SPACING)
SNAIL_PARAMETERS = {
    "soft-palate": {
        "width_int": mm_to_percent(1.5),
        "width_ext": mm_to_percent(3.5),
        "width_apex_int": mm_to_percent(2.5),
        "width_apex_ext": mm_to_percent(1.)
    },
    "epiglottis": {
        "width_int": mm_to_percent(2.5),
        "width_ext": mm_to_percent(2.5),
        "width_apex_int": mm_to_percent(1.5),
        "width_apex_ext": mm_to_percent(1.5)
    }
}


def closest_point_between_arrays(arr_1, arr_2):
    """
    Calculates the closest point between two np.ndarray and return the indices for each array

    Args:
    arr_1 (np.ndarray): (N,) shaped array
    arr_2 (np.ndarray): (M,) shaped array
    """

    dist_mtx = distance_matrix(arr_1, arr_2)
    nrows, ncols = dist_mtx.shape

    amin_array_1 = dist_mtx.argmin() // nrows
    amin_array_2 = dist_mtx.argmin() % ncols

    return amin_array_1, amin_array_2


def interpolate_points(pt1, pt2):
    """
    Calculate the middle x and y coordinates of two points.

    Args:
    pt1 (Tuple): First point.
    pt2 (Tuple): Second point.
    """
    x1, y1 = pt1
    x2, y2 = pt2

    new_x = min(x1, x2) + abs(x1 - x2) / 2
    new_y = min(y1, y2) + abs(y1 - y2) / 2

    return new_x, new_y


def upsample_curve(curve, approx_n_samples):
    new_curve = deepcopy(curve)

    consecutive_points_distances = funcy.lmap(
        lambda ps: euclidean(*ps),
        zip(new_curve[:-1], new_curve[1:])
    )

    linear_curve_length = np.sum(consecutive_points_distances)
    segment_length = linear_curve_length / (approx_n_samples - 1)

    curve_segments = list(zip(consecutive_points_distances, new_curve[:-1], new_curve[1:]))
    while any([d > segment_length for d, _, _ in curve_segments]):
        interp_curve = []
        for dist, pt1, pt2 in curve_segments:
            if dist <= segment_length:
                interp_curve.append((pt1, pt2))
            else:
                new_point = interpolate_points(pt1, pt2)
                interp_curve.append((pt1, new_point))
                interp_curve.append((new_point, pt2))
        new_curve = [pt1 for (pt1, _) in interp_curve]

        consecutive_points_distances = funcy.lmap(
            lambda ps: euclidean(*ps),
            zip(new_curve[:-1], new_curve[1:])
        )
        curve_segments = list(zip(consecutive_points_distances, new_curve[:-1], new_curve[1:]))

    return np.array(new_curve)


def load_articulator_arrays(articulators_filepaths):
    """
    Load the articulators, processing the snail structures accordingly.

    Args:
    articulators_filepaths (Dict[str, str]): Dictionary containing the articulator name in the key
    and the filepath of the .npy file.
    """
    # Load soft palate and reconstruct snail
    fp_soft_palate = articulators_filepaths["soft-palate"]
    midline_soft_palate = ArtSpeechDataset.load_target_array(fp_soft_palate, norm=False)
    params_soft_palate = SNAIL_PARAMETERS["soft-palate"]
    snail_soft_palate = reconstruct_snail_from_midline(midline_soft_palate, **params_soft_palate)

    # Load epiglottis and reconstruct snail
    fp_epiglottis = articulators_filepaths["epiglottis"]
    midline_epiglottis = ArtSpeechDataset.load_target_array(fp_epiglottis, norm=False)
    params_epiglottis = SNAIL_PARAMETERS["epiglottis"]
    snail_epiglottis = reconstruct_snail_from_midline(midline_epiglottis, **params_epiglottis)

    # Load non-snail articulators
    articulators_arrays = {}
    for articulator, fp_articulator in articulators_filepaths.items():
        if articulator not in SNAIL_PARAMETERS:
            articulators_arrays[articulator] = ArtSpeechDataset.load_target_array(fp_articulator, norm=False)

    articulators_arrays["soft-palate"] = snail_soft_palate
    articulators_arrays["epiglottis"] = snail_epiglottis

    return articulators_arrays


def find_lip_end(lip_array):
    """
    Finds the point where the absissas starts decreasing.
    """
    lip_array_0 = lip_array[:-1]
    lip_array_1 = lip_array[1:]

    offsets = list(enumerate(zip(lip_array_0, lip_array_1)))
    decreasing_absissas = funcy.lfilter(lambda t: t[1][0][0] < t[1][1][0], offsets)
    if len(decreasing_absissas) > 0:
        idx, (_, _) = min(decreasing_absissas, key=lambda t: t[0])
    else:
        idx = -1

    return idx


def connect_articulators(articulators_dict, eps=0.004, load=True):
    """
    Connect the articulators to produce a single shape for the entire vocal tract.

    Args:
    articulators_dict (Dict): Dictionary containing the articulator name in the key and the filepath
    of the .npy file.
    eps (float): The maximum distance between two points that determine a contact.
        TODO: Put the eps in milimiters.
    load (bool): If should load the articulators
    """
    if load:
        articulators_arrays = load_articulator_arrays(articulators_dict)
    else:
        articulators_arrays = articulators_dict

    # Internal vocal tract wall

    # Determine the closest point between the vocal folds and the epiglottis
    vocal_folds_end, epiglottis_start = closest_point_between_arrays(
        articulators_arrays["vocal-folds"],
        articulators_arrays["epiglottis"]
    )

    # Determine the closest point between the epiglottis and the tongue
    tongue_start, epiglottis_end = closest_point_between_arrays(
        articulators_arrays["tongue"],
        articulators_arrays["epiglottis"]
    )

    # Determine the closest point between the tongue and the lower incisor
    tongue_end, lower_incisor_start = closest_point_between_arrays(
        articulators_arrays["tongue"],
        articulators_arrays["lower-incisor"]
    )

    # Determine the closest point between the lower incisor and the lower lip
    lower_incisor_end, lower_lip_start = closest_point_between_arrays(
        articulators_arrays["lower-incisor"],
        articulators_arrays["lower-lip"]
    )

    lower_lip_end = find_lip_end(articulators_arrays["lower-lip"])

    internal_wall = np.concatenate([
        np.array([articulators_arrays["vocal-folds"][vocal_folds_end]]),
        articulators_arrays["epiglottis"][epiglottis_start:epiglottis_end + 1],
        articulators_arrays["tongue"][tongue_start:tongue_end + 1],
        articulators_arrays["lower-incisor"][lower_incisor_start:lower_incisor_end + 1],
        articulators_arrays["lower-lip"][lower_lip_start:lower_lip_end]
    ])

    internal_wall = upsample_curve(internal_wall, approx_n_samples=300)
    internal_wall_tensor = torch.from_numpy(internal_wall).T.unsqueeze(dim=0)
    internal_wall_tensor = F.interpolate(
        internal_wall_tensor,
        size=100,
        mode="linear",
        align_corners=True
    )
    internal_wall = internal_wall_tensor.squeeze(dim=0).T.numpy()

    # External vocal tract wall

    # Determine the closest point between pharynx and soft palate

    # The contact between the pharynx and the soft palate is a special case because it might have
    # zero, one or two intersection points. If there are zero contact points, we select the same
    # way we do for the remaining articulators. If there are one or more, we choose the one with
    # the lowest index in the soft palate array.

    dist_mtx = distance_matrix(
        articulators_arrays["pharynx"],
        articulators_arrays["soft-palate"]
    )

    contact_points = np.where(dist_mtx < eps)
    contact_points = np.concatenate(
        funcy.lmap(lambda arr: np.expand_dims(arr, axis=1), contact_points
    ), axis=1)

    if len(contact_points):
        pharynx_end, soft_palate_end = sorted(contact_points, key=lambda p: p[0])[0]
    else:
        pharynx_end, soft_palate_end = closest_point_between_arrays(
            articulators_arrays["pharynx"],
            articulators_arrays["soft-palate"]
        )

    upper_lip_end = find_lip_end(articulators_arrays["upper-lip"])

    external_wall = np.concatenate([
        np.flip(articulators_arrays["arytenoid-muscle"], axis=0),
        articulators_arrays["pharynx"][:pharynx_end + 1],
        np.flip(articulators_arrays["soft-palate"][:soft_palate_end + 1], axis=0),
        articulators_arrays["upper-incisor"],
        articulators_arrays["upper-lip"][:upper_lip_end]
    ])

    external_wall = upsample_curve(external_wall, approx_n_samples=300)
    external_wall_tensor = torch.from_numpy(external_wall).T.unsqueeze(dim=0)
    external_wall_tensor = F.interpolate(
        external_wall_tensor,
        size=100,
        mode="linear",
        align_corners=True
    )
    external_wall = external_wall_tensor.squeeze(dim=0).T.numpy()

    return internal_wall, external_wall
