import torch


ART_SLICES = {
    "tongue-tip": (30, 45),
    "tongue-body": (10, 30),
    "upper-incisor": (25, 50),
    "hard-palate": (0, 25),
    "soft-palate": (35, 50),
    "velum": (0, 15)
}


def _calculate_TV(arr1, arr2):
    TV_cdist = torch.cdist(arr1, arr2)

    min_d0, argmin_d0 = TV_cdist.min(dim=0)
    min_d1, argmin_d1 = min_d0.min(dim=0)
    arr1_argmin = argmin_d0[argmin_d1].item()
    arr2_argmin = argmin_d1.item()

    TV = min_d1.item()
    TV_point_arr1 = arr1[arr1_argmin]
    TV_point_arr2 = arr2[arr2_argmin]

    return TV, TV_point_arr1, TV_point_arr2


def _calculate_LA(llip_arr, ulip_arr):
    LA, LA_llip, LA_ulip = _calculate_TV(llip_arr, ulip_arr)

    return LA, LA_llip, LA_ulip


def _calculate_TTCD(tongue_arr, uincisor_arr):
    tongue_tip_arr = tongue_arr[slice(*ART_SLICES["tongue-tip"])]
    teeth_arr = uincisor_arr[slice(*ART_SLICES["upper-incisor"])]
    tt_start, _ = ART_SLICES["tongue-tip"]

    TTCD, TTCD_tongue, TTDC_uincisor = _calculate_TV(tongue_tip_arr, teeth_arr)

    return TTCD, TTCD_tongue, TTDC_uincisor


def _calculate_TBCD(tongue_arr, uincisor_arr, soft_palate_velum_arr):
    tongue_body_arr = tongue_arr[slice(*ART_SLICES["tongue-body"])]
    hard_palate_arr = uincisor_arr[slice(*ART_SLICES["hard-palate"])]
    soft_palate_arr = soft_palate_velum_arr[slice(*ART_SLICES["soft-palate"])]
    palate_arr = torch.cat([hard_palate_arr, soft_palate_arr], axis=0)

    TBCD, TBCD_tongue, TBCD_palate = _calculate_TV(tongue_body_arr, palate_arr)

    return TBCD, TBCD_tongue, TBCD_palate


def _calculate_VEL(soft_palate_velum_arr, pharynx_arr):
    velum_arr = soft_palate_velum_arr[slice(*ART_SLICES["velum"])]

    VEL, VEL_velum, VEL_pharynx = _calculate_TV(velum_arr, pharynx_arr)

    return VEL, VEL_velum, VEL_pharynx


def calculate_vocal_tract_variables(inputs_dict):
    """
    inputs_dict: Dictionary containing the articulator name as key and the articulator points as value.

    return: Dictionary containing the TV name as key and the value and location as value.

    Measured vocal tract variables are:
    LA - Lip aperture
    LP - Lip protusion
    TTCD - Tongue tip constrict degree
    TTCL - Tongue tip constrict location
    TBCD - Tongue body constrict degree
    TBCL - Tongue body constrict location
    VEL - Velum
    GLO - Glottis
    """
    LA, LA_llip, LA_ulip = _calculate_LA(inputs_dict["lower-lip"], inputs_dict["upper-lip"])
    TTCD, TTCD_tongue, TTDC_uincisor = _calculate_TTCD(inputs_dict["tongue"], inputs_dict["upper-incisor"])
    TBCD, TBCD_tongue, TBCD_palate = _calculate_TBCD(inputs_dict["tongue"], inputs_dict["upper-incisor"], inputs_dict["soft-palate"])
    VEL, VEL_velum, VEL_pharynx = _calculate_VEL(inputs_dict["soft-palate"], inputs_dict["pharynx"])

    # PoC stands for Place of Constriction. For each TV, there are two PoCs,
    # from which the TV is measured.
    # TODO: Implement the functions to calculate LP, TTCL, TBCL, GLO
    TVs = {
        "LA": {
            "value": LA,
            "poc_1": LA_llip,
            "poc_2": LA_ulip
        },
        "LP": None,
        "TTCD": {
            "value": TTCD,
            "poc_1": TTCD_tongue,
            "poc_2": TTDC_uincisor
        },
        "TTCL": None,
        "TBCD": {
            "value": TBCD,
            "poc_1": TBCD_tongue,
            "poc_2": TBCD_palate
        },
        "TBCL": None,
        "VEL": {
            "value": VEL,
            "poc_1": VEL_velum,
            "poc_2": VEL_pharynx
        },
        "GLO": None
    }

    return TVs
