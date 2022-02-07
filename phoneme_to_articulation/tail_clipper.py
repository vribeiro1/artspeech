import numpy as np

from vt_tools import LOWER_INCISOR, UPPER_INCISOR, EPIGLOTTIS
from vt_tools.bs_regularization import regularize_Bsplines

from settings import DatasetConfig


class TailClipper:
    TAIL_CLIP_REFERENCES = [LOWER_INCISOR, UPPER_INCISOR, EPIGLOTTIS]

    @classmethod
    def clip_tongue_tails(cls, tongue, lower_incisor, epiglottis, reg_out=True, **kwargs):
        # Remove the front tail of the tongue using the lower incisor as the reference
        ref_idx = lower_incisor[:, 1].argmax()
        reference = lower_incisor[ref_idx]

        tongue_cp = tongue.copy()

        tongue_1st_half = tongue_cp[:25]
        tongue_2nd_half = tongue_cp[25:]

        keep_indices = np.where(tongue_2nd_half[:, 1] < reference[1])

        tailless_tongue = np.concatenate([
            tongue_1st_half,
            tongue_2nd_half[keep_indices]
        ])

        # Remove the back tail of the tongue using the epiglottis as the reference
        ref_idx = epiglottis[:, 1].argmin()
        reference = epiglottis[ref_idx]

        tongue_cp = tailless_tongue.copy()

        tongue_1st_half = tongue_cp[:25]
        tongue_2nd_half = tongue_cp[25:]

        keep_indices = np.where(tongue_1st_half[:, 1] < reference[1] + (10 / DatasetConfig.PIXEL_SPACING / DatasetConfig.RES))

        tailless_tongue = np.concatenate([
            tongue_1st_half[keep_indices],
            tongue_2nd_half
        ])

        if reg_out:
            reg_x, reg_y = regularize_Bsplines(tailless_tongue, 3)
            tailless_tongue = np.array([reg_x, reg_y]).T

        return tailless_tongue

    @classmethod
    def clip_lower_lip_tails(cls, lower_lip, lower_incisor, reg_out=True, **kwargs):
        # Remove the front tail of the lower lip using the lower incisor as the reference
        ref_idx = lower_incisor[:, 1].argmax()
        reference = lower_incisor[ref_idx]

        llip_cp = lower_lip.copy()

        llip_1st_half = llip_cp[:25]
        llip_2nd_half = llip_cp[25:]

        keep_indices = np.where(llip_2nd_half[:, 1] < reference[1] + (5 / DatasetConfig.PIXEL_SPACING / DatasetConfig.RES))

        tailless_llip = np.concatenate([
            llip_1st_half,
            llip_2nd_half[keep_indices]
        ])

        if reg_out:
            reg_x, reg_y = regularize_Bsplines(tailless_llip, 3)
            tailless_llip = np.array([reg_x, reg_y]).T

        # Remove the back tail of the lower lip using the lower incisor as the reference
        ref_idx = lower_incisor[:, 1].argmax()
        reference = lower_incisor[ref_idx]

        llip_cp = tailless_llip.copy()

        llip_1st_half = llip_cp[:25]
        llip_2nd_half = llip_cp[25:]

        keep_indices = np.where(llip_1st_half[:, 1] < reference[1])

        tailless_llip = np.concatenate([
            llip_1st_half[keep_indices],
            llip_2nd_half
        ])

        if reg_out:
            reg_x, reg_y = regularize_Bsplines(tailless_llip, 3)
            tailless_llip = np.array([reg_x, reg_y]).T

        return tailless_llip

    @classmethod
    def clip_upper_lip_tails(cls, upper_lip, upper_incisor, reg_out=True, **kwargs):
        # Remove the front tail of the upper lip using the upper incisor as the reference
        ref_idx = -1
        reference = upper_incisor[ref_idx]

        ulip_cp = upper_lip.copy()

        ulip_1st_half = ulip_cp[:25]
        ulip_2nd_half = ulip_cp[25:]

        keep_indices = np.where(ulip_2nd_half[:, 1] > reference[1] - (10 / DatasetConfig.PIXEL_SPACING))

        tailless_ulip = np.concatenate([
            ulip_1st_half,
            ulip_2nd_half[keep_indices]
        ])

        # Remove the back tail of the upper lip using the upper incisor as the reference
        ref_idx = -1
        reference = upper_incisor[ref_idx]

        ulip_cp = tailless_ulip.copy()

        ulip_1st_half = ulip_cp[:25]
        ulip_2nd_half = ulip_cp[25:]

        keep_indices = np.where(ulip_1st_half[:, 1] > reference[1] - (5 / DatasetConfig.PIXEL_SPACING))

        tailless_ulip = np.concatenate([
            ulip_1st_half[keep_indices],
            ulip_2nd_half
        ])

        if reg_out:
            reg_x, reg_y = regularize_Bsplines(tailless_ulip, 3)
            tailless_ulip = np.array([reg_x, reg_y]).T

        return tailless_ulip
