import numpy as np
import torch
import torch.nn.functional as F

from vt_tools import LOWER_INCISOR, UPPER_INCISOR, EPIGLOTTIS
from vt_tools.bs_regularization import regularize_Bsplines

from settings import DatasetConfig


class TailClipper:
    TAIL_CLIP_REFERENCES = [LOWER_INCISOR, UPPER_INCISOR, EPIGLOTTIS]

    @classmethod
    def clip_tongue_tails(cls, tongue, lower_incisor, epiglottis, **kwargs):
        # Remove the front tail of the tongue using the lower incisor as the reference
        ref_idx = lower_incisor[:, 1].argmax()
        reference = lower_incisor[ref_idx]

        tongue_cp = tongue.clone()

        tongue_1st_half = tongue_cp[:25]
        tongue_2nd_half = tongue_cp[25:]

        keep_indices = torch.where(tongue_2nd_half[:, 1] < reference[1])

        tailless_tongue = torch.cat([
            tongue_1st_half,
            tongue_2nd_half[keep_indices]
        ])

        # Remove the back tail of the tongue using the epiglottis as the reference
        ref_idx = epiglottis[:, 1].argmin()
        reference = epiglottis[ref_idx]

        tongue_cp = tailless_tongue.clone()

        tongue_1st_half = tongue_cp[:25]
        tongue_2nd_half = tongue_cp[25:]

        keep_indices = torch.where(tongue_1st_half[:, 1] < reference[1] + (10 / DatasetConfig.PIXEL_SPACING / DatasetConfig.RES))

        tailless_tongue = torch.cat([
            tongue_1st_half[keep_indices],
            tongue_2nd_half
        ])

        tailless_tongue = tailless_tongue.T.unsqueeze(dim=0)
        tailless_tongue = F.interpolate(tailless_tongue, size=50).squeeze(dim=0).T

        return tailless_tongue

    @classmethod
    def clip_lower_lip_tails(cls, lower_lip, lower_incisor, **kwargs):
        # Remove the front tail of the lower lip using the lower incisor as the reference
        ref_idx = lower_incisor[:, 1].argmax()
        reference = lower_incisor[ref_idx]

        llip_cp = lower_lip.clone()

        llip_1st_half = llip_cp[:25]
        llip_2nd_half = llip_cp[25:]

        keep_indices = torch.where(llip_2nd_half[:, 1] < reference[1] + (5 / DatasetConfig.PIXEL_SPACING / DatasetConfig.RES))

        tailless_llip = torch.cat([
            llip_1st_half,
            llip_2nd_half[keep_indices]
        ])

        tailless_llip = tailless_llip.T.unsqueeze(dim=0)
        tailless_llip = F.interpolate(tailless_llip, size=50).squeeze(dim=0).T

        # Remove the back tail of the lower lip using the lower incisor as the reference
        ref_idx = lower_incisor[:, 1].argmax()
        reference = lower_incisor[ref_idx]

        llip_cp = tailless_llip.clone()

        llip_1st_half = llip_cp[:25]
        llip_2nd_half = llip_cp[25:]

        keep_indices = torch.where(llip_1st_half[:, 1] < reference[1])

        tailless_llip = torch.cat([
            llip_1st_half[keep_indices],
            llip_2nd_half
        ])

        tailless_llip = tailless_llip.T.unsqueeze(dim=0)
        tailless_llip = F.interpolate(tailless_llip, size=50).squeeze(dim=0).T

        return tailless_llip

    @classmethod
    def clip_upper_lip_tails(cls, upper_lip, upper_incisor, **kwargs):
        # Remove the front tail of the upper lip using the upper incisor as the reference
        ref_idx = -1
        reference = upper_incisor[ref_idx]

        ulip_cp = upper_lip.clone()

        ulip_1st_half = ulip_cp[:25]
        ulip_2nd_half = ulip_cp[25:]

        keep_indices = torch.where(ulip_2nd_half[:, 1] > reference[1] - (10 / DatasetConfig.PIXEL_SPACING))

        tailless_ulip = torch.cat([
            ulip_1st_half,
            ulip_2nd_half[keep_indices]
        ])

        # Remove the back tail of the upper lip using the upper incisor as the reference
        ref_idx = -1
        reference = upper_incisor[ref_idx]

        ulip_cp = tailless_ulip.clone()

        ulip_1st_half = ulip_cp[:25]
        ulip_2nd_half = ulip_cp[25:]

        keep_indices = torch.where(ulip_1st_half[:, 1] > reference[1] - (5 / DatasetConfig.PIXEL_SPACING))

        tailless_ulip = torch.cat([
            ulip_1st_half[keep_indices],
            ulip_2nd_half
        ])

        tailless_ulip = tailless_ulip.T.unsqueeze(dim=0)
        tailless_ulip = F.interpolate(tailless_ulip, size=50).squeeze(dim=0).T

        return tailless_ulip
