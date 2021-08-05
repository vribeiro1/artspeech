import torch
import torch.nn as nn

from torch.distributions import MultivariateNormal

from helpers import assert_expression
from loss import EuclideanDistanceLoss


# lower incisor, lower lip, pharynx, soft palate, tongue, upper incisor, upper lip
default_critical_articulator = ("soft-palate", "pharynx")
critical_articulators = {
    "p": [
        ("lower-lip", "upper-lip"),
        default_critical_articulator
    ],
    "b": [
        ("lower-lip", "upper-lip"),
        default_critical_articulator
    ],
    "l": [
        ("tongue", "upper-incisor"),
        default_critical_articulator
    ],
    "t": [
        ("tongue", "upper-incisor"),
        default_critical_articulator
    ],
    "k": [
        ("tongue", "upper-incisor"),
        default_critical_articulator
    ],
    "g": [
        ("tongue", "upper-incisor"),
        default_critical_articulator
    ],
    "nasal": [],
    "#": [],
    "_": []
}


def make_critical_mask(phone, articulators):
    n_articulators = len(articulators)
    art_dict = {art: i for i, art in enumerate(articulators)}

    mask = torch.zeros(size=(n_articulators, n_articulators), dtype=torch.long)

    if "~" in phone:
        phone = "nasal"
    phone_critical_articulators = critical_articulators.get(phone, [default_critical_articulator])

    for art_1, art_2 in phone_critical_articulators:
        if art_1 not in art_dict or art_2 not in art_dict:
            continue

        i = art_dict[art_1]
        j = art_dict[art_2]
        mask[i][j] = mask[j][i] = 1

    return mask


class CriticalLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(CriticalLoss, self).__init__()

        self.reduction = getattr(torch, reduction, lambda x: x)

    def forward(self, outputs, masks):
        bs, seq_len, N_art, _, N_samples = outputs.shape  # torch.Size([bs, seq_len, N_art, 2, N_samples])
        assert_expression(masks.shape == torch.Size([bs, seq_len, N_art, N_art]), ValueError, "Masks should have the shape (B, S, N, N)")

        outputs_reshaped = outputs.view(bs * seq_len, N_art, 2, N_samples)  # torch.Size([bs * seq_len, N_art, 2, N_samples])
        outputs_reshaped = outputs_reshaped.permute(1, 0, 3, 2)  # torch.Size([N_art, bs * seq_len, N_samples, 2])

        min_dists = torch.zeros((N_art, N_art, bs * seq_len), dtype=torch.float, device=outputs.device)  # torch.Size([N_art, N_art, bs * seq_len])
        for art_i in range(N_art):
            xi = outputs_reshaped[art_i, :, :, :]
            for art_j in range(N_art):
                xj = outputs_reshaped[art_j, :, :, :]
                dist_ij = torch.cdist(xi, xj, p=2)  # Norm 2 between art_i and art_j
                dist_ij = dist_ij.view(bs * seq_len, N_samples ** 2)
                min_dist_ij, _ = dist_ij.min(axis=1)

                min_dists[art_i, art_j, :] = min_dist_ij

        min_dists = min_dists.permute(2, 0, 1).view(bs, seq_len, N_art, N_art)
        masked_dists = masks * min_dists

        return self.reduction(masked_dists)


class CombinedLoss(nn.Module):
    def __init__(self, a1=1.0, reduction="mean"):
        super(CombinedLoss, self).__init__()

        self.euclidean_loss = EuclideanDistanceLoss(reduction=reduction)
        self.critical_loss = CriticalLoss(reduction=reduction)

        self.a1 = a1

    def forward(self, outputs, targets, masks):
        euclidean = self.euclidean_loss(outputs, targets)
        critical = self.critical_loss(outputs, masks)

        return euclidean + self.a1 * critical


class NegLogProbLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(NegLogProbLoss, self).__init__()

        self.reduction = getattr(torch, reduction, lambda x: x)

    def forward(self, mean_, var_, targets_):
        # mean: torch.Size([bs, seq_len, N_art, 2, N_samples])
        # std: torch.Size([bs, seq_len, N_art, 2, N_samples])
        # targets: torch.Size([bs, seq_len, N_art, 2, N_samples])
        bs, seq_len, N_art, _, N_samples = targets_.shape

        mean = mean_.clone().transpose(4, 3)
        var = var_.clone().transpose(4, 3)
        targets = targets_.clone().transpose(4, 3)

        covar = torch.zeros([bs, seq_len, N_art, N_samples, 2, 2])
        covar[:, :, :, :, 0, 0] = var[:, :, :, :, 0]
        covar[:, :, :, :, 1, 1] = var[:, :, :, :, 1]

        P = MultivariateNormal(mean, covar)
        neg_log_prob = -P.log_prob(targets)

        return self.reduction(neg_log_prob)
