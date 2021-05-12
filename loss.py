import pdb

import numpy as np
import torch
import torch.nn as nn


class EuclideanDistanceLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(EuclideanDistanceLoss, self).__init__()

        self.reduction = getattr(torch, reduction, lambda x: x)

    def forward(self, outputs, targets):
        # outputs: torch.Size([bs, seq_len, N_art, 2, N_samples])
        # targets: torch.Size([bs, seq_len, N_art, 2, N_samples])

        x_outputs = outputs[:, :, :, 0, :].clone()
        y_outputs = outputs[:, :, :, 1, :].clone()

        x_targets = targets[:, :, :, 0, :].clone()
        y_targets = targets[:, :, :, 1, :].clone()

        dist = torch.sqrt((x_outputs - x_targets) ** 2 + (y_outputs - y_targets) ** 2)
        return self.reduction(dist)


def pearsons_correlation(outputs, targets):
    eps = 1e-5
    bs, seq_len, n_articulators, _, n_samples = targets.shape

    x_outputs = outputs[:, :, :, 0, :]
    y_outputs = outputs[:, :, :, 1, :]

    x_targets = targets[:, :, :, 0, :]
    y_targets = targets[:, :, :, 1, :]

    x_outputs_mean = x_outputs.mean(dim=1).unsqueeze(1).repeat_interleave(seq_len, dim=1)
    vx_outputs = x_outputs - x_outputs_mean

    x_targets_mean = x_outputs.mean(dim=1).unsqueeze(1).repeat_interleave(seq_len, dim=1)
    vx_targets = x_targets - x_targets_mean

    x_corr = torch.sum(vx_outputs * vx_targets, dim=1) / (torch.sqrt(torch.sum(vx_outputs ** 2, dim=1)) * torch.sqrt(torch.sum(vx_targets ** 2, dim=1)) + eps)

    y_outputs_mean = y_outputs.mean(dim=1).unsqueeze(1).repeat_interleave(seq_len, dim=1)
    vy_outputs = y_outputs - y_outputs_mean

    y_targets_mean = y_targets.mean(dim=1).unsqueeze(1).repeat_interleave(seq_len, dim=1)
    vy_targets = y_targets - y_targets_mean

    y_corr = torch.sum(vy_outputs * vy_targets, dim=1) / (torch.sqrt(torch.sum(vy_outputs ** 2, dim=1)) * torch.sqrt(torch.sum(vy_targets ** 2, dim=1)) + eps)

    return x_corr, y_corr
