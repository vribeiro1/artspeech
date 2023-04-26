import pdb
import torch

from vt_tools.metrics import p2cp_mean

from phoneme_to_articulation.metrics import EuclideanDistance, MeanP2CPDistance


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


def p2cp_distance(outputs, targets):
    """
    Args:
        outputs (torch.tensor): Torch tensor with shape (bs, seq_len, N_art, 2, N_samples).
        targets (torch.tensor): Torch tensor with shape (bs, seq_len, N_art, 2, N_samples).

    Return:
        p2cp distance (torch.tensor): Torch tensor with shape (bs, seq_len, N_art)
    """
    p2cp_distance_fn = MeanP2CPDistance(reduction="none")
    p2cp = p2cp_distance_fn(
        outputs.transpose(-1, -2),
        targets.transpose(-1, -2)
    )
    return p2cp

def euclidean_distance(outputs, targets):
    """
    Args:
        outputs (torch.tensor): Torch tensor with shape (bs, seq_len, N_art, 2, N_samples).
        targets (torch.tensor): Torch tensor with shape (bs, seq_len, N_art, 2, N_samples).

    Return:
        euclidean (torch.tensor): Torch tensor with shape (bs, seq_len, N_art)
    """
    euclidean_distance_fn = EuclideanDistance(reduction="none")
    euclidean = euclidean_distance_fn(
        outputs,
        targets
    ).mean(dim=-1)
    return euclidean
