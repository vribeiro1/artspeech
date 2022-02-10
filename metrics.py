import torch

from vt_tools.metrics import p2cp_mean

from loss import EuclideanDistanceLoss


def p2cp_distance(outputs, targets):
    """
    Args:
    outputs (torch.tensor): Torch tensor with shape (bs, seq_len, N_art, 2, N_samples).
    targets (torch.tensor): Torch tensor with shape (bs, seq_len, N_art, 2, N_samples).
    """
    bs, seq_len, N_art, _, N_samples = outputs.shape

    results = torch.zeros(0, seq_len, N_art)
    for batch_out, batch_target in zip(outputs, targets):
        batch_results = torch.zeros(0, N_art)
        for seq_out, seq_target in zip(batch_out, batch_target):
            seq_results = []
            for output, target in zip(seq_out, seq_target):
                output_transpose = output.transpose(1, 0)
                target_transpose = target.transpose(1, 0)

                p2cp = p2cp_mean(output_transpose.numpy(), target_transpose.numpy())
                seq_results.append(p2cp)

            batch_results = torch.cat([batch_results, torch.tensor(seq_results).unsqueeze(dim=0)])
        results = torch.cat([results, batch_results.unsqueeze(dim=0)])

    return results


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


def euclidean_distance(outputs, targets):
    """
    Args:
    outputs (torch.tensor): Torch tensor with shape (bs, seq_len, N_art, 2, N_samples).
    targets (torch.tensor): Torch tensor with shape (bs, seq_len, N_art, 2, N_samples).
    """
    euclidean_distance_fn = EuclideanDistanceLoss(reduction="none")
    return euclidean_distance_fn(outputs, targets).mean(dim=-1)
