import torch
import torch.nn as nn


class EuclideanDistance(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()

        self.reduction = getattr(torch, reduction, lambda x: x)

    def forward(self, outputs, targets):
        """
        Args:
        outputs (torch.tensor): Torch tensor with shape (bs, seq_len, N_art, 2, N_samples).
        targets (torch.tensor): Torch tensor with shape (bs, seq_len, N_art, 2, N_samples).
        """
        x_outputs = outputs[..., 0, :].clone()
        y_outputs = outputs[..., 1, :].clone()

        x_targets = targets[..., 0, :].clone()
        y_targets = targets[..., 1, :].clone()

        dist = torch.sqrt((x_outputs - x_targets) ** 2 + (y_outputs - y_targets) ** 2)
        return self.reduction(dist)


class MeanP2CPDistance(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = getattr(torch, reduction, lambda x: x)

    def forward(self, u_, v_):
        """
        Args:
        u_ (torch.tensor): Tensor of shape (*, N, 2)
        v_ (torch.tensor): Tensor of shape (*, M, 2)
        """
        n = u_.shape[-2]
        m = v_.shape[-2]

        dist_matrix = torch.cdist(u_, v_)
        u2cp, _ = dist_matrix.min(axis=-1)
        v2cp, _ = dist_matrix.min(axis=-2)
        mean_p2cp = (torch.sum(u2cp, dim=-1) / n + torch.sum(v2cp, dim=-1) / m) / 2

        return self.reduction(mean_p2cp)
