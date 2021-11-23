import pdb

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


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

        self.mel_loss_fn = nn.MSELoss()
        self.gate_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        mel_spec_targets, gate_targets = targets
        mel_spec_targets.requires_grad = False
        gate_targets.requires_grad = False
        gate_targets = gate_targets.view(-1, 1)

        mel_specs, mel_specs_postnet, gate_outputs, _ = outputs
        gate_outputs = gate_outputs.view(-1, 1)

        mel_loss = self.mel_loss_fn(mel_specs, mel_spec_targets)
        mel_loss_postnet = self.mel_loss_fn(mel_specs_postnet, mel_spec_targets)
        gate_loss = self.gate_loss_fn(gate_outputs, gate_targets)

        return mel_loss + mel_loss_postnet + gate_loss
