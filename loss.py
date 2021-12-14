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
