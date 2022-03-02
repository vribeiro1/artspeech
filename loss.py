import pdb

import torch.nn as nn

from vt_tools import *


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super().__init__()

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
