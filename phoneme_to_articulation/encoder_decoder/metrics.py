import torch
import torch.nn as nn

from phoneme_to_articulation.metrics import MeanP2CPDistance


class P2CPDistance(nn.Module):
    def __init__(
        self,
        dataset_config,
    ):
        super().__init__()
        self.dataset_config = dataset_config
        self.to_mm = self.dataset_config.RES * self.dataset_config.PIXEL_SPACING

        self.mean_p2cp = MeanP2CPDistance(reduction="none")

    def forward(self, outputs, targets, lengths):
        p2cp = self.mean_p2cp(outputs, targets)
        p2cp_mm = p2cp * self.to_mm
        p2cp_mm = torch.concat([p2cp_mm[i, :l, :] for i, l in enumerate(lengths)])
        mean_p2cp = p2cp_mm.mean()
        return mean_p2cp