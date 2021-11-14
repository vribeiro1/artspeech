import torch
import torch.nn as nn
import torch.nn.functional as F


class ArticulatorsEmbedding(nn.Module):
    def __init__(self, n_curves, w_init_gain="linear"):
        super(ArticulatorsEmbedding, self).__init__()

        # Performs a 3D-convolution along the articulators, combining the x and y coordinates into
        # a single channel.
        kernel_size = torch.tensor([1, 3, 1])
        padding = torch.div(kernel_size - 1, 2, rounding_mode="trunc")

        self.conv1 = nn.Conv3d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=1,
            bias=True
        )

        torch.nn.init.xavier_uniform_(
            self.conv1.weight,
            gain=nn.init.calculate_gain(w_init_gain)
        )

        # Performs a 3D-convolution along the articulators, combining the articulator curves
        # into a single channel.
        kernel_size = torch.tensor([1, 3])
        padding = torch.div(kernel_size - 1, 2, rounding_mode="trunc")

        self.conv2 = nn.Conv2d(
            in_channels=n_curves,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=1,
            bias=True
        )

        torch.nn.init.xavier_uniform_(
            self.conv2.weight,
            gain=nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        """
        Args:
        x (torch.tensor): Tensor of shape (bs, 2, seq_len, n_samples, n_curves)
        """
        conv1_out = F.relu(self.conv1(x))  # (bs, 1, seq_len, n_samples, n_curves)

        conv2_in = conv1_out.squeeze(dim=1)  # (bs, seq_len, n_samples, n_curves)
        conv2_in = conv2_in.permute(0, 3, 1, 2)  # (bs, n_curves, seq_len, n_samples)
        conv2_out = self.conv2(conv2_in)  # (bs, 1, seq_len, n_samples)

        outputs = conv2_out.squeeze(dim=1)  # (bs, seq_len, n_samples)

        return outputs
