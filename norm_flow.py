"""
The code for Affine Coupling and RealNVP was based on the tutorial available at
https://github.com/xqding/RealNVP
"""

import pdb

import torch
import torch.nn as nn
import torch.nn.init as init


class AffineCoupling(nn.Module):
    def __init__(self, mask, hidden_size):
        super(AffineCoupling, self).__init__()

        self.n_dims = len(mask)
        self.hidden_size = hidden_size

        self.mask = nn.Parameter(mask, requires_grad=False)

        self.scale_fc1 = nn.Linear(self.n_dims, self.hidden_size)
        self.scale_fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.scale_fc3 = nn.Linear(self.hidden_size, self.n_dims)

        self.scale = nn.Parameter(torch.Tensor(self.n_dims))
        init.normal_(self.scale)

        self.translation_fc1 = nn.Linear(self.n_dims, self.hidden_size)
        self.translation_fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.translation_fc3 = nn.Linear(self.hidden_size, self.n_dims)

    def _scale(self, x):
        s = torch.sigmoid(self.scale_fc1(x))
        s = torch.sigmoid(self.scale_fc2(s))
        s = torch.sigmoid(self.scale_fc3(s)) * self.scale

        return s

    def _translation(self, x):
        t = torch.relu(self.translation_fc1(x))
        t = torch.relu(self.translation_fc2(t))
        t = self.translation_fc3(t)

        return t

    def forward(self, x):
        s = self._scale(x * self.mask)
        t = self._translation(x * self.mask)

        y = self.mask * x + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det = torch.sum((1 - self.mask) * s, -1)

        return y, log_det

    def inverse(self, y):
        s = self._scale(y * self.mask)
        t = self._translation(y * self.mask)

        x = self.mask * y + (1 - self.mask) * ((y - t) * torch.exp(-s))
        log_det = torch.sum((1 - self.mask) * -s, dim=-1)

        return x, log_det


class RealNVP(nn.Module):
    """
    Density Estimation using Real NVP
    Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio
    ICLR 2017
    https://arxiv.org/abs/1605.08803
    """
    def __init__(self, masks, hidden_size):
        super(RealNVP, self).__init__()

        self.hidden_size = hidden_size
        self.masks = nn.ParameterList([
            nn.Parameter(torch.Tensor(mask),requires_grad=False)
            for mask in masks
        ])

        self.flows = nn.ModuleList([
            AffineCoupling(mask, self.hidden_size)
            for mask in self.masks
        ])

    def forward(self, x):
        y = x
        cumulative_log_det = 0
        for flow in self.flows:
            y, log_det = flow(y)
            cumulative_log_det += log_det

        # Normalization layer
        # log_det = torch.sum(
        #     torch.log(torch.abs(4 * (1 - torch.tanh(y) ** 2))),
        #     dim=-1
        # )
        # y = 4 * torch.tanh(y)
        # cumulative_log_det += log_det

        return y, cumulative_log_det

    def inverse(self, y):
        x = y
        cumulative_log_det = 0

        # Invert the normalization layer
        # log_det = torch.sum(
        #     torch.log(torch.abs((1. / 4.) * (1. / (1. - (x / 4.) ** 2.)))),
        #     dim=-1
        # )
        # x  = 0.5 * torch.log((1 + (x / 4.)) / (1 - (x / 4.)))
        # cumulative_log_det += log_det

        # Invert affine coupling layers
        for flow in reversed(self.flows):
            x, log_det = flow.inverse(x)
            cumulative_log_det += log_det

        return x, cumulative_log_det


class ArtSpeech(nn.Module):
    def __init__(self, vocab_size, n_articulators, embed_dim=64, hidden_size=128, n_samples=50, n_flows=8):
        super(ArtSpeech, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_size, num_layers=2, bidirectional=True)

        self.linear = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU()
        )

        # Uncomment for masking the same axis for all samples (x or y)
        # zeros = torch.zeros(n_samples)
        # ones = torch.ones(n_samples)
        # masks = torch.zeros((n_flows, 2 * n_samples))
        # masks[0::2] = torch.cat([ones, zeros])
        # masks[1::2] = torch.cat([zeros, ones])

        # Uncomment for masking half of the samples
        # zeros = torch.zeros(n_samples // 2)
        # ones = torch.ones(n_samples // 2)
        # masks = torch.zeros((n_flows, 2 * n_samples))
        # masks[0::2] = torch.cat([ones, zeros, ones, zeros])
        # masks[1::2] = torch.cat([zeros, ones, zeros, ones])

        # Uncomment for masking half of the samples with intercalation
        mask_0 = torch.zeros(2 * n_samples)
        mask_0[0::2] = torch.cat([torch.ones(n_samples // 2), torch.ones(n_samples // 2)])

        mask_1 = torch.zeros(2 * n_samples)
        mask_1[1::2] = torch.cat([torch.ones(n_samples // 2), torch.ones(n_samples // 2)])

        masks = torch.zeros((n_flows, 2 * n_samples))
        masks[0::2] = mask_0
        masks[1::2] = mask_1

        self.reshape = nn.Linear(hidden_size, 2 * n_samples)
        self.predictors = nn.ModuleList([
            RealNVP(masks, hidden_size) for i in range(n_articulators)
        ])

    def forward(self, x):
        """
        input: torch.Size([bs, seq_len])
        output: torch.Size([bs, seq_len, n_articulators, 2, n_samples])
        """
        embed = self.embedding(x)

        # Reshape as batch second
        embed = embed.transpose(1, 0)
        rnn_out, _ = self.rnn(embed)
        # Reshape again as batch first
        rnn_out = rnn_out.transpose(1, 0)

        linear_out = self.linear(rnn_out)  # torch.Size([bs, seq_len, embed_dim])
        reshape_out = self.reshape(linear_out)  # torch.Size([bs, seq_len, 2 * n_samples])

        outs = []
        log_dets = []
        for predictor in self.predictors:
            p_out, log_det = predictor(reshape_out)

            outs.append(p_out)
            log_dets.append(log_det)

        output = torch.stack(outs, dim=2)  # torch.Size([bs, seq_len, n_articulators, 2 * n_samples])
        log_dets = torch.stack(log_dets, dim=2)  # torch.Size([bs, seq_len, n_articulators])

        bs, seq_len, n_articulators, n_samples2 = output.shape
        output = output.view(bs, seq_len, n_articulators, 2, n_samples2 // 2)

        return torch.sigmoid(output), log_dets
