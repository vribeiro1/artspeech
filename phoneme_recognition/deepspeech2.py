import pdb

import os
import torch
import torch.nn.functional as F
import torch.nn as nn

from settings import BASE_DIR

DEEPSPEECH2_PRETRAINED_LIBRISPEECH_FILEPATH = os.path.join(
    BASE_DIR,
    "data",
    "deepspeech2_pretrained_librispeech.pt"
)


class ResidualCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout, num_features):
        super().__init__()

        padding = kernel_size // 2

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.layer_norm1 = nn.LayerNorm(num_features)

        self.cnn2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.layer_norm2 = nn.LayerNorm(num_features)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = x.transpose(2, 3).contiguous()
        out = self.layer_norm1(out)
        out = out.transpose(2, 3).contiguous()

        out = F.gelu(out)
        out = self.dropout(out)
        out = self.cnn1(out)

        out = out.transpose(2, 3).contiguous()
        out = self.layer_norm2(out)
        out = out.transpose(2, 3).contiguous()

        out = F.gelu(out)
        out = self.dropout(out)
        out = self.cnn2(out)

        out += x
        return out


class RecurrentBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=False,
            batch_first=False
        )

        self.layer_norm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.layer_norm(x)
        out = F.gelu(out)
        out, _ = self.rnn(out)
        out = self.dropout(out)
        return out


class Adapter(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.Linear(out_features, out_features)
        )

    def forward(self, x):
        x = torch.transpose(x, 3, 2)  # (B, C, T, D)
        x = self.adapter(x)
        x = torch.transpose(x, 3, 2)  # (B, C, D, T)
        return x


class DeepSpeech2(nn.Module):
    def __init__(
        self,
        in_channels,
        num_residual_layers,
        num_rnn_layers,
        rnn_hidden_size,
        num_classes=31,  # 31 LibriSpeech classes including blank, silence and unknown
        num_features=80,
        dropout=0.1,
        adapter_out_features=None
    ):
        super().__init__()

        out_channels = 32
        kernel_size = 3
        padding = kernel_size // 2

        if adapter_out_features is not None:
            self.adapter = Adapter(num_features, adapter_out_features)
            num_features = adapter_out_features
        else:
            self.adapter = None
        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding)

        self.residual_layers = nn.ModuleList([
            ResidualCNN(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                dropout=dropout,
                num_features=num_features
            ) for _ in range(num_residual_layers)
        ])

        self.linear = nn.Linear(num_features * out_channels, rnn_hidden_size)

        self.recurrent_layers = nn.ModuleList([
            RecurrentBlock(
                input_size=rnn_hidden_size,
                hidden_size=rnn_hidden_size,
                dropout=dropout
            ) for _ in range(num_rnn_layers)
        ])

        self.feature_extractor = nn.Sequential(
            nn.Linear(rnn_hidden_size, rnn_hidden_size),
            nn.GELU(),
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(rnn_hidden_size, num_classes)

    @staticmethod
    def get_noise_logits(x, factor):
        out = x + factor * torch.randn_like(x)
        return out

    @staticmethod
    def get_normalized_outputs(x, use_log_prob=False):
        norm_fn = F.log_softmax if use_log_prob else F.softmax
        out = norm_fn(x, dim=-1)
        return out

    def forward(self, x, return_features=False):
        """
        Args:
            x (torch.tensor): Tensor of shape (B, C, D, T)

        Return:
            outputs (torch.tensor): Tensor of shape (batch_size, time, classes)
            features (torch.tensor): Tensor of shape (batch_size, time, dim)
        """
        if self.adapter is not None:
            x = self.adapter(x)

        out = self.cnn(x)
        for residual_layer in self.residual_layers:
            out = residual_layer(out)

        batch_size, channels, features, seq_len = out.shape
        out = out.view(batch_size, channels * features, seq_len)
        out = out.permute(2, 0, 1)  # time, batch_size, features
        out = self.linear(out)
        for recurrent_layer in self.recurrent_layers:
            out = recurrent_layer(out)
        out = out.permute(1, 0, 2)  # batch, time, features

        features = self.feature_extractor(out)
        out = self.classifier(self.dropout(features))

        if return_features:
            return out, features
        else:
            return out

    @classmethod
    def load_librispeech_model(cls, num_features=80, adapter_out_features=None):
        print("Loading Librispeech-pretrained model")
        model = cls(
            in_channels=2,
            num_residual_layers=5,
            num_rnn_layers=3,
            rnn_hidden_size=128,
            num_classes=31,
            num_features=num_features,
            dropout=0.05,
            adapter_out_features=adapter_out_features
        )

        state_dict = torch.load(
            DEEPSPEECH2_PRETRAINED_LIBRISPEECH_FILEPATH,
            map_location=torch.device("cpu")
        )
        model.load_state_dict(state_dict, strict=False)

        return model
