import pdb

import torch
import torch.nn as nn


class ArticulatorPredictor(nn.Module):
    def __init__(self, in_features, n_samples):
        super(ArticulatorPredictor, self).__init__()
        self.linear = nn.Sequential(
            nn.LayerNorm([128]),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.LayerNorm([256]),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
        )

        self.x_coords = nn.Linear(256, n_samples)
        self.y_coords = nn.Linear(256, n_samples)

    def forward(self, inputs):
        """
        input: torch.Size([bs, seq_len, embed_size])
        output: torch.Size([bs, seq_len, 2, n_samples])
        """
        linear_out = self.linear(inputs)
        x_pos = self.x_coords(linear_out)
        y_pos = self.y_coords(linear_out)
        out = torch.stack([x_pos, y_pos], dim=2)

        return out


class ArtSpeech(nn.Module):
    def __init__(
        self, vocab_size, n_articulators, embed_dim=64, hidden_size=128, n_samples=50, gru_dropout=0.
    ):
        super(ArtSpeech, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_size, num_layers=2, bidirectional=True, dropout=gru_dropout)

        self.linear = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU()
        )

        self.predictors = nn.ModuleList([
            ArticulatorPredictor(hidden_size, n_samples) for i in range(n_articulators)
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
        out = torch.stack([
            predictor(linear_out) for predictor in self.predictors
        ], dim=2)

        return torch.sigmoid(out)
