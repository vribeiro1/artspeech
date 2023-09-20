import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ArticulatorPredictor(nn.Module):
    def __init__(self, in_features, n_samples):
        super().__init__()
        self.linear = nn.Sequential(
            nn.LayerNorm([in_features]),
            nn.Linear(in_features=in_features, out_features=256),
            nn.ReLU(),
            nn.LayerNorm([256]),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.LayerNorm(256),
        )

        self.x_coords = nn.Linear(in_features=256, out_features=n_samples)
        self.y_coords = nn.Linear(in_features=256, out_features=n_samples)

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


class Decoder(nn.Module):
    def __init__(self, n_articulators, hidden_size, n_samples):
        super(Decoder, self).__init__()

        self.predictors = nn.ModuleList([
            ArticulatorPredictor(hidden_size, n_samples) for i in range(n_articulators)
        ])

    def forward(self, x):
        out = torch.stack([
            predictor(x) for predictor in self.predictors
        ], dim=2)

        return torch.sigmoid(out)


class SimpleArtSpeech(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_articulators,
        embed_dim=64,
        hidden_size=128,
        num_samples=50,
        dropout=0.,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.ReLU()
        )

        self.predictors = nn.ModuleList([
            ArticulatorPredictor(hidden_size, num_samples) for _ in range(n_articulators)
        ])

    def forward(self, x, lengths):
        """
        Args:
            x (torch.tensor): Torch tensor of shape (bs, seq_len).
            lengths (list): Lengths of the input sequences sorted in decreasing order.

        Return:
            output (torch.tensor): Torch tensor of shape (bs, seq_len, n_articulators, 2, n_samples)
        """
        embed = self.embedding(x)  # (bs, seq_len, embed_dim)

        linear_out = self.linear(self.dropout(embed))  # (bs, seq_len, embed_dim)
        out = torch.stack([
            predictor(linear_out) for predictor in self.predictors
        ], dim=2)

        return torch.sigmoid(out)


class ArtSpeech(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_articulators,
        embed_dim=64,
        hidden_size=128,
        n_samples=50,
        dropout=0.,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_size, num_layers=2, bidirectional=True, dropout=dropout, batch_first=True)

        self.linear = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU()
        )

        self.predictors = nn.ModuleList([
            ArticulatorPredictor(hidden_size, n_samples) for _ in range(n_articulators)
        ])

    def forward(self, x, lengths):
        """
        Args:
            x (torch.tensor): Torch tensor of shape (bs, seq_len).
            lengths (list): Lengths of the input sequences sorted in decreasing order.

        Return:
            output (torch.tensor): Torch tensor of shape (bs, seq_len, n_articulators, 2, n_samples)
        """
        embed = self.embedding(x)
        packed_embed = pack_padded_sequence(embed, lengths, batch_first=True)
        packed_rnn_out, _ = self.rnn(packed_embed)
        rnn_out, _ = pad_packed_sequence(packed_rnn_out, batch_first=True)

        linear_out = self.linear(rnn_out)  # torch.Size([bs, seq_len, embed_dim])
        out = torch.stack([
            predictor(linear_out) for predictor in self.predictors
        ], dim=2)

        return torch.sigmoid(out)
