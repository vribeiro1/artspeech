import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from vt_tools import *


class HiddenBlock(nn.Module):
    def __init__(self, hidden_features):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(in_features=hidden_features, out_features=hidden_features),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_features, n_components, hidden_blocks=1, hidden_features=64):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.ReLU()
        )

        self.hidden_layers = nn.ModuleList([
            HiddenBlock(hidden_features=hidden_features)
            for _ in range(hidden_blocks)]
        )

        self.output_layer = nn.Linear(in_features=hidden_features, out_features=n_components)

    def forward(self, x):
        out_input = self.input_layer(x)

        out_hidden = out_input
        for hidden_layer in self.hidden_layers:
            out_hidden = hidden_layer(out_hidden)

        out = self.output_layer(out_hidden)

        return out


class Decoder(nn.Module):
    def __init__(self, n_components, out_features, hidden_blocks=1, hidden_features=64):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(in_features=n_components, out_features=hidden_features),
            nn.ReLU()
        )

        self.hidden_layers = nn.ModuleList([
            HiddenBlock(hidden_features=hidden_features)
            for _ in range(hidden_blocks)]
        )

        self.output_layer = nn.Linear(in_features=hidden_features, out_features=out_features)

    def forward(self, x):
        out_input = self.input_layer(x)

        out_hidden = out_input
        for hidden_layer in self.hidden_layers:
            out_hidden = hidden_layer(out_hidden)

        out = self.output_layer(out_hidden)

        return out


class Autoencoder(nn.Module):
    def __init__(self, in_features, n_components, hidden_blocks=1, hidden_features=64):
        super().__init__()

        self.encoder = Encoder(
            in_features=in_features,
            n_components=n_components,
            hidden_blocks=hidden_blocks,
            hidden_features=hidden_features
        )

        self.decoder = Decoder(
            n_components=n_components,
            out_features=in_features,
            hidden_blocks=hidden_blocks,
            hidden_features=hidden_features
        )

    def forward(self, x):
        latents = torch.tanh(self.encoder(x))
        outputs = self.decoder(latents)
        return outputs, latents


class ArticulatorPrincipalComponentsPredictor(nn.Module):
    def __init__(self, in_features, n_components):
        super().__init__()

        self.linear = nn.Sequential(
            nn.LayerNorm([in_features]),
            nn.Linear(in_features=in_features, out_features=256),
            nn.ReLU(),
            nn.LayerNorm([256]),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(in_features=128, out_features=n_components)
        )

    def forward(self, inputs):
        linear_out = self.linear(inputs)
        return linear_out


class PrincipalComponentsArtSpeech(nn.Module):
    def __init__(self, vocab_size, n_components, embed_dim=64, hidden_size=128, gru_dropout=0.):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_size, num_layers=2, bidirectional=True, dropout=gru_dropout, batch_first=True)

        self.linear = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU()
        )

        self.predictor = ArticulatorPrincipalComponentsPredictor(
            in_features=hidden_size,
            n_components=n_components
        )

    def forward(self, x, lengths):
        """
        Args:
        x (torch.tensor): Torch tensor of shape (bs, seq).
        lengths (list): Lengths of the input sequences sorted in decreasing order.

        Return:
        (torch.tensor): Torch tensor of shape (bs, seq_len, n_components).
        """
        embed = self.embedding(x)

        packed_embed = pack_padded_sequence(embed, lengths, batch_first=True)
        packed_rnn_out, _ = self.rnn(packed_embed)
        rnn_out, _ = pad_packed_sequence(packed_rnn_out, batch_first=True)

        linear_out = self.linear(rnn_out)  # (bs, seq_len, embed_dim)
        components = torch.tanh(self.predictor(linear_out))

        return components
