import funcy
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


class MultiArticulatorAutoencoder(nn.Module):
    def __init__(self, in_features, indices_dict, hidden_blocks=1, hidden_features=64):
        super().__init__()

        self.indices_dict = indices_dict
        self.shared_latent_size = max(funcy.flatten(self.indices_dict.values())) + 1
        self.sorted_articulators = sorted(self.indices_dict.keys())

        self.encoders = nn.ModuleDict({
            articulator: Encoder(
                in_features=in_features,
                n_components=len(indices),
                hidden_blocks=hidden_blocks,
                hidden_features=hidden_features
            )
            for articulator, indices in self.indices_dict.items()
        })

        self.decoders = nn.ModuleDict({
            articulator: Decoder(
                n_components=len(indices),
                out_features=in_features,
                hidden_blocks=hidden_blocks,
                hidden_features=hidden_features
            )
            for articulator, indices in self.indices_dict.items()
        })

    def forward(self, x):
        """
        Args:
        x (torch.tensor): Torch tensor of shape (bs, N_articulators, in_features)
        """
        bs, _, _ = x.shape

        articulators_latent_space = {
            articulator: -torch.inf * torch.ones(
                size=(bs, self.shared_latent_size),
                dtype=torch.float, device=x.device
            ) for articulator in self.sorted_articulators
        }

        for i, articulator in enumerate(self.sorted_articulators):
            indices = self.indices_dict[articulator]
            encoder = self.encoders[articulator]
            articulators_latent_space[articulator][:, indices] = torch.tanh(encoder(x[:, i, :]))

        latent_space = torch.stack([
            articulators_latent_space[articulator] for articulator in self.sorted_articulators
        ], dim=1)

        shared_latent_space, _ = torch.max(latent_space, dim=1)
        outputs = torch.concat([
            self.decoders[articulator](shared_latent_space[:, self.indices_dict[articulator]]).unsqueeze(dim=1)
            for articulator in self.sorted_articulators
        ], dim=1)

        return outputs, shared_latent_space
