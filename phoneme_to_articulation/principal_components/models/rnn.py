import torch
import torch.nn as nn

from functools import reduce
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from phoneme_to_articulation import RNNType


class PrincipalComponentsPredictor(nn.Module):
    def __init__(
        self,
        in_features,
        num_components,
        hidden_features=256
    ):
        super().__init__()

        self.linear = nn.Sequential(
            nn.LayerNorm([in_features]),
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.ReLU(),
            nn.LayerNorm([hidden_features]),
            nn.Linear(in_features=hidden_features, out_features=hidden_features),
            nn.ReLU(),
            nn.LayerNorm([hidden_features]),
            nn.Linear(in_features=hidden_features, out_features=hidden_features),
            nn.ReLU(),
            nn.LayerNorm(hidden_features),
            nn.Linear(in_features=hidden_features, out_features=hidden_features // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_features // 2),
            nn.Linear(in_features=hidden_features // 2, out_features=hidden_features // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_features // 2),
            nn.Linear(in_features=hidden_features // 2, out_features=num_components)
        )

    def forward(self, inputs):
        linear_out = self.linear(inputs)
        return linear_out


class PrincipalComponentsArtSpeech(nn.Module):
    def __init__(
        self,
        vocab_size,
        indices_dict,
        embed_dim=64,
        hidden_size=128,
        rnn_dropout=0.,
        rnn=RNNType.GRU,
    ):
        super().__init__()

        self.latent_size = 1 + max(set(
            reduce(lambda l1, l2: l1 + l2,
            indices_dict.values())
        ))
        self.indices_dict = indices_dict
        self.sorted_articulators = sorted(indices_dict.keys())

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        rnn_class = rnn.value
        self.rnn = rnn_class(
            embed_dim,
            hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=rnn_dropout,
            batch_first=True
        )

        self.linear = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU()
        )

        self.predictors = nn.ModuleDict({
            articulator: PrincipalComponentsPredictor(
                in_features=hidden_size,
                num_components=len(components),
                hidden_features=256,
            ) for articulator, components in indices_dict.items()
        })

    def forward(self, x, lengths):
        """
        Args:
        x (torch.tensor): Torch tensor of shape (bs, seq).
        lengths (list): Lengths of the input sequences sorted in decreasing order.

        Return:
        (torch.tensor): Torch tensor of shape (bs, seq_len, num_components).
        """
        embed = self.embedding(x)
        packed_embed = pack_padded_sequence(embed, lengths, batch_first=True)
        packed_rnn_out, _ = self.rnn(packed_embed)
        rnn_out, _ = pad_packed_sequence(packed_rnn_out, batch_first=True)

        linear_out = self.linear(rnn_out)  # (bs, seq_len, embed_dim)

        bs, seq_len = x.shape
        articulators_components = {
            articulator: -torch.inf * torch.ones(
                size=(bs, seq_len, self.latent_size),
                dtype=torch.float, device=x.device
            ) for articulator in self.sorted_articulators
        }

        for articulator in self.sorted_articulators:
            indices = self.indices_dict[articulator]
            predictor = self.predictors[articulator]
            articulators_components[articulator][..., indices] = predictor(linear_out)

        components = torch.stack([
            articulators_components[articulator] for articulator in self.sorted_articulators
        ], dim=2)
        components, _ = torch.max(components, dim=2)
        components = torch.tanh(components)

        return components
