import torch
import torch.nn as nn

from functools import reduce
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from helpers import make_indices_dict
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
            nn.Linear(in_features=hidden_features, out_features=hidden_features // 2),
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

        if isinstance(list(indices_dict.values())[0], int):
            indices_dict = make_indices_dict(indices_dict)

        self.latent_size = 1 + max(set(
            reduce(lambda l1, l2: l1 + l2,
            indices_dict.values())
        ))

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        if isinstance(rnn, str):
            rnn = RNNType[rnn.upper()]
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

        self.predictor = PrincipalComponentsPredictor(
            in_features=hidden_size,
            num_components=self.latent_size,
            hidden_features=256,
        )

    @property
    def total_parameters(self):
        return sum(p.numel() for p in self.parameters())


    def forward(self, x, lengths):
        """
        Args:
        x (torch.tensor): Torch tensor of shape (bs, seq).
        lengths (list): Lengths of the input sequences sorted in decreasing order.

        Return:
        (torch.tensor): Torch tensor of shape (bs, seq_len, num_components).
        """
        embed = self.embedding(x)
        packed_embed = pack_padded_sequence(
            embed,
            lengths,
            batch_first=True
        )
        packed_rnn_out, _ = self.rnn(packed_embed)
        rnn_out, _ = pad_packed_sequence(
            packed_rnn_out,
            batch_first=True
        )

        linear_out = self.linear(rnn_out)  # (bs, seq_len, embed_dim)
        components = torch.tanh(self.predictor(linear_out))
        return components
