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
        linear_out = self.linear(inputs)

        x_pos = self.x_coords(linear_out)
        x_pos = torch.sigmoid(x_pos)

        y_pos = self.y_coords(linear_out)
        y_pos = torch.sigmoid(y_pos)

        return torch.stack([x_pos, y_pos], dim=2)


class ArtSpeech(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_size=128, n_samples=50):
        super(ArtSpeech, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_size, num_layers=2, bidirectional=True)

        self.linear = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU()
        )

        self.llip_predictor = ArticulatorPredictor(hidden_size, n_samples)
        self.velum_predictor = ArticulatorPredictor(hidden_size, n_samples)
        self.tongue_predictor = ArticulatorPredictor(hidden_size, n_samples)
        self.ulip_predictor = ArticulatorPredictor(hidden_size, n_samples)

    def forward(self, x):
        embed = self.embedding(x)

        # Reshape as batch second
        embed = embed.transpose(1, 0)
        rnn_out, h_n = self.rnn(embed)
        # Reshape again as batch first
        rnn_out = rnn_out.transpose(1, 0)

        linear_out = self.linear(rnn_out)

        llip = self.llip_predictor(linear_out)
        velum = self.velum_predictor(linear_out)
        tongue = self.tongue_predictor(linear_out)
        ulip = self.ulip_predictor(linear_out)

        out = torch.stack([llip, velum, tongue, ulip], dim=2)
        return out
