import funcy
import torch
import torch.nn as nn

from helpers import make_indices_dict


class Encoder(nn.Module):
    def __init__(self, in_features, num_components, hidden_features):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.ReLU(),
            nn.Linear(in_features=hidden_features, out_features=hidden_features // 2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_features // 2, out_features=num_components)
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, num_components, out_features, hidden_features):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(in_features=num_components, out_features=hidden_features // 2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_features // 2, out_features=hidden_features),
            nn.ReLU(),
            nn.Linear(in_features=hidden_features, out_features=out_features)
        )

    def forward(self, x):
        return self.decoder(x)


class MultiEncoder(nn.Module):
    def __init__(
        self,
        indices_dict,
        in_features,
        hidden_features,
    ):
        super().__init__()

        if isinstance(list(indices_dict.values())[0], int):
            indices_dict = make_indices_dict(indices_dict)
            
        self.indices_dict = indices_dict
        self.latent_size = max(funcy.flatten(self.indices_dict.values())) + 1
        self.sorted_articulators = sorted(self.indices_dict.keys())
        self.encoders = nn.ModuleDict({
            articulator: Encoder(
                in_features=in_features,
                num_components=len(indices),
                hidden_features=hidden_features,
            )
            for articulator, indices in self.indices_dict.items()
        })

    def forward(self, x):
        bs, _, _ = x.shape
        articulators_latent_space = {
            articulator: -torch.inf * torch.ones(
                size=(bs, self.latent_size),
                dtype=torch.float, device=x.device
            ) for articulator in self.sorted_articulators
        }

        for i, articulator in enumerate(self.sorted_articulators):
            indices = self.indices_dict[articulator]
            encoder = self.encoders[articulator]
            articulators_latent_space[articulator][..., indices] = encoder(x[..., i, :])

        latent_space = torch.stack([
            articulators_latent_space[articulator] for articulator in self.sorted_articulators
        ], dim=1)
        latent_space, _ = torch.max(latent_space, dim=1)
        return latent_space


class MultiDecoder(nn.Module):
    def __init__(
        self,
        indices_dict,
        in_features,
        hidden_features,
    ):
        super().__init__()

        if isinstance(list(indices_dict.values())[0], int):
            indices_dict = make_indices_dict(indices_dict)

        self.indices_dict = indices_dict
        self.latent_size = max(funcy.flatten(self.indices_dict.values())) + 1
        self.sorted_articulators = sorted(self.indices_dict.keys())
        self.decoders = nn.ModuleDict({
            articulator: Decoder(
                num_components=len(indices),
                out_features=in_features,
                hidden_features=hidden_features,
            )
            for articulator, indices in self.indices_dict.items()
        })

    def forward(self, x):
        output_list = []
        for articulator in self.sorted_articulators:
            decoder = self.decoders[articulator]
            indices = self.indices_dict[articulator]
            decoder_input = x[..., indices]
            decoder_output = decoder(decoder_input).unsqueeze(dim=-2)
            output_list.append(decoder_output)
        outputs = torch.concat(output_list, dim=-2)

        return outputs


class MultiArticulatorAutoencoder(nn.Module):
    def __init__(
        self,
        in_features,
        indices_dict,
        hidden_features=64,
    ):
        super().__init__()

        if isinstance(list(indices_dict.values())[0], int):
            indices_dict = make_indices_dict(indices_dict)

        self.indices_dict = indices_dict
        self.latent_size = max(funcy.flatten(self.indices_dict.values())) + 1
        self.sorted_articulators = sorted(self.indices_dict.keys())
        self.encoders = MultiEncoder(
            indices_dict=indices_dict,
            in_features=in_features,
            hidden_features=hidden_features,
        )
        self.decoders = MultiDecoder(
            indices_dict=indices_dict,
            in_features=in_features,
            hidden_features=hidden_features,
        )

    def forward(self, x):
        """
        Args:
            x (torch.tensor): Torch tensor of shape (bs, N_articulators, in_features)
        """
        latent_space = torch.tanh(self.encoders(x))
        outputs = self.decoders(latent_space)
        return outputs, latent_space
