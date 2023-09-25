import funcy
import torch
import torch.nn as nn

from enum import Enum

from helpers import make_indices_dict


class PCAEncoder(nn.Module):
    def __init__(
        self,
        in_features,
        num_components,
        mean=None,
        whiten=False,
        **kwargs,
    ):
        super().__init__()
        self.eigenvalues = nn.Parameter(
            torch.rand(
                size=(num_components,),
            )
        )
        self.eigenvectors = nn.Parameter(
            torch.rand(
                size=(num_components, in_features),
            )
        )
        self.mean = mean or torch.zeros(size=(in_features,))
        self.whiten = whiten

    def forward(self, x):
        x = x - self.mean.to(x.device)
        z = torch.mm(x, self.eigenvectors.T)
        if self.whiten:
            z /= torch.sqrt(self.eigenvalues)

        return z


class PCADecoder(nn.Module):
    def __init__(
        self,
        out_features,
        num_components,
        mean=None,
        whiten=False,
        **kwargs,
    ):
        super().__init__()
        self.eigenvalues = nn.Parameter(
            torch.rand(
                size=(num_components, 1)
            )
        )
        self.eigenvectors = nn.Parameter(
            torch.rand(
                size=(num_components, out_features),
            )
        )
        self.mean = mean or torch.zeros(size=(out_features,))
        self.whiten = whiten

    def forward(self, z):
        bs, length, dim1 = z.shape
        z = z.reshape(bs * length, dim1)

        if self.whiten:
            out = torch.mm(z, torch.sqrt(self.eigenvalues)) * self.eigenvectors
            out += self.mean
        else:
            out = torch.mm(z, self.eigenvectors)
            out += self.mean.to(z.device)

        _, dim2 = out.shape
        out = out.reshape(bs, length, dim2)

        return out


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


class EncoderType(Enum):
    AE = Encoder
    PCA = PCAEncoder


class DecoderType(Enum):
    AE = Decoder
    PCA = PCADecoder


class MultiEncoder(nn.Module):
    def __init__(
        self,
        indices_dict,
        in_features,
        hidden_features,
        encoder_cls=Encoder,
    ):
        super().__init__()

        if isinstance(list(indices_dict.values())[0], int):
            indices_dict = make_indices_dict(indices_dict)

        self.indices_dict = indices_dict
        self.latent_size = max(funcy.flatten(self.indices_dict.values())) + 1
        self.sorted_articulators = sorted(self.indices_dict.keys())

        if isinstance(encoder_cls, str):
            encoder_cls = EncoderType[encoder_cls].value

        self.encoders = nn.ModuleDict({
            articulator: encoder_cls(
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
        decoder_cls=Decoder,
    ):
        super().__init__()

        if isinstance(list(indices_dict.values())[0], int):
            indices_dict = make_indices_dict(indices_dict)

        self.indices_dict = indices_dict
        self.latent_size = max(funcy.flatten(self.indices_dict.values())) + 1
        self.sorted_articulators = sorted(self.indices_dict.keys())

        if isinstance(decoder_cls, str):
            decoder_cls = DecoderType[decoder_cls].value

        self.decoders = nn.ModuleDict({
            articulator: decoder_cls(
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

    @property
    def total_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        """
        Args:
            x (torch.tensor): Torch tensor of shape (bs, N_articulators, in_features)
        """
        latent_space = torch.tanh(self.encoders(x))
        outputs = self.decoders(latent_space)
        return outputs, latent_space
