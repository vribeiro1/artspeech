import pdb

import torch
import torch.nn as nn

from phoneme_to_articulation.metrics import EuclideanDistance, MeanP2CPDistance
from phoneme_to_articulation.principal_components.models import Decoder, MultiDecoder
from phoneme_to_articulation.principal_components.transforms import Decode, InputTransform


# Problem in metrics calculation since it is taking padding into account, which creates unrealistic
# good results.
# TODO: Think about how to fix this problem.


class DecoderEuclideanDistance(nn.Module):
    def __init__(self, decoder_filepath, n_components, n_samples, reduction, device, denorm_fn=None):
        super().__init__()
        self.n_samples = n_samples
        self.denorm_fn = denorm_fn

        self.decode = Decode(
            decoder_cls=Decoder,
            state_dict_filepath=decoder_filepath,
            device=device,
            n_components=n_components,
            out_features=2*n_samples
        )

        self.euclidean = EuclideanDistance(reduction=reduction)

    def forward(self, outputs, targets):
        bs, seq_len, _, _, _ = targets.shape
        output_shapes = self.decode(outputs)
        output_shapes = output_shapes.reshape(bs, seq_len, 2, self.n_samples).unsqueeze(dim=2)

        if self.denorm_fn is not None:
            targets = self.denorm_fn(targets)
            output_shapes = self.denorm_fn(output_shapes)

        euclidean = self.euclidean(output_shapes, targets).mean(dim=-1)

        return euclidean


class DecoderMeanP2CPDistance(nn.Module):
    def __init__(self, decoder_filepath, n_components, n_samples, reduction, device, denorm_fn=None):
        super().__init__()
        self.n_samples = n_samples
        self.denorm_fn = denorm_fn

        self.decode = Decode(
            decoder_cls=Decoder,
            state_dict_filepath=decoder_filepath,
            device=device,
            n_components=n_components,
            out_features=2*n_samples
        )

        self.mean_p2cp = MeanP2CPDistance(reduction=reduction)

    def forward(self, outputs, targets):
        bs, seq_len, _, _, _ = targets.shape
        output_shapes = self.decode(outputs.clone())
        output_shapes = output_shapes.reshape(bs, seq_len, 2, self.n_samples).unsqueeze(dim=2)

        if self.denorm_fn is not None:
            targets = self.denorm_fn(targets.clone())
            output_shapes = self.denorm_fn(output_shapes)

        outputs_u = output_shapes.permute(0, 1, 2, 4, 3)
        targets_v = targets.permute(0, 1, 2, 4, 3)
        mean_p2cp = self.mean_p2cp(outputs_u, targets_v)

        return mean_p2cp


class DecoderMeanP2CPDistance2(nn.Module):
    def __init__(
            self,
            dataset_config,
            decoder_state_dict_filepath,
            indices_dict,
            autoencoder_kwargs,
            denorm_fns,
            device,
        ):
        super().__init__()

        self.dataset_config = dataset_config
        decoder = MultiDecoder(
            indices_dict,
            **autoencoder_kwargs,
        )
        decoder_state_dict = torch.load(
            decoder_state_dict_filepath,
            map_location=device
        )
        decoder.load_state_dict(decoder_state_dict)
        self.decode = InputTransform(
            transform=decoder,
            device=device
        )

        self.mean_p2cp = MeanP2CPDistance(reduction="none")
        self.denorm_fns = denorm_fns

    def forward(self, outputs, targets, lengths):
        bs, seq_len, num_articulators, _, num_samples = targets.shape

        outputs = outputs.clone()
        targets = targets.clone()

        outputs_shapes = self.decode(outputs)
        outputs_shapes = outputs_shapes.reshape(bs, seq_len, num_articulators, 2, num_samples)
        for i in range(num_articulators):
            outputs_shapes[..., i, :, :] = self.denorm_fns[i](outputs_shapes[..., i, :, :])
            targets[..., i, :, :] = self.denorm_fns[i](targets[..., i, :, :])

        p2cp = self.mean_p2cp(outputs_shapes, targets)
        p2cp_mm = p2cp * self.dataset_config.RES * self.dataset_config.PIXEL_SPACING
        mean_p2cp = torch.concat([p2cp_mm[i, :l, :] for i, l in enumerate(lengths)]).mean()
        return mean_p2cp


class AutoencoderP2CPDistance(nn.Module):
    def __init__(self, reduction, denorm_fn=None):
        super().__init__()

        self.denorm_fn = denorm_fn
        self.mean_p2cp = MeanP2CPDistance(reduction=reduction)

    def forward(self, outputs, targets):
        bs, in_features = targets.shape

        outputs = outputs.clone().reshape(bs, 2, in_features // 2)
        if self.denorm_fn is not None:
            outputs = self.denorm_fn(outputs)
        outputs = outputs.reshape(0, 2, 1)

        targets = targets.clone().reshape(bs, 2, in_features // 2)
        if self.denorm_fn is not None:
            targets = self.denorm(targets)
        targets = targets.reshape(0, 2, 1)

        mean_p2cp = self.mean_p2cp(outputs, targets)

        return mean_p2cp
