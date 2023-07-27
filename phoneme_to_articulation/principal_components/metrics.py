import torch
import torch.nn as nn

from phoneme_to_articulation.metrics import MeanP2CPDistance
from phoneme_to_articulation.principal_components.models.autoencoder import (
    Decoder,
    MultiDecoder
)
from phoneme_to_articulation.principal_components.transforms import InputTransform


class DecoderMeanP2CPDistance2(nn.Module):
    def __init__(
        self,
        dataset_config,
        decoder_state_dict_filepath,
        indices_dict,
        autoencoder_kwargs,
        denorm_fns,
        device,
        decoder_cls=Decoder
    ):
        super().__init__()

        self.articulators = sorted(indices_dict.keys())
        self.dataset_config = dataset_config
        self.to_mm = self.dataset_config.RES * self.dataset_config.PIXEL_SPACING
        decoder = MultiDecoder(
            indices_dict,
            decoder_cls=decoder_cls,
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

        outputs_shapes = self.decode(outputs)
        outputs_shapes = outputs_shapes.reshape(bs, seq_len, num_articulators, 2, num_samples)
        for i, articulator in enumerate(self.articulators):
            outputs_shapes[..., i, :, :] = self.denorm_fns[articulator](outputs_shapes[..., i, :, :])
            targets[..., i, :, :] = self.denorm_fns[articulator](targets[..., i, :, :])
        outputs_shapes = outputs_shapes.transpose(-1, -2)
        targets = targets.transpose(-1, -2)

        p2cp = self.mean_p2cp(outputs_shapes, targets)
        p2cp_mm = p2cp * self.to_mm
        p2cp_mm = torch.concat([p2cp_mm[i, :l, :] for i, l in enumerate(lengths)])
        mean_p2cp = p2cp_mm.mean()
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
