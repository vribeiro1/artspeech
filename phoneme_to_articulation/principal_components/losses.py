import torch
import torch.nn as nn

from vt_tools import (
    LOWER_LIP,
    PHARYNX,
    SOFT_PALATE,
    TONGUE,
    UPPER_INCISOR,
    UPPER_LIP
)

from helpers import make_padding_mask
from phoneme_to_articulation.principal_components.models.autoencoder import (
    MultiEncoder,
    MultiDecoder
)
from phoneme_to_articulation.principal_components.transforms import InputTransform


class CriticalLoss(nn.Module):
    TV_TO_ARTICULATOR_MAP = {
        "LA": [LOWER_LIP, UPPER_LIP],
        "TTCD": [TONGUE, UPPER_INCISOR],
        "TBCD": [TONGUE, UPPER_INCISOR],
        "VEL": [SOFT_PALATE, PHARYNX]
    }

    def __init__(
        self,
        TVs,
        articulators,
        denormalize_fn=None,
    ):
        super().__init__()

        self.TVs = sorted(TVs)
        self.inject_reference = UPPER_INCISOR not in articulators

        if UPPER_INCISOR not in articulators:
            articulators = sorted(articulators + [UPPER_INCISOR])

        self.articulators_indices = {
            articulator: i
            for i, articulator in enumerate(articulators)
        }

        self.denorm_fn = denormalize_fn

    def forward(
        self,
        output_shapes,
        target_shapes,
        reference_arrays,
        critical_mask,
    ):
        if len(self.TVs) < 0:
            critical_loss = torch.tensor(0, device=target_shapes.device, dtype=torch.float)
            return critical_loss

        if self.inject_reference:
            ref_index = self.articulators_indices[UPPER_INCISOR]
            output_shapes = torch.concat([
                output_shapes[:, :, :ref_index, :, :],
                reference_arrays,
                output_shapes[:, :, ref_index:, :, :]
            ], dim=2)

        bs, seq_len, _, _, num_samples = target_shapes.shape
        critical_dists = []
        for TV in self.TVs:
            articulator_1, articulator_2 = self.TV_TO_ARTICULATOR_MAP[TV]

            idx_articulator_1 = self.articulators_indices[articulator_1]
            articulator_array_1 = output_shapes[..., idx_articulator_1, :, :]
            if self.denorm_fn and articulator_1 != UPPER_INCISOR:
                denorm_fn = self.denorm_fn[articulator_1]
                articulator_array_1 = denorm_fn(articulator_array_1)
            articulator_array_1 = articulator_array_1.transpose(2, 3)

            idx_articulator_2 = self.articulators_indices[articulator_2]
            articulator_array_2 = output_shapes[..., idx_articulator_2, :, :]
            if self.denorm_fn and articulator_2 != UPPER_INCISOR:
                denorm_fn = self.denorm_fn[articulator_2]
                articulator_array_2 = denorm_fn(articulator_array_2)
            articulator_array_2 = articulator_array_2.transpose(2, 3)

            dist = torch.cdist(articulator_array_1, articulator_array_2)
            critical_dists.append(dist)

        critical_loss = torch.stack(critical_dists, dim=0)  # (Ntvs, B, T, D, D)
        critical_loss = critical_loss.permute(1, 0, 2, 3, 4)
        critical_loss = critical_loss.reshape(bs, len(self.TVs), seq_len, num_samples * num_samples)
        critical_loss, _ = critical_loss.min(dim=-1)
        critical_loss = critical_loss[critical_mask == 1].mean()

        return critical_loss


class AutoencoderLoss2(nn.Module):
    """
    AutoencoderLoss adapted for the case of multiple articulators.
    """
    def __init__(
        self,
        indices_dict,
        TVs,
        in_features,
        hidden_features,
        encoder_state_dict_filepath,
        decoder_state_dict_filepath,
        device,
        denormalize_fn=None,
        beta1=1.0,
        beta2=1.0,
        beta3=1.0,
        **kwargs,
    ):
        super().__init__()

        beta1 = beta1
        beta2 = beta2
        beta3 = beta3
        self.beta1, self.beta2, self.beta3 = self.normalize_betas([beta1, beta2, beta3])

        encoder = MultiEncoder(
            indices_dict,
            in_features,
            hidden_features,
        )
        encoder_state_dict = torch.load(
            encoder_state_dict_filepath,
            map_location=device
        )
        encoder.load_state_dict(encoder_state_dict)
        self.encode = InputTransform(
            transform=encoder,
            device=device,
            activation=torch.tanh
        )

        decoder = MultiDecoder(
            indices_dict,
            in_features,
            hidden_features,
        )
        decoder_state_dict = torch.load(
            decoder_state_dict_filepath,
            map_location=device
        )
        decoder.load_state_dict(decoder_state_dict)
        self.decode = InputTransform(
            transform=decoder,
            device=device,
        )

        self.latent = nn.MSELoss(reduction="none")
        self.reconstruction = nn.MSELoss(reduction="none")
        articulators = sorted(indices_dict.keys())
        self.critical = CriticalLoss(TVs, articulators, denormalize_fn)

    @staticmethod
    def normalize_betas(betas):
        # betas = torch.softmax(torch.tensor(betas), dim=0)
        return betas

    def forward(
        self,
        output_pcs,
        target_shapes,
        reference_arrays,
        lengths,
        critical_mask,
    ):
        """
        Args:
            output_pcs (torch.tensor): tensor of shape (B, Nart, Npc)
            target_shapes (torch.tensor): tensor of shape (B, T, Nart, 2, D)
            critical_mask (torch.tensor): tensor of shape (B, Ntv, T)
        """
        padding_mask = make_padding_mask(lengths)

        bs, seq_len, num_articulators, _, num_samples = target_shapes.shape
        encoder_inputs = target_shapes.reshape(bs * seq_len, num_articulators, 2 * num_samples)
        target_pcs = self.encode(encoder_inputs)
        _, num_pcs = target_pcs.shape
        target_pcs = target_pcs.reshape(bs, seq_len, num_pcs)

        output_shapes = self.decode(output_pcs)
        output_shapes = output_shapes.reshape(bs, seq_len, num_articulators, 2, num_samples)

        # outputs_pcs : (bs, seq_len, num_components)
        # target_pcs : (bs, seq_len, num_components)
        # output_shapes : (bs, seq_len, Nart, 2, D)
        # target_shaoes : (bs, seq_len, Nart, 2, D)

        # Mean squared error loss in the level of the principal components
        latent_loss = self.latent(output_pcs, target_pcs)
        latent_loss = latent_loss.view(bs * seq_len, num_pcs)
        latent_loss = latent_loss[padding_mask.view(bs * seq_len)]
        latent_loss = latent_loss.mean()

        # Euclidean distance loss in the level of the shapes
        reconstruction_loss = self.reconstruction(output_shapes, target_shapes)
        reconstruction_loss = reconstruction_loss.view(bs * seq_len, num_articulators, 2, num_samples)
        reconstruction_loss = reconstruction_loss[padding_mask.view(bs * seq_len)].mean()

        # Critical loss
        critical_loss = self.critical(output_shapes, target_shapes, reference_arrays, critical_mask)

        return (
            self.beta1 * latent_loss +
            self.beta2 * reconstruction_loss +
            self.beta3 * critical_loss
        )


class RegularizedLatentsMSELoss2(nn.Module):
    def __init__(
        self,
        alpha,
        indices_dict
    ):
        super().__init__()

        self.alpha = alpha
        self.mse = nn.MSELoss(reduction="none")
        self.latents_cov_mse = nn.MSELoss(reduction="mean")
        self.indices_dict = indices_dict

    def forward(
        self,
        outputs,
        latents,
        target,
        sample_weights=None
    ):
        mse = self.mse(outputs, target)
        if sample_weights is not None:
            mse = mse.permute(2, 1, 0)
            mse = (sample_weights * mse).permute(2, 1, 0)
        mse = mse.mean()

        latents_mean_loss = latents.mean(dim=0).square().sum()

        _, latent_size = latents.shape
        latents_cov_loss = self.latents_cov_mse(
            torch.cov(latents.T),
            0.3 * torch.eye(latent_size).to(latents.device)
        )

        return mse + latents_mean_loss + latents_cov_loss
