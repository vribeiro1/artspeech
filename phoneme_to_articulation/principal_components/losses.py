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
from phoneme_to_articulation.principal_components.models import (
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
        if len(self.TVs) == 0:
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
        beta4=0.0,
        recognizer=None,
        **kwargs,
    ):
        super().__init__()

        (
            self.beta1,
            self.beta2,
            self.beta3,
            self.beta4
        ) = self.normalize_betas([beta1, beta2, beta3, beta4])

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
        self.recognition = nn.MSELoss(reduction="none")
        self.recognizer = recognizer

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
        voicing=None,
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

        # Recognition loss
        bs, seq_len, n_art, chann, n_samples = target_shapes.shape
        if self.recognizer is not None:
            _, target_features = self.recognizer(
                target_shapes.view(bs, chann, n_art * n_samples, seq_len),
                voicing,
                return_features=True
            )
            _, output_features = self.recognizer(
                output_shapes.view(bs, chann, n_art * n_samples, seq_len),
                voicing,
                return_features=True
            )
            recognition_loss = self.recognition(output_features, target_features)
            _, _, features = recognition_loss.shape
            recognition_loss = recognition_loss.view(bs * seq_len, features)
            recognition_loss = recognition_loss[padding_mask.view(bs * seq_len)].mean()
        else:
            recognition_loss = torch.tensor(0, device=target_shapes.device, dtype=torch.float)

        return (
            self.beta1 * latent_loss +
            self.beta2 * reconstruction_loss +
            self.beta3 * critical_loss +
            self.beta4 * recognition_loss
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
        sample_weights=None,
    ):
        mse = self.mse(outputs, target)
        if sample_weights is not None:
            mse = mse.permute(2, 1, 0)
            mse = (sample_weights * mse).permute(2, 1, 0)
        mse = mse.mean()

        cov_loss = torch.tensor([
            torch.cov(latents.T[indices]).square().sum() - torch.cov(latents.T[indices]).diag().square().sum()
            for _, indices in self.indices_dict.items() if len(indices) > 1
        ]).sum()

        return mse + self.alpha * cov_loss
