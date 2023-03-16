import pdb

import torch
import torch.nn as nn

from vt_tools import (
    LOWER_LIP,
    PHARYNX,
    SOFT_PALATE,
    TONGUE,
    UPPER_LIP
)

from phoneme_to_articulation.metrics import EuclideanDistance
from phoneme_to_articulation.principal_components.models import (
    Encoder,
    Decoder,
    MultiEncoder,
    MultiDecoder
)
from phoneme_to_articulation.principal_components.transforms import Encode, Decode, InputTransform


class AutoencoderLoss(nn.Module):
    def __init__(
        self,
        in_features,
        n_components,
        encoder_state_dict_fpath,
        decoder_state_dict_fpath,
        device,
        alpha=1.,
        beta=1.
    ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta

        self.encode = Encode(
            encoder_cls=Encoder,
            state_dict_filepath=encoder_state_dict_fpath,
            device=device,
            in_features=in_features,
            n_components=n_components
        )

        self.decode = Decode(
            decoder_cls=Decoder,
            state_dict_filepath=decoder_state_dict_fpath,
            device=device,
            n_components=n_components,
            out_features=in_features
        )

        self.mse = nn.MSELoss()
        self.euclidean = EuclideanDistance()

    def forward(
        self,
        outputs_pcs,
        targets_shapes,
        references,
        critical_mask
    ):
        """
        Args:
            outputs_pcs (torch.tensor): tensor of shape
            targets_shapes (torch.tensor): tensor of shape (B, T, 1, 2, D)
            references (torch.tensor): tensor of shape
            critical_masks (torch.tensor): tensor of shape
        """
        bs, seq_len, _, _, n_samples = targets_shapes.shape
        encoder_inputs = targets_shapes.squeeze(dim=2).reshape(bs, seq_len, 2 * n_samples)
        target_pcs = torch.tanh(self.encode(encoder_inputs))

        output_shapes = self.decode(outputs_pcs)
        output_shapes = output_shapes.reshape(bs, seq_len, 2, n_samples)

        # Mean squared error loss in the level of the principal components
        mse_loss = self.mse(outputs_pcs, target_pcs)

        # Euclidean distance loss in the level of the shapes
        euclidean_loss = self.euclidean(output_shapes.unsqueeze(dim=2), targets_shapes)

        # Critical loss
        output_shapes = output_shapes.permute(0, 1, 3, 2)
        references = references.permute(2, 0, 1, 4, 3)
        n_refs, _, _, _, _ = references.shape

        critical_loss = torch.stack([
            torch.cdist(output_shapes, reference) for reference in references
        ], dim=0)
        critical_loss = critical_loss.permute(1, 0, 2, 3, 4)
        critical_loss = critical_loss.reshape(bs, n_refs, seq_len, n_samples * n_samples)
        min_critical_loss, _ = critical_loss.min(dim=-1)
        mean_min_critical_loss = min_critical_loss[critical_mask == 1].mean()

        return mse_loss + self.alpha * euclidean_loss + self.beta * mean_min_critical_loss


class AutoencoderLoss2(nn.Module):
    """
    AutoencoderLoss adapted for the case of multiple articulators.
    """
    TV_TO_ARTICULATOR_MAP = {
        "LA": [LOWER_LIP, UPPER_LIP],
        "VEL": [SOFT_PALATE, PHARYNX]
    }
    def __init__(
        self,
        indices_dict,
        TVs,
        in_features,
        hidden_blocks,
        hidden_features,
        encoder_state_dict_filepath,
        decoder_state_dict_filepath,
        device,
        beta1=1.0,
        beta2=1.0,
    ):
        super().__init__()

        beta0 = 1.0
        beta1 = beta1
        beta2 = beta2
        self.beta0, self.beta1, self.beta2 = self.normalize_betas([beta0, beta1, beta2])

        self.articulators = sorted(indices_dict.keys())
        self.articulators_indices = {
            articulator: i
            for i, articulator in enumerate(self.articulators)
        }
        self.TVs = sorted(TVs)

        encoder = MultiEncoder(
            indices_dict,
            in_features,
            hidden_blocks,
            hidden_features,
        )
        encoder_state_dict = torch.load(
            encoder_state_dict_filepath,
            map_location=device
        )
        encoder.load_state_dict(encoder_state_dict)
        self.encode = InputTransform(
            transform=encoder,
            device=device
        )

        decoder = MultiDecoder(
            indices_dict,
            in_features,
            hidden_blocks,
            hidden_features,
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

        self.latent = nn.MSELoss(reduction="none")
        self.reconstruction = EuclideanDistance(reduction="none")

    @staticmethod
    def normalize_betas(betas):
        betas = torch.softmax(torch.tensor(betas), dim=0)
        return betas

    @staticmethod
    def make_padding_mask(lengths):
        """
        Make a padding mask from a tensor lengths.

        Args:
            lengths (torch.tensor): tensor of shape (B,)
        """
        bs = len(lengths)
        max_length = lengths.max()
        mask = torch.ones(size=(bs, max_length))
        mask = torch.cumsum(mask, dim=1)
        mask = mask <= lengths.unsqueeze(dim=1)
        return mask

    def forward(
        self,
        output_pcs,
        target_shapes,
        lengths,
        critical_mask,
    ):
        """
        Args:
            output_pcs (torch.tensor): tensor of shape (B, Nart, Npc)
            target_shapes (torch.tensor): tensor of shape (B, T, Nart, 2, D)
            critical_mask (torch.tensor): tensor of shape (B, Ntv, T)
        """
        padding_mask = self.make_padding_mask(lengths)

        bs, seq_len, num_articulators, _, num_samples = target_shapes.shape
        encoder_inputs = target_shapes.reshape(bs * seq_len, num_articulators, 2 * num_samples)
        target_pcs = torch.tanh(self.encode(encoder_inputs))
        _, num_pcs = target_pcs.shape
        target_pcs = target_pcs.reshape(bs, seq_len, num_pcs)

        output_shapes = self.decode(output_pcs)
        output_shapes = output_shapes.reshape(bs, seq_len, num_articulators, 2, num_samples)

        # Mean squared error loss in the level of the principal components
        latent_loss = self.latent(output_pcs, target_pcs)
        latent_loss = latent_loss.view(bs * seq_len, num_pcs)
        latent_loss = latent_loss[padding_mask.view(bs * seq_len)].mean()

        # Euclidean distance loss in the level of the shapes
        reconstruction_loss = self.reconstruction(output_shapes, target_shapes)
        reconstruction_loss = reconstruction_loss.view(bs * seq_len, num_articulators, num_samples)
        reconstruction_loss = reconstruction_loss[padding_mask.view(bs * seq_len)].mean()

        # Critical loss
        num_TVs = len(self.TVs)
        output_shapes = output_shapes.permute(0, 1, 2, 4, 3)  # (B, T, Nart, D, 2)
        critical_loss = torch.stack([
            torch.cdist(
                output_shapes[..., self.articulators_indices[self.TV_TO_ARTICULATOR_MAP[TV][0]], :, :],
                output_shapes[..., self.articulators_indices[self.TV_TO_ARTICULATOR_MAP[TV][1]], :, :]
            ) for TV in self.TVs
        ], dim=0)  # (Ntvs, B, T, D, D)
        critical_loss = critical_loss.permute(1, 0, 2, 3, 4)
        critical_loss = critical_loss.reshape(bs, num_TVs, seq_len, num_samples * num_samples)
        critical_loss, _ = critical_loss.min(dim=-1)
        critical_loss = critical_loss[critical_mask == 1].mean()

        return (
            self.beta0 * latent_loss +
            self.beta1 * reconstruction_loss +
            self.beta2 * critical_loss
        )


class RegularizedLatentsMSELoss(nn.Module):
    def __init__(
        self,
        alpha
    ):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss(reduction="none")

    def forward(
        self,
        outputs,
        latents,
        target,
        sample_weights=None
    ):
        mse = self.mse(outputs, target)
        if sample_weights is not None:
            mse = (sample_weights * mse.T).T
        mse = mse.mean()

        cov_mtx = torch.cov(latents.T)
        diag = cov_mtx.diag()
        cov_loss = cov_mtx.square().sum() - diag.square().sum()

        return mse + self.alpha * cov_loss


class MultiArtRegularizedLatentsMSELoss(nn.Module):
    def __init__(
        self,
        alpha,
        indices_dict
    ):
        super().__init__()

        self.alpha = alpha
        self.mse = nn.MSELoss(reduction="none")
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

        cov_loss = torch.tensor([
            torch.cov(latents.T[indices]).square().sum() - torch.cov(latents.T[indices]).diag().square().sum()
            for _, indices in self.indices_dict.items() if len(indices) > 1
        ]).sum()

        return mse + self.alpha * cov_loss
