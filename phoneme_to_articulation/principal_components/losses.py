import pdb

import torch
import torch.nn as nn

from phoneme_to_articulation.metrics import EuclideanDistance
from phoneme_to_articulation.principal_components.models import Encoder, Decoder
from phoneme_to_articulation.principal_components.transforms import Encode, Decode


class AutoencoderLoss(nn.Module):
    def __init__(
        self,
        in_features,
        n_components,
        encoder_state_dict_fpath,
        decoder_state_dict_fpath, device,
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
            for _, indices in self.indices_dict.items()
        ]).sum()

        return mse + self.alpha * cov_loss
