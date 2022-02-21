import torch
import torch.nn as nn

from loss import EuclideanDistanceLoss
from phoneme_to_articulation.principal_components.models import Encoder, Decoder


class AutoencoderLoss(nn.Module):
    def __init__(self, in_features, n_components, encoder_state_dict_fpath, decoder_state_dict_fpath, device):
        super().__init__()

        self.encoder = Encoder(in_features=in_features, n_components=n_components)
        encoder_state_dict = torch.load(encoder_state_dict_fpath, map_location=device)
        self.encoder.load_state_dict(encoder_state_dict)
        self.encoder.to(device)
        self.encoder.eval()

        self.decoder = Decoder(n_components=n_components, out_features=in_features)
        decoder_state_dict = torch.load(decoder_state_dict_fpath, map_location=device)
        self.decoder.load_state_dict(decoder_state_dict)
        self.decoder.to(device)
        self.decoder.eval()

        # The encoder and the decoder should not be trained during this phase. They should only be
        # used for evaluation without learning anything new. Therefore, we have to turn off the
        # gradients of these two networks.

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        for parameter in self.decoder.parameters():
            parameter.requires_grad = False

        self.mse = nn.MSELoss()
        self.euclidean = EuclideanDistanceLoss()

    def forward(self, outputs, targets, references, critical_mask):
        bs, seq_len, _, _, n_samples = targets.shape
        encoder_inputs = targets.squeeze(dim=2).reshape(bs, seq_len, 2 * n_samples)
        target_pcs = self.encoder(encoder_inputs)

        output_shapes = self.decoder(outputs)
        output_shapes = output_shapes.reshape(bs, seq_len, 2, n_samples).unsqueeze(dim=2)

        # Critical loss
        output_shapes = output_shapes.permute(0, 1, 3, 2)
        references = references.reshape(bs, seq_len, 2, n_samples)
        references = references.permute(0, 1, 3, 2)

        critical_loss = torch.cdist(output_shapes, references)
        critical_loss = critical_loss.reshape(bs, seq_len, n_samples * n_samples)

        min_critical_loss, _ = critical_loss.min(dim=-1)
        mean_min_critical_loss = min_critical_loss[critical_mask == 1].mean()

        # Mean squared error loss in the level of the principal components
        mse_loss = self.mse(outputs, target_pcs)
        # Euclidean distance loss in the level of the shapes
        euclidean_loss = self.euclidean(targets, output_shapes)

        return mse_loss + euclidean_loss + mean_min_critical_loss


class RegularizedLatentsMSELoss(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, anchor_outputs, anchor_latents, anchor_target, sample_weights=None):
        mse = self.mse(anchor_outputs, anchor_target)
        if sample_weights is not None:
            mse = (sample_weights * mse.T).T
        mse = mse.mean()

        reg_latents = torch.norm(anchor_latents, p=2, dim=1).mean()
        cov_features = torch.cov(anchor_latents.T).square().sum()

        return mse + self.alpha * reg_latents + self.beta * cov_features
