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

    def forward(self, outputs, targets):
        bs, seq_len, _, _, n_samples = targets.shape
        encoder_inputs = targets.squeeze(dim=2).reshape(bs, seq_len, 2 * n_samples)
        target_pcs = self.encoder(encoder_inputs)

        output_shapes = self.decoder(outputs)
        output_shapes = output_shapes.reshape(bs, seq_len, 2, n_samples).unsqueeze(dim=2)

        # Mean squared error loss in the level of the principal components
        mse_loss = self.mse(outputs, target_pcs)
        # Euclidean distance loss in the level of the shapes
        euclidean_loss = self.euclidean(targets, output_shapes)

        return mse_loss + euclidean_loss
