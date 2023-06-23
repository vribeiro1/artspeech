import os
import torch.nn as nn

from phoneme_to_articulation.principal_components.models.autoencoder import (
    Encoder,
    Decoder,
    MultiEncoder,
    MultiDecoder,
    MultiArticulatorAutoencoder
)
from phoneme_to_articulation.principal_components.models.rnn import (
    PrincipalComponentsPredictor,
    PrincipalComponentsArtSpeech
)


class PrincipalComponentsArtSpeechWrapper(nn.Module):
    def __init__(self, rnn, decoder, denorm):
        super().__init__()

        self.rnn = rnn
        self.decoder = decoder
        self.denorm = denorm

    def forward(self, x, lengths):
        """
        Args:
            x (torch.tensor): (bs, seq_len)
            lengths (List)
        """
        components = self.rnn(x, lengths)  # (bs, seq_len, n_components)
        outputs = self.decoder(components)  # (bs, seq_len, n_articulators, 2 * dim)
        bs, seq_len, n_articulators, features = outputs.shape
        outputs = outputs.reshape(bs, seq_len, n_articulators, 2, features // 2)

        for i, articulator in enumerate(self.decoder.sorted_articulators):
            denorm_fn = self.denorm[articulator]
            outputs[:, :, i, :, :] = denorm_fn(outputs[:, :, i, :, :])

        return outputs
