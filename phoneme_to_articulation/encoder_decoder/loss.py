import torch.nn as nn

from phoneme_to_articulation.metrics import EuclideanDistance


class ArtSpeechLoss(nn.Module):
    def __init__(
            self,
            recognizer=None,
            reduction="none"
        ):
        super().__init__()

        self.euclidean = EuclideanDistance(reduction=reduction)
        self.recognition = nn.MSELoss(reduction=reduction)
        self.recognizer = recognizer

    def forward(self, outputs, targets, voicing):
        euclidean_loss = self.euclidean(outputs, targets)

        bs, seq_len, n_art, chann, n_samples = targets.shape
        if self.recognizer:
            _, target_features = self.recognizer(
                targets.view(bs, chann, n_art * n_samples, seq_len),
                voicing,
                return_features=True
            )
            _, output_features = self.recognizer(
                outputs.view(bs, chann, n_art * n_samples, seq_len),
                voicing,
                return_features=True
            )
            recognition_loss = self.recognition(output_features, target_features)
        else:
            recognition_loss = None

        return euclidean_loss, recognition_loss
