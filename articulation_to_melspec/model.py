import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from glow_tts.models import FlowGenerator
from torchaudio.models import Tacotron2

from articulation_to_melspec import (
    NVIDIA_TACOTRON2_WEIGHTS_FILEPATH,
    GLOW_TTS_WEIGHTS_FILEPATH,
    GLOW_ATS_EMBEDDING_WEIGHTS_FILEPATH
)


class ArticulatorsEmbedding(nn.Module):
    def __init__(self, n_curves, n_samples, embed_size=512, w_init_gain="linear"):
        super(ArticulatorsEmbedding, self).__init__()

        # Performs a 3D-convolution along the articulators, combining the x and y coordinates into
        # a single channel.
        kernel_size = torch.tensor([1, 3, 1])
        padding = torch.div(kernel_size - 1, 2, rounding_mode="trunc")

        self.conv1 = nn.Conv3d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=1,
            bias=True
        )

        torch.nn.init.xavier_uniform_(
            self.conv1.weight,
            gain=nn.init.calculate_gain(w_init_gain)
        )

        # Performs a 3D-convolution along the articulators, combining the articulator curves
        # into a single channel.
        kernel_size = torch.tensor([1, n_curves])
        padding = torch.div(kernel_size - 1, 2, rounding_mode="trunc")

        self.conv2 = nn.Conv2d(
            in_channels=n_curves,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=1,
            bias=True
        )

        torch.nn.init.xavier_uniform_(
            self.conv2.weight,
            gain=nn.init.calculate_gain(w_init_gain)
        )

        self.conv2 = nn.Conv2d(
            in_channels=n_curves,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=1,
            bias=True
        )

        torch.nn.init.xavier_uniform_(
            self.conv2.weight,
            gain=nn.init.calculate_gain(w_init_gain)
        )

        kernel_size = torch.tensor([5])
        padding = torch.div(kernel_size - 1, 2, rounding_mode="trunc")

        self.conv3 = nn.Conv1d(
            in_channels=n_samples,
            out_channels=embed_size,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=1,
            bias=True
        )

        torch.nn.init.xavier_uniform_(
            self.conv2.weight,
            gain=nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        """
        Args:
        x (torch.tensor): Tensor of shape (bs, seq_len, n_curves, 2, n_samples)
        """
        x = x.permute(0, 3, 1, 4, 2)  # (bs, 2, seq_len, n_samples, n_curves)

        conv1_out = F.relu(self.conv1(x))  # (bs, 1, seq_len, n_samples, n_curves)
        conv1_out = conv1_out.squeeze(dim=1)  # (bs, seq_len, n_samples, n_curves)

        conv2_in = conv1_out.permute(0, 3, 1, 2)  # (bs, n_curves, seq_len, n_samples)
        conv2_out = F.relu(self.conv2(conv2_in))  # (bs, 1, seq_len, n_samples)
        conv2_out = conv2_out.squeeze(dim=1)  # (bs, seq_len, n_samples)

        conv3_in  = conv2_out.permute(0, 2, 1)  # (bs, n_samples, seq_len)
        conv3_out = F.relu(self.conv3(conv3_in))  # (bs, embed_size, seq_len)

        outputs = conv3_out.permute(0, 2, 1)  # (bs, seq_len, embed_size)

        return outputs


class ArticulatoryTacotron2(Tacotron2):
    def __init__(self, n_articulators, *args, n_samples=50, pretrained=False, **kwargs):
        super(ArticulatoryTacotron2, self).__init__(*args, **kwargs)

        if pretrained:
            self._load_pretrained_weigths()

        self.embedding = ArticulatorsEmbedding(n_curves=n_articulators, n_samples=n_samples)

    def _load_pretrained_weigths(self):
        tacotron2_state_dict = torch.load(
            NVIDIA_TACOTRON2_WEIGHTS_FILEPATH, map_location=torch.device("cpu")
        )
        self.load_state_dict(tacotron2_state_dict)

    def infer(self, sequence, lengths):
        """
        Args:
        sequence (torch.tensor): Tensor of shape (bs, seq_len, n_curves, 2, n_samples)
        """
        n_batch, max_length, _, _, _ = sequence.shape
        if lengths is None:
            lengths = torch.tensor([max_length]).expand(n_batch).to(sequence.device, sequence.dtype)

        assert lengths is not None  # For TorchScript compiler

        embedded_inputs = self.embedding(sequence).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, lengths)
        mel_specgram, mel_specgram_lengths, _, alignments = self.decoder.infer(
            encoder_outputs, lengths
        )

        mel_outputs_postnet = self.postnet(mel_specgram)
        mel_outputs_postnet = mel_specgram + mel_outputs_postnet

        alignments = alignments.unfold(1, n_batch, n_batch).transpose(0, 2)

        return mel_outputs_postnet, mel_specgram_lengths, alignments


class GlowATS(FlowGenerator):
    def __init__(self, n_articulators, *args, n_samples=50, pretrained=False, pretrained_encoder=False, **kwargs):
        """
        Glow Articulatory-to-Speech. A modified version of Glow-TTS to handle articulatory data
        instead of text.

        Args:
        n_articulators (int): Number of articulators involved.
        n_samples (int): Number of samples in each articulator curve.
        pretrained (bool): Use Glow-TTS pretraining.
        pretrained_encoder (bool): Use articulatory autoencoder pretraining.
        """
        super(GlowATS, self).__init__(*args, **kwargs)

        if pretrained:
            self._load_pretrained_glow_tts()

        self.encoder.emb = ArticulatorsEmbedding(n_curves=n_articulators, n_samples=n_samples, embed_size=192)

        if pretrained_encoder:
            self._load_pretrained_encoder()

    def _load_pretrained_glow_tts(self):
        glow_tts_state_dict = torch.load(
            GLOW_TTS_WEIGHTS_FILEPATH, map_location=torch.device("cpu")
        )
        self.load_state_dict(glow_tts_state_dict)

    def _load_pretrained_encoder(self):
        embedding_state_dict = torch.load(
            GLOW_ATS_EMBEDDING_WEIGHTS_FILEPATH, map_location=torch.device("cpu")
        )

        self.encoder.emb.load_state_dict(embedding_state_dict)
