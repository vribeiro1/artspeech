"""
The WaveGlow code inside this was extracted from NVIDIA's Waveglow repository, which is available
at https://github.com/NVIDIA/waveglow.
"""
import torch

from functools import lru_cache

from articul_to_melspec import NVIDIA_WAVEGLOW_WEIGHTS_FILEPATH
from articul_to_melspec.waveglow.denoiser import Denoiser
from articul_to_melspec.waveglow.glow import WaveGlow

MAX_WAV_VALUE = 32768.0


@lru_cache()
def load_waveglow():
    WN_config = dict(
        n_layers=8,
        n_channels=256,
        kernel_size=3
    )

    waveglow = WaveGlow(
        n_mel_channels=80,
        n_flows=12,
        n_group=8,
        n_early_every=4,
        n_early_size=2,
        WN_config=WN_config
    )

    state_dict = torch.load(NVIDIA_WAVEGLOW_WEIGHTS_FILEPATH, map_location=torch.device("cpu"))
    waveglow.load_state_dict(state_dict)

    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.eval()

    return waveglow


def melspec_to_audio(melspectograms, sigma=1.0, denoiser_strength=0.1):
    """
    Args:
    melspectograms (torch.tensor): Tensor with shape (batch_size, resolution, n_mels)
    """
    waveglow = load_waveglow()
    waveglow = waveglow.to(melspectograms.device)
    denoiser = Denoiser(waveglow).to(melspectograms.device)

    with torch.no_grad():
        audios = waveglow.infer(melspectograms, sigma)
        if denoiser_strength > 0.:
            audios = denoiser(audios, denoiser_strength)
        audios = audios * MAX_WAV_VALUE

    return audios.squeeze(dim=1)
