import os
import logging

ARTICUL_TO_MELSPEC_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(ARTICUL_TO_MELSPEC_DIR, "assets")
NVIDIA_TACOTRON2_WEIGHTS_FILEPATH = os.path.join(ASSETS_DIR, "nvidia_tacotron2.pt")
NVIDIA_WAVEGLOW_WEIGHTS_FILEPATH = os.path.join(ASSETS_DIR, "nvidia_waveglow.pt")

if not os.path.isfile(NVIDIA_TACOTRON2_WEIGHTS_FILEPATH):
    logging.warning("NVidia Tacotron2 weights file not available.")

if not os.path.isfile(NVIDIA_WAVEGLOW_WEIGHTS_FILEPATH):
    logging.warning("WaveGlow weights file not available.")
