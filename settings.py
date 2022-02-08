import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class ArtSpeechConfig:
    RES = 136
    PIXEL_SPACING = 1.6176470518112


class GottingenConfig:
    RES = 136
    PIXEL_SPACING = 1.4117647409439


DatasetConfig = GottingenConfig
