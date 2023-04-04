import os

TRAIN = "train"
VALID = "validation"
TEST = "test"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class ArtSpeechConfig:
    RES = 136
    PIXEL_SPACING = 1.6176470518112
    FRAMERATE = 50
    SYNC_SHIFT = 2


class GottingenConfig:
    RES = 136
    PIXEL_SPACING = 1.4117647409439
    FRAMERATE = 55
    SYNC_SHIFT = 0


DatasetConfig = GottingenConfig

DATASET_CONFIG = {
    "artspeech2": ArtSpeechConfig,
    "gottingen": GottingenConfig
}
