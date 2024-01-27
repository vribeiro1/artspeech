import os

SIL = "#"
UNKNOWN = "<unk>"
BLANK = "<blank>"

TRAIN = "train"
VALID = "validation"
TEST = "test"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class ArtSpeechConfig:
    RES = 136
    PIXEL_SPACING = 1.6176470518112
    FRAMERATE = 50
    SYNC_SHIFT = 2


class ArtSpeech2Config:
    RES = 136
    PIXEL_SPACING = 1.6176470518112
    FRAMERATE = 50
    SYNC_SHIFT = -20


class GottingenConfig:
    RES = 136
    PIXEL_SPACING = 1.4117647409439
    FRAMERATE = 55
    SYNC_SHIFT = 0


class TextgridOnlyConfig:
    RES = 136
    PIXEL_SPACING = 1.6176470518112
    FRAMERATE = 50
    SYNC_SHIFT = 0


DATASET_CONFIG = {
    "artspeech": ArtSpeechConfig,
    "artspeech2": ArtSpeech2Config,
    "gottingen": GottingenConfig,
    "textgrid_only": TextgridOnlyConfig,
}
