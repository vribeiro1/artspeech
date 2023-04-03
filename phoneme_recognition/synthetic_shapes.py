import pdb
import os
import torch

from glob import glob
from tgt.io3 import read_textgrid
from torch.utils.data import Dataset
from vt_tools import (
    ARYTENOID_CARTILAGE,
    EPIGLOTTIS,
    LOWER_INCISOR,
    LOWER_LIP,
    PHARYNX,
    SOFT_PALATE,
    SOFT_PALATE_MIDLINE,
    THYROID_CARTILAGE,
    TONGUE,
    UPPER_INCISOR,
    UPPER_LIP,
    VOCAL_FOLDS
)
from vt_shape_gen.vocal_tract_tube import generate_vocal_tract_tube

ARTICULATORS = [
    ARYTENOID_CARTILAGE,
    EPIGLOTTIS,
    LOWER_INCISOR,
    LOWER_LIP,
    PHARYNX,
    SOFT_PALATE,
    SOFT_PALATE_MIDLINE,
    THYROID_CARTILAGE,
    TONGUE,
    UPPER_INCISOR,
    UPPER_LIP,
    VOCAL_FOLDS
]


class SyntheticPhonemeRecognitionDataset(Dataset):
    def __init__(
        self,
        datadir,
        vocabulary,
        voiced_tokens=None,
        **kwargs
    ):
        self.datadir = datadir
        self.vocabulary = vocabulary
        self.data = self._collect_data(datadir)
        self.voiced_tokens = voiced_tokens or []

    @staticmethod
    def _collect_data(datadir):
        targets_filepaths = glob(os.path.join(datadir, "*", "*", "target_sequence.txt"))
        data = []
        for filepath in targets_filepaths:
            dirname = os.path.dirname(filepath)
            sentence_name = os.path.basename(dirname)
            sequence_name = os.path.basename(os.path.dirname(dirname))
            sentence_dirname = os.path.dirname(filepath)

            with open(filepath) as f:
                phonemes = f.read().strip().split()

            tongue_filepaths = glob(os.path.join(sentence_dirname, "contours", "*_tongue.npy"))
            frame_ids = [os.path.basename(fp).split("_")[0] for fp in tongue_filepaths]

            item = {
                "sequence_name": sequence_name,
                "sentence_name": sentence_name,
                "phonemes": phonemes,
                "frame_ids": sorted(frame_ids),
            }
            data.append(item)

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        item = self.data[item]

        sequence_name = item["sequence_name"]
        sentence_name = item["sentence_name"]
        frame_ids = item["frame_ids"]
        phonemes = item["phonemes"]
        targets = torch.tensor(
            [self.vocabulary.get(phon, 0) for phon in phonemes],
            dtype=torch.long
        )

        inputs = torch.zeros(size=(0, 2, 2, 100))
        for frame_id in frame_ids:
            articulators_dict = {}
            contours_dir = os.path.join(self.datadir, sequence_name, sentence_name, "contours")
            for articulator in ARTICULATORS:
                if articulator == SOFT_PALATE:
                    articulators_dict[SOFT_PALATE] = os.path.join(
                        contours_dir,
                        f"{frame_id}_{SOFT_PALATE_MIDLINE}.npy"
                    )
                else:
                    articulators_dict[articulator] = os.path.join(
                        contours_dir,
                        f"{frame_id}_{articulator}.npy"
                    )

            internal_wall, external_wall = generate_vocal_tract_tube(articulators_dict)
            internal_wall = torch.from_numpy(internal_wall)
            external_wall = torch.from_numpy(external_wall)

            air_column = torch.stack([internal_wall.T, external_wall.T], dim=0)
            inputs = torch.concat([inputs, air_column.unsqueeze(dim=0)])

        inputs = inputs.permute(2, 1, 3, 0)
        channels, n_walls, features, time = inputs.shape
        inputs = inputs.reshape(channels, n_walls * features, time)
        inputs = inputs.type(torch.float)

        # Voicing information
        voicing = torch.tensor(
            [phoneme in self.voiced_tokens for phoneme in phonemes],
            dtype=torch.float
        )

        ctc_target = torch.unique_consecutive(targets)
        ctc_target_length = len(ctc_target)

        sample = {
            "air_column": inputs,
            "air_column_length": time,
            "articulatory_target": targets,
            "articulatory_target_length": len(targets),
            "voicing": voicing,
            "ctc_target": ctc_target,
            "ctc_target_length": ctc_target_length,
        }

        return sample
