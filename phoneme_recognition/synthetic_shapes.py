import logging
import numpy as np
import os
import torch

from glob import glob
from torch.utils.data import Dataset
from tqdm import tqdm
from vt_tools import (
    ARYTENOID_CARTILAGE,
    EPIGLOTTIS,
    LOWER_INCISOR,
    LOWER_LIP,
    PHARYNX,
    SOFT_PALATE_MIDLINE,
    THYROID_CARTILAGE,
    TONGUE,
    UPPER_LIP,
    VOCAL_FOLDS
)

from settings import DATASET_CONFIG

ARTICULATORS = [
    ARYTENOID_CARTILAGE,
    EPIGLOTTIS,
    LOWER_INCISOR,
    LOWER_LIP,
    PHARYNX,
    SOFT_PALATE_MIDLINE,
    THYROID_CARTILAGE,
    TONGUE,
    UPPER_LIP,
    VOCAL_FOLDS
]


class SyntheticPhonemeRecognitionDataset(Dataset):
    def __init__(
        self,
        database_name,
        datadir,
        vocabulary,
        sequences,
        voiced_tokens,
        **kwargs,
    ):
        self.datadir = datadir
        self.database_config = DATASET_CONFIG[database_name]
        self.data = self._collect_data(sequences)
        self.articulators = sorted(ARTICULATORS)
        self.vocabulary = vocabulary
        self.voiced_tokens = voiced_tokens or []

    def __len__(self):
        return len(self.data)

    def get_frame_ids(self, subject, sentence_name):
        sequence_dir = os.path.join(self.datadir, subject, sentence_name)
        frame_filepaths = glob(os.path.join(sequence_dir, "air_column", "*.npy"))
        frame_ids = sorted(map(lambda s: s.split(".")[0], map(os.path.basename, frame_filepaths)))
        return frame_ids

    def _collect_data(self, sequences):
        data = []
        for subject, sentence_name in tqdm(sequences, desc="Collecting data"):
            sequence_dir = os.path.join(self.datadir, subject, sentence_name)
            frame_ids = self.get_frame_ids(subject, sentence_name)
            if len(frame_ids) == 0:
                logging.warning(f"Skipping {subject}/{sentence_name} - Empty frame sequence")
                continue

            phonemes_filepath = os.path.join(sequence_dir, "target_sequence.txt")
            with open(phonemes_filepath) as f:
                phonemes = f.read().strip().split()

            item = {
                "subject": subject,
                "sentence_name": sentence_name,
                "phonemes": phonemes,
                "frame_ids": frame_ids,
            }
            data.append(item)
        return data

    def __getitem__(self, item):
        item = self.data[item]
        subject = item["subject"]
        sentence_name = item["sentence_name"]
        frame_ids = item["frame_ids"]
        phonemes = item["phonemes"]

        sentence_numerized = torch.tensor(
            [self.vocabulary.get(phon, 0) for phon in phonemes],
            dtype=torch.long
        )

        air_columns_filepaths = [
            os.path.join(
                self.datadir,
                subject,
                sentence_name,
                "air_column",
                f"{frame_id}.npy"
            ) for frame_id in frame_ids
        ]
        air_columns = torch.stack([
            torch.from_numpy(np.load(filepath)).type(torch.float)
            for filepath in air_columns_filepaths
        ])  # (T, Nwalls, C, D)

        vocal_tract_shapes = torch.zeros(size=(0, len(self.articulators), 2, 50))
        for frame_id in frame_ids:
            frame_arrays = torch.zeros(size=(0, 2, 50))
            for articulator in self.articulators:
                articulator_filepath = os.path.join(
                    self.datadir,
                    subject,
                    sentence_name,
                    "inference_contours",
                    f"{frame_id}_{articulator}.npy"
                )
                articulator_array = torch.from_numpy(np.load(articulator_filepath))
                articulator_array = articulator_array.type(torch.float).unsqueeze(dim=0)
                frame_arrays = torch.concat([frame_arrays, articulator_array], dim=0)
            frame_arrays = frame_arrays.unsqueeze(dim=0)
            vocal_tract_shapes = torch.concat([vocal_tract_shapes, frame_arrays], dim=0)

        air_columns = air_columns.permute(2, 1, 3, 0)
        channels, n_walls, features, time = air_columns.shape
        air_columns = air_columns.reshape(channels, n_walls * features, time)

        vocal_tract_shapes = vocal_tract_shapes.permute(2, 1, 3, 0)
        channels, n_articulators, features, time = vocal_tract_shapes.shape
        vocal_tract_shapes = vocal_tract_shapes.reshape(channels, n_articulators * features, time)

        # Voicing information
        voicing = torch.tensor(
            [phoneme in self.voiced_tokens for phoneme in phonemes],
            dtype=torch.float
        )

        ctc_target = torch.unique_consecutive(sentence_numerized)
        ctc_target_length = len(ctc_target)

        sample = {
            "air_column": air_columns,
            "air_column_length": len(frame_ids),
            "vocal_tract": vocal_tract_shapes,
            "vocal_tract_length": len(frame_ids),
            "articulatory_target": sentence_numerized,
            "articulatory_target_length": len(sentence_numerized),
            "voicing": voicing,
            "ctc_target": ctc_target,
            "ctc_target_length": ctc_target_length,
        }

        return sample
