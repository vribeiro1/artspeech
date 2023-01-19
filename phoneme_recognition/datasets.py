import pdb

import numpy as np
import os
import torch
import torchaudio
import torchaudio.functional as F

from enum import Enum
from itertools import groupby
from tempfile import mkdtemp
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchaudio import transforms
from typing import Dict, List, Optional
from vt_tools import (
    ARYTENOID_MUSCLE,
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

from database_collector import DATABASE_COLLECTORS
from phoneme_recognition import Feature, Target
from phoneme_to_articulation.tail_clipper import TailClipper
from settings import BASE_DIR
from vocal_tract_loader import VocalTractShapeLoader

ARTICULATORS = [
    # ARYTENOID_MUSCLE,
    # EPIGLOTTIS,
    # LOWER_INCISOR,
    LOWER_LIP,
    # PHARYNX,
    SOFT_PALATE,
    # SOFT_PALATE_MIDLINE,
    # THYROID_CARTILAGE,
    TONGUE,
    # UPPER_INCISOR,
    UPPER_LIP,
    # VOCAL_FOLDS
]


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


class PhonemeRecognitionDataset(Dataset):
    def __init__(
        self,
        datadir: str,
        database: str,
        sequences,
        vocabulary,
        framerate: int,
        sync_shift: int,
        features: List[Feature],
        sample_rate: Optional[int] = 16000,
        n_fft: Optional[int] = 1024,
        win_length: Optional[int] = 1024,
        hop_length: Optional[int] = 256,
        n_mels: Optional[int] = 80,
        f_min: Optional[int] = 0,
        f_max: Optional[int] = None,
        sil_token: Optional[str] = "#",
        blank_token: Optional[str] = None,
        unknown_token: Optional[str] = "<unk>",
        tmp_dir: Optional[str] = None,
    ):
        super().__init__()

        self.datadir = datadir
        self.vocabulary = vocabulary
        self.sil_token = sil_token
        self.blank_token = blank_token
        self.unknown_token = unknown_token
        self.sample_rate = sample_rate
        self.features = features

        self.melspectrogram = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max
        )

        self.tmp_dir = tmp_dir
        if tmp_dir is not None:
            save_audio_dir = os.path.join(self.tmp_dir, "audios")
            if not os.path.exists(save_audio_dir):
                os.makedirs(save_audio_dir)
        else:
            save_audio_dir = None
        collector = DATABASE_COLLECTORS[database](datadir, save_audio_dir)
        self.data = collector.collect_data(sequences)

        self.vocal_tract_loader = VocalTractShapeLoader(
            datadir=self.datadir,
            articulators=ARTICULATORS,
            num_samples=50,
            dataset_config=collector.dataset_config
        )

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_audio_interval(audio, start, end, duration, num_samples):
        time = np.linspace(0., duration, num_samples)
        ge_start, = np.where(time >= start)  # Greater than or equal to the start
        lt_end, = np.where(time < end)  # Lower than the end
        indices = list(set(ge_start) & set(lt_end))
        audio_interval = audio[:, indices]
        return torch.tensor(time[indices], dtype=torch.float), audio_interval

    def load_melspec(self, filepath):
        audio, sample_rate = torchaudio.load(filepath)
        if sample_rate != self.sample_rate:
            audio = F.resample(audio, orig_freq=sample_rate, new_freq=self.sample_rate)
        audio = torch.concat([audio, audio], dim=0)  # mono to stereo

        _, num_samples = audio.shape
        duration = num_samples / self.sample_rate
        melspec = dynamic_range_compression(self.melspectrogram(audio))
        _, _, melspec_length = melspec.shape

        return melspec, melspec_length

    def load_air_column(self, subject, sequence, frame_ids, skip_missing=False):
        sentence_targets = torch.zeros(size=(0, 2, 2, 100))
        for frame_id in frame_ids:
            filepath = os.path.join(self.datadir, subject, sequence, "air_column", f"{frame_id}.npy")
            air_column_array = torch.from_numpy(np.load(filepath)).type(torch.float)
            air_column_array = air_column_array.unsqueeze(dim=0)
            sentence_targets = torch.cat([sentence_targets, air_column_array], dim=0)
        sentence_length = len(frame_ids)
        return sentence_targets, sentence_length

    def __getitem__(self, index):
        item = self.data[index]
        subject = item["subject"]
        sequence = item["sequence"]
        frame_ids = item["frame_ids"]
        phonemes = item["phonemes"]
        phonemes_with_time = item["phonemes_with_time"]
        wav_filepath = item["wav_filepath"]
        audio_duration = item["audio_duration"]

        sample = {}
        if Feature.MELSPEC in self.features:
            melspec, melspec_length = self.load_melspec(wav_filepath)
            sample[Feature.MELSPEC.value] = melspec
            sample[f"{Feature.MELSPEC.value}_length"] = melspec_length

        if Feature.VOCAL_TRACT in self.features:
            vocal_tract, _, sequence_length = self.vocal_tract_loader.load_vocal_tract_shapes(
                subject, sequence, frame_ids
            )
            # vocal_tract has shape (time, num_articulators, channels, features)
            # we want it to have shape (channels, num_articulators, features, time)
            vocal_tract = vocal_tract.permute(2, 1, 3, 0)
            channels, n_art, features, time = vocal_tract.shape
            vocal_tract = vocal_tract.reshape(channels, n_art * features, time)
            sample[Feature.VOCAL_TRACT.value] = vocal_tract
            sample[f"{Feature.VOCAL_TRACT.value}_length"] = sequence_length

        if Feature.AIR_COLUMN in self.features:
            air_column, sequence_length = self.load_air_column(subject, sequence, frame_ids)
            # air_column has shape (time, num_walls, channels, features)
            # we want it to have shape (channels, num_walls, features, time)
            air_column = air_column.permute(2, 1, 3, 0)
            channels, n_walls, features, time = air_column.shape
            air_column = air_column.reshape(channels, n_walls * features, time)
            sample[Feature.AIR_COLUMN.value] = air_column
            sample[f"{Feature.AIR_COLUMN.value}_length"] = sequence_length

        unknown_token = self.vocabulary[self.unknown_token]
        if Feature.MELSPEC.value in sample:
            # Acoustic cross entropy targets
            # If the melspectrogram was computed, we can include the acoustic targets.
            melspec_length = sample[f"{Feature.MELSPEC.value}_length"]
            acoustic_target = torch.zeros(size=(melspec_length,), dtype=torch.long)
            for phoneme, start_time, end_time in phonemes_with_time:
                token = self.vocabulary.get(phoneme, unknown_token)
                start_spec = int(np.around((start_time * melspec_length) / audio_duration))
                end_spec = int(np.around((end_time * melspec_length) / audio_duration))
                acoustic_target[start_spec:end_spec] = token
                sample[Target.ACOUSTIC.value] = acoustic_target
                sample[f"{Target.ACOUSTIC.value}_length"] = melspec_length

        # Articulatory cross entropy targets
        articulatory_tokens = [
            self.vocabulary.get(phoneme, unknown_token)
            for phoneme in phonemes
        ]
        articulatory_target = torch.tensor(articulatory_tokens, dtype=torch.long)
        articulatory_target_length = len(articulatory_target)
        sample[Target.ARTICULATORY.value] = articulatory_target
        sample[f"{Target.ARTICULATORY.value}_length"] = articulatory_target_length

        # CTC targets
        ctc_phonemes = [k for k, _ in groupby(phonemes)]
        ctc_tokens = [
            self.vocabulary.get(phoneme, unknown_token)
            for phoneme in ctc_phonemes
        ]
        ctc_target = torch.tensor(ctc_tokens, dtype=torch.long)
        ctc_target_length = len(ctc_target)
        sample[Target.CTC.value] = ctc_target
        sample[f"{Target.CTC.value}_length"] = ctc_target_length

        return sample


def collate_fn(batch: List[Dict], features_names: List[Feature]):
    # Each feature item has the shape (channels, features, time). After permutation,
    # the shape is (time, channels, features).
    collated_batch = {}
    for feature_name in features_names:
        features = [item[feature_name.value].permute(2, 0, 1) for item in batch]
        features = pad_sequence(features, batch_first=True, padding_value=-1)  # (B, T, C, D)
        features = features.permute(0, 2, 3, 1)  # (B, C, D, T)
        feature_lengths = torch.tensor(
            [item[f"{feature_name.value}_length"] for item in batch],
            dtype=torch.long
        )
        collated_batch[feature_name.value] = features
        collated_batch[f"{feature_name.value}_length"] = feature_lengths

    if Feature.MELSPEC in features_names:
        acoustic_targets = [item[Target.ACOUSTIC.value] for item in batch]
        acoustic_targets = pad_sequence(acoustic_targets, batch_first=True, padding_value=-1)
        acoustic_target_lengths = torch.tensor(
            [item[f"{Target.ACOUSTIC.value}_length"] for item in batch],
            dtype=torch.long
        )

        collated_batch[Target.ACOUSTIC.value] = acoustic_targets
        collated_batch[f"{Target.ACOUSTIC.value}_length"] = acoustic_target_lengths

    articulatory_targets = [item[Target.ARTICULATORY.value] for item in batch]
    articulatory_targets = pad_sequence(articulatory_targets, batch_first=True, padding_value=-1)
    articulatory_target_lengths = torch.tensor(
        [item[f"{Target.ARTICULATORY.value}_length"] for item in batch],
        dtype=torch.long
    )

    collated_batch[Target.ARTICULATORY.value] = articulatory_targets
    collated_batch[f"{Target.ARTICULATORY.value}_length"] = articulatory_target_lengths

    ctc_targets = [item[Target.CTC.value] for item in batch]
    ctc_targets = pad_sequence(ctc_targets, batch_first=True, padding_value=-1)
    ctc_target_lengths = torch.tensor(
        [item[f"{Target.CTC.value}_length"] for item in batch],
        dtype=torch.long
    )

    collated_batch[Target.CTC.value] = ctc_targets
    collated_batch[f"{Target.CTC.value}_length"] = ctc_target_lengths
    return collated_batch
