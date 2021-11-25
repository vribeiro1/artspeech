from gevent import monkey
monkey.patch_all()

import funcy
import gevent
import logging
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
import re

from collections import namedtuple
from gevent.pool import Pool
from glob import glob
from tgt.io import read_textgrid
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm

from video import Video

MAX_GREENLETS = 5

DataItem = namedtuple("DataItem", [
    "sentence_name",
    "n_frames",
    "audio",
    "articulators_filepaths"
])


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    Tacotron2's dynamic range compression.

    Args:
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    Tacontron2's dynamic range decompression.

    Args:
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def pad_sequence_collate_fn(batch):
    batch_size = len(batch)
    sentence_names = [item[0] for item in batch]

    sentence_articulators = [item[1] for item in batch]
    len_sentences = torch.tensor(funcy.lmap(len, sentence_articulators), dtype=torch.int)
    len_sentences_sorted, sentences_sorted_indices = len_sentences.sort(descending=True)
    padded_sentence_articulators = pad_sequence(sentence_articulators, batch_first=True)
    padded_sentence_articulators_sorted = padded_sentence_articulators[sentences_sorted_indices]

    targets = [item[2].T for item in batch]
    len_targets = torch.tensor(funcy.lmap(len, targets), dtype=torch.int)
    len_targets_sorted = len_targets[sentences_sorted_indices]
    padded_targets = pad_sequence(targets, batch_first=True)
    padded_targets_sorted = padded_targets[sentences_sorted_indices]
    padded_targets_sorted = padded_targets_sorted.permute(0, 2, 1)

    padded_gate = torch.zeros(size=(batch_size, len_targets.max()), dtype=torch.float)
    for i, len_target in enumerate(len_targets_sorted):
        padded_gate[i][len_target - 1:] = 1.

    return sentence_names, padded_sentence_articulators_sorted, len_sentences_sorted, padded_targets_sorted, padded_gate, len_targets_sorted


class ArticulToMelSpecDataset(Dataset):
    RES = 136.
    def __init__(
        self, datadir, sequences, articulators, fps_art=55, fps_spec=86, sync_shift=0,
        sample_rate=16e3, n_fft=1024, win_length=1024, hop_length=256, n_mels=80,
        f_min=0.0, f_max=None, n_greenlets=MAX_GREENLETS
    ):
        super(ArticulToMelSpecDataset, self).__init__()

        self.datadir = datadir
        self.data = self._collect_data(datadir, sequences, sync_shift, fps_art)

        self.interp_factor = fps_spec / fps_art
        self.articulators = articulators

        self.mel_spectogram = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            normalized=True
        )

        if n_greenlets > MAX_GREENLETS:
            logging.warning(
                f"Number of loading processes set to a value greater than the maximum allowed."
                "Clipping to {MAX_LOAD_PROC}."
            )
        n_greenlets = n_greenlets or MAX_GREENLETS
        self.n_greenlets = min(n_greenlets, MAX_GREENLETS)

    def load_frame_articulators(self, filepath):
        """
        Loads the target array with the proper orientation (right to left).

        Args:
        filepath (str): Path to the npy array.
        """
        frame_articulators = torch.zeros(size=(0, 2, 50))
        target_dict = np.load(filepath, allow_pickle=True).item()

        for articulator in self.articulators:
            articulator_array = target_dict[articulator] / ArticulToMelSpecDataset.RES

            # All the countors should be oriented from right to left. If it is the opposite,
            # we flip the array.
            if articulator_array[0][0] < articulator_array[-1][0]:
                articulator_array = np.flip(articulator_array, axis=0)
            articulator_array = torch.tensor(articulator_array.copy(), dtype=torch.float).T

            frame_articulators = torch.cat([
                frame_articulators,
                articulator_array.unsqueeze(dim=0)
            ])

        return frame_articulators

    @staticmethod
    def _collect_data(datadir, sequences, sync_shift, framerate):
        data = []
        for subject, sequence in tqdm(sequences, desc="Collecting data"):
            seq_dir = os.path.join(datadir, subject, sequence)

            textgrid_filepath = os.path.join(seq_dir, f"vol_{subject}_{sequence}.textgrid")
            wav_filepath = os.path.join(seq_dir, f"vol_{subject}_{sequence}.wav")

            articul_filepaths = sorted(filter(
                lambda fp: re.match(r"^[0-9]{4}.npy", os.path.basename(fp)) is not None,
                glob(os.path.join(seq_dir, "inference_contours", f"*.npy"))
            ))
            video = Video(articul_filepaths[sync_shift:], wav_filepath, framerate=framerate)

            textgrid = read_textgrid(textgrid_filepath)
            sentence_tier = textgrid.get_tier_by_name("SentenceTier")
            for i, sentence in enumerate(sentence_tier):
                audio_interval, articulators_filepaths = video.get_interval(
                    sentence.start_time,
                    sentence.end_time
                )

                wav_filename, _  = os.path.basename(wav_filepath).split(".")
                sentence_name = f"{wav_filename}_{i}"

                data_item = DataItem(
                    sentence_name=sentence_name,
                    n_frames=len(articul_filepaths),
                    audio=audio_interval,
                    articulators_filepaths=articulators_filepaths
                )

                data.append(data_item)

        return data

    def load_with_greenlets(self, articulators_filepaths):
        pool = Pool(self.n_greenlets)
        tasks = [pool.spawn(self.load_frame_articulators, fp) for fp in articulators_filepaths]
        gevent.joinall(tasks)

        return [task.value for task in tasks]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_item = self.data[index]

        if self.n_greenlets == 0:
            sentence_articulators = funcy.lmap(self.load_frame_articulators, data_item.articulators_filepaths)
        else:
            sentence_articulators = self.load_with_greenlets(data_item.articulators_filepaths)
        sentence_articulators = torch.stack(sentence_articulators)

        melspec = self.mel_spectogram(data_item.audio)
        melspec = dynamic_range_compression(melspec)

        interp_size = int(np.ceil(self.interp_factor * data_item.n_frames))
        target = F.interpolate(
            melspec.unsqueeze(dim=0),
            size=(interp_size,),
            mode="linear",
            align_corners=True
        ).squeeze(dim=0)

        return data_item.sentence_name, sentence_articulators, target
