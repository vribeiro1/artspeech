import pdb

import funcy
import numpy as np
import os
import torch
import torch.nn.functional as F

from collections import namedtuple
from glob import glob
from tgt.io import read_textgrid
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm

from articul_to_melspec.waveglow.tacotron2.audio_processing import dynamic_range_compression
from video import Video

DataItem = namedtuple("DataItem", [
    "sentence_name",
    "n_frames",
    "audio",
    "articul_filepaths"
])


def pad_sequence_collate_fn(batch):
    sentence_names = [item[0] for item in batch]

    sentence_articulators = [item[1] for item in batch]
    len_sentences = torch.tensor(funcy.lmap(len, sentence_articulators), dtype=torch.int)
    padded_sentence_articulators = pad_sequence(sentence_articulators, batch_first=True)

    targets = [item[2].T for item in batch]
    len_targets = torch.tensor(funcy.lmap(len, targets), dtype=torch.int)
    padded_targets = pad_sequence(targets, batch_first=True)
    padded_targets = padded_targets.permute(0, 2, 1)

    return sentence_names, padded_sentence_articulators, len_sentences, padded_targets, len_targets


class ArticulToMelSpecDataset(Dataset):
    RES = 136.
    def __init__(
        self, datadir, sequences, articulators, fps_art=55, fps_spec=86, sync_shift=0,
        sample_rate=16e3, n_fft=2048, hop_length=None, n_mels=80
    ):
        super(ArticulToMelSpecDataset, self).__init__()

        self.datadir = datadir
        self.data = self._collect_data(datadir, sequences, articulators, sync_shift, fps_art)

        self.interp_factor = fps_spec / fps_art
        self.n_articulators = len(articulators)

        self.mel_spectogram = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            normalized=True
        )


    @staticmethod
    def load_target_array(filepath):
        """
        Loads the target array with the proper orientation (right to left).

        Args:
        filepath (str): Path to the npy array.
        """
        target_array = np.load(filepath) / ArticulToMelSpecDataset.RES

        # All the countors should be oriented from right to left. If it is the opposite,
        # we flip the array.
        if target_array[0][0] < target_array[-1][0]:
            target_array = np.flip(target_array, axis=0)

        return target_array.copy()


    @staticmethod
    def _collect_data(datadir, sequences, articulators, sync_shift, framerate):
        data = []
        for subject, sequence in tqdm(sequences, desc="Collecting data"):
            seq_dir = os.path.join(datadir, subject, sequence)

            textgrid_filepath = os.path.join(seq_dir, f"vol_{subject}_{sequence}.textgrid")
            wav_filepath = os.path.join(seq_dir, f"vol_{subject}_{sequence}.wav")
            frames_filepaths = sorted(glob(os.path.join(seq_dir, "dicoms", "*.dcm")))[sync_shift:]

            video = Video(frames_filepaths, wav_filepath, framerate=framerate)

            textgrid = read_textgrid(textgrid_filepath)
            sentence_tier = textgrid.get_tier_by_name("SentenceTier")
            for i, sentence in enumerate(sentence_tier):
                audio_interval, frames_interval = video.get_interval(
                    sentence.start_time,
                    sentence.end_time
                )

                articul_filepaths = []
                for filepath in frames_interval:
                    filename, _ = os.path.basename(filepath).split(".")

                    articul_filepaths.append({
                        articulator: os.path.join(
                            seq_dir,
                            "inference_contours",
                            f"{filename}_{articulator}.npy"
                        ) for articulator in sorted(articulators)
                    })

                wav_filename, _  = os.path.basename(wav_filepath).split(".")
                sentence_name = f"{wav_filename}_{i}"

                data_item = DataItem(
                    sentence_name=sentence_name,
                    n_frames=len(frames_interval),
                    audio=audio_interval,
                    articul_filepaths=articul_filepaths
                )

                data.append(data_item)

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_item = self.data[index]

        sentence_articulators = torch.zeros(size=(0, self.n_articulators, 2, 50))
        for frame_articulators_fps in data_item.articul_filepaths:
            frame_articulators = torch.zeros(size=(0, 2, 50))

            for _, filepath in sorted(frame_articulators_fps.items(), key=lambda t: t[0]):
                articul_arr = torch.tensor(self.load_target_array(filepath).T, dtype=torch.float)
                frame_articulators = torch.cat([
                    frame_articulators,
                    articul_arr.unsqueeze(dim=0)
                ])

            sentence_articulators = torch.cat([
                sentence_articulators,
                frame_articulators.unsqueeze(dim=0)
            ])

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
