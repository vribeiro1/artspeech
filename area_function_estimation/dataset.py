import numpy as np
import os
import re
import torch

from glob import glob
from tgt.io import read_textgrid
from torch.utils.data import Dataset
from tqdm import tqdm

from helpers import sequences_from_dict
from phoneme_to_articulation.dataset import TailClipper
from settings import DatasetConfig
from video import Video


class AreaFunctionDataset2(Dataset):
    def __init__(self, datadir, sequences, sync_shift=0, fps_art=55, clip_tails=True):
        super(AreaFunctionDataset2, self).__init__()

        sequences = sequences_from_dict(datadir, sequences)
        self.frames = self._collect_data(datadir, sequences, sync_shift, fps_art)
        self.clip_tails = clip_tails

    @staticmethod
    def _collect_data(datadir, sequences, sync_shift, framerate):
        frames = []
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

            for sentence in sentence_tier.intervals:
                _, articulators_filepaths = video.get_interval(sentence.start_time, sentence.end_time)
                frames.extend(articulators_filepaths)

        return frames

    @staticmethod
    def flip_and_reshape(array):
        n, _ = array.shape
        if n == 2:
            array = array.T

        # All the countors should be oriented from right to left. If it is the opposite,
        # we flip the array.
        if array[0][0] < array[-1][0]:
            array = np.flip(array, axis=0)

        return array

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame_filepath = self.frames[index]

        filename, _ = os.path.basename(frame_filepath).split(".")
        dir_area_function = os.path.join(os.path.dirname(os.path.dirname(frame_filepath)), "area_function")
        area_function_filepath = os.path.join(dir_area_function, f"{filename}.npy")
        x, fx = np.load(area_function_filepath)
        x = x / x.max()

        frame_articulators = torch.zeros(size=(0, 2, 50), dtype=torch.float)
        target_dict = np.load(frame_filepath, allow_pickle=True).item()

        target_dict = {
            articulator: self.flip_and_reshape(articulator_array) / DatasetConfig.RES
            for articulator, articulator_array in target_dict.items()
        }

        for art, articulator_array in target_dict.items():
            if self.clip_tails:
                tail_clip_method = getattr(
                    TailClipper, f"clip_{art.replace('-', '_')}_tails", None
                )

                if tail_clip_method:
                    articulator_array = tail_clip_method(
                        articulator_array,
                        lower_incisor=target_dict["lower-incisor"],
                        upper_incisor=target_dict["upper-incisor"],
                        epiglottis=target_dict["epiglottis"]
                    )

            articulator_array = torch.from_numpy(articulator_array).T
            articulator_array = articulator_array.type(torch.float)

            frame_articulators = torch.cat([
                frame_articulators,
                articulator_array.unsqueeze(dim=0)
            ])

        return frame_articulators, x, fx
