import pdb

import funcy
import json
import numpy as np
import os
import torch

from scipy.spatial.distance import euclidean
from torch.utils.data import Dataset


class ArtSpeechDataset(Dataset):
    def __init__(
        self, datadir, filepath, vocabulary, articulators, n_samples=50, size=136,
        register=False, lazy_load=False
    ):
        """
        ArtSpeech Dataset class

        Keyword arguments:

        datadir (str): Dirname of the dataset directory.
        filepath (str): Path to the data file.
        vocabulary (Dict[str, int]): Dictionary mapping tokens to their numerical values.
        articulators (List[str]): List of articulators.
        n_samples (int): Number of samples in the contour.
        size (int): Interval domain of the contours' x- and y- coordinates.
        register (bool): If should register the contours in relation to the complete sentences.
        lazy_load (bool): If should load the data on demand.
        """

        self.vocabulary = vocabulary
        self.datadir = datadir
        self.articulators = articulators
        self.n_articulators = len(articulators)
        self.n_samples = n_samples
        self.size = size
        self.register_targets = register

        with open(filepath) as f:
            data = funcy.lfilter(self._exclude_missing_data, json.load(f))

        self.data = data if lazy_load else self._collect_data(data)
        self.load_fn = self._collect_sentence if lazy_load else lambda x: x

    @staticmethod
    def _get_frames_interval(start, end, timed_frame_keys):
        on_interval = filter(lambda d: start <= d[0] < end, timed_frame_keys)
        frame_keys = [d[1] for d in on_interval]
        return frame_keys

    def _exclude_missing_data(self, item):
        """
        Returns False, if any articulator in any frame in the sentence is None,
        returns True otherwise.
        """
        contours_filepaths = item["contours_filepaths"]

        frame_is_missing = [
            any([
                art_fp is None
                for art, art_fp in art_filepaths.items()
                if art in self.articulators
            ]) for frame_number, art_filepaths in contours_filepaths.items()
        ]

        return not any(frame_is_missing)

    def _collect_data(self, data):
        dataset = funcy.lmap(self._collect_sentence, data)
        return dataset

    def _collect_sentence(self, item):
        phonemes = item["phonemes"]

        first_phoneme = phonemes[0]
        sentence_start = first_phoneme["start_time"]

        last_phoneme = phonemes[-1]
        sentence_end = last_phoneme["end_time"]

        contours_filepaths = item["contours_filepaths"]
        frame_keys = list(contours_filepaths.keys())

        time = np.linspace(sentence_start, sentence_end, len(frame_keys))
        timed_frame_keys = list(zip(time, frame_keys))

        sentence_tokens = []
        sentence_targets = torch.zeros(size=(0, self.n_articulators, 2, self.n_samples))
        for phoneme in phonemes:
            phone_start = phoneme["start_time"]
            phone_end = phoneme["end_time"]
            phoneme_frame_keys = self._get_frames_interval(phone_start, phone_end, timed_frame_keys)
            sentence_tokens.extend([phoneme["text"]] * len(phoneme_frame_keys))

            for frame_key in phoneme_frame_keys:
                frame_contours_filepaths = contours_filepaths[frame_key]
                frame_targets = torch.zeros(size=(0, 2, self.n_samples))
                for art, contour_fp in sorted(frame_contours_filepaths.items(), key=lambda t: t[0]):
                    if art not in self.articulators:
                        continue

                    abs_contour_fp = os.path.join(self.datadir, contour_fp)
                    contour = torch.tensor(np.load(abs_contour_fp)) / self.size  # torch.Size([self.n_samples, 2])
                    contour = contour.transpose(1, 0)  # torch.Size([2, self.n_samples])
                    contour = contour.unsqueeze(dim=0)  # torch.Size([1, 2, self.n_samples]

                    frame_targets = torch.cat([frame_targets, contour])

                frame_targets = frame_targets.unsqueeze(dim=0)
                sentence_targets = torch.cat([sentence_targets, frame_targets])

        sentence_numerized = torch.tensor([
            self.vocabulary[token] for token in sentence_tokens
        ], dtype=torch.long)

        return sentence_numerized, sentence_targets, sentence_tokens

    @staticmethod
    def register(targets):
        n_frames, n_art, _, n_samples = targets.shape

        first_frame = targets[0]

        _, _, tongue, _ = first_frame
        first, last = tongue[:, 0], tongue[:, -1]
        first_x, first_y = first
        last_x, last_y = last

        # We want to translate the vocal tract from the position (x_disp, y_disp) to the center of
        # first quadrant in the cartesian plane (0.5, 0.5).
        x_, y_ = 0.5, 0.5
        x_disp = (first_frame[:, 0].min() + (first_frame[:, 0].max() - first_frame[:, 0].min()) / 2).item()
        y_disp = (first_frame[:, 1].min() + (first_frame[:, 1].max() - first_frame[:, 1].min()) / 2).item()

        dist = euclidean(first, last)
        theta_sin = abs(last_y - first_y) / dist
        theta_cos = abs(last_x - first_x) / dist
        theta = np.arctan2(theta_sin, theta_cos)
        rot_theta = -1 * theta

        R = torch.tensor([
            [np.cos(rot_theta), -np.sin(rot_theta)],
            [np.sin(rot_theta), np.cos(rot_theta)]
        ])

        new_targets = torch.zeros(size=(0, n_art, 2, n_samples), dtype=targets.dtype, device=targets.device)
        for target in targets:
            new_target = torch.zeros(size=(0, 2, n_samples), dtype=target.dtype, device=target.device)
            for art in target:
                new_art = torch.zeros_like(art)
                new_art[0] = art[0] - x_disp
                new_art[1] = art[1] - y_disp

                new_art = torch.matmul(R, new_art)
                new_art[0] = new_art[0] + x_
                new_art[1] = new_art[1] + y_
                new_art = new_art.unsqueeze(dim=0)

                new_target = torch.cat([new_target, new_art])

            new_target = new_target.unsqueeze(dim=0)
            new_targets = torch.cat([new_targets, new_target])

        return new_targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sentence_numerized, sentence_targets, phonemes = self.load_fn(self.data[item])

        # Centralize the targets.
        # Subtract the mean and add 0.5 to centralize in the 0-1 plane.
        sentence_targets_mean = sentence_targets.mean(dim=(1, 3))
        sentence_targets_mean = sentence_targets_mean.unsqueeze(dim=1).unsqueeze(dim=-1)
        sentence_targets = sentence_targets - sentence_targets_mean + 0.5

        if self.register_targets:
            sentence_targets = self.register(sentence_targets)

        return sentence_numerized, sentence_targets, phonemes
