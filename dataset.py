import pdb

import funcy
import json
import numpy as np
import os
import torch

from scipy.spatial.distance import euclidean
from torch.utils.data import Dataset


class ArtSpeechDataset(Dataset):
    def __init__(self, datadir, filepath, vocabulary, n_articulators=4, n_samples=50, size=136, register=False, save_missing=None):
        self.vocabulary = vocabulary
        self.datadir = datadir
        self.n_articulators = n_articulators
        self.n_samples = n_samples
        self.size = size
        self.register_targets = register

        with open(filepath) as f:
            data = json.load(f)
            self.data = self._collect_data(data)

    def _collect_data(self, data):
        dataset = []
        missing_data = []
        for item in data:
            missing = False

            phonemes = item["phonemes"]
            sentence_tokens = funcy.lflatten([
                [phoneme["text"]] * phoneme["n_frames"] for phoneme in phonemes
            ])

            sentence_numerized = torch.tensor([
                self.vocabulary[token] for token in sentence_tokens
            ], dtype=torch.long)

            contours_filepaths = item["contours_filepaths"]
            sentence_targets = torch.zeros(size=(0, self.n_articulators, 2, self.n_samples))
            for i_number, filepaths in contours_filepaths.items():
                target = torch.zeros(size=(0, 2, self.n_samples))
                for art, filepath in sorted(filepaths.items(), key=lambda t: t[0]):
                    if filepath is None:
                        missing_data.append({
                            "subject": item["metadata"]["subject"],
                            "sequence": item["metadata"]["sequence"],
                            "instance_number": i_number,
                            "articulator": art
                        })

                        missing = True
                        continue
                    abs_filepath = os.path.join(self.datadir, filepath)

                    contour = torch.tensor(np.load(abs_filepath)) / self.size  # torch.Size([self.n_samples, 2])
                    contour = contour.transpose(1, 0)  # torch.Size([2, self.n_samples])
                    contour = contour.unsqueeze(dim=0)  # torch.Size([1, 2, self.n_samples])

                    target = torch.cat([target, contour])

                n_art, _, _ = target.shape
                if n_art != self.n_articulators:
                    continue

                target = target.unsqueeze(dim=0)
                sentence_targets = torch.cat([sentence_targets, target])

            if not missing:
                len_sentence, = sentence_numerized.shape
                len_target, _, _, _ = sentence_targets.shape

                if len_sentence == len_target:
                    dataset.append((
                        sentence_numerized,
                        sentence_targets.float(),
                        sentence_tokens
                    ))

        if save_missing is not None:
            with open(save_missing, "w") as f:
                json.dump(missing_art, f)

        return dataset

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
        sentence_numerized, sentence_targets, phonemes = self.data[item]

        if self.register_targets:
            sentence_targets = self.register(sentence_targets)

        return sentence_numerized, sentence_targets, phonemes
