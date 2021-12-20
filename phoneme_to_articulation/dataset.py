import pdb

import funcy
import json
import numpy as np
import os
import torch

from functools import lru_cache, reduce
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from bs_regularization import regularize_Bsplines
from settings import DatasetConfig


def get_token_intervals(token_sequence, break_token):
    token_intervals = []

    seq_start = seq_end = None
    for i, token in enumerate(token_sequence):
        if seq_start is None:
            if token == break_token:
                seq_start = i

                if i == len(token_sequence) - 1:
                    token_intervals.append((seq_start, i + 1))
        else:
            if token != break_token:
                seq_end = i

                token_intervals.append((seq_start, seq_end))
                seq_start = seq_end = None

    return token_intervals


def get_break_points(break_intervals):
    break_points = funcy.lmap(int, map(
        lambda tup: np.ceil(tup[0] + (tup[1] - tup[0]) / 2),
        break_intervals
    ))

    return break_points


def break_sequence(sequence, break_points):
    segments = []
    i = 0
    for break_point in break_points:
        segment = sequence[i:break_point]
        segments.append(segment)

        i = break_point

    segments.append(sequence[break_point:])

    return segments


class TailClipper:
    @classmethod
    def clip_tongue_tails(cls, tongue, lower_incisor, epiglottis, reg_out=True, **kwargs):
        # Remove the front tail of the tongue using the lower incisor as the reference
        ref_idx = lower_incisor[:, 1].argmax()
        reference = lower_incisor[ref_idx]

        tongue_cp = tongue.copy()

        tongue_1st_half = tongue_cp[:25]
        tongue_2nd_half = tongue_cp[25:]

        keep_indices = np.where(tongue_2nd_half[:, 1] < reference[1])

        tailless_tongue = np.concatenate([
            tongue_1st_half,
            tongue_2nd_half[keep_indices]
        ])

        # Remove the back tail of the tongue using the epiglottis as the reference
        ref_idx = epiglottis[:, 1].argmin()
        reference = epiglottis[ref_idx]

        tongue_cp = tailless_tongue.copy()

        tongue_1st_half = tongue_cp[:25]
        tongue_2nd_half = tongue_cp[25:]

        keep_indices = np.where(tongue_1st_half[:, 1] < reference[1] + (10 / DatasetConfig.PIXEL_SPACING / DatasetConfig.RES))

        tailless_tongue = np.concatenate([
            tongue_1st_half[keep_indices],
            tongue_2nd_half
        ])

        if reg_out:
            reg_x, reg_y = regularize_Bsplines(tailless_tongue, 3)
            tailless_tongue = np.array([reg_x, reg_y]).T

        return tailless_tongue

    @classmethod
    def clip_lower_lip_tails(cls, lower_lip, lower_incisor, reg_out=True, **kwargs):
        # Remove the front tail of the lower lip using the lower incisor as the reference
        ref_idx = lower_incisor[:, 1].argmax()
        reference = lower_incisor[ref_idx]

        llip_cp = lower_lip.copy()

        llip_1st_half = llip_cp[:25]
        llip_2nd_half = llip_cp[25:]

        keep_indices = np.where(llip_2nd_half[:, 1] < reference[1] + (5 / DatasetConfig.PIXEL_SPACING / DatasetConfig.RES))

        tailless_llip = np.concatenate([
            llip_1st_half,
            llip_2nd_half[keep_indices]
        ])

        if reg_out:
            reg_x, reg_y = regularize_Bsplines(tailless_llip, 3)
            tailless_llip = np.array([reg_x, reg_y]).T

        # Remove the back tail of the lower lip using the lower incisor as the reference
        ref_idx = lower_incisor[:, 1].argmax()
        reference = lower_incisor[ref_idx]

        llip_cp = tailless_llip.copy()

        llip_1st_half = llip_cp[:25]
        llip_2nd_half = llip_cp[25:]

        keep_indices = np.where(llip_1st_half[:, 1] < reference[1])

        tailless_llip = np.concatenate([
            llip_1st_half[keep_indices],
            llip_2nd_half
        ])

        if reg_out:
            reg_x, reg_y = regularize_Bsplines(tailless_llip, 3)
            tailless_llip = np.array([reg_x, reg_y]).T

        return tailless_llip

    @classmethod
    def clip_upper_lip_tails(cls, upper_lip, upper_incisor, reg_out=True, **kwargs):
        # Remove the front tail of the upper lip using the upper incisor as the reference
        ref_idx = -1
        reference = upper_incisor[ref_idx]

        ulip_cp = upper_lip.copy()

        ulip_1st_half = ulip_cp[:25]
        ulip_2nd_half = ulip_cp[25:]

        keep_indices = np.where(ulip_2nd_half[:, 1] > reference[1] - (10 / cls.PIXEL_SPACING))

        tailless_ulip = np.concatenate([
            ulip_1st_half,
            ulip_2nd_half[keep_indices]
        ])

        # Remove the back tail of the upper lip using the upper incisor as the reference
        ref_idx = -1
        reference = upper_incisor[ref_idx]

        ulip_cp = tailless_ulip.copy()

        ulip_1st_half = ulip_cp[:25]
        ulip_2nd_half = ulip_cp[25:]

        keep_indices = np.where(ulip_1st_half[:, 1] > reference[1] - (5 / cls.PIXEL_SPACING))

        tailless_ulip = np.concatenate([
            ulip_1st_half[keep_indices],
            ulip_2nd_half
        ])

        if reg_out:
            reg_x, reg_y = regularize_Bsplines(tailless_ulip, 3)
            tailless_ulip = np.array([reg_x, reg_y]).T

        return tailless_ulip


def pad_sequence_collate_fn(batch):
    sentence_numerized = [item[0] for item in batch]
    len_sentences = torch.tensor(funcy.lmap(len, sentence_numerized), dtype=torch.int)
    len_sentences_sorted, sentences_sorted_indices = len_sentences.sort(descending=True)
    padded_sentence_numerized = pad_sequence(sentence_numerized, batch_first=True)
    padded_sentence_numerized = padded_sentence_numerized[sentences_sorted_indices]

    sentence_targets = [item[1] for item in batch]
    padded_sentence_targets = pad_sequence(sentence_targets, batch_first=True)
    padded_sentence_targets = padded_sentence_targets[sentences_sorted_indices]

    phonemes = [batch[i][2] for i in sentences_sorted_indices]

    return padded_sentence_numerized, padded_sentence_targets, len_sentences_sorted, phonemes


class ArtSpeechDataset(Dataset):
    N_SAMPLES = 50
    def __init__(
        self, datadir, filepath, vocabulary, articulators, n_samples=50, size=136,
        lazy_load=False, p_aug=0., clip_tails=False
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
        lazy_load (bool): If should load the data on demand.
        p_aug (float): Probability of data augmentation.
        clip_tails (bool): If should clip the tails of some articulators to keep only the
            acoustically relevant parts of the articulator.
        """

        self.vocabulary = vocabulary
        self.datadir = datadir
        self.articulators = sorted(articulators)
        self.n_articulators = len(articulators)
        self.n_samples = n_samples
        self.size = size
        self.p_aug = p_aug

        if "upper-incisor" in self.articulators:
            self.upper_incisor_index = self.articulators.index("upper-incisor")

        with open(filepath) as f:
            data = funcy.lfilter(self._exclude_missing_data, json.load(f))

        tail_clip_refs = ["lower-incisor", "upper-incisor", "epiglottis"]
        if clip_tails and not all(map(lambda art: art in articulators, tail_clip_refs)):
            raise ValueError(
                f"clip_tails == True requires that all the references are available."
                f"References are {tail_clip_refs}"
            )

        self.clip_tails = clip_tails

        self.data = data if lazy_load else self._collect_data(data)
        self.load_fn = self._collect_sentence if lazy_load else lambda x: x

    @staticmethod
    def _get_frames_interval(start, end, timed_frame_keys):
        on_interval = filter(lambda d: start <= d[0] < end, timed_frame_keys)
        frame_keys = [d[1] for d in on_interval]
        return frame_keys

    @staticmethod
    @lru_cache()
    def load_target_array(filepath, norm=True):
        """
        Loads the target array with the proper orientation (right to left)
        """
        target_array = np.load(filepath)
        n_rows, n_cols = target_array.shape
        if n_rows == 2:
            target_array = target_array.T

        # All the countors should be oriented from right to left. If it is the opposite,
        # we flip the array.
        if target_array[0][0] < target_array[-1][0]:
            target_array = np.flip(target_array, axis=0)

        if norm:
            target_array = target_array.copy() / DatasetConfig.RES

        return target_array

    @staticmethod
    def load_frame_targets(datadir, frame_targets_filepaths, articulators, clip_tails=False):
        # References for tail clipping
        if clip_tails:
            lower_incisor_fp = os.path.join(datadir, frame_targets_filepaths["lower-incisor"])
            lower_incisor = ArtSpeechDataset.load_target_array(lower_incisor_fp)

            upper_incisor_fp = os.path.join(datadir, frame_targets_filepaths["upper-incisor"])
            upper_incisor = ArtSpeechDataset.load_target_array(upper_incisor_fp)

            epiglottis_fp = os.path.join(datadir, frame_targets_filepaths["epiglottis"])
            epiglottis = ArtSpeechDataset.load_target_array(epiglottis_fp)
        else:
            lower_incisor = upper_incisor = epiglottis = None

        frame_targets = torch.zeros(size=(0, 2, ArtSpeechDataset.N_SAMPLES))
        for art, contour_fp in sorted(frame_targets_filepaths.items(), key=lambda t: t[0]):
            if art not in articulators:
                continue

            abs_contour_fp = os.path.join(datadir, contour_fp)
            contour_arr = ArtSpeechDataset.load_target_array(abs_contour_fp)

            if clip_tails:
                tail_clip_method = getattr(
                    TailClipper, f"clip_{art.replace('-', '_')}_tails", None
                )

                if tail_clip_method:
                    contour_arr = tail_clip_method(
                        contour_arr,
                        lower_incisor=lower_incisor,
                        upper_incisor=upper_incisor,
                        epiglottis=epiglottis
                    )

            contour = torch.from_numpy(contour_arr)  # torch.Size([self.n_samples, 2])
            contour = contour.transpose(1, 0)  # torch.Size([2, self.n_samples])
            contour = contour.unsqueeze(dim=0)  # torch.Size([1, 2, self.n_samples]

            frame_targets = torch.cat([frame_targets, contour])

        return frame_targets

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
            ]) for _, art_filepaths in contours_filepaths.items()
        ]

        return not any(frame_is_missing)

    def _collect_data(self, data):
        dataset = funcy.lmap(self._collect_sentence, tqdm(data, desc="Collecting data"))
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

                frame_targets = self.load_frame_targets(
                    self.datadir,
                    frame_contours_filepaths,
                    self.articulators,
                    self.clip_tails
                )

                frame_targets = frame_targets.unsqueeze(dim=0)
                sentence_targets = torch.cat([sentence_targets, frame_targets])

        sentence_numerized = torch.tensor([
            self.vocabulary[token] for token in sentence_tokens
        ], dtype=torch.long)

        return sentence_numerized, sentence_targets, sentence_tokens

    def augment(self, sentence_numerized, sentence_targets, phonemes):
        # Get silence intervals in the original sentence
        orig_token_intervals = get_token_intervals(phonemes, "#")
        orig_break_points = get_break_points(orig_token_intervals)

        # If there are no points to break in the original sentence, return it unchanged
        if len(orig_break_points) == 0:
            return sentence_numerized, sentence_targets, phonemes

        # Break the original sentence
        orig_segments_numerized = break_sequence(sentence_numerized, orig_break_points)
        orig_segments_targets = break_sequence(sentence_targets, orig_break_points)
        orig_segments_phonemes = break_sequence(phonemes, orig_break_points)

        rand_break_points = []
        while len(rand_break_points) == 0:
            # Randomly select a sentence in the dataset
            rand_idx = np.random.randint(0, len(self.data))
            rand_sentence_numerized, rand_sentence_targets, rand_phonemes = self.load_fn(self.data[rand_idx])

            # Get silence intervals in the selected sentence
            rand_token_intervals = get_token_intervals(rand_phonemes, "#")
            rand_break_points = get_break_points(rand_token_intervals)

        # Break the selected sentence
        rand_segments_numerized = break_sequence(rand_sentence_numerized, rand_break_points)
        rand_segments_targets = break_sequence(rand_sentence_targets, rand_break_points)
        rand_segments_phonemes = break_sequence(rand_phonemes, rand_break_points)

        # Randomly select a new segment to include
        new_segment_idx = np.random.randint(0, len(rand_segments_phonemes))
        new_segment_numerized = rand_segments_numerized[new_segment_idx]
        new_segment_targets = rand_segments_targets[new_segment_idx]
        new_segment_phonemes = rand_segments_phonemes[new_segment_idx]

        # Randomly select an original segment to remove
        orig_segment_idx = np.random.randint(low=0, high=len(orig_segments_phonemes))

        # Replace the original segment by the new segment
        orig_segments_numerized[orig_segment_idx] = new_segment_numerized
        orig_segments_targets[orig_segment_idx] = new_segment_targets
        orig_segments_phonemes[orig_segment_idx] = new_segment_phonemes

        # Reconstruct the data item
        aug_sentence_numerized = torch.cat(orig_segments_numerized)
        aug_segments_targets = torch.cat(orig_segments_targets)
        aug_segments_phonemes = reduce(lambda l1, l2: l1 + l2, orig_segments_phonemes)

        return aug_sentence_numerized, aug_segments_targets, aug_segments_phonemes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sentence_numerized, sentence_targets, phonemes = self.load_fn(self.data[item])

        if np.random.rand() < self.p_aug:
            sentence_numerized, sentence_targets, phonemes = self.augment(
                sentence_numerized, sentence_targets, phonemes
            )

        # Centralize the targets using the upper incisor as the reference of the coordinates system
        if "upper-incisor" in self.articulators:
            upper_incisor_last_samples = sentence_targets[:, self.upper_incisor_index, :, -1]
            subtract_array = upper_incisor_last_samples.unsqueeze(1).unsqueeze(-1)

            sentence_targets = sentence_targets - subtract_array
            sentence_targets[:, :, 0, :] = sentence_targets[:, :, 0, :] + 0.3
            sentence_targets[:, :, 1, :] = sentence_targets[:, :, 1, :] + 0.3

        return sentence_numerized, sentence_targets, phonemes
