import funcy
import numpy as np
import os
import pandas as pd
import torch

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from database_collector import DATABASE_COLLECTORS
from phoneme_to_articulation import InputLoaderMixin
from phoneme_to_articulation.transforms import Normalize
from settings import DATASET_CONFIG, UNKNOWN

phoneme_weights = {
    "l": 3,
    "d": 3,
    "t": 3,
    "n": 3,
    "k": 3,
    "g": 3,
    "#": 0.1,
    "-": 0.1,
    "ih": 0.1,
    "yh": 0.1,
    "uh": 0.1,
}


class PrincipalComponentsAutoencoderDataset2(Dataset):
    def __init__(
        self,
        database_name,
        datadir,
        sequences,
        articulators,
        clip_tails=True
    ):
        self.datadir = datadir
        self.dataset_config = DATASET_CONFIG[database_name]
        self.clip_tails = clip_tails
        self.articulators = sorted(articulators)

        collector = DATABASE_COLLECTORS[database_name](datadir)
        sentence_data = collector.collect_data(sequences)
        data = []
        for sentence in sentence_data:
            for frame_id, phoneme in zip(sentence["frame_ids"], sentence["phonemes"]):
                data.append({
                    "subject": sentence["subject"],
                    "sequence": sentence["sequence"],
                    "frame_id": frame_id,
                    "phoneme": phoneme,
                })
        self.data = pd.DataFrame(data)

        self.normalize = {}
        for articulator in self.articulators:
            mean_filepath = os.path.join(
                datadir,
                "normalization_statistics",
                f"{articulator}_mean.npy"
            )
            mean = torch.from_numpy(np.load(mean_filepath))
            std_filepath = os.path.join(
                datadir,
                "normalization_statistics",
                f"{articulator}_std.npy"
            )
            std = torch.from_numpy(np.load(std_filepath))
            self.normalize[articulator] = Normalize(mean, std)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        phoneme = item["phoneme"]

        subject = item["subject"]
        sequence = item["sequence"]
        frame_id = item["frame_id"]

        weight = torch.tensor(phoneme_weights.get(phoneme, 1), dtype=torch.float)
        frame_name = f"{subject}_{sequence}_{frame_id}"

        articulators = torch.stack([
            InputLoaderMixin.prepare_articulator_array(
                self.datadir,
                subject,
                sequence,
                frame_id,
                articulator,
                self.dataset_config,
                self.normalize[articulator],
                self.clip_tails,
            )[0]
            for articulator in self.articulators
        ], dim=0)

        l, n, m = articulators.shape
        articulators = articulators.reshape(l, n * m).type(torch.float)

        return frame_name, articulators, weight, phoneme


class PrincipalComponentsPhonemeToArticulationDataset2(Dataset):
    """
    Dataset for phoneme to principal components adapted for the multiarticulators case.
    """
    def __init__(
        self,
        database_name,
        datadir,
        sequences,
        vocabulary,
        articulators,
        TV_to_phoneme_map,
        num_samples=50,
        clip_tails=True,
    ):
        self.datadir = datadir
        self.dataset_config = DATASET_CONFIG[database_name]
        self.vocabulary = vocabulary
        self.articulators = sorted(articulators)
        self.num_samples = num_samples
        self.clip_tails = clip_tails
        self.TV_to_phoneme_map = TV_to_phoneme_map

        collector = DATABASE_COLLECTORS[database_name](datadir)
        self.data = collector.collect_data(sequences)

        self.normalize = {}
        for articulator in self.articulators:
            mean_filepath = os.path.join(
                datadir,
                "normalization_statistics",
                f"{articulator}_mean.npy"
            )
            mean = torch.from_numpy(np.load(mean_filepath))
            std_filepath = os.path.join(
                datadir,
                "normalization_statistics",
                f"{articulator}_std.npy"
            )
            std = torch.from_numpy(np.load(std_filepath))
            self.normalize[articulator] = Normalize(mean, std)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        sentence_name = item["sentence_name"]
        subject = item["subject"]
        sequence = item["sequence"]
        frame_ids = item["frame_ids"]

        reference_arrays = torch.zeros(size=(0, 1, 2, self.num_samples))
        sentence_targets = torch.zeros(size=(0, len(self.articulators), 2, self.num_samples))
        for frame_id in frame_ids:
            frame_targets = torch.zeros(size=(0, 2, self.num_samples))
            for articulator in self.articulators:
                articulator_array, reference_array = InputLoaderMixin.prepare_articulator_array(
                    self.datadir,
                    subject,
                    sequence,
                    frame_id,
                    articulator,
                    self.dataset_config,
                    self.normalize[articulator],
                    self.clip_tails
                )  # (2, D)
                articulator_array = articulator_array.unsqueeze(dim=0)  # (1, 2, D)
                frame_targets = torch.cat([frame_targets, articulator_array], dim=0)
            frame_targets = frame_targets.unsqueeze(dim=0)  # (1, Nart, 2, D)
            sentence_targets = torch.cat([sentence_targets, frame_targets], dim=0)
            reference_array = reference_array.unsqueeze(dim=0).unsqueeze(dim=0)
            reference_arrays = torch.cat([reference_arrays, reference_array], dim=0)

        sentence_tokens = item["phonemes"]
        sentence_numerized = torch.tensor([
            self.vocabulary.get(token, self.vocabulary[UNKNOWN])
            for token in sentence_tokens
        ], dtype=torch.long)

        if len(self.TV_to_phoneme_map) > 0:
            critical_mask = torch.stack([
                torch.tensor(
                    [
                        int(p in self.TV_to_phoneme_map[TV])
                        for p in sentence_tokens
                    ],
                    dtype=torch.int
                )
                for TV in sorted(self.TV_to_phoneme_map.keys())
            ])
        else:
            critical_mask = torch.zeros(size=(0, len(sentence_tokens)))

        return (
            sentence_name,
            sentence_numerized,
            sentence_targets,
            sentence_tokens,
            critical_mask,
            reference_arrays,
            frame_ids,
        )


def pad_sequence_collate_fn(batch):
    sentence_numerized = [item[1] for item in batch]
    len_sentences = torch.tensor(funcy.lmap(len, sentence_numerized), dtype=torch.int)
    len_sentences_sorted, sentences_sorted_indices = len_sentences.sort(descending=True)
    padded_sentence_numerized = pad_sequence(sentence_numerized, batch_first=True)
    padded_sentence_numerized = padded_sentence_numerized[sentences_sorted_indices]

    sentences_ids = [batch[i][0] for i in sentences_sorted_indices]

    sentence_targets = [item[2] for item in batch]
    padded_sentence_targets = pad_sequence(sentence_targets, batch_first=True)
    padded_sentence_targets = padded_sentence_targets[sentences_sorted_indices]

    phonemes = [batch[i][3] for i in sentences_sorted_indices]

    critical_masks = [item[4].T for item in batch]
    padded_critical_masks = pad_sequence(critical_masks, batch_first=True)
    padded_critical_masks = padded_critical_masks[sentences_sorted_indices]
    padded_critical_masks = padded_critical_masks.permute(0, 2, 1)

    references = [item[5] for item in batch]
    padded_references = pad_sequence(references, batch_first=True)
    padded_references = padded_references[sentences_sorted_indices]

    sentence_frames = [batch[i][6] for i in sentences_sorted_indices]

    return (
        sentences_ids,
        padded_sentence_numerized,
        padded_sentence_targets,
        len_sentences_sorted,
        phonemes,
        padded_critical_masks,
        padded_references,
        sentence_frames
    )
