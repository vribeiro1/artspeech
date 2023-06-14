import pdb

import funcy
import os
import torch

from functools import lru_cache
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from vt_shape_gen.helpers import load_articulator_array
from vt_tools import UPPER_INCISOR

from database_collector import DATABASE_COLLECTORS
from phoneme_to_articulation import InputLoaderMixin
from phoneme_to_articulation.tail_clipper import TailClipper
from settings import DATASET_CONFIG, UNKNOWN

phonemes_per_TV = {
    "LA": lambda p: p in ["p", "b", "m"],
    "TTCD": lambda p: p in ["l", "d", "n", "t"],
    "TBCD": lambda p: p in ["k", "g"],
    "VEL": lambda p: False
}


def pad_sequence_collate_fn(batch):
    sentence_numerized = [item[1] for item in batch]
    len_sentences = torch.tensor(funcy.lmap(len, sentence_numerized), dtype=torch.int)
    len_sentences_sorted, sentences_sorted_indices = len_sentences.sort(descending=True)
    padded_sentence_numerized = pad_sequence(sentence_numerized, batch_first=True)
    padded_sentence_numerized = padded_sentence_numerized[sentences_sorted_indices]

    sentence_targets = [item[2] for item in batch]
    padded_sentence_targets = pad_sequence(sentence_targets, batch_first=True)
    padded_sentence_targets = padded_sentence_targets[sentences_sorted_indices]

    phonemes = [batch[i][3] for i in sentences_sorted_indices]

    references = [item[4] for item in batch]
    padded_references = pad_sequence(references, batch_first=True)
    padded_references = padded_references[sentences_sorted_indices]

    # critical_masks = [item[5].T for item in batch]
    # padded_critical_masks = pad_sequence(critical_masks, batch_first=True)
    # padded_critical_masks = padded_critical_masks[sentences_sorted_indices]
    # padded_critical_masks = padded_critical_masks.permute(0, 2, 1)

    sentence_frames = [batch[i][6] for i in sentences_sorted_indices]
    sentences_ids = [batch[i][0] for i in sentences_sorted_indices]

    voicing = [batch[i][7] for i in sentences_sorted_indices]
    voicing = pad_sequence(voicing, batch_first=True, padding_value=-1)

    return (
        sentences_ids,
        padded_sentence_numerized,
        padded_sentence_targets,
        len_sentences_sorted,
        phonemes,
        # padded_critical_masks,
        padded_references,
        sentence_frames,
        voicing,
    )


@lru_cache(maxsize=None)
def cached_load_articulator_array(filepath, norm_value):
    return torch.from_numpy(load_articulator_array(filepath, norm_value)).type(torch.float)


class ArtSpeechDataset(Dataset):
    def __init__(
        self,
        datadir,
        database_name,
        sequences,
        vocabulary,
        articulators,
        n_samples=50,
        clip_tails=False,
        TVs=None,
        voiced_tokens=None,
    ):
        self.vocabulary = vocabulary
        self.datadir = datadir
        self.articulators = sorted(articulators)
        self.num_articulators = len(articulators)
        self.num_samples = n_samples
        self.clip_tails = clip_tails
        self.TVs = TVs or []
        self.voiced_tokens = voiced_tokens or []

        collector = DATABASE_COLLECTORS[database_name](datadir)
        data = collector.collect_data(sequences)
        self.data = funcy.lfilter(lambda d: d["has_all"], data)
        self.dataset_config = DATASET_CONFIG[database_name]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        sentence_name = item["sentence_name"]
        subject = item["subject"]
        sequence = item["sequence"]
        frame_ids = item["frame_ids"]
        sentence_tokens = item["phonemes"]

        sentence_targets = torch.zeros(size=(0, self.num_articulators, 2, self.num_samples))
        reference_arrays = torch.zeros(size=(0, 1, 2, self.num_samples))
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
                    clip_tails=self.clip_tails
                )  # (2, D)
                articulator_array = articulator_array.unsqueeze(dim=0)  # (1, 2, D)
                frame_targets = torch.cat([frame_targets, articulator_array], dim=0)

            frame_targets = frame_targets.unsqueeze(dim=0)  # (1, Nart, 2, D)
            sentence_targets = torch.cat([sentence_targets, frame_targets], dim=0)
            reference_array = reference_array.unsqueeze(dim=0).unsqueeze(dim=0)
            reference_arrays = torch.cat([reference_arrays, reference_array], dim=0)

        if len(self.TVs) == 0:
            critical_masks = torch.tensor([], dtype=torch.int)
        else:
            critical_masks = torch.stack([
                torch.tensor([
                    int(phonemes_per_TV[TV](phoneme)) for phoneme in sentence_tokens
                ], dtype=torch.int)
                for TV in self.TVs
            ])

        sentence_targets = sentence_targets.type(torch.float)
        reference_arrays = reference_arrays.type(torch.float)

        sentence_numerized = torch.tensor([
            self.vocabulary.get(token, self.vocabulary[UNKNOWN])
            for token in sentence_tokens
        ], dtype=torch.long)

        # Voicing information
        voicing = torch.tensor(
            [phoneme in self.voiced_tokens for phoneme in sentence_tokens],
            dtype=torch.float
        )

        return (
            sentence_name,
            sentence_numerized,
            sentence_targets,
            sentence_tokens,
            reference_arrays,
            critical_masks,
            frame_ids,
            voicing,
        )
