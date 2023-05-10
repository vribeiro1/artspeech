import funcy
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F

from functools import lru_cache
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm
from vt_tools import UPPER_INCISOR
from vt_shape_gen.helpers import load_articulator_array

from database_collector import DATABASE_COLLECTORS
from phoneme_to_articulation.tail_clipper import TailClipper
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


@lru_cache(maxsize=None)
def cached_load_articulator_array(filepath, norm_value):
    return torch.from_numpy(load_articulator_array(filepath, norm_value)).type(torch.float)


class InputLoaderMixin:
    @staticmethod
    def prepare_articulator_array(
        datadir,
        subject,
        sequence,
        frame_id,
        articulator,
        dataset_config,
        normalize_fn=None,
        clip_tails=True
    ):
        fp_articulator = os.path.join(
            datadir, subject, sequence, "inference_contours", f"{frame_id}_{articulator}.npy"
        )
        articulator_array = cached_load_articulator_array(
            fp_articulator,
            norm_value=dataset_config.RES
        )

        if clip_tails:
            tail_clip_refs = {}
            tail_clipper = TailClipper(dataset_config)
            for reference in TailClipper.TAIL_CLIP_REFERENCES:
                fp_reference = os.path.join(
                    datadir, subject, sequence, "inference_contours", f"{frame_id}_{reference}.npy"
                )
                reference_array = cached_load_articulator_array(
                    fp_reference,
                    norm_value=dataset_config.RES
                )
                tail_clip_refs[reference.replace("-", "_")] = reference_array

            tail_clip_method_name = f"clip_{articulator.replace('-', '_')}_tails"
            tail_clip_method = getattr(tail_clipper, tail_clip_method_name, None)
            if tail_clip_method:
                articulator_array = tail_clip_method(articulator_array, **tail_clip_refs)

        fp_coord_system_reference = os.path.join(
            datadir, subject, sequence, "inference_contours", f"{frame_id}_{UPPER_INCISOR}.npy"
        )
        coord_system_reference_array = cached_load_articulator_array(
            fp_coord_system_reference,
            norm_value=dataset_config.RES
        )
        coord_system_reference = coord_system_reference_array.T[:, -1]
        coord_system_reference = coord_system_reference.unsqueeze(dim=-1)

        coord_system_reference_array = coord_system_reference_array.T
        coord_system_reference_array = coord_system_reference_array - coord_system_reference
        coord_system_reference_array[0, :] = coord_system_reference_array[0, :] + 0.3
        coord_system_reference_array[1, :] = coord_system_reference_array[1, :] + 0.3

        articulator_array = articulator_array.T
        articulator_array = articulator_array - coord_system_reference
        articulator_array[0, :] = articulator_array[0, :] + 0.3
        articulator_array[1, :] = articulator_array[1, :] + 0.3

        if normalize_fn is not None:
            articulator_array = normalize_fn(articulator_array)
            coord_system_reference_array = normalize_fn(coord_system_reference_array)

        return articulator_array, coord_system_reference_array


class PrincipalComponentsAutoencoderDataset(Dataset):
    def __init__(
        self,
        database_name,
        datadir,
        sequences,
        articulator,
        clip_tails=True
    ):
        self.datadir = datadir
        self.dataset_config = DATASET_CONFIG[database_name]
        self.articulator = articulator
        self.clip_tails = clip_tails

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
        self.normalize = Normalize(mean, std)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """

        Return:
            frame_name (str): unique reference to the frame, created by concatenating the subject,
                sequence and frame number
            articulator (torch.tensor): tensor of shape (2 * D,), where D is the dimension of the
                articulator array.
            weight (float): phoneme weight
            phoneme (str): phoneme
        """
        item = self.data.iloc[index]
        phoneme = item["phoneme"]

        subject = item["subject"]
        sequence = item["sequence"]
        frame_id = item["frame_id"]

        weight = torch.tensor(phoneme_weights.get(phoneme, 1), dtype=torch.float)
        frame_name = f"{subject}_{sequence}_{frame_id}"

        articulator, _ = InputLoaderMixin.prepare_articulator_array(
            self.datadir,
            subject,
            sequence,
            frame_id,
            self.articulator,
            self.dataset_config,
            self.normalize,
            self.clip_tails,
        )
        n, m = articulator.shape
        articulator = articulator.reshape(n * m).type(torch.float)

        return frame_name, articulator, weight, phoneme


class PrincipalComponentsAutoencoderDataset2(PrincipalComponentsAutoencoderDataset):
    def __init__(
        self,
        database_name,
        datadir,
        sequences,
        articulators,
        clip_tails=True
    ):
        super().__init__(
            database_name,
            datadir,
            sequences,
            articulators[0],
            clip_tails
        )
        self.articulators = sorted(articulators)

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


class PrincipalComponentsPhonemeToArticulationDataset(Dataset):
    critical_phonemes = {
        "TTCD": lambda p: p in ["l", "d", "n", "t"],
        "TBCD": lambda p: p in ["k", "g"]
    }

    def __init__(
        self,
        database_name,
        datadir,
        sequences,
        vocabulary,
        articulator,
        sync_shift,
        framerate,
        n_samples=50,
        clip_tails=True
    ):
        self.datadir = datadir
        self.dataset_config = DATASET_CONFIG[database_name]
        self.vocabulary = vocabulary
        self.articulator = articulator
        self.n_samples = n_samples
        self.clip_tails = clip_tails
        self.TVs = ["TBCD", "TTCD"]

        collector = DATABASE_COLLECTORS[database_name](datadir)
        self.data = collector.collect_data(sequences)

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
        self.normalize = Normalize(mean, std)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        sentence_name = item["sentence_name"]
        subject = item["subject"]
        sequence = item["sequence"]
        frame_ids = item["frame_ids"]

        sentence_targets = torch.zeros(size=(0, 1, 2, self.n_samples))
        sentence_critical_references = torch.zeros(size=(0, len(self.TVs), 2, self.n_samples))
        for frame_id in frame_ids:
            articulator_array, coord_system_reference_array = InputLoaderMixin.prepare_articulator_array(
                self.datadir,
                subject,
                sequence,
                frame_id,
                self.articulator,
                self.dataset_config,
                self.normalize,
                self.clip_tails,
            )
            articulator_array = articulator_array.unsqueeze(dim=0).unsqueeze(dim=0)
            coord_system_reference_array = coord_system_reference_array.unsqueeze(dim=0)
            sentence_targets = torch.cat([sentence_targets, articulator_array], dim=0)

            denorm_coord_system_reference_array = self.normalize.inverse(
                coord_system_reference_array
            )
            hard_palate = denorm_coord_system_reference_array[:, :, :20]
            hard_palate = F.interpolate(
                hard_palate,
                size=self.n_samples,
                mode="linear",
                align_corners=True
            )
            norm_hard_palate = self.normalize(hard_palate)

            alveolar_region = denorm_coord_system_reference_array[:, :, 20:42]
            alveolar_region = F.interpolate(
                alveolar_region,
                size=self.n_samples,
                mode="linear",
                align_corners=True
            )
            norm_alveolar_region = self.normalize(alveolar_region)

            critical_reference = torch.cat([norm_hard_palate, norm_alveolar_region], dim=0)
            critical_reference = critical_reference.unsqueeze(dim=0)
            sentence_critical_references = torch.cat([
                sentence_critical_references,
                critical_reference
            ])

        sentence_targets = sentence_targets.type(torch.float)
        sentence_critical_references = sentence_critical_references.type(torch.float)

        critical_mask = torch.stack([
            torch.tensor([int(self.critical_phonemes[TV](p)) for p in item["phonemes"]], dtype=torch.int)
             for TV in sorted(self.TVs)
        ])

        sentence_tokens = item["phonemes"]
        sentence_numerized = torch.tensor([
            self.vocabulary.get(token, self.vocabulary[UNKNOWN])
            for token in sentence_tokens
        ], dtype=torch.long)

        return (
            sentence_name,
            sentence_numerized,
            sentence_targets,
            sentence_tokens,
            critical_mask,
            sentence_critical_references,
            frame_ids
        )


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
        self.data = collector.collect_data(sequences)[:1]

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

        sentence_targets = torch.zeros(size=(0, len(self.articulators), 2, self.num_samples))
        for frame_id in frame_ids:
            frame_targets = torch.zeros(size=(0, 2, self.num_samples))
            for articulator in self.articulators:
                articulator_array, _ = InputLoaderMixin.prepare_articulator_array(
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
            frame_ids
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

    # critical_references = [item[5] for item in batch]
    # padded_critical_references = pad_sequence(critical_references, batch_first=True)
    # padded_critical_references = padded_critical_references[sentences_sorted_indices]

    sentence_frames = [batch[i][5] for i in sentences_sorted_indices]

    return (
        sentences_ids,
        padded_sentence_numerized,
        padded_sentence_targets,
        len_sentences_sorted,
        phonemes,
        padded_critical_masks,
        # padded_critical_references,
        sentence_frames
    )
