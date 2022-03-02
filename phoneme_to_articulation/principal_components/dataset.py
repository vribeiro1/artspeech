import pdb

import funcy
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F

from functools import lru_cache
from glob import glob
from tgt import read_textgrid
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm
from vt_tools import UPPER_INCISOR
from vt_shape_gen.helpers import load_articulator_array

from phoneme_to_articulation.tail_clipper import TailClipper
from phoneme_to_articulation.transforms import Normalize
from settings import BASE_DIR, DatasetConfig
from video import Video

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


def collect_data(datadir, sequences, sync_shift, framerate):
    data = []
    for subject, sequence in tqdm(sequences, desc="Collecting data"):
        seq_dir = os.path.join(datadir, subject, sequence)

        # Collect all of the articulators files
        articulators_filepaths = glob(os.path.join(seq_dir, "inference_contours", "*.npy"))
        # Since each frame have more than one articulator, we extract the frame ids and remove
        # repetitions.
        articulators_basenames = map(os.path.basename, articulators_filepaths)
        articulators_filenames = map(lambda s: s.split(".")[0], articulators_basenames)
        frame_ids = sorted(set(map(lambda s: s.split("_")[0], articulators_filenames)))

        wav_filepath = os.path.join(seq_dir, f"vol_{subject}_{sequence}.wav")

        video = Video(
            frames_filepaths=frame_ids[sync_shift:],
            audio_filepath=wav_filepath,
            framerate=framerate
        )

        textgrid_filepath = os.path.join(seq_dir, f"vol_{subject}_{sequence}.textgrid")
        textgrid = read_textgrid(textgrid_filepath)
        phone_tier = textgrid.get_tier_by_name("PhonTier")
        sentence_tier = textgrid.get_tier_by_name("SentenceTier")

        for sentence_interval in sentence_tier.intervals:
            sentence_phone_intervals = filter(lambda phone: (
                phone.start_time >= sentence_interval.start_time and
                phone.end_time <= sentence_interval.end_time
            ), phone_tier)

            sentence_phone_intervals = sorted(
                sentence_phone_intervals,
                key=lambda interval: interval.start_time
            )

            sentence_phonemes = []
            sentence_frame_ids = []
            for phone_interval in sentence_phone_intervals:
                _, phoneme_frame_ids = video.get_frames_interval(phone_interval.start_time, phone_interval.end_time)
                repeated_phoneme = [phone_interval.text] * len(phoneme_frame_ids)

                sentence_frame_ids.extend(phoneme_frame_ids)
                sentence_phonemes.extend(repeated_phoneme)

            start_str = "%.04f" % sentence_interval.start_time
            end_str = "%.04f" % sentence_interval.end_time
            sentence_name = f"vol_{subject}_{sequence}-{start_str}_{end_str}"

            data.append({
                "subject": subject,
                "sequence": sequence,
                "sentence_name": sentence_name,
                "n_frames": len(sentence_frame_ids),
                "frame_ids": sentence_frame_ids,
                "phonemes": sentence_phonemes
            })

    return data


class PrincipalComponentsAutoencoderDataset(Dataset):
    def __init__(self, datadir, sequences, articulator, sync_shift, framerate, clip_tails=True):
        self.datadir = datadir
        self.articulator = articulator
        self.clip_tails = clip_tails

        sentence_data = collect_data(datadir, sequences, sync_shift, framerate)
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

        mean = torch.from_numpy(np.load(os.path.join(BASE_DIR, "data", f"{articulator}_mean.npy")))
        std = torch.from_numpy(np.load(os.path.join(BASE_DIR, "data", f"{articulator}_std.npy")))
        self.normalize = Normalize(mean, std)

    def __len__(self):
        return len(self.data)

    def prepare_articulator_array(self, subject, sequence, frame_id):
        fp_articulator = os.path.join(
            self.datadir, subject, sequence, "inference_contours", f"{frame_id}_{self.articulator}.npy"
        )

        articulator_array = cached_load_articulator_array(fp_articulator, norm_value=DatasetConfig.RES)

        if self.clip_tails:
            tail_clip_refs = {}
            for reference in TailClipper.TAIL_CLIP_REFERENCES:
                fp_reference = os.path.join(
                    self.datadir, subject, sequence, "inference_contours", f"{frame_id}_{reference}.npy"
                )

                reference_array = cached_load_articulator_array(fp_reference, norm_value=DatasetConfig.RES)
                tail_clip_refs[reference.replace("-", "_")] = reference_array

            tail_clip_method_name = f"clip_{self.articulator.replace('-', '_')}_tails"
            tail_clip_method = getattr(TailClipper, tail_clip_method_name, None)

            if tail_clip_method:
                articulator_array = tail_clip_method(articulator_array, **tail_clip_refs)

        articulator_array = articulator_array.T

        fp_coord_system_reference = os.path.join(
            self.datadir, subject, sequence, "inference_contours", f"{frame_id}_{UPPER_INCISOR}.npy"
        )

        coord_system_reference_array = cached_load_articulator_array(fp_coord_system_reference, norm_value=DatasetConfig.RES)
        coord_system_reference = coord_system_reference_array.T[:, -1]
        coord_system_reference = coord_system_reference.unsqueeze(dim=-1)

        articulator_array = articulator_array - coord_system_reference
        articulator_array[0, :] = articulator_array[0, :] + 0.3
        articulator_array[1, :] = articulator_array[1, :] + 0.3

        articulator_norm = self.normalize(articulator_array)

        return articulator_norm

    def __getitem__(self, index):
        item = self.data.iloc[index]
        phoneme = item["phoneme"]

        subject = item["subject"]
        sequence = item["sequence"]
        frame_id = item["frame_id"]

        weight = torch.tensor(phoneme_weights.get(phoneme, 1), dtype=torch.float)
        frame_name = f"{subject}_{sequence}_{frame_id}"

        articulator = self.prepare_articulator_array(subject, sequence, frame_id)
        n, m = articulator.shape
        articulator = articulator.reshape(n * m).type(torch.float)

        return frame_name, articulator, weight, phoneme


class PrincipalComponentsPhonemeToArticulationDataset(Dataset):
    critical_phonemes = {
        "TTCD": lambda p: p in ["l", "d", "n", "t"],
        "TBCD": lambda p: p in ["k", "g"]
    }

    def __init__(self, datadir, sequences, vocabulary, articulator, sync_shift, framerate, n_samples=50, clip_tails=True):
        self.datadir = datadir
        self.vocabulary = vocabulary
        self.articulator = articulator
        self.n_samples = n_samples
        self.clip_tails = clip_tails

        self.TVs = ["TBCD", "TTCD"]
        self.data = collect_data(datadir, sequences, sync_shift, framerate)

        mean = torch.from_numpy(np.load(os.path.join(BASE_DIR, "data", f"{articulator}_mean.npy")))
        std = torch.from_numpy(np.load(os.path.join(BASE_DIR, "data", f"{articulator}_std.npy")))
        self.normalize = Normalize(mean, std)

    def __len__(self):
        return len(self.data)

    def prepare_articulator_array(self, subject, sequence, frame_id):
        fp_articulator = os.path.join(
            self.datadir, subject, sequence, "inference_contours", f"{frame_id}_{self.articulator}.npy"
        )

        articulator_array = cached_load_articulator_array(fp_articulator, norm_value=DatasetConfig.RES)

        if self.clip_tails:
            tail_clip_refs = {}
            for reference in TailClipper.TAIL_CLIP_REFERENCES:
                fp_reference = os.path.join(
                    self.datadir, subject, sequence, "inference_contours", f"{frame_id}_{reference}.npy"
                )

                reference_array = cached_load_articulator_array(fp_reference, norm_value=DatasetConfig.RES)
                tail_clip_refs[reference.replace("-", "_")] = reference_array

            tail_clip_method_name = f"clip_{self.articulator.replace('-', '_')}_tails"
            tail_clip_method = getattr(TailClipper, tail_clip_method_name, None)

            if tail_clip_method:
                articulator_array = tail_clip_method(articulator_array, **tail_clip_refs)

        articulator_array = articulator_array.T

        fp_coord_system_reference = os.path.join(
            self.datadir, subject, sequence, "inference_contours", f"{frame_id}_{UPPER_INCISOR}.npy"
        )

        coord_system_reference_array = cached_load_articulator_array(fp_coord_system_reference, norm_value=DatasetConfig.RES).T
        coord_system_reference = coord_system_reference_array[:, -1]
        coord_system_reference = coord_system_reference.unsqueeze(dim=-1)

        coord_system_reference_array = coord_system_reference_array - coord_system_reference
        coord_system_reference_array[0, :] = coord_system_reference_array[0, :] + 0.3
        coord_system_reference_array[1, :] = coord_system_reference_array[1, :] + 0.3

        articulator_array = articulator_array - coord_system_reference
        articulator_array[0, :] = articulator_array[0, :] + 0.3
        articulator_array[1, :] = articulator_array[1, :] + 0.3

        articulator_array = self.normalize(articulator_array)
        coord_system_reference_array = self.normalize(coord_system_reference_array)

        return articulator_array, coord_system_reference_array

    def __getitem__(self, index):
        item = self.data[index]
        sentence_name = item["sentence_name"]

        critical_mask = []
        sentence_frames = []
        sentence_targets = torch.zeros(size=(0, 1, 2, self.n_samples))
        sentence_critical_references = torch.zeros(size=(0, len(self.TVs), 2, self.n_samples))
        for frame_id in item["frame_ids"]:
            subject = item["subject"]
            sequence = item["sequence"]

            articulator_array, coord_system_reference_array = self.prepare_articulator_array(subject, sequence, frame_id)
            articulator_array = articulator_array.unsqueeze(dim=0).unsqueeze(dim=0)
            coord_system_reference_array = coord_system_reference_array.unsqueeze(dim=0)
            sentence_targets = torch.cat([sentence_targets, articulator_array], dim=0)

            denorm_coord_system_reference_array = self.normalize.inverse(coord_system_reference_array)
            hard_palate = denorm_coord_system_reference_array[:, :, :20]
            hard_palate = F.interpolate(hard_palate, size=self.n_samples, mode="linear", align_corners=True)
            norm_hard_palate = self.normalize(hard_palate)

            alveolar_region = denorm_coord_system_reference_array[:, :, 20:42]
            alveolar_region = F.interpolate(alveolar_region, size=self.n_samples, mode="linear", align_corners=True)
            norm_alveolar_region = self.normalize(alveolar_region)

            critical_reference = torch.cat([norm_hard_palate, norm_alveolar_region], dim=0)
            critical_reference = critical_reference.unsqueeze(dim=0)
            sentence_critical_references = torch.cat([sentence_critical_references, critical_reference])

            sentence_frames.append(frame_id)

        sentence_targets = sentence_targets.type(torch.float)
        sentence_critical_references = sentence_critical_references.type(torch.float)

        critical_mask = torch.stack([
            torch.tensor([int(self.critical_phonemes[TV](p)) for p in item["phonemes"]], dtype=torch.int)
             for TV in sorted(self.TVs)
        ])

        sentence_tokens = item["phonemes"]
        sentence_numerized = torch.tensor([
            self.vocabulary[token] for token in sentence_tokens
        ], dtype=torch.long)

        return (
            sentence_name,
            sentence_numerized,
            sentence_targets,
            sentence_tokens,
            critical_mask,
            sentence_critical_references,
            sentence_frames
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

    critical_references = [item[5] for item in batch]
    padded_critical_references = pad_sequence(critical_references, batch_first=True)
    padded_critical_references = padded_critical_references[sentences_sorted_indices]

    sentence_frames = [batch[i][6] for i in sentences_sorted_indices]

    return (
        sentences_ids,
        padded_sentence_numerized,
        padded_sentence_targets,
        len_sentences_sorted,
        phonemes,
        padded_critical_masks,
        padded_critical_references,
        sentence_frames
    )