import pdb

import funcy
import numpy as np
import os
import torch

from functools import lru_cache
from glob import glob
from tgt import read_textgrid
from torch.utils.data import Dataset
from tqdm import tqdm
from vt_tools import UPPER_INCISOR
from vt_shape_gen.helpers import load_articulator_array

from phoneme_to_articulation.tail_clipper import TailClipper
from phoneme_to_articulation.transforms import Normalize
from settings import BASE_DIR, DatasetConfig
from video import Video


@lru_cache(maxsize=None)
def cached_load_articulator_array(filepath, norm_value):
    return load_articulator_array(filepath, norm_value)


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
            sentence_phone_intervals = funcy.lfilter(lambda phone: (
                phone.start_time >= sentence_interval.start_time and
                phone.end_time <= sentence_interval.end_time
            ), phone_tier)

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
        self.data = funcy.lflatten([(
            {
                "subject": sentence["subject"],
                "sequence": sentence["sequence"],
                "frame_id": frame_id
            }
            for frame_id in sentence["frame_ids"]
        ) for sentence in sentence_data])

        mean = torch.from_numpy(np.load(os.path.join(BASE_DIR, "data", f"{articulator}_mean.npy")))
        std = torch.from_numpy(np.load(os.path.join(BASE_DIR, "data", f"{articulator}_std.npy")))
        self.normalize = Normalize(mean, std)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        frame_name = f"{item['subject']}_{item['sequence']}_{item['frame_id']}"

        fp_articulator = os.path.join(
            self.datadir,
            item["subject"],
            item["sequence"],
            "inference_contours",
            f"{item['frame_id']}_{self.articulator}.npy"
        )

        articulator_array = cached_load_articulator_array(
            fp_articulator,
            norm_value=DatasetConfig.RES
        )

        if self.clip_tails:
            tail_clip_refs = {}
            for reference in TailClipper.TAIL_CLIP_REFERENCES:
                fp_reference = os.path.join(
                    self.datadir, item["subject"], item["sequence"], "inference_contours",
                    f"{item['frame_id']}_{reference}.npy"
                )

                reference_array = cached_load_articulator_array(
                    fp_reference, norm_value=DatasetConfig.RES
                )

                tail_clip_refs[reference.replace("-", "_")] = reference_array

            tail_clip_method = getattr(
                TailClipper, f"clip_{self.articulator.replace('-', '_')}_tails", None
            )

            if tail_clip_method:
                articulator_array = tail_clip_method(articulator_array, **tail_clip_refs)

        articulator_array = articulator_array.T

        # Centralize the targets using the upper incisor as the reference of the coordinates system
        fp_coord_system_reference = os.path.join(
            self.datadir, item["subject"], item["sequence"], "inference_contours",
            f"{item['frame_id']}_{UPPER_INCISOR}.npy"
        )

        coord_system_reference_array = cached_load_articulator_array(
            fp_coord_system_reference, norm_value=DatasetConfig.RES
        ).T
        coord_system_reference = coord_system_reference_array[:, -1]
        coord_system_reference = np.expand_dims(coord_system_reference, axis=-1)

        articulator_array = articulator_array - coord_system_reference
        articulator_array[0, :] = articulator_array[0, :] + 0.3
        articulator_array[1, :] = articulator_array[1, :] + 0.3

        articulator_tensor = torch.from_numpy(articulator_array)
        articulator_norm = self.normalize(articulator_tensor)
        n, m = articulator_norm.shape
        articulator_norm = articulator_norm.reshape(n * m).type(torch.float)

        return frame_name, articulator_norm


class PrincipalComponentsPhonemeToArticulationDataset(Dataset):
    def __init__(self, datadir, sequences, vocabulary, articulator, sync_shift, framerate, n_samples=50, clip_tails=True):
        self.datadir = datadir
        self.vocabulary = vocabulary
        self.articulator = articulator
        self.n_samples = n_samples
        self.clip_tails = clip_tails

        self.data = collect_data(datadir, sequences, sync_shift, framerate)

        mean = torch.from_numpy(np.load(os.path.join(BASE_DIR, "data", f"{articulator}_mean.npy")))
        std = torch.from_numpy(np.load(os.path.join(BASE_DIR, "data", f"{articulator}_std.npy")))
        self.normalize = Normalize(mean, std)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        sentence_name = item["sentence_name"]

        sentence_targets = torch.zeros(size=(0, 1, 2, self.n_samples))
        for frame_id in item["frame_ids"]:
            fp_articulator = os.path.join(
                self.datadir,
                item["subject"],
                item["sequence"],
                "inference_contours",
                f"{frame_id}_{self.articulator}.npy"
            )

            articulator_array = cached_load_articulator_array(
                fp_articulator,
                norm_value=DatasetConfig.RES
            )

            if self.clip_tails:
                tail_clip_refs = {}
                for reference in TailClipper.TAIL_CLIP_REFERENCES:
                    fp_reference = os.path.join(
                        self.datadir, item["subject"], item["sequence"], "inference_contours",
                        f"{frame_id}_{reference}.npy"
                    )

                    reference_array = cached_load_articulator_array(
                        fp_reference, norm_value=DatasetConfig.RES
                    )

                    tail_clip_refs[reference.replace("-", "_")] = reference_array

                tail_clip_method = getattr(
                    TailClipper, f"clip_{self.articulator.replace('-', '_')}_tails", None
                )

                if tail_clip_method:
                    articulator_array = tail_clip_method(articulator_array, **tail_clip_refs)

            articulator_array = torch.from_numpy(articulator_array).type(torch.float).T

            # Centralize the targets using the upper incisor as the reference of the coordinates system
            fp_coord_system_reference = os.path.join(
                self.datadir, item["subject"], item["sequence"], "inference_contours",
                f"{frame_id}_{UPPER_INCISOR}.npy"
            )

            coord_system_reference_array = cached_load_articulator_array(
                fp_coord_system_reference, norm_value=DatasetConfig.RES
            ).T
            coord_system_reference = torch.from_numpy(coord_system_reference_array[:, -1])
            coord_system_reference = coord_system_reference.unsqueeze(dim=-1)

            articulator_array = articulator_array - coord_system_reference
            articulator_array[0, :] = articulator_array[0, :] + 0.3
            articulator_array[1, :] = articulator_array[1, :] + 0.3

            articulator_norm = self.normalize(articulator_array)
            articulator_norm = articulator_norm.unsqueeze(dim=0).unsqueeze(dim=0)
            sentence_targets = torch.cat([sentence_targets, articulator_norm], dim=0)

        sentence_targets = sentence_targets.type(torch.float)

        sentence_tokens = item["phonemes"]
        sentence_numerized = torch.tensor([
            self.vocabulary[token] for token in sentence_tokens
        ], dtype=torch.long)

        return sentence_name, sentence_numerized, sentence_targets, sentence_tokens
