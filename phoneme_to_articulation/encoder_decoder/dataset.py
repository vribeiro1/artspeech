import pdb

import funcy
import os
import torch

from functools import lru_cache
from glob import glob
from tgt.io import read_textgrid
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from vt_shape_gen.helpers import load_articulator_array
from vt_tools import UPPER_INCISOR

from phoneme_to_articulation.tail_clipper import TailClipper
from settings import DatasetConfig
from video import Video

phonemes_per_TV = {
    "LA": lambda p: p in ["p", "b", "m"],
    "TTCD": lambda p: p in ["l", "d", "n", "t"],
    "TBCD": lambda p: p in ["k", "g"],
    "VEL": lambda p: False
}


def pad_sequence_collate_fn(batch):
    sentences_ids = [item[0] for item in batch]

    sentence_numerized = [item[1] for item in batch]
    len_sentences = torch.tensor(funcy.lmap(len, sentence_numerized), dtype=torch.int)
    len_sentences_sorted, sentences_sorted_indices = len_sentences.sort(descending=True)
    padded_sentence_numerized = pad_sequence(sentence_numerized, batch_first=True)
    padded_sentence_numerized = padded_sentence_numerized[sentences_sorted_indices]

    sentence_targets = [item[2] for item in batch]
    padded_sentence_targets = pad_sequence(sentence_targets, batch_first=True)
    padded_sentence_targets = padded_sentence_targets[sentences_sorted_indices]

    phonemes = [batch[i][3] for i in sentences_sorted_indices]

    critical_masks = [item[5].T for item in batch]
    padded_critical_masks = pad_sequence(critical_masks, batch_first=True)
    padded_critical_masks = padded_critical_masks[sentences_sorted_indices]
    padded_critical_masks = padded_critical_masks.permute(0, 2, 1)

    sentence_frames = [batch[i][6] for i in sentences_sorted_indices]

    return (
        sentences_ids,
        padded_sentence_numerized,
        padded_sentence_targets,
        len_sentences_sorted,
        phonemes,
        padded_critical_masks,
        sentence_frames
    )


@lru_cache(maxsize=None)
def cached_load_articulator_array(filepath, norm_value):
    return torch.from_numpy(load_articulator_array(filepath, norm_value)).type(torch.float)


def collect_data(datadir, sequences, sync_shift, framerate, required_articulators):
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

            has_all_required_articulators = all([
                all([
                    os.path.exists(
                        os.path.join(seq_dir, "inference_contours", f"{frame_id}_{articulator}.npy")
                    ) for articulator in required_articulators
                ]) for frame_id in sentence_frame_ids
            ])

            start_str = "%.04f" % sentence_interval.start_time
            end_str = "%.04f" % sentence_interval.end_time
            sentence_name = f"vol_{subject}_{sequence}-{start_str}_{end_str}"

            data.append({
                "subject": subject,
                "sequence": sequence,
                "sentence_name": sentence_name,
                "n_frames": len(sentence_frame_ids),
                "frame_ids": sentence_frame_ids,
                "phonemes": sentence_phonemes,
                "has_all": has_all_required_articulators
            })

    return data


class ArtSpeechDataset(Dataset):
    def __init__(
        self, datadir, sequences, vocabulary, articulators, n_samples=50,
        sync_shift=0, framerate=DatasetConfig.FRAMERATE, clip_tails=False,
        TVs=None
    ):
        self.vocabulary = vocabulary
        self.datadir = datadir
        self.articulators = sorted(articulators)
        self.n_articulators = len(articulators)
        self.n_samples = n_samples
        self.clip_tails = clip_tails
        self.TVs = TVs or []

        data = collect_data(datadir, sequences, sync_shift, framerate, articulators)
        self.data = funcy.lfilter(lambda d: d["has_all"], data)

    def prepare_articulator_array(self, subject, sequence, frame_id, articulator):
        fp_articulator = os.path.join(
            self.datadir, subject, sequence, "inference_contours", f"{frame_id}_{articulator}.npy"
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

            tail_clip_method_name = f"clip_{articulator.replace('-', '_')}_tails"
            tail_clip_method = getattr(TailClipper, tail_clip_method_name, None)

            if tail_clip_method:
                articulator_array = tail_clip_method(articulator_array, **tail_clip_refs)

        articulator_array = articulator_array.T

        return articulator_array

    def get_frame_coordinate_system_reference(self, subject, sequence, frame_id):
        fp_coord_system_reference = os.path.join(
            self.datadir, subject, sequence, "inference_contours", f"{frame_id}_{UPPER_INCISOR}.npy"
        )

        coord_system_reference_array = cached_load_articulator_array(
            fp_coord_system_reference,
            norm_value=DatasetConfig.RES
        ).T

        return coord_system_reference_array

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        sentence_name = item["sentence_name"]
        subject = item["subject"]
        sequence = item["sequence"]
        frame_ids = item["frame_ids"]

        sentence_targets = torch.zeros(size=(0, self.n_articulators, 2, self.n_samples))
        sentence_references = torch.zeros(size=(0, 2, self.n_samples))
        for frame_id in item["frame_ids"]:
            coord_system_reference_array = self.get_frame_coordinate_system_reference(
                subject, sequence, frame_id
            )
            coord_system_reference = coord_system_reference_array[:, -1]
            coord_system_reference = coord_system_reference.unsqueeze(dim=-1)

            coord_system_reference_array = coord_system_reference_array - coord_system_reference
            coord_system_reference_array[0, :] = coord_system_reference_array[0, :] + 0.3
            coord_system_reference_array[1, :] = coord_system_reference_array[1, :] + 0.3

            frame_targets = torch.stack([
                self.prepare_articulator_array(subject, sequence, frame_id, articulator)
                for articulator in self.articulators
            ], dim=0).unsqueeze(dim=0)

            frame_targets = frame_targets - coord_system_reference
            frame_targets[..., 0, :] = frame_targets[..., 0, :] + 0.3
            frame_targets[..., 1, :] = frame_targets[..., 1, :] + 0.3

            sentence_targets = torch.cat([sentence_targets, frame_targets], dim=0)
            coord_system_reference_array = coord_system_reference_array.unsqueeze(dim=0)
            sentence_references = torch.cat([sentence_references, coord_system_reference_array], dim=0)

        critical_masks = torch.stack([
            torch.tensor([
                int(phonemes_per_TV[TV](phoneme)) for phoneme in item["phonemes"]
            ], dtype=torch.int)
            for TV in self.TVs
        ])

        sentence_targets = sentence_targets.type(torch.float)
        sentence_references = sentence_references.type(torch.float)

        sentence_tokens = item["phonemes"]
        sentence_numerized = torch.tensor([
            self.vocabulary[token] for token in sentence_tokens
        ], dtype=torch.long)

        return (
            sentence_name,
            sentence_numerized,
            sentence_targets,
            sentence_tokens,
            sentence_references,
            critical_masks,
            frame_ids
        )
