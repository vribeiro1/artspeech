import pdb

import argparse
import funcy
import json
import os
from numpy import save
import yaml

from glob import glob
from sklearn.model_selection import KFold
from tgt.io3 import read_textgrid
from tqdm import tqdm
from vt_tracker import (
    ARYTENOID_MUSCLE,
    EPIGLOTTIS,
    LOWER_INCISOR,
    LOWER_LIP,
    PHARYNX,
    SOFT_PALATE,
    THYROID_CARTILAGE,
    TONGUE,
    UPPER_INCISOR,
    UPPER_LIP,
    VOCAL_FOLDS
)

from video import Video

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ARTICULATORS = sorted([
    ARYTENOID_MUSCLE,
    EPIGLOTTIS,
    LOWER_INCISOR,
    LOWER_LIP,
    PHARYNX,
    SOFT_PALATE,
    THYROID_CARTILAGE,
    TONGUE,
    UPPER_INCISOR,
    UPPER_LIP,
    VOCAL_FOLDS
])

def rm_basedir(filepath, basedir):
    dir_to_remove = os.path.dirname(basedir) + os.sep

    if filepath is None:
        return

    return filepath.replace(dir_to_remove, "")


def exists_or_none(filepath):
    return filepath if os.path.isfile(filepath) else None


def get_contours_for_frames(datadir, subject, sequence, instance_numbers):
    contours = {}
    for inumber in instance_numbers:
        articulator_filepath = lambda art: os.path.join(
            datadir,
            subject,
            sequence,
            "inference_contours",
            f"{'%04d' % inumber}_{art}.npy"
        )

        contours[inumber] = {
            art: rm_basedir(exists_or_none(articulator_filepath(art)), datadir)
            for art in ARTICULATORS
        }

    return contours


def get_instance_number_from_filepath(filepath):
    basename = os.path.basename(filepath)
    inumber_str, _ = basename.split(".")
    inumber = int(inumber_str)

    return inumber


def make_sentences_data(sequences, datadir, sync_shift=0, framerate=50, progbar_desc=""):
    vocabulary = set()
    data = []

    for subject, sequence in tqdm(sequences, desc=progbar_desc):
        frames_filepaths = sorted(glob(os.path.join(datadir, subject, sequence, "dicoms", "*.dcm")))
        sync_frames_filepaths = frames_filepaths[sync_shift:]

        audio_filepath = os.path.join(datadir, subject, sequence, f"vol_{subject}_{sequence}.wav")
        textgrid_filepath = os.path.join(datadir, subject, sequence, f"vol_{subject}_{sequence}.textgrid")

        video = Video(sync_frames_filepaths, audio_filepath, framerate=framerate)
        textgrid = read_textgrid(textgrid_filepath)

        phone_intervals = textgrid.get_tier_by_name("PhonTier")
        sentence_intervals = textgrid.get_tier_by_name("SentenceTier")

        for sentence_interval in sentence_intervals:
            start_time = sentence_interval.start_time
            end_time = sentence_interval.end_time
            in_interval = lambda interval: interval.start_time >= start_time and interval.end_time <= end_time

            phonemes = [
                {
                    "text": phone.text,
                    "start_time": phone.start_time,
                    "end_time": phone.end_time,
                    "n_frames": len(video.get_frames_interval(
                        phone.start_time,
                        phone.end_time,
                        load_frames=False)
                    )
                }
                for phone in filter(in_interval, phone_intervals)
            ]

            if len(phonemes) == 0:
                continue

            [vocabulary.add(p["text"]) for p in phonemes]

            frames = video.get_frames_interval(start_time, end_time)
            instance_numbers = funcy.lmap(get_instance_number_from_filepath, frames)
            contours = get_contours_for_frames(datadir, subject, sequence, instance_numbers)

            item = {
                "start_time": start_time,
                "end_time": end_time,
                "phonemes": phonemes,
                "frames_filepaths": funcy.lmap(lambda fp: rm_basedir(fp, datadir), frames),
                "contours_filepaths": contours,
                "metadata": {
                    "subject": subject,
                    "sequence": sequence,
                    "audio_filepath": rm_basedir(audio_filepath, datadir),
                    "textgrid_filepath": rm_basedir(textgrid_filepath, datadir)
                }
            }

            data.append(item)

    return data, vocabulary


def make_kfold(cfg):
    vocabulary = set()

    prefix = cfg["prefix"]
    suffix = cfg["suffix"]
    datadir = cfg["datadir"]
    sync_shift = cfg["sync_shift"]
    framerate = cfg["framerate"]

    kfold_dirname = os.path.join(BASE_DIR, "data", "kfold")
    if not os.path.exists(kfold_dirname):
        os.makedirs(kfold_dirname)

    test_subj_sequences = cfg["test_seqs"]
    test_sequences = []
    for subj, seqs in test_subj_sequences.items():
        test_sequences.extend([(subj, seq) for seq in seqs])

    print(f"""
Number of test sequences: {len(test_sequences)}
""")

    test_data, test_vocab = make_sentences_data(
        test_sequences,
        datadir,
        sync_shift=sync_shift,
        framerate=framerate,
        progbar_desc="Making test"
    )
    vocabulary.update(test_vocab)

    filename = f"{prefix}test{suffix}.json"
    with open(os.path.join(kfold_dirname, filename), "w") as f:
        json.dump(test_data, f)

    n = 5
    kfold = KFold(n_splits=n)

    train_valid_subj_sequences = cfg["train_valid_seqs"]
    train_valid_sequences = []
    for subj, seqs in train_valid_subj_sequences.items():
        train_valid_sequences.extend([(subj, seq) for seq in seqs])

    for fold_i, (train_idx, valid_idx) in enumerate(kfold.split(train_valid_sequences)):
        train_sequences = [train_valid_sequences[i] for i in train_idx]
        valid_sequences = [train_valid_sequences[i] for i in valid_idx]

        print(f"""
Fold {fold_i + 1}
Number of train sequences: {len(train_sequences)}
Number of validation sequences: {len(valid_sequences)}
""")

        train_data, train_vocab = make_sentences_data(
            train_sequences,
            datadir,
            sync_shift=sync_shift,
            framerate=framerate,
            progbar_desc=f"Making fold {fold_i} train"
        )
        vocabulary.update(train_vocab)

        valid_data, valid_vocab = make_sentences_data(
            valid_sequences,
            datadir,
            sync_shift=sync_shift,
            framerate=framerate,
            progbar_desc=f"Making fold {fold_i} valid"
        )
        vocabulary.update(valid_vocab)

        filename = f"{prefix}train_fold_{fold_i + 1}{suffix}.json"
        with open(os.path.join(kfold_dirname, filename), "w") as f:
            json.dump(train_data, f)

        filename = f"{prefix}valid_fold_{fold_i + 1}{suffix}.json"
        with open(os.path.join(kfold_dirname, filename), "w") as f:
            json.dump(valid_data, f)

    return vocabulary


def make_data_efficiency(cfg):
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="cfg_filepath")
    args = parser.parse_args()

    with open(args.cfg_filepath) as f:
        cfg = yaml.safe_load(f.read())

    scheme = cfg["scheme"]
    save_to = cfg["save_to"]
    if not os.path.exists(save_to):
        os.makedirs(save_to)

    if scheme == "kfold":
        vocabulary = make_kfold(cfg)
    elif scheme == "data-efficiency":
        vocabulary = make_data_efficiency(cfg)
    else:
        raise Exception(f"Unavailable scheme '{scheme}'")

    vocab_len = len(vocabulary)
    print(f"Vocabulary length: {vocab_len}")

    suffix = cfg["suffix"]
    with open(os.path.join(save_to, f"vocabulary{suffix}.json"), "w") as f:
        json.dump(sorted(vocabulary), f)
