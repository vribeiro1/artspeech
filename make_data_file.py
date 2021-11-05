import argparse
import funcy
import json
import os
import xml.etree.ElementTree as ET
import yaml

from glob import glob
from sklearn.model_selection import KFold
from tgt.io import read_textgrid
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
ARTSPEECH_DIR = "/home/vsouzari/Documents/loria/datasets/ArtSpeech_Database"

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


def exists_or_none(filepath):
    return filepath if os.path.isfile(filepath) else None


def rm_basedir(filepath):
    dir_to_remove = os.path.dirname(ARTSPEECH_DIR) + os.sep

    if filepath is None:
        return

    return filepath.replace(dir_to_remove, "")


def get_contours_for_frames(subject, sequence, instance_numbers):
    contours = {}
    for inumber in instance_numbers:
        articulator_filepath = lambda art: os.path.join(
            ARTSPEECH_DIR,
            subject,
            sequence,
            "inference_contours",
            f"{'%04d' % inumber}_{art}.npy"
        )

        contours[inumber] = {
            art: rm_basedir(exists_or_none(articulator_filepath(art)))
            for art in ARTICULATORS
        }

    return contours


def all_contours_exist(filepath):
    basename = os.path.basename(filepath)
    instance_number = int(basename.split(".")[0])

    dirname = os.path.dirname(os.path.dirname(filepath))
    sequence = os.path.basename(dirname)
    subject = os.path.basename(os.path.dirname(dirname))

    contours = get_contours_for_frames(subject, sequence, [instance_number])

    articulators = [
        contours[instance_number][art]
        for art in ARTICULATORS
    ]

    return all([fp is not None for fp in articulators])


def get_intervals_from_trs_file(filepath):
    etree = ET.parse(filepath)

    root = etree.getroot()
    episode = root.find("Episode")
    section = episode.find("Section")
    turn = section.find("Turn")
    syncs = turn.findall("Sync")
    sync_times = funcy.lmap(float, [sync.get("time") for sync in syncs])

    return list(zip(sync_times[0:-1], sync_times[1:]))


def get_instance_number_from_filepath(filepath):
    basename = os.path.basename(filepath)
    inumber_str, _ = basename.split(".")
    inumber = int(inumber_str)

    return inumber


def make_sentences_data(sequences, wav_suffix, textgrid_suffix, trs_suffix):
    vocabulary = set()
    data = []
    for subject, sequence in tqdm(sequences):
        frames_filepaths = sorted(glob(os.path.join(ARTSPEECH_DIR, subject, sequence, "dicoms", "*.dcm")))
        audio_filepath = os.path.join(ARTSPEECH_DIR, subject, sequence, f"{sequence}{wav_suffix}.wav")

        textgrid_filepath = os.path.join(ARTSPEECH_DIR, subject, sequence, f"{sequence}{textgrid_suffix}.textgrid")
        trs_filepath = os.path.join(ARTSPEECH_DIR, subject, sequence, f"{sequence}{trs_suffix}.trs")

        video = Video(frames_filepaths, audio_filepath)

        try:
            textgrid = read_textgrid(textgrid_filepath)
        except Exception as e:
            print(textgrid_filepath)
            raise e

        sentence_intervals = get_intervals_from_trs_file(trs_filepath)
        phones_intervals = textgrid.tiers[1].intervals

        for start_time, end_time in sentence_intervals:
            in_interval = lambda interval: interval.start_time >= start_time and interval.end_time <= end_time
            phonemes = [
                {
                    "text": phone.text,
                    "start_time": phone.start_time,
                    "end_time": phone.end_time,
                    "n_frames": len(video.get_frames_interval(phone.start_time, phone.end_time, load_frames=False))
                }
                for phone in filter(in_interval, phones_intervals)
            ]

            if len(phonemes) == 0:
                continue

            [vocabulary.add(p["text"]) for p in phonemes]

            _, frames = video.get_frames_interval(start_time, end_time)
            instance_numbers = funcy.lmap(get_instance_number_from_filepath, frames)
            contours = get_contours_for_frames(subject, sequence, instance_numbers)

            item = {
                "start_time": start_time,
                "end_time": end_time,
                "phonemes": phonemes,
                "frames_filepaths": funcy.lmap(rm_basedir, frames),
                "contours_filepaths": contours,
                "metadata": {
                    "subject": subject,
                    "sequence": sequence,
                    "audio_filepath": rm_basedir(audio_filepath),
                    "textgrid_filepath": rm_basedir(textgrid_filepath),
                    "trs_filepath": rm_basedir(trs_filepath)
                }
            }

            data.append(item)

    return data, vocabulary


def make_kfold(cfg):
    vocabulary = set()
    prefix = cfg["prefix"]
    suffix = cfg["suffix"]

    wav_suffix = cfg["wav_suffix"]
    textgrid_suffix = cfg["textgrid_suffix"]
    trs_suffix = cfg["trs_suffix"]

    test_subj_sequences = cfg["test_seqs"]
    test_sequences = []
    for subj, seqs in test_subj_sequences.items():
        test_sequences.extend([(subj, seq) for seq in seqs])
    test_data, test_vocab = make_sentences_data(
        test_sequences,
        wav_suffix,
        textgrid_suffix,
        trs_suffix
    )
    vocabulary.update(test_vocab)

    kfold_dirname = os.path.join(BASE_DIR, "data", "kfold")
    if not os.path.exists(kfold_dirname):
        os.makedirs(kfold_dirname)

    filename = f"{prefix}test{suffix}.json"
    with open(os.path.join(kfold_dirname, filename), "w") as f:
        json.dump(test_data, f)

    n = 7
    kfold = KFold(n_splits=n)
    train_valid_seqs = cfg["train_valid_seqs"]["1662"]
    for fold_i, (train_idx, valid_idx) in enumerate(kfold.split(train_valid_seqs)):
        if fold_i >= 5:
            break

        train_seqs = [train_valid_seqs[i] for i in train_idx]
        valid_seqs = [train_valid_seqs[i] for i in valid_idx]

        train_subj_sequences = {
            "1662": train_seqs
        }
        train_sequences = []
        for subj, seqs in train_subj_sequences.items():
            train_sequences.extend([(subj, seq) for seq in seqs])
        train_data, train_vocab = make_sentences_data(
            train_sequences,
            wav_suffix,
            textgrid_suffix,
            trs_suffix
        )
        vocabulary.update(train_vocab)

        filename = f"{prefix}train_fold_{fold_i + 1}{suffix}.json"
        with open(os.path.join(kfold_dirname, filename), "w") as f:
            json.dump(train_data, f)

        valid_subj_sequences = {
            "1662": valid_seqs
        }
        valid_sequences = []
        for subj, seqs in valid_subj_sequences.items():
            valid_sequences.extend([(subj, seq) for seq in seqs])
        valid_data, valid_vocab = make_sentences_data(
            valid_sequences,
            wav_suffix,
            textgrid_suffix,
            trs_suffix
        )
        vocabulary.update(valid_vocab)

        filename = f"{prefix}valid_fold_{fold_i + 1}{suffix}.json"
        with open(os.path.join(kfold_dirname, filename), "w") as f:
            json.dump(valid_data, f)

    return vocabulary

def make_data_efficiency(cfg):
    vocabulary = set()
    datadir = cfg["datadir"]
    prefix = cfg["prefix"]
    suffix = cfg["suffix"]

    wav_suffix = cfg["wav_suffix"]
    textgrid_suffix = cfg["textgrid_suffix"]
    trs_suffix = cfg["trs_suffix"]

    # Make validation
    valid_subj_sequences = cfg["valid_seqs"]

    valid_sequences = []
    for subj, seqs in valid_subj_sequences.items():
        use_seqs = seqs
        if len(seqs) == 0:
            # Use all sequences
            use_seqs = filter(
                lambda s: s.startswith("S") and os.path.isdir(os.path.join(datadir, subj, s)),
                os.listdir(os.path.join(datadir, subj))
            )

        valid_sequences.extend([(subj, seq) for seq in use_seqs])

    valid_data, valid_vocab = make_sentences_data(
        valid_sequences,
        wav_suffix,
        textgrid_suffix,
        trs_suffix
    )
    vocabulary.update(valid_vocab)

    filename = f"{prefix}valid{suffix}.json"
    with open(os.path.join(BASE_DIR, "data", "data_efficiency", filename), "w") as f:
        json.dump(valid_data, f)

    # Make test
    test_subj_sequences = cfg["test_seqs"]
    test_sequences = []
    for subj, seqs in test_subj_sequences.items():
        use_seqs = seqs
        if len(seqs) == 0:
            # Use all sequences
            use_seqs = filter(
                lambda s: s.startswith("S") and os.path.isdir(os.path.join(datadir, subj, s)),
                os.listdir(os.path.join(datadir, subj))
            )

        test_sequences.extend([(subj, seq) for seq in use_seqs])

    test_data, test_vocab = make_sentences_data(
        test_sequences,
        wav_suffix,
        textgrid_suffix,
        trs_suffix
    )
    vocabulary.update(test_vocab)

    data_eff_dirname = os.path.join(BASE_DIR, "data", "data_efficiency")
    if not os.path.exists(data_eff_dirname):
        os.makedirs(data_eff_dirname)

    filename = "{prefix}test{suffix}.json"
    with open(os.path.join(data_eff_dirname, filename), "w") as f:
        json.dump(test_data, f)

    # Make train
    for train_subj_sequences in cfg["train_seqs"]:
        train_sequences = []
        for subj, seqs in train_subj_sequences.items():
            use_seqs = seqs
            if len(seqs) == 0:
                # Use all sequences
                use_seqs = filter(
                    lambda s: s.startswith("S") and os.path.isdir(os.path.join(datadir, subj, s)),
                    os.listdir(os.path.join(datadir, subj))
                )

            train_sequences.extend([(subj, seq) for seq in use_seqs])

        # train_data = make_words_data(train_sequences)
        train_data, train_vocab = make_sentences_data(
            train_sequences,
            wav_suffix,
            textgrid_suffix,
            trs_suffix
        )
        vocabulary.update(train_vocab)

        filename = f"{prefix}train_{len(train_sequences)}_acquisitions{suffix}.json"
        with open(os.path.join(data_eff_dirname, filename), "w") as f:
            json.dump(train_data, f)

    return vocabulary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="cfg_filepath")
    args = parser.parse_args()

    with open(args.cfg_filepath) as f:
        cfg = yaml.safe_load(f.read())

    scheme = cfg["scheme"]

    if scheme == "kfold":
        vocabulary = make_kfold(cfg)
    elif scheme == "data-efficiency":
        vocabulary = make_data_efficiency(cfg)
    else:
        raise Exception(f"Unavailable scheme '{scheme}'")

    vocab_len = len(vocabulary)
    print(f"Vocabulary length: {vocab_len}")

    suffix = cfg["suffix"]
    with open(os.path.join(BASE_DIR, "data", f"vocabulary{suffix}.json"), "w") as f:
        json.dump(sorted(vocabulary), f)
