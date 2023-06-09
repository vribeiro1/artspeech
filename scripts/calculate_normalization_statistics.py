import argparse
import os
import numpy as np
import torch
import yaml

from random import sample
from tqdm import tqdm

from database_collector import DATABASE_COLLECTORS
from helpers import sequences_from_dict
from phoneme_to_articulation.principal_components.dataset import InputLoaderMixin
from settings import DATASET_CONFIG


def proc(item):
    datadir = item["datadir"]
    database_name = item["database_name"]
    subject = item["subject"]
    sequence = item["sequence"]
    frame_id = item["frame_id"]
    dataset_config = DATASET_CONFIG[database_name]

    articulator, _ = InputLoaderMixin.prepare_articulator_array(
        datadir,
        subject,
        sequence,
        frame_id,
        articulator_name,
        dataset_config
    )
    return articulator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_filepath")
    args = parser.parse_args()

    with open(args.config_filepath) as f:
        config = yaml.safe_load(f)

    database_name = config["database_name"]
    datadir = config["datadir"]
    save_to = config["save_to"]
    sequences_dict = config["sequences_dict"]
    articulators_list = config["articulators"]
    num_samples = config.get("num_samples")
    sequences = sequences_from_dict(datadir, sequences_dict)

    collector = DATABASE_COLLECTORS[database_name](datadir)
    sentence_data = collector.collect_data(sequences)

    data = []
    for sentence in sentence_data:
        for frame_id, phoneme in zip(sentence["frame_ids"], sentence["phonemes"]):
            data.append({
                "datadir": datadir,
                "database_name": database_name,
                "subject": sentence["subject"],
                "sequence": sentence["sequence"],
                "frame_id": frame_id,
                "phoneme": phoneme
            })

    if num_samples is not None:
        sampled_data = sample(data, num_samples)
    else:
        sampled_data = data

    for articulator_name in articulators_list:
        articulators = [proc(item) for item in tqdm(sampled_data, desc=articulator_name)]
        articulators = torch.stack(articulators, dim=0)
        train_mean = articulators.mean(axis=0).numpy()
        train_std = articulators.std(axis=0).numpy()

        if not os.path.exists(save_to):
            os.makedirs(save_to)
        with open(os.path.join(save_to, f"{articulator_name}_mean.npy"), "wb") as f:
            np.save(f, train_mean)
        with open(os.path.join(save_to, f"{articulator_name}_std.npy"), "wb") as f:
            np.save(f, train_std)
