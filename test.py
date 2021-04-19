import pdb

import argparse
import json
import os
import torch
import yaml

from torch.utils.data import DataLoader

from dataset import ArtSpeechDataset
from evaluation import run_test
from helpers import set_seeds
from loss import EuclideanDistanceLoss
from model import ArtSpeech


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(cfg["vocab_filepath"]) as f:
        tokens = json.load(f)
        vocabulary = {token: i for i, token in enumerate(tokens)}

    articulators = cfg["articulators"]
    n_articulators = len(articulators)

    test_dataset = ArtSpeechDataset(
        os.path.dirname(cfg["datadir"]),
        cfg["test_filepath"],
        vocabulary,
        articulators
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        worker_init_fn=set_seeds
    )

    best_model = ArtSpeech(len(vocabulary), n_articulators)
    state_dict = torch.load(cfg["state_dict_fpath"], map_location=device)
    best_model.load_state_dict(state_dict)
    best_model.to(device)

    test_outputs_dir = os.path.join(cfg["save_to"], "test_outputs")
    if not os.path.exists(test_outputs_dir):
        os.makedirs(test_outputs_dir)

    loss_fn = EuclideanDistanceLoss()

    test_results = run_test(
        epoch=0,
        model=best_model,
        dataloader=test_dataloader,
        criterion=loss_fn,
        outputs_dir=test_outputs_dir,
        articulators=articulators,
        device=device,
        regularize_out=True
    )

    test_results_filepath = os.path.join(cfg["save_to"], "test_results.json")
    with open(test_results_filepath, "w") as f:
        json.dump(test_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="cfg_filepath")
    args = parser.parse_args()

    with open(args.cfg_filepath) as f:
        cfg = yaml.safe_load(f.read())

    main(cfg)
