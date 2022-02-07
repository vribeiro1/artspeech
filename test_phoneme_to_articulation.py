import pdb

import argparse
import json
import os
import pandas as pd
import torch
import yaml

from torch.utils.data import DataLoader

from helpers import set_seeds
from loss import EuclideanDistanceLoss
from phoneme_to_articulation.encoder_decoder.dataset import ArtSpeechDataset, pad_sequence_collate_fn
from phoneme_to_articulation.encoder_decoder.evaluation import run_test
from phoneme_to_articulation.encoder_decoder.models import ArtSpeech


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
        articulators,
        p_aug=0.,
        lazy_load=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        worker_init_fn=set_seeds,
        collate_fn=pad_sequence_collate_fn
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

    results_item = {
        "exp": None,
        "loss": test_results["loss"],
    }

    for articulator in test_dataset.articulators:
        results_item[f"p2cp_{articulator}"] = test_results[articulator]["p2cp"]
        results_item[f"med_{articulator}"] = test_results[articulator]["med"]
        results_item[f"x_corr_{articulator}"] = test_results[articulator]["x_corr"]
        results_item[f"y_corr_{articulator}"] = test_results[articulator]["y_corr"]

    df = pd.DataFrame([results_item])
    df_filepath = os.path.join(cfg["save_to"], "test_results.csv")
    df.to_csv(df_filepath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="cfg_filepath")
    args = parser.parse_args()

    with open(args.cfg_filepath) as f:
        cfg = yaml.safe_load(f.read())

    main(cfg)
