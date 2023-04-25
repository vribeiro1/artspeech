import pdb

import argparse
import os
import pandas as pd
import torch
import yaml
import ujson

from torch.utils.data import DataLoader

from helpers import set_seeds, sequences_from_dict
from phoneme_recognition import UNKNOWN
from phoneme_to_articulation.metrics import EuclideanDistance
from phoneme_to_articulation.encoder_decoder.dataset import ArtSpeechDataset, pad_sequence_collate_fn
from phoneme_to_articulation.encoder_decoder.evaluation import run_test
from phoneme_to_articulation.encoder_decoder.models import ArtSpeech
from phoneme_to_articulation.metrics import EuclideanDistance


def main(
    datadir,
    database_name,
    batch_size,
    test_seq_dict,
    state_dict_fpath,
    vocab_filepath,
    articulators,
    save_to,
    clip_tails=True,
    num_workers=0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    default_tokens = [UNKNOWN]
    vocabulary = {token: i for i, token in enumerate(default_tokens)}
    with open(vocab_filepath) as f:
        tokens = ujson.load(f)
        for i, token in enumerate(tokens, start=len(vocabulary)):
            vocabulary[token] = i

    test_sequences = sequences_from_dict(datadir, test_seq_dict)
    test_dataset = ArtSpeechDataset(
        datadir,
        database_name,
        test_sequences,
        vocabulary,
        articulators,
        clip_tails=clip_tails
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=set_seeds,
        collate_fn=pad_sequence_collate_fn
    )

    num_articulators = len(articulators)
    best_model = ArtSpeech(len(vocabulary), num_articulators, gru_dropout=0.2)
    state_dict = torch.load(state_dict_fpath, map_location=device)
    best_model.load_state_dict(state_dict)
    best_model.to(device)

    test_outputs_dir = os.path.join(save_to, "test_outputs")
    if not os.path.exists(test_outputs_dir):
        os.makedirs(test_outputs_dir)

    loss_fn = EuclideanDistance("none")

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

    test_results_filepath = os.path.join(save_to, "test_results.json")
    with open(test_results_filepath, "w") as f:
        ujson.dump(test_results, f)

    results_item = {
        "exp": None,
        "loss": test_results["loss"],
    }

    for articulator in test_dataset.articulators:
        results_item[f"p2cp_{articulator}"] = test_results[articulator]["p2cp"]
        results_item[f"p2cp_mm_{articulator}"] = test_results[articulator]["p2cp_mm"]
        results_item[f"med_{articulator}"] = test_results[articulator]["med"]
        results_item[f"med_mm_{articulator}"] = test_results[articulator]["med_mm"]
        results_item[f"x_corr_{articulator}"] = test_results[articulator]["x_corr"]
        results_item[f"y_corr_{articulator}"] = test_results[articulator]["y_corr"]

    df = pd.DataFrame([results_item])
    df_filepath = os.path.join(save_to, "test_results.csv")
    df.to_csv(df_filepath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="cfg_filepath")
    args = parser.parse_args()

    with open(args.cfg_filepath) as f:
        cfg = yaml.safe_load(f.read())

    main(**cfg)
