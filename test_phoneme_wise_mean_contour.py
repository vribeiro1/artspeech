import pdb

import argparse
import os
import pandas as pd
import ujson
import yaml

from helpers import sequences_from_dict
from phoneme_recognition import UNKNOWN
from phoneme_to_articulation.encoder_decoder.dataset import ArtSpeechDataset
from phoneme_to_articulation.phoneme_wise_mean_contour import test


def main(
    database_name,
    datadir,
    seq_dict,
    vocab_filepath,
    articulators,
    state_dict_filepath,
    save_to,
    clip_tails=True,
    weighted=False,
):
    default_tokens = [UNKNOWN]
    vocabulary = {token: i for i, token in enumerate(default_tokens)}
    with open(vocab_filepath) as f:
        tokens = ujson.load(f)
        for i, token in enumerate(tokens, start=len(vocabulary)):
            vocabulary[token] = i

    test_sequences = sequences_from_dict(datadir, seq_dict)
    test_dataset = ArtSpeechDataset(
        datadir,
        database_name,
        test_sequences,
        vocabulary,
        articulators,
        clip_tails=clip_tails,
    )

    df = pd.read_csv(state_dict_filepath)
    for articulator in test_dataset.articulators:
        df[articulator] = df[articulator].apply(eval)

    test_outputs_dir = os.path.join(save_to, "test_outputs")
    if not os.path.exists(test_outputs_dir):
        os.makedirs(test_outputs_dir)

    test_results = test(
        test_dataset,
        df,
        test_outputs_dir,
        weighted=weighted
    )

    test_results_filepath = os.path.join(save_to, "test_results.json")
    with open(test_results_filepath, "w") as f:
        ujson.dump(test_results, f)

    results_item = {
        "loss": test_results["loss"],
    }

    for articulator in test_dataset.articulators:
        results_item[f"x_corr_{articulator}"] = test_results[articulator]["x_corr"]
        results_item[f"y_corr_{articulator}"] = test_results[articulator]["y_corr"]

    df = pd.DataFrame([results_item])
    df_filepath = os.path.join(save_to, "test_results.csv")
    df.to_csv(df_filepath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_filepath")
    args = parser.parse_args()

    with open(args.config_filepath) as f:
        cfg = yaml.safe_load(f)

    main(**cfg)
