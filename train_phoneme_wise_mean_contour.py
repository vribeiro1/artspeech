import pdb

import argparse
import logging
import mlflow
import os
import pandas as pd
import shutil
import tempfile
import ujson
import yaml

from helpers import sequences_from_dict
from phoneme_recognition import UNKNOWN
from phoneme_to_articulation.encoder_decoder.dataset import ArtSpeechDataset
from phoneme_wise_mean_contour import train, test
from settings import BASE_DIR

TMPFILES = os.path.join(BASE_DIR, "tmp")
TMP_DIR = tempfile.mkdtemp(dir=TMPFILES)
RESULTS_DIR = os.path.join(TMP_DIR, "results")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


def main(
    database_name,
    datadir,
    train_seq_dict,
    test_seq_dict,
    vocab_filepath,
    articulators,
    state_dict_filepath=None,
    clip_tails=True,
    weighted=False,
):
    default_tokens = [UNKNOWN]
    vocabulary = {token: i for i, token in enumerate(default_tokens)}
    with open(vocab_filepath) as f:
        tokens = ujson.load(f)
        for i, token in enumerate(tokens, start=len(vocabulary)):
            vocabulary[token] = i

    train_sequences = sequences_from_dict(datadir, train_seq_dict)
    train_dataset = ArtSpeechDataset(
        datadir,
        database_name,
        train_sequences,
        vocabulary,
        articulators,
        clip_tails=clip_tails,
    )

    if state_dict_filepath is None:
        save_to = os.path.join(RESULTS_DIR, "phoneme_wise_articulators.csv")
        df = train(train_dataset, save_to=save_to, weighted=weighted)
        mlflow.log_artifact(save_to)
    else:
        df = pd.read_csv(state_dict_filepath)
        for articulator in train_dataset.articulators:
            df[articulator] = df[articulator].apply(eval)
        mlflow.log_artifact(state_dict_filepath)
    logging.info("Finished training phoneme wise mean contour")

    test_sequences = sequences_from_dict(datadir, test_seq_dict)
    test_dataset = ArtSpeechDataset(
        datadir,
        database_name,
        test_sequences,
        vocabulary,
        articulators,
        clip_tails=clip_tails,
    )

    test_outputs_dir = os.path.join(RESULTS_DIR, "test_outputs")
    if not os.path.exists(test_outputs_dir):
        os.makedirs(test_outputs_dir)

    test_results = test(
        test_dataset,
        df,
        test_outputs_dir,
        weighted=weighted
    )
    mlflow.log_artifact(test_outputs_dir)

    test_results_filepath = os.path.join(RESULTS_DIR, "test_results.json")
    with open(test_results_filepath, "w") as f:
        ujson.dump(test_results, f)
    mlflow.log_artifact(test_results_filepath)

    results_item = {
        "loss": test_results["loss"],
    }

    for articulator in test_dataset.articulators:
        results_item[f"x_corr_{articulator}"] = test_results[articulator]["x_corr"]
        results_item[f"y_corr_{articulator}"] = test_results[articulator]["y_corr"]

    df = pd.DataFrame([results_item])
    df_filepath = os.path.join(RESULTS_DIR, "test_results.csv")
    df.to_csv(df_filepath, index=False)
    mlflow.log_artifact(df_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_filepath")
    parser.add_argument("--mlflow", dest="mlflow_tracking_uri", default=None)
    parser.add_argument("--experiment", dest="experiment_name", default="phoneme_wise_mean_contour")
    parser.add_argument("--run", dest="run_name", default=None)
    args = parser.parse_args()

    if args.mlflow_tracking_uri is not None:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    with open(args.config_filepath) as f:
        cfg = yaml.safe_load(f)

    experiment = mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=args.run_name
    ):
        mlflow.log_artifact(args.config_filepath)
        try:
            main(**cfg)
        finally:
            shutil.rmtree(TMP_DIR)
