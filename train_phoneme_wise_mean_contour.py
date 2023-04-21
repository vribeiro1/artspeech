import argparse
import mlflow
import os
import pandas as pd
import shutil
import tempfile
import ujson
import yaml

from sacred import Experiment
from sacred.observers import FileStorageObserver

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
    valid_seq_dict,
    test_seq_dict,
    vocab_filepath,
    articulators,
    state_dict_fpath=None,
    clip_tails=True,
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

    valid_sequences = sequences_from_dict(datadir, valid_seq_dict)
    valid_dataset = ArtSpeechDataset(
        datadir,
        database_name,
        valid_sequences,
        vocabulary,
        articulators,
        clip_tails=clip_tails,
    )

    if state_dict_fpath is None:
        save_to = os.path.join(RESULTS_DIR, "phoneme_wise_articulators.csv")
        df = train(train_dataset, save_to)
    else:
        df = pd.read_csv(state_dict_fpath)
        for articulator in train_dataset.articulators:
            df[articulator] = df[articulator].apply(eval)

    test_sequences = sequences_from_dict(datadir, test_seq_dict)
    test_dataset = ArtSpeechDataset(
        datadir,
        database_name,
        test_filepath,
        vocabulary,
        articulators,
        clip_tails=clip_tails,
    )

    test_outputs_dir = os.path.join(RESULTS_DIR, "test_outputs")
    if not os.path.exists(test_outputs_dir):
        os.makedirs(test_outputs_dir)
    test_results = test(test_dataset, df, test_outputs_dir)
    mlflow.log_artifact(test_outputs_dir)

    test_results_filepath = os.path.join(RESULTS_DIR, "test_results.json")
    with open(test_results_filepath, "w") as f:
        ujson.dump(test_results, f)
    mlflow.log_artifact(df_fitest_results_filepathlepath)

    results_item = {
        "exp": _run._id,
        "loss": test_results["loss"],
    }

    for articulator in test_dataset.articulators:
        results_item[f"p2cp_{articulator}"] = test_results[articulator]["p2cp"]
        results_item[f"med_{articulator}"] = test_results[articulator]["med"]
        results_item[f"x_corr_{articulator}"] = test_results[articulator]["x_corr"]
        results_item[f"y_corr_{articulator}"] = test_results[articulator]["y_corr"]

    df = pd.DataFrame([results_item])
    df_filepath = os.path.join(RESULTS_DIR, "test_results.csv")
    df.to_csv(df_filepath, index=False)
    mlflow.log_artifact(df_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_filepath")
    parser.add_argument("--experiment", dest="experiment_name", default="phoneme_wise_mean_contour")
    parser.add_argument("--run", dest="run_name", default=None)
    args = parser.parse_args()

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
