import pdb

import argparse
import json
import logging
import mlflow
import numpy as np
import os
import pandas as pd
import tempfile
import torch
import yaml

from sacred import Experiment
from sacred.observers import FileStorageObserver
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from helpers import set_seeds, sequences_from_dict
from metrics import pearsons_correlation
from phoneme_to_articulation.encoder_decoder.dataset import ArtSpeechDataset, pad_sequence_collate_fn
from phoneme_to_articulation.encoder_decoder.evaluation import run_test
from phoneme_to_articulation.encoder_decoder.models import ArtSpeech
from phoneme_to_articulation.metrics import EuclideanDistance
from settings import BASE_DIR, TRAIN, VALID, TEST

TMPFILES = os.path.join(BASE_DIR, "tmp")
TMP_DIR = tempfile.mkdtemp(dir=TMPFILES)
RESULTS_DIR = os.path.join(TMP_DIR, "results")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


def run_epoch(
    phase,
    epoch,
    model,
    dataloader,
    optimizer,
    criterion,
    articulators,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training = phase == TRAIN

    if training:
        model.train()
    else:
        model.eval()

    losses = []
    x_corrs = [[] for _ in articulators]
    y_corrs = [[] for _ in articulators]
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - {phase}")
    for i, (_, sentence, targets, lengths, _, _) in enumerate(progress_bar):
        sentence = sentence.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            outputs = model(sentence, lengths)
            loss = criterion(outputs, targets)

            if training:
                loss.backward()
                optimizer.step()

            x_corr, y_corr = pearsons_correlation(outputs, targets)
            x_corr = x_corr.mean(dim=-1)[0]
            y_corr = y_corr.mean(dim=-1)[0]

            for i, _ in enumerate(articulators):
                x_corrs[i].append(x_corr[i].item())
                y_corrs[i].append(y_corr[i].item())

            losses.append(loss.item())
            progress_bar.set_postfix(
                loss=np.mean(losses),
                x_corr_1st=np.mean(x_corrs[0]),
                y_corr_1st=np.mean(y_corrs[0])
            )

    mean_loss = np.mean(losses)
    info = {
        "loss": mean_loss
    }

    info.update({
        art: {
            "x_corr": np.mean(x_corrs[i_art]),
            "y_corr": np.mean(y_corrs[i_art])
        }
        for i_art, art in enumerate(articulators)
    })

    return info


def main(
    datadir,
    num_epochs,
    batch_size,
    patience,
    learning_rate,
    weight_decay,
    train_seq_dict,
    valid_seq_dict,
    test_seq_dict,
    vocab_filepath,
    articulators,
    clip_tails=True,
    state_dict_filepath=None,
    checkpoint_filepath=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on '{device.type}'")

    best_model_path = os.path.join(RESULTS_DIR, "best_model.pt")
    last_model_path = os.path.join(RESULTS_DIR, "last_model.pt")
    save_checkpoint_path = os.path.join(RESULTS_DIR, "checkpoint.pt")

    outputs_dir = os.path.join(RESULTS_DIR, "outputs")
    if not os.path.exists(outputs_dir):
        os.mkdir(outputs_dir)

    with open(vocab_filepath) as f:
        tokens = json.load(f)
        vocabulary = {token: i for i, token in enumerate(tokens)}

    n_articulators = len(articulators)

    model = ArtSpeech(len(vocabulary), n_articulators, gru_dropout=0.2)
    if state_dict_filepath is not None:
        state_dict = torch.load(state_dict_filepath, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)

    loss_fn = EuclideanDistance()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=0.1,
        patience=10,
    )

    num_workers=5
    train_sequences = sequences_from_dict(datadir, train_seq_dict)
    train_dataset = ArtSpeechDataset(
        datadir,
        train_sequences,
        vocabulary,
        articulators,
        clip_tails=clip_tails
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=set_seeds,
        collate_fn=pad_sequence_collate_fn
    )

    valid_sequences = sequences_from_dict(datadir, valid_seq_dict)
    valid_dataset = ArtSpeechDataset(
        datadir,
        valid_sequences,
        vocabulary,
        articulators,
        clip_tails=clip_tails
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=set_seeds,
        collate_fn=pad_sequence_collate_fn
    )

    info = {}
    epochs = range(1, num_epochs + 1)
    best_metric = np.inf
    epochs_since_best = 0

    if checkpoint_filepath is not None:
        checkpoint = torch.load(checkpoint_filepath, map_location=device)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        epoch = checkpoint["epoch"]
        epochs = range(epoch, num_epochs + 1)
        best_metric = checkpoint["best_metric"]
        epochs_since_best = checkpoint["epochs_since_best"]
        best_model_path = checkpoint["best_model_path"]
        last_model_path = checkpoint["last_model_path"]

        print(f"""
Loaded checkpoint -- Launching training from epoch {epoch} with best metric
so far {best_metric} seen {epochs_since_best} epochs ago.
""")

    for epoch in epochs:
        info_train = run_epoch(
            phase=TRAIN,
            epoch=epoch,
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=loss_fn,
            articulators=train_dataset.articulators,
            device=device
        )

        mlflow.log_metrics(
            {"train_loss": info_train["loss"]},
            step=epoch
        )

        info_valid = run_epoch(
            phase=VALID,
            epoch=epoch,
            model=model,
            dataloader=valid_dataloader,
            optimizer=optimizer,
            criterion=loss_fn,
            articulators=valid_dataset.articulators,
            device=device
        )

        mlflow.log_metrics(
            {"valid_loss": info_valid["loss"]},
            step=epoch
        )

        scheduler.step(info_valid["loss"])

        if info_valid["loss"] < best_metric:
            best_metric = info_valid["loss"]
            torch.save(model.state_dict(), best_model_path)
            mlflow.log_artifact(best_model_path)
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        torch.save(model.state_dict(), last_model_path)
        mlflow.log_artifact(last_model_path)

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_metric": best_metric,
            "epochs_since_best": epochs_since_best,
            "best_model_path": best_model_path,
            "last_model_path": last_model_path
        }
        torch.save(checkpoint, save_checkpoint_path)
        mlflow.log_artifact(save_checkpoint_path)

        print(f"""
Finished training epoch {epoch}
Best metric: {'%0.4f' % best_metric}, Epochs since best: {epochs_since_best}
""")

        if epochs_since_best > patience:
            break

    test_sequences = sequences_from_dict(datadir, test_seq_dict)
    test_dataset = ArtSpeechDataset(
        datadir,
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

    best_model = ArtSpeech(len(vocabulary), n_articulators, gru_dropout=0.2)
    state_dict = torch.load(best_model_path, map_location=device)
    best_model.load_state_dict(state_dict)
    best_model.to(device)

    test_outputs_dir = os.path.join(RESULTS_DIR, "test_outputs")
    if not os.path.exists(test_outputs_dir):
        os.makedirs(test_outputs_dir)

    test_results = run_test(
        epoch=0,
        model=best_model,
        dataloader=test_dataloader,
        criterion=loss_fn,
        outputs_dir=test_outputs_dir,
        articulators=test_dataset.articulators,
        device=device,
        regularize_out=True
    )

    test_results_filepath = os.path.join(RESULTS_DIR, "test_results.json")
    with open(test_results_filepath, "w") as f:
        json.dump(test_results, f)
    mlflow.log_artifact(test_results_filepath)

    results_item = {
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
    parser.add_argument("--experiment", dest="experiment_name", default="phoneme_to_articulation")
    parser.add_argument("--run", dest="run_name", default=None)
    args = parser.parse_args()

    with open(args.config_filepath) as f:
        cfg = yaml.safe_load(f)

    experiment = mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=args.run_name
    ):
        mlflow.log_params(cfg)
        mlflow.log_dict(cfg, "config.json")
        main(**cfg)
