import argparse
import logging
import mlflow
import numpy as np
import os
import torch
import yaml

from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from helpers import set_seeds, sequences_from_dict
from phoneme_to_articulation.principal_components import run_autoencoder_epoch, TRAIN, VALID
from phoneme_to_articulation.principal_components.dataset import PrincipalComponentsAutoencoderDataset
from phoneme_to_articulation.principal_components.evaluation import run_autoencoder_test
from phoneme_to_articulation.principal_components.losses import RegularizedLatentsMSELoss
from phoneme_to_articulation.principal_components.models import Autoencoder


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


def main(
    datadir, n_epochs, batch_size, patience, learning_rate, weight_decay,
    train_seq_dict, valid_seq_dict, test_seq_dict, articulator, n_components,
    clip_tails=True, state_dict_fpath=None, alpha=1e-2
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on '{device.type}'")

    best_encoder_path = os.path.join(RESULTS_DIR, "best_encoder.pt")
    best_decoder_path = os.path.join(RESULTS_DIR, "best_decoder.pt")
    last_encoder_path = os.path.join(RESULTS_DIR, "last_encoder.pt")
    last_decoder_path = os.path.join(RESULTS_DIR, "last_decoder.pt")

    autoencoder = Autoencoder(
        in_features=100,
        n_components=n_components
    )
    if state_dict_fpath is not None:
        state_dict = torch.load(state_dict_fpath, map_location=device)
        autoencoder.load_state_dict(state_dict)
    autoencoder.to(device)

    num_workers = 5
    train_sequences = sequences_from_dict(datadir, train_seq_dict)
    train_dataset = PrincipalComponentsAutoencoderDataset(
        datadir=datadir,
        sequences=train_sequences,
        articulator=articulator,
        sync_shift=0,
        framerate=55,
        clip_tails=clip_tails
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=set_seeds
    )

    valid_sequences = sequences_from_dict(datadir, valid_seq_dict)
    valid_dataset = PrincipalComponentsAutoencoderDataset(
        datadir=datadir,
        sequences=valid_sequences,
        articulator=articulator,
        sync_shift=0,
        framerate=55,
        clip_tails=clip_tails
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=set_seeds
    )

    loss_fn = RegularizedLatentsMSELoss(alpha=alpha)
    optimizer = Adam(autoencoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=int(len(train_dataloader)),
        epochs=n_epochs,
        anneal_strategy="linear"
    )

    best_metric = np.inf
    epochs_since_best = 0

    epochs = range(1, n_epochs + 1)
    for epoch in epochs:
        info_train = run_autoencoder_epoch(
            phase=TRAIN,
            epoch=epoch,
            model=autoencoder,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=loss_fn,
            device=device
        )

        mlflow.log_metrics({
            f"train_{metric}": value for metric, value in info_train.items()
        }, step=epoch)

        info_valid = run_autoencoder_epoch(
            phase=VALID,
            epoch=epoch,
            model=autoencoder,
            dataloader=valid_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=loss_fn,
            device=device
        )

        mlflow.log_metrics({
            f"valid_{metric}": value for metric, value in info_valid.items()
        }, step=epoch)

        if info_valid["loss"] < best_metric:
            best_metric = info_valid["loss"]
            epochs_since_best = 0
            torch.save(autoencoder.encoder.state_dict(), best_encoder_path)
            torch.save(autoencoder.decoder.state_dict(), best_decoder_path)

            mlflow.log_artifact(best_encoder_path)
            mlflow.log_artifact(best_decoder_path)
        else:
            epochs_since_best += 1

        torch.save(autoencoder.encoder.state_dict(), last_encoder_path)
        torch.save(autoencoder.decoder.state_dict(), last_decoder_path)

        mlflow.log_artifact(last_encoder_path)
        mlflow.log_artifact(last_decoder_path)

        if epochs_since_best > patience:
            break

    test_sequences = sequences_from_dict(datadir, test_seq_dict)
    test_dataset = PrincipalComponentsAutoencoderDataset(
        datadir=datadir,
        sequences=test_sequences,
        articulator=articulator,
        sync_shift=0,
        framerate=55,
        clip_tails=clip_tails
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=set_seeds
    )

    best_autoencoder = Autoencoder(
        in_features=100,
        n_components=n_components
    )
    best_encoder_state_dict = torch.load(best_encoder_path, map_location=device)
    best_autoencoder.encoder.load_state_dict(best_encoder_state_dict)
    best_decoder_state_dict = torch.load(best_decoder_path, map_location=device)
    best_autoencoder.decoder.load_state_dict(best_decoder_state_dict)
    best_autoencoder.to(device)

    test_outputs_dir = os.path.join(RESULTS_DIR, "test_outputs")
    if not os.path.exists(test_outputs_dir):
        os.makedirs(test_outputs_dir)

    info_test = run_autoencoder_test(
        epoch=0,
        model=best_autoencoder,
        dataloader=test_dataloader,
        criterion=loss_fn,
        outputs_dir=test_outputs_dir,
        device=device
    )

    mlflow.log_artifacts(test_outputs_dir, "test_outputs")


if __name__ == "__main__":
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", dest="config_filepath")
        args = parser.parse_args()

    with open(args.config_filepath) as f:
        cfg = yaml.safe_load(f)

    alpha = 1e-2
    with mlflow.start_run():
        mlflow.log_param(key="alpha", value=alpha)
        mlflow.log_params(cfg)
        mlflow.log_dict(cfg, "config.json")

        main(**cfg, alpha=alpha)
