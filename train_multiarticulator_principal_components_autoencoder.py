import argparse
import logging
import mlflow
import numpy as np
import os
import tempfile
import torch
import yaml

from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from vt_tools import *

from helpers import set_seeds, sequences_from_dict
from phoneme_to_articulation.principal_components import run_autoencoder_epoch, TRAIN, VALID
from phoneme_to_articulation.principal_components.dataset import PrincipalComponentsMultiArticulatorAutoencoderDataset
from phoneme_to_articulation.principal_components.evaluation import run_multiart_autoencoder_test
from phoneme_to_articulation.principal_components.losses import MultiArtRegularizedLatentsMSELoss
from phoneme_to_articulation.principal_components.models import MultiArticulatorAutoencoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

TMP_DIR = tempfile.mkdtemp(dir=RESULTS_DIR)


def main(
    datadir, n_epochs, batch_size, patience, learning_rate, weight_decay,
    train_seq_dict, valid_seq_dict, test_seq_dict, articulators_indices_dict,
    hidden_blocks, hidden_features, clip_tails=True, state_dict_fpath=None
):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    logging.info(f"Running on '{device.type}'")

    best_encoders_path = os.path.join(TMP_DIR, "best_encoders.pt")
    best_decoders_path = os.path.join(TMP_DIR, "best_decoders.pt")
    last_encoders_path = os.path.join(TMP_DIR, "last_encoders.pt")
    last_decoders_path = os.path.join(TMP_DIR, "last_decoders.pt")

    articulators = sorted(articulators_indices_dict.keys())

    model_kwargs = dict(
        in_features=100,
        indices_dict=articulators_indices_dict,
        hidden_blocks=hidden_blocks,
        hidden_features=hidden_features
    )

    autoencoder = MultiArticulatorAutoencoder(**model_kwargs)
    if state_dict_fpath is not None:
        state_dict = torch.load(state_dict_fpath, map_location=device)
        autoencoder.load_state_dict(state_dict)
    autoencoder.to(device)

    num_workers = 5
    train_sequences = sequences_from_dict(datadir, train_seq_dict)
    train_dataset = PrincipalComponentsMultiArticulatorAutoencoderDataset(
        datadir=datadir,
        sequences=train_sequences,
        articulators=articulators,
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
    valid_dataset = PrincipalComponentsMultiArticulatorAutoencoderDataset(
        datadir=datadir,
        sequences=valid_sequences,
        articulators=articulators,
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

    loss_fn = MultiArtRegularizedLatentsMSELoss(alpha=1e-2, indices_dict=articulators_indices_dict)
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
            criterion=loss_fn,
            device=device
        )

        mlflow.log_metrics({
            f"valid_{metric}": value for metric, value in info_valid.items()
        }, step=epoch)

        if info_valid["loss"] < best_metric:
            best_metric = info_valid["loss"]
            epochs_since_best = 0
            torch.save(autoencoder.encoders.state_dict(), best_encoders_path)
            torch.save(autoencoder.decoders.state_dict(), best_decoders_path)

            mlflow.log_artifact(best_encoders_path)
            mlflow.log_artifact(best_decoders_path)
        else:
            epochs_since_best += 1

        torch.save(autoencoder.encoders.state_dict(), last_encoders_path)
        torch.save(autoencoder.decoders.state_dict(), last_decoders_path)

        mlflow.log_artifact(last_encoders_path)
        mlflow.log_artifact(last_decoders_path)

        if epochs_since_best > patience:
            break

    test_sequences = sequences_from_dict(datadir, test_seq_dict)
    test_dataset = PrincipalComponentsMultiArticulatorAutoencoderDataset(
        datadir=datadir,
        sequences=test_sequences,
        articulators=articulators,
        sync_shift=0,
        framerate=55,
        clip_tails=clip_tails
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=set_seeds
    )

    best_autoencoder = MultiArticulatorAutoencoder(
        **model_kwargs
    )
    best_encoders_state_dict = torch.load(best_encoders_path, map_location=device)
    best_autoencoder.encoders.load_state_dict(best_encoders_state_dict)
    best_decoders_state_dict = torch.load(best_decoders_path, map_location=device)
    best_autoencoder.decoders.load_state_dict(best_decoders_state_dict)
    best_autoencoder.to(device)

    test_outputs_dir = os.path.join(TMP_DIR, "test_outputs")
    if not os.path.exists(test_outputs_dir):
        os.makedirs(test_outputs_dir)

    info_test = run_multiart_autoencoder_test(
        epoch=0,
        model=best_autoencoder,
        dataloader=test_dataloader,
        criterion=loss_fn,
        outputs_dir=test_outputs_dir,
        device=device
    )

    mlflow.log_metrics({
        f"test_{metric}": value for metric, value in info_test.items()
    }, step=epoch)

    mlflow.log_artifacts(test_outputs_dir, "test_outputs")


if __name__ == "__main__":
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", dest="config_filepath")
        args = parser.parse_args()

    with open(args.config_filepath) as f:
        cfg = yaml.safe_load(f)

    with mlflow.start_run():
        mlflow.log_params(cfg)
        mlflow.log_dict(cfg, "config.json")

        main(**cfg)
