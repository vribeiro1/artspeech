import argparse
import logging
import mlflow
import numpy as np
import os
import tempfile
import torch
import yaml

from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader
from vt_tools import *

from helpers import set_seeds, sequences_from_dict
from phoneme_to_articulation.principal_components import run_autoencoder_epoch
from phoneme_to_articulation.principal_components.dataset import PrincipalComponentsMultiArticulatorAutoencoderDataset
from phoneme_to_articulation.principal_components.evaluation import run_multiart_autoencoder_test
from phoneme_to_articulation.principal_components.losses import MultiArtRegularizedLatentsMSELoss
from phoneme_to_articulation.principal_components.models import MultiArticulatorAutoencoder
from phoneme_to_articulation.principal_components.metrics import MeanP2CPDistance
from settings import BASE_DIR, DatasetConfig, TRAIN, VALID, TEST

TMPFILES = os.path.join(BASE_DIR, "tmp")
TMP_DIR = tempfile.mkdtemp(dir=TMPFILES)
RESULTS_DIR = os.path.join(TMP_DIR, "results")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


def reconstruction_error(outputs, targets, denorm_fn_dict, px_space=1, res=1):
    p2cp_fn = MeanP2CPDistance(reduction="mean")

    batch_size, num_articulators, n_features = outputs.shape
    outputs = outputs.reshape(batch_size, num_articulators, 2, n_features // 2)
    targets = targets.reshape(batch_size, num_articulators, 2, n_features // 2)

    p2cps = []
    for i, (_, denorm_fn) in enumerate(denorm_fn_dict.items()):
        outputs[:, i, :] = denorm_fn(outputs[:, i, :])
        targets[:, i, :] = denorm_fn(targets[:, i, :])

        p2cp = p2cp_fn(
            outputs[:, i, :].permute(0, 2, 1),
            targets[:, i, :].permute(0, 2, 1)
        )
        p2cp_mm = p2cp * px_space * res
        p2cps.append(p2cp_mm.item())
    return np.mean(p2cps)


def main(
    datadir,
    n_epochs,
    batch_size,
    patience,
    learning_rate,
    weight_decay,
    train_seq_dict,
    valid_seq_dict,
    test_seq_dict,
    model_params,
    alpha,
    clip_tails=True,
    state_dict_fpath=None,
    num_workers=0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on '{device.type}'")

    best_encoders_path = os.path.join(RESULTS_DIR, "best_encoders.pt")
    best_decoders_path = os.path.join(RESULTS_DIR, "best_decoders.pt")
    last_encoders_path = os.path.join(RESULTS_DIR, "last_encoders.pt")
    last_decoders_path = os.path.join(RESULTS_DIR, "last_decoders.pt")

    articulators_indices_dict = model_params["indices_dict"]
    articulators = sorted(articulators_indices_dict.keys())

    autoencoder = MultiArticulatorAutoencoder(**model_params)
    if state_dict_fpath is not None:
        state_dict = torch.load(state_dict_fpath, map_location=device)
        autoencoder.load_state_dict(state_dict)
    autoencoder.to(device)

    train_sequences = sequences_from_dict(datadir, train_seq_dict)
    train_dataset = PrincipalComponentsMultiArticulatorAutoencoderDataset(
        datadir=datadir,
        dataset_config=DatasetConfig,
        sequences=train_sequences,
        articulators=articulators,
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
        dataset_config=DatasetConfig,
        sequences=valid_sequences,
        articulators=articulators,
        clip_tails=clip_tails
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=set_seeds
    )

    loss_fn = MultiArtRegularizedLatentsMSELoss(
        indices_dict=articulators_indices_dict,
        alpha=alpha,
    )
    optimizer = Adam(autoencoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CyclicLR(
        optimizer,
        base_lr=learning_rate / 25,
        max_lr=learning_rate,
        cycle_momentum=False
    )

    denorm_fn_dict = {
        articulator: denorm_fn.inverse
        for articulator, denorm_fn
        in train_dataset.normalize.items()
    }
    metrics = {
        "p2cp_mm": lambda outputs, targets: reconstruction_error(
            outputs, targets,
            denorm_fn_dict=denorm_fn_dict,
            px_space=DatasetConfig.PIXEL_SPACING,
            res=DatasetConfig.RES
        )
    }

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
            fn_metrics=metrics,
            device=device
        )

        mlflow.log_metrics({
            f"valid_{metric}": value for metric, value in info_valid.items()
        }, step=epoch)

        if info_valid["p2cp_mm"] < best_metric:
            best_metric = info_valid["p2cp_mm"]
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

        print(f"""
Finished training epoch {epoch}
Best metric: {best_metric}, Epochs since best: {epochs_since_best}
""")

        if epochs_since_best > patience:
            break

    test_sequences = sequences_from_dict(datadir, test_seq_dict)
    test_dataset = PrincipalComponentsMultiArticulatorAutoencoderDataset(
        datadir=datadir,
        dataset_config=DatasetConfig,
        sequences=test_sequences,
        articulators=articulators,
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
        **model_params
    )
    best_encoders_state_dict = torch.load(best_encoders_path, map_location=device)
    best_autoencoder.encoders.load_state_dict(best_encoders_state_dict)
    best_decoders_state_dict = torch.load(best_decoders_path, map_location=device)
    best_autoencoder.decoders.load_state_dict(best_decoders_state_dict)
    best_autoencoder.to(device)

    test_outputs_dir = os.path.join(RESULTS_DIR, "test_outputs")
    if not os.path.exists(test_outputs_dir):
        os.makedirs(test_outputs_dir)

    info_test = run_multiart_autoencoder_test(
        epoch=0,
        model=best_autoencoder,
        dataloader=test_dataloader,
        criterion=loss_fn,
        outputs_dir=test_outputs_dir,
        plots_dir=RESULTS_DIR,
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
        parser.add_argument("--experiment", dest="experiment_name", default="multiarticulator_autoencoder")
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
