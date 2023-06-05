import argparse
import logging
import mlflow
import numpy as np
import os
import shutil
import tempfile
import torch
import yaml

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from vt_tools import *

from helpers import set_seeds, sequences_from_dict, make_indices_dict
from phoneme_to_articulation.principal_components import run_autoencoder_epoch
from phoneme_to_articulation.principal_components.dataset import PrincipalComponentsAutoencoderDataset2
from phoneme_to_articulation.principal_components.evaluation import run_multiart_autoencoder_test
from phoneme_to_articulation.principal_components.losses import RegularizedLatentsMSELoss2
from phoneme_to_articulation.principal_components.models.autoencoder import MultiArticulatorAutoencoder
from phoneme_to_articulation.principal_components.metrics import MeanP2CPDistance
from settings import BASE_DIR, DATASET_CONFIG, TRAIN, VALID, TEST

TMPFILES = os.path.join(BASE_DIR, "tmp")
TMP_DIR = tempfile.mkdtemp(dir=TMPFILES)
RESULTS_DIR = os.path.join(TMP_DIR, "results")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


def reconstruction_error(
    outputs,
    targets,
    denorm_fn_dict,
    px_space=1,
    res=1
):
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
    database_name,
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
    num_workers=0,
    clip_tails=True,
    state_dict_fpath=None,
    checkpoint_filepath=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on '{device.type}'")

    best_encoders_path = os.path.join(RESULTS_DIR, "best_encoders.pt")
    best_decoders_path = os.path.join(RESULTS_DIR, "best_decoders.pt")
    last_encoders_path = os.path.join(RESULTS_DIR, "last_encoders.pt")
    last_decoders_path = os.path.join(RESULTS_DIR, "last_decoders.pt")
    save_checkpoint_path = os.path.join(RESULTS_DIR, "checkpoint.pt")

    articulators_indices_dict = model_params["indices_dict"]
    if isinstance(list(articulators_indices_dict.values())[0], int):
        articulators_indices_dict = make_indices_dict(articulators_indices_dict)
        model_params["indices_dict"] = articulators_indices_dict
    articulators = sorted(articulators_indices_dict.keys())

    autoencoder = MultiArticulatorAutoencoder(**model_params)
    if state_dict_fpath is not None:
        state_dict = torch.load(state_dict_fpath, map_location=device)
        autoencoder.load_state_dict(state_dict)
    autoencoder.to(device)

    dataset_config = DATASET_CONFIG[database_name]
    train_sequences = sequences_from_dict(datadir, train_seq_dict)
    train_dataset = PrincipalComponentsAutoencoderDataset2(
        database_name=database_name,
        datadir=datadir,
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
    valid_dataset = PrincipalComponentsAutoencoderDataset2(
        database_name=database_name,
        datadir=datadir,
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

    loss_fn = RegularizedLatentsMSELoss2(
        indices_dict=articulators_indices_dict,
        alpha=alpha,
    )
    optimizer = Adam(autoencoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=0.1,
        patience=10,
        min_lr=learning_rate / 1000,
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
            px_space=dataset_config.PIXEL_SPACING,
            res=dataset_config.RES
        )
    }

    best_metric = np.inf
    epochs_since_best = 0
    epochs = range(1, n_epochs + 1)

    if checkpoint_filepath is not None:
        checkpoint = torch.load(checkpoint_filepath, map_location=device)

        autoencoder.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        epoch = checkpoint["epoch"] + 1
        epochs = range(epoch, n_epochs + 1)
        best_metric = checkpoint["best_metric"]
        epochs_since_best = checkpoint["epochs_since_best"]

        logging.info(f"""
Loaded checkpoint -- Launching training from epoch {epoch} with best metric
so far {best_metric} seen {epochs_since_best} epochs ago.
""")
    for epoch in epochs:
        info_train = run_autoencoder_epoch(
            phase=TRAIN,
            epoch=epoch,
            model=autoencoder,
            dataloader=train_dataloader,
            optimizer=optimizer,
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

        checkpoint = {
            "epoch": epoch,
            "model": autoencoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_metric": best_metric,
            "epochs_since_best": epochs_since_best,
            "best_encoders_path": best_encoders_path,
            "best_decoders_path": best_decoders_path,
            "last_encoders_path": last_encoders_path,
            "last_decoders_path": last_decoders_path,
        }

        torch.save(checkpoint, save_checkpoint_path)
        mlflow.log_artifact(save_checkpoint_path)

        print(f"""
Finished training epoch {epoch}
Best metric: {best_metric}, Epochs since best: {epochs_since_best}
""")

        if epochs_since_best > patience:
            break

    test_sequences = sequences_from_dict(datadir, test_seq_dict)
    test_dataset = PrincipalComponentsAutoencoderDataset2(
        database_name=database_name,
        datadir=datadir,
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
        dataset_config=dataset_config,
        outputs_dir=test_outputs_dir,
        plots_dir=RESULTS_DIR,
        indices_dict=articulators_indices_dict,
        device=device,
    )

    mlflow.log_metrics({
        f"test_{metric}": value for metric, value in info_test.items()
    }, step=epoch)

    mlflow.log_artifacts(test_outputs_dir, "test_outputs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_filepath")
    parser.add_argument("--mlflow", dest="mlflow_tracking_uri", default=None)
    parser.add_argument("--experiment", dest="experiment_name", default="multiarticulator_autoencoder")
    parser.add_argument("--run_id", dest="run_id", default=None)
    parser.add_argument("--run_name", dest="run_name", default=None)
    parser.add_argument("--checkpoint", dest="checkpoint_filepath", default=None)
    args = parser.parse_args()

    if args.mlflow_tracking_uri is not None:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    with open(args.config_filepath) as f:
        cfg = yaml.safe_load(f)

    experiment = mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(
        run_id=args.run_id,
        experiment_id=experiment.experiment_id,
        run_name=args.run_name
    ) as run:
        print(f"Experiment ID: {experiment.experiment_id}\nRun ID: {run.info.run_id}")
        try:
            mlflow.log_artifact(args.config_filepath)
        except shutil.SameFileError:
            logging.info("Skipping logging config file since it already exists.")
        try:
            main(
                **cfg,
                checkpoint_filepath=args.checkpoint_filepath,
            )
        finally:
            shutil.rmtree(TMP_DIR)
