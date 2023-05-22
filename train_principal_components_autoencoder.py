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
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader

from helpers import set_seeds, sequences_from_dict
from phoneme_to_articulation.principal_components import run_autoencoder_epoch
from phoneme_to_articulation.principal_components.dataset import PrincipalComponentsAutoencoderDataset
from phoneme_to_articulation.principal_components.evaluation import run_autoencoder_test
from phoneme_to_articulation.principal_components.losses import RegularizedLatentsMSELoss
from phoneme_to_articulation.principal_components.metrics import MeanP2CPDistance
from phoneme_to_articulation.principal_components.models.autoencoder import Autoencoder
from settings import DATASET_CONFIG, BASE_DIR, TRAIN, VALID, TEST

TMPFILES = os.path.join(BASE_DIR, "tmp")
TMP_DIR = tempfile.mkdtemp(dir=TMPFILES)
RESULTS_DIR = os.path.join(TMP_DIR, "results")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


def reconstruction_error(outputs, targets, denorm_fn, px_space=1, res=1):
    p2cp_fn = MeanP2CPDistance(reduction="mean")

    batch_size, n_features = outputs.shape
    outputs = outputs.reshape(batch_size, 2, n_features // 2)
    targets = targets.reshape(batch_size, 2, n_features // 2)

    outputs = denorm_fn(outputs).permute(0, 2, 1)
    targets = denorm_fn(targets).permute(0, 2, 1)

    p2cp = p2cp_fn(outputs, targets)
    p2cp_mm = p2cp * px_space * res

    return p2cp_mm


def main(
    database_name,
    datadir,
    num_epochs,
    batch_size,
    patience,
    learning_rate,
    weight_decay,
    train_seq_dict,
    valid_seq_dict,
    test_seq_dict,
    articulator,
    model_params,
    clip_tails=True,
    encoder_state_dict_fpath=None,
    decoder_state_dict_fpath=None,
    checkpoint_filepath=None,
    alpha=1.,
    num_workers=0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on '{device.type}'")

    best_encoder_path = os.path.join(RESULTS_DIR, "best_encoder.pt")
    best_decoder_path = os.path.join(RESULTS_DIR, "best_decoder.pt")
    last_encoder_path = os.path.join(RESULTS_DIR, "last_encoder.pt")
    last_decoder_path = os.path.join(RESULTS_DIR, "last_decoder.pt")
    save_checkpoint_path = os.path.join(RESULTS_DIR, "checkpoint.pt")

    autoencoder = Autoencoder(**model_params)
    if encoder_state_dict_fpath is not None:
        encoder_state_dict = torch.load(encoder_state_dict_fpath, map_location=device)
        autoencoder.encoder.load_state_dict(encoder_state_dict)
    if decoder_state_dict_fpath is not None:
        decoder_state_dict = torch.load(decoder_state_dict_fpath, map_location=device)
        autoencoder.decoder.load_state_dict(decoder_state_dict)
    autoencoder.to(device)

    dataset_config = DATASET_CONFIG[database_name]
    train_sequences = sequences_from_dict(datadir, train_seq_dict)
    train_dataset = PrincipalComponentsAutoencoderDataset(
        database_name=database_name,
        datadir=datadir,
        sequences=train_sequences,
        articulator=articulator,
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
        database_name=database_name,
        datadir=datadir,
        sequences=valid_sequences,
        articulator=articulator,
        clip_tails=clip_tails
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=set_seeds
    )

    loss_fn = RegularizedLatentsMSELoss(alpha=alpha)
    optimizer = Adam(autoencoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CyclicLR(
        optimizer,
        base_lr=learning_rate / 25,
        max_lr=learning_rate,
        cycle_momentum=False
    )

    metrics = {
        "p2cp_mm": lambda outputs, targets: reconstruction_error(
            outputs, targets,
            denorm_fn=train_dataset.normalize.inverse,
            px_space=dataset_config.PIXEL_SPACING,
            res=dataset_config.RES
        )
    }

    best_metric = np.inf
    epochs_since_best = 0
    epochs = range(1, num_epochs + 1)

    if checkpoint_filepath is not None:
        # TODO: Save and load the scheduler state dict when the following change is released
        # https://github.com/Lightning-AI/lightning/issues/15901
        checkpoint = torch.load(checkpoint_filepath, map_location=device)

        autoencoder.encoder.load_state_dict(checkpoint["encoder"])
        autoencoder.decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # scheduler.load_state_dict(checkpoint["scheduler"])
        epoch = checkpoint["epoch"] + 1
        epochs = range(epoch, num_epochs + 1)
        best_metric = checkpoint["best_metric"]
        epochs_since_best = checkpoint["epochs_since_best"]

        print(f"""
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
            fn_metrics=metrics,
            device=device
        )

        mlflow.log_metrics({
            f"valid_{metric}": value for metric, value in info_valid.items()
        }, step=epoch)

        if info_valid["p2cp_mm"] < best_metric:
            best_metric = info_valid["p2cp_mm"]
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

        checkpoint = {
            "epoch": epoch,
            "encoder": autoencoder.encoder.state_dict(),
            "decoder": autoencoder.decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            # "scheduler": scheduler.state_dict(),
            "best_metric": best_metric,
            "epochs_since_best": epochs_since_best,
            "best_encoder_path": best_encoder_path,
            "best_decoder_path": best_decoder_path,
            "last_encoder_path": last_encoder_path,
            "last_decoder_path": last_decoder_path,
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
    test_dataset = PrincipalComponentsAutoencoderDataset(
        database_name=database_name,
        datadir=datadir,
        sequences=test_sequences,
        articulator=articulator,
        clip_tails=clip_tails
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=set_seeds
    )

    best_autoencoder = Autoencoder(**model_params)
    best_encoder_state_dict = torch.load(best_encoder_path, map_location=device)
    best_autoencoder.encoder.load_state_dict(best_encoder_state_dict)
    best_decoder_state_dict = torch.load(best_decoder_path, map_location=device)
    best_autoencoder.decoder.load_state_dict(best_decoder_state_dict)
    best_autoencoder.to(device)

    plots_dir = os.path.join(TMP_DIR, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    test_outputs_dir = os.path.join(TMP_DIR, "test_outputs")
    if not os.path.exists(test_outputs_dir):
        os.makedirs(test_outputs_dir)

    info_test = run_autoencoder_test(
        epoch=0,
        model=best_autoencoder,
        dataloader=test_dataloader,
        criterion=loss_fn,
        # outputs_dir=test_outputs_dir,
        plots_dir=plots_dir,
        fn_metrics=metrics,
        device=device
    )

    mlflow.log_metrics({
        f"test_{metric}": value for metric, value in info_test.items()
    }, step=epoch)

    mlflow.log_artifacts(plots_dir, "plots_dir")
    mlflow.log_artifacts(test_outputs_dir, "test_outputs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_filepath")
    parser.add_argument("--mlflow", dest="mlflow_tracking_uri", default=None)
    parser.add_argument("--experiment", dest="experiment_name", default="principal_components_autoencoder")
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
