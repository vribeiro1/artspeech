import pdb

import argparse
import logging
import mlflow
import numpy as np
import os
import tempfile
import torch
import ujson
import yaml

from collections import OrderedDict
from functools import reduce
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from helpers import set_seeds, sequences_from_dict
from phoneme_recognition import UNKNOWN
from phoneme_to_articulation.principal_components.dataset import (
    PrincipalComponentsPhonemeToArticulationDataset2,
    pad_sequence_collate_fn
)
from phoneme_to_articulation.principal_components.losses import AutoencoderLoss2
from phoneme_to_articulation.principal_components.metrics import (
    DecoderEuclideanDistance,
    DecoderMeanP2CPDistance
)
from phoneme_to_articulation.principal_components.models import PrincipalComponentsArtSpeech
from settings import BASE_DIR, TRAIN, VALID, TEST, GottingenConfig

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
    fn_metrics=None,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if fn_metrics is None:
        fn_metrics = {}
    training = phase == TRAIN

    if training:
        model.train()
    else:
        model.eval()

    losses = []
    metrics_values = {metric_name: [] for metric_name in fn_metrics}
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - {phase}")
    for (
        sentence_ids,
        inputs,
        targets,
        len_inputs,
        phonemes,
        critical_masks,
        sentence_frames
    ) in progress_bar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            outputs = model(inputs, len_inputs)
            loss = criterion(outputs, targets, critical_masks)

            if training:
                loss.backward()
                optimizer.step()
            for metric_name, fn_metric in fn_metrics.items():
                metric_val = fn_metric(outputs, targets)
                metrics_values[metric_name].append(metric_val.item())
            losses.append(loss.item())

        postfixes = {
            "loss": np.mean(losses)
        }
        postfixes.update({
            metric_name: np.mean(metric_vals)
            for metric_name, metric_vals in metrics_values.items()
        })
        progress_bar.set_postfix(OrderedDict(postfixes))

    info = {
        "loss": np.mean(losses),
    }
    info.update({
        metric_name: np.mean(metric_values)
        for metric_name, metric_values in fn_metrics.items()
    })

    return info


def run_test(
    epoch,
    model,
    dataloader,
    criterion,
    fn_metrics=None,
    outputs_dir=None,
    decode_transform=None,
    articulators=None,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if fn_metrics is None:
        fn_metrics={}

    model.eval()
    losses = []
    metrics_values = {metric_name: [] for metric_name in fn_metrics}
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - test")
    for (
        sentence_ids,
        inputs,
        targets,
        len_inputs,
        phonemes,
        critical_masks,
        sentence_frames
    ) in progress_bar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs, len_inputs)
            loss = criterion(outputs, targets, critical_masks)

            for metric_name, fn_metric in fn_metrics.items():
                metric_val = fn_metric(outputs, targets)
                metrics_values[metric_name].append(metric_val.item())
            losses.append(loss.item())

        postfixes = {
            "loss": np.mean(losses)
        }
        postfixes.update({
            metric_name: np.mean(metric_vals)
            for metric_name, metric_vals in metrics_values.items()
        })
        progress_bar.set_postfix(OrderedDict(postfixes))

        if outputs_dir is not None:
            output_shapes = decode_transform(outputs)  # (B, Nart, T, 2 * D)
            output_shapes = output_shapes.permute(0, 2, 1, 3)
            bs, seq_len, num_articulators, features = output_shapes.shape
            output_shapes = output_shapes.reshape(bs, seq_len, num_articulators, 2, features // 2)
            for sentence_id, sentence_shapes in zip(sentence_ids, output_shapes):
                sentence_dir = os.path.join(outputs_dir, sentence_id)
                if not os.path.exists(sentence_dir):
                    os.makedirs(sentence_dir)

                for timestep, articulators_arrays in enumerate(sentence_shapes):
                    for i, articulator_array in enumerate(articulators_arrays):
                        articulator_name = articulators[i]
                        denorm_fn = dataloader.dataset.normalize[articulator_name].inverse
                        frame = "%04d" % timestep
                        articulator_array = denorm_fn(articulator_array)
                        articulator_array = articulator_array.detach().cpu().numpy()
                        npy_filepath = os.path.join(sentence_dir, f"{frame}_{articulator_name}.npy")
                        np.save(npy_filepath, articulator_array)

    info = {
        "loss": np.mean(losses),
    }
    info.update({
        metric_name: np.mean(metric_values)
        for metric_name, metric_values in fn_metrics.items()
    })

    return info


def main(
    datadir,
    database,
    num_epochs,
    batch_size,
    patience,
    learning_rate,
    weight_decay,
    train_seq_dict,
    valid_seq_dict,
    test_seq_dict,
    indices_dict,
    vocab_filepath,
    modelkwargs,
    autoencoder_kwargs,
    encoder_state_dict_filepath,
    decoder_state_dict_filepath,
    alpha=1.0,
    beta=1.0,
    clip_tails=True,
    num_workers=0,
    state_dict_filepath=None,
    checkpoint_filepath=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on '{device.type}'")

    best_model_path = os.path.join(RESULTS_DIR, "best_model.pt")
    last_model_path = os.path.join(RESULTS_DIR, "last_model.pt")
    save_checkpoint_path = os.path.join(RESULTS_DIR, "checkpoint.pt")

    vocabulary = {UNKNOWN: 0}
    with open(vocab_filepath) as f:
        tokens = ujson.load(f)
        for i, token in enumerate(tokens, start=len(vocabulary)):
            vocabulary[token] = i

    TV_to_phoneme_map = {
        "LA": [
            "p",
            "b",
            "m"
        ],
        "VEL": [
            token
            for token in vocabulary
            if "~" not in token
        ]
    }

    articulators_indices_dict = indices_dict
    articulators = sorted(articulators_indices_dict.keys())
    num_components = 1 + max(set(
        reduce(lambda l1, l2: l1 + l2,
        articulators_indices_dict.values())
    ))
    model = PrincipalComponentsArtSpeech(
        vocab_size=len(vocabulary),
        num_components=num_components,
        **modelkwargs,
    )
    if state_dict_filepath is not None:
        state_dict = torch.load(state_dict_filepath, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)

    TVs = [
        "LA",
        "VEL"
    ]
    loss_fn = AutoencoderLoss2(
        indices_dict=indices_dict,
        TVs=TVs,
        device=device,
        encoder_state_dict_filepath=encoder_state_dict_filepath,
        decoder_state_dict_filepath=decoder_state_dict_filepath,
        alpha=alpha,
        beta=beta,
        **autoencoder_kwargs,
    )
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

    train_sequences = sequences_from_dict(datadir, train_seq_dict)
    train_dataset = PrincipalComponentsPhonemeToArticulationDataset2(
        datadir,
        GottingenConfig,
        train_sequences,
        vocabulary,
        articulators,
        TV_to_phoneme_map
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=set_seeds,
        collate_fn=pad_sequence_collate_fn,
    )

    valid_sequences = sequences_from_dict(datadir, valid_seq_dict)
    valid_dataset = PrincipalComponentsPhonemeToArticulationDataset2(
        datadir,
        GottingenConfig,
        valid_sequences,
        vocabulary,
        articulators,
        TV_to_phoneme_map
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=set_seeds,
        collate_fn=pad_sequence_collate_fn,
    )

    best_metric = np.inf
    epochs_since_best = 0
    epochs = range(1, num_epochs + 1)

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

        logging.info(f"""
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
            device=device,
        )

        mlflow.log_metrics(
            {
                f"train_{metric}": value
                for metric, value in info_train.items()
            },
            step=epoch
        )

        info_valid = run_epoch(
            phase=VALID,
            epoch=epoch,
            model=model,
            dataloader=valid_dataloader,
            optimizer=optimizer,
            criterion=loss_fn,
            device=device,
        )

        mlflow.log_metrics(
            {
                f"valid_{metric}": value
                for metric, value in info_valid.items()
            },
            step=epoch
        )

        scheduler.step(info_valid["loss"])

        if info_valid["loss"] < best_metric:
            best_metric = info_valid["loss"]
            epochs_since_best = 0
            torch.save(model.state_dict(), best_model_path)
            mlflow.log_artifact(best_model_path)
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
    test_dataset = PrincipalComponentsPhonemeToArticulationDataset2(
        datadir,
        GottingenConfig,
        test_sequences,
        vocabulary,
        articulators,
        TV_to_phoneme_map
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=set_seeds,
        collate_fn=pad_sequence_collate_fn,
    )

    best_model = PrincipalComponentsArtSpeech(
        vocab_size=len(vocabulary),
        num_components=num_components,
        **modelkwargs,
    )
    best_model_state_dict = torch.load(best_model_path, map_location=device)
    best_model.load_state_dict(best_model_state_dict)
    best_model.to(device)

    test_outputs_dir = os.path.join(RESULTS_DIR, "test_outputs")
    if not os.path.exists(test_outputs_dir):
        os.makedirs(test_outputs_dir)

    info_test = run_test(
        epoch=0,
        model=best_model,
        dataloader=test_dataloader,
        criterion=loss_fn,
        outputs_dir=test_outputs_dir,
        decode_transform=loss_fn.decode,
        articulators=articulators,
        device=device
    )
    mlflow.log_artifact(test_outputs_dir)
    mlflow.log_metrics(
        {
            f"test_{metric}": value
            for metric, value in info_test.items()
        },
        step=0
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_filepath")
    parser.add_argument("--experiment", dest="experiment_name", default="phoneme_to_principal_components")
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
