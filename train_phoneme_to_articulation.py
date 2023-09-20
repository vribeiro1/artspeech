import argparse
import logging
import mlflow
import numpy as np
import os
import pandas as pd
import random
import shutil
import tempfile
import torch
import ujson
import yaml

from collections import OrderedDict
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from helpers import set_seeds, sequences_from_dict, make_padding_mask
from phoneme_to_articulation.encoder_decoder.dataset import ArtSpeechDataset, pad_sequence_collate_fn
from phoneme_to_articulation.encoder_decoder.evaluation import run_test
from phoneme_to_articulation.encoder_decoder.metrics import P2CPDistance
from phoneme_to_articulation.encoder_decoder.models import ArtSpeech, SimpleArtSpeech
from phoneme_to_articulation.metrics import EuclideanDistance
from settings import BASE_DIR, TRAIN, VALID, DATASET_CONFIG, BLANK, UNKNOWN

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
    scheduler=None,
    device=None,
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
        _,  # sentence_ids
        sentence,
        targets,
        lengths,
        _,  # phonemes
        _,  # references
        _,  # sentence_frames
        _,  # voicing
    ) in progress_bar:
        sentence = sentence.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            outputs = model(sentence, lengths)
            loss = criterion(outputs, targets)
            padding_mask = make_padding_mask(lengths)
            bs, max_len, num_articulators, features = loss.shape
            loss = loss.view(bs * max_len, num_articulators, features)
            loss = loss[padding_mask.view(bs * max_len)].mean()

            if training:
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            for metric_name, fn_metric in fn_metrics.items():
                metric_val = fn_metric(outputs, targets, lengths)
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

    mean_loss = np.mean(losses)
    info = {
        "loss": mean_loss
    }
    info.update({
        metric_name: np.mean(metric_values)
        for metric_name, metric_values in metrics_values.items()
    })

    return info


def main(
    datadir,
    database_name,
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
    model_kwargs=None,
    num_workers=0,
    clip_tails=True,
    state_dict_filepath=None,
    checkpoint_filepath=None,
    seed=0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on '{device.type}'")

    best_model_path = os.path.join(RESULTS_DIR, "best_model.pt")
    last_model_path = os.path.join(RESULTS_DIR, "last_model.pt")
    save_checkpoint_path = os.path.join(RESULTS_DIR, "checkpoint.pt")

    default_tokens = [BLANK, UNKNOWN]
    vocabulary = {token: i for i, token in enumerate(default_tokens)}
    with open(vocab_filepath) as f:
        tokens = ujson.load(f)
        for i, token in enumerate(tokens, start=len(vocabulary)):
            vocabulary[token] = i

    num_articulators = len(articulators)
    model_kwargs = model_kwargs or {}
    model = ArtSpeech(
        len(vocabulary),
        num_articulators,
        **model_kwargs,
    )
    if state_dict_filepath is not None:
        state_dict = torch.load(state_dict_filepath, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)

    loss_fn = EuclideanDistance(reduction="none")
    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=0.1,
        patience=10,
    )

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    train_sequences = sequences_from_dict(datadir, train_seq_dict)
    train_dataset = ArtSpeechDataset(
        datadir,
        database_name,
        train_sequences,
        vocabulary,
        articulators,
        clip_tails=clip_tails,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=set_seeds,
        collate_fn=pad_sequence_collate_fn,
        generator=gen,
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
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=set_seeds,
        collate_fn=pad_sequence_collate_fn,
        generator=gen,
    )

    dataset_config = DATASET_CONFIG[database_name]
    fn_metrics = {
        "p2cp_mean": P2CPDistance(
            dataset_config=dataset_config,
        )
    }

    epochs = range(1, num_epochs + 1)
    best_metric = np.inf
    epochs_since_best = 0

    if checkpoint_filepath is not None:
        checkpoint = torch.load(checkpoint_filepath, map_location=device)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        epoch = checkpoint["epoch"] + 1
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
            device=device,
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
            device=device,
            fn_metrics=fn_metrics,
        )

        mlflow.log_metrics(
            {"valid_loss": info_valid["loss"]},
            step=epoch
        )

        scheduler.step(info_valid["loss"])

        if info_valid["p2cp_mean"] < best_metric:
            best_metric = info_valid["p2cp_mean"]
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
        database_name,
        test_sequences,
        vocabulary,
        articulators,
        clip_tails=clip_tails,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=set_seeds,
        collate_fn=pad_sequence_collate_fn,
        generator=gen,
    )

    best_model = SimpleArtSpeech(
        len(vocabulary),
        num_articulators,
        **model_kwargs,
    )
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
        regularize_out=True,
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
        results_item[f"p2cp_{articulator}"] = test_results[articulator]["p2cp"]
        results_item[f"med_{articulator}"] = test_results[articulator]["med"]

    df = pd.DataFrame([results_item])
    df_filepath = os.path.join(RESULTS_DIR, "test_results.csv")
    df.to_csv(df_filepath, index=False)
    mlflow.log_artifact(df_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_filepath")
    parser.add_argument("--mlflow", dest="mlflow_tracking_uri", default=None)
    parser.add_argument("--experiment", dest="experiment_name", default="phoneme_to_articulation")
    parser.add_argument("--run_id", dest="run_id", default=None)
    parser.add_argument("--run_name", dest="run_name", default=None)
    parser.add_argument("--checkpoint", dest="checkpoint_filepath", default=None)
    args = parser.parse_args()

    seed = 0
    rs = RandomState(MT19937(SeedSequence(seed)))
    random.seed(seed)
    torch.manual_seed(seed)

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
                seed=seed,
            )
        finally:
            shutil.rmtree(TMP_DIR)
