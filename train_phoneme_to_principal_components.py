####################################################################################################
#
# Train the autoencoder-based phoneme-to-articulation
#
####################################################################################################
import argparse
import logging
import mlflow
import numpy as np
import os
import random
import shutil
import tempfile
import torch
import ujson
import yaml
import shutil

from collections import OrderedDict
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from helpers import set_seeds, sequences_from_dict, make_indices_dict
from phoneme_to_articulation import RNNType
from phoneme_to_articulation.principal_components.dataset import (
    PrincipalComponentsPhonemeToArticulationDataset2,
    pad_sequence_collate_fn
)
from phoneme_to_articulation.principal_components.evaluation import run_phoneme_to_principal_components_test
from phoneme_to_articulation.principal_components.losses import AutoencoderLoss2
from phoneme_to_articulation.principal_components.metrics import DecoderMeanP2CPDistance2
from phoneme_to_articulation.principal_components.models import (
    EncoderType,
    DecoderType,
    PrincipalComponentsArtSpeech
)
from phoneme_recognition.deepspeech2 import DeepSpeech2
from settings import (
    BASE_DIR,
    BLANK,
    UNKNOWN,
    TRAIN,
    VALID,
    DATASET_CONFIG
)

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
    scheduler=None,
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
        _,  # sentence_name
        inputs,  # sentence_numerized
        targets,  # sentence_targets
        len_inputs,  # lengths
        _,  # phonemes
        critical_masks,  # critical masks
        reference_arrays,  # reference_arrays
        _,  # sentence_frames
        voicing,  # voicing
    ) in progress_bar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        reference_arrays = reference_arrays.to(device)
        voicing = voicing.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            outputs = model(inputs, len_inputs)
            loss = criterion(
                outputs,
                targets,
                reference_arrays,
                len_inputs,
                critical_masks,
                voicing,
            )

            if training:
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            for metric_name, fn_metric in fn_metrics.items():
                metric_val = fn_metric(outputs, targets, len_inputs)
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
        for metric_name, metric_values in metrics_values.items()
    })

    return info


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
    indices_dict,
    vocab_filepath,
    modelkwargs,
    autoencoder_kwargs,
    encoder_state_dict_filepath,
    decoder_state_dict_filepath,
    encoder_type="AE",
    decoder_type="AE",
    rnn_type="GRU",
    beta1=1.0,
    beta2=1.0,
    beta3=1.0,
    beta4=0.0,
    rescale_factor=1.0,
    recognizer_filepath=None,
    recognizer_params=None,
    voicing_filepath=None,
    TV_to_phoneme_map=None,
    clip_tails=True,
    num_workers=0,
    state_dict_filepath=None,
    checkpoint_filepath=None,
    seed=0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on '{device.type}'")
    dataset_config = DATASET_CONFIG[database_name]

    best_model_path = os.path.join(RESULTS_DIR, "best_model.pt")
    last_model_path = os.path.join(RESULTS_DIR, "last_model.pt")
    save_checkpoint_path = os.path.join(RESULTS_DIR, "checkpoint.pt")

    default_tokens = [BLANK, UNKNOWN]
    vocabulary = {token: i for i, token in enumerate(default_tokens)}
    with open(vocab_filepath) as f:
        tokens = ujson.load(f)
        for i, token in enumerate(tokens, start=len(vocabulary)):
            vocabulary[token] = i
    if voicing_filepath is not None:
        with open(voicing_filepath) as f:
            voiced_tokens = ujson.load(f)
    else:
        voiced_tokens = None

    if isinstance(list(indices_dict.values())[0], int):
        indices_dict = make_indices_dict(indices_dict)
    articulators = sorted(indices_dict.keys())

    model = PrincipalComponentsArtSpeech(
        vocab_size=len(vocabulary),
        indices_dict=indices_dict,
        rnn=RNNType[rnn_type.upper()],
        **modelkwargs,
    )
    if state_dict_filepath is not None:
        state_dict = torch.load(state_dict_filepath, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)

    print(f"""
PrincipalComponentsArtSpeech -- {model.total_parameters} parameters
""")

    mlflow.log_param("num_network_params", model.total_parameters)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    train_sequences = sequences_from_dict(datadir, train_seq_dict)
    train_dataset = PrincipalComponentsPhonemeToArticulationDataset2(
        database_name,
        datadir,
        train_sequences,
        vocabulary,
        articulators,
        TV_to_phoneme_map,
        clip_tails=clip_tails,
        voiced_tokens=voiced_tokens,
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
    valid_dataset = PrincipalComponentsPhonemeToArticulationDataset2(
        database_name,
        datadir,
        valid_sequences,
        vocabulary,
        articulators,
        TV_to_phoneme_map,
        clip_tails=clip_tails,
        voiced_tokens=voiced_tokens,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=set_seeds,
        collate_fn=pad_sequence_collate_fn,
        generator=gen,
    )

    if TV_to_phoneme_map is None:
        TV_to_phoneme_map = {}
    TVs = sorted(TV_to_phoneme_map.keys())

    if recognizer_filepath:
        recognizer = DeepSpeech2(num_classes=len(vocabulary), **recognizer_params)
        recog_state_dict = torch.load(recognizer_filepath, map_location=device)
        recognizer.load_state_dict(recog_state_dict)
        recognizer.to(device)

        for p in recognizer.parameters():
            p.requires_grad = False
    else:
        recognizer = None
    denorm_fn = {
        articulator: normalize.inverse
        for articulator, normalize in train_dataset.normalize.items()
    }

    encoder_cls = EncoderType[encoder_type.upper()].value
    decoder_cls = DecoderType[decoder_type.upper()].value

    loss_fn = AutoencoderLoss2(
        indices_dict=indices_dict,
        TVs=TVs,
        device=device,
        encoder_state_dict_filepath=encoder_state_dict_filepath,
        decoder_state_dict_filepath=decoder_state_dict_filepath,
        denormalize_fn=denorm_fn,
        beta1=beta1,
        beta2=beta2,
        beta3=beta3,
        beta4=beta4,
        rescale_factor=rescale_factor,
        encoder_cls=encoder_cls,
        decoder_cls=decoder_cls,
        recognizer=recognizer,
        **autoencoder_kwargs,
    )
    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=0.1,
        patience=10
    )

    fn_metrics = {
        "p2cp_mean": DecoderMeanP2CPDistance2(
            dataset_config=dataset_config,
            decoder_state_dict_filepath=decoder_state_dict_filepath,
            indices_dict=indices_dict,
            autoencoder_kwargs=autoencoder_kwargs,
            device=device,
            decoder_cls=decoder_cls,
            denorm_fns={
                articulator: train_dataset.normalize[articulator].inverse
                for articulator in articulators
            }
        )
    }

    test_outputs_dir = os.path.join(RESULTS_DIR, "test_outputs")
    if not os.path.exists(test_outputs_dir):
        os.makedirs(test_outputs_dir)

    best_metric = np.inf
    epochs_since_best = 0
    epochs = range(1, num_epochs + 1)

    if checkpoint_filepath is not None:
        checkpoint = torch.load(checkpoint_filepath, map_location=device)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        epoch = checkpoint["epoch"] + 1
        epochs = range(epoch, num_epochs + 1)
        best_metric = checkpoint["best_metric"]
        epochs_since_best = checkpoint["epochs_since_best"]

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
            fn_metrics=fn_metrics,
        )

        mlflow.log_metrics(
            {
                f"valid_{metric}": value
                for metric, value in info_valid.items()
            },
            step=epoch
        )

        scheduler.step(info_valid["loss"])

        if info_valid["p2cp_mean"] < best_metric:
            best_metric = info_valid["p2cp_mean"]
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
        database_name,
        datadir,
        test_sequences,
        vocabulary,
        articulators,
        TV_to_phoneme_map,
        clip_tails=clip_tails,
        voiced_tokens=voiced_tokens,
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

    best_model = PrincipalComponentsArtSpeech(
        vocab_size=len(vocabulary),
        indices_dict=indices_dict,
        rnn=RNNType[rnn_type.upper()],
        **modelkwargs,
    )
    best_model_state_dict = torch.load(best_model_path, map_location=device)
    best_model.load_state_dict(best_model_state_dict)
    best_model.to(device)

    info_test = run_phoneme_to_principal_components_test(
        epoch=0,
        model=best_model,
        dataloader=test_dataloader,
        criterion=loss_fn,
        outputs_dir=test_outputs_dir,
        decode_transform=loss_fn.decode,
        device=device,
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
    parser.add_argument("--mlflow", dest="mlflow_tracking_uri", default=None)
    parser.add_argument("--experiment", dest="experiment_name", default="phoneme_to_principal_components")
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
