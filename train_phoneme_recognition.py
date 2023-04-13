import pdb

import argparse
import logging
import mlflow
import numpy as np
import os
import shutil
import tempfile
import torch
import torch.nn as nn
import ujson
import yaml

from functools import partial
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader
from torchaudio.models.decoder import ctc_decoder

from helpers import set_seeds, sequences_from_dict
from phoneme_recognition import (
    run_epoch,
    run_test,
    Criterion,
    Feature,
    Target,
    SIL,
    BLANK,
    UNKNOWN
)
from phoneme_recognition.datasets import  PhonemeRecognitionDataset, collate_fn
from phoneme_recognition.decoders import TopKDecoder
from phoneme_recognition.deepspeech2 import DeepSpeech2
from phoneme_recognition.metrics import EditDistance, Accuracy, AUROC
from settings import DATASET_CONFIG, BASE_DIR, TRAIN, VALID, TEST

TMPFILES = os.path.join(BASE_DIR, "tmp")
TMP_DIR = tempfile.mkdtemp(dir=TMPFILES)
RESULTS_DIR = os.path.join(TMP_DIR, "results")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


def main(
    database_name,
    datadir,
    num_epochs,
    batch_size,
    patience,
    learning_rate,
    weight_decay,
    feature,
    target,
    vocab_filepath,
    train_seq_dict,
    valid_seq_dict,
    test_seq_dict,
    model_params,
    loss,
    loss_params,
    num_workers=0,
    logits_large_margins=0.0,
    pretrained=False,
    voicing_filepath=None,
    state_dict_filepath=None,
    checkpoint_filepath=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on '{device.type}'")

    best_model_path = os.path.join(RESULTS_DIR, "best_model.pt")
    last_model_path = os.path.join(RESULTS_DIR, "last_model.pt")
    save_checkpoint_path = os.path.join(RESULTS_DIR, "checkpoint.pt")

    dataset_config = DATASET_CONFIG[database_name]
    feature = Feature(feature)
    target = Target(target)
    criterion = Criterion[loss]
    criterion_cls = criterion.value

    default_tokens = [BLANK, UNKNOWN] if criterion == Criterion.CTC else [UNKNOWN]
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

    tokens = [k for k, v in sorted(vocabulary.items(), key=lambda t: t[1])]
    decoder_fn = ctc_decoder if criterion == Criterion.CTC else TopKDecoder
    decoder = decoder_fn(
        lexicon=None,
        tokens=tokens,
        sil_token=SIL,
        blank_token=BLANK if criterion == Criterion.CTC else None,
        unk_word=UNKNOWN,
    )

    if pretrained:
        model = DeepSpeech2.load_librispeech_model(
            model_params["num_features"],
            adapter_out_features=model_params.get("adapter_out_features")
        )
        hidden_size = model_params["rnn_hidden_size"]
        model.classifier = nn.Linear(hidden_size, len(vocabulary))
    else:
        model = DeepSpeech2(num_classes=len(vocabulary), **model_params)

    if state_dict_filepath is not None:
        state_dict = torch.load(state_dict_filepath, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    model.to(device)

    train_sequences = sequences_from_dict(datadir, train_seq_dict)
    train_dataset = PhonemeRecognitionDataset(
        datadir=datadir,
        database_name=database_name,
        sequences=train_sequences,
        vocabulary=vocabulary,
        features=[feature],
        tmp_dir=TMP_DIR,
        voiced_tokens=voiced_tokens,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=set_seeds,
        collate_fn=partial(collate_fn, features_names=[feature]),
    )

    valid_sequences = sequences_from_dict(datadir, valid_seq_dict)
    valid_dataset = PhonemeRecognitionDataset(
        datadir=datadir,
        database_name=database_name,
        sequences=valid_sequences,
        vocabulary=vocabulary,
        features=[feature],
        tmp_dir=TMP_DIR,
        voiced_tokens=voiced_tokens,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=set_seeds,
        collate_fn=partial(collate_fn, features_names=[feature]),
    )

    loss_fn = criterion_cls(**loss_params)
    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = CyclicLR(
        optimizer,
        base_lr=learning_rate / 25,
        max_lr=learning_rate,
        cycle_momentum=False
    )

    metrics = {
        "edit_distance": EditDistance(decoder),
        "accuracy": Accuracy(len(vocabulary)),
        "auroc": AUROC(len(vocabulary))
    }

    best_metric = np.inf
    epochs_since_best = 0
    epochs = range(1, num_epochs + 1)

    if checkpoint_filepath is not None:
        # TODO: Save and load the scheduler state dict when the following change is released
        # https://github.com/Lightning-AI/lightning/issues/15901
        checkpoint = torch.load(checkpoint_filepath, map_location=device)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # scheduler.load_state_dict(checkpoint["scheduler"])
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
            logits_large_margins=logits_large_margins,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=loss_fn,
            device=device,
            feature=feature,
            target=target,
            use_voicing=(voicing_filepath is not None),
            use_log_prob=(criterion == Criterion.CTC),
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
            scheduler=scheduler,
            criterion=loss_fn,
            fn_metrics=metrics,
            device=device,
            feature=feature,
            target=target,
            use_voicing=(voicing_filepath is not None),
            use_log_prob=(criterion == Criterion.CTC),
        )

        mlflow.log_metrics(
            {
                f"valid_{metric}": value
                for metric, value in info_valid.items()
            },
            step=epoch
        )

        if 1 - info_valid["accuracy"] < best_metric:
            best_metric = 1 - info_valid["accuracy"]
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
            # "scheduler": scheduler.state_dict(),
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
    test_dataset = PhonemeRecognitionDataset(
        datadir=datadir,
        database_name=database_name,
        sequences=test_sequences,
        vocabulary=vocabulary,
        features=[feature],
        tmp_dir=TMP_DIR,
        voiced_tokens=voiced_tokens,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=set_seeds,
        collate_fn=partial(collate_fn, features_names=[feature]),
    )

    if pretrained:
        best_model = DeepSpeech2.load_librispeech_model(
            model_params["num_features"],
            adapter_out_features=model_params.get("adapter_out_features")
        )
        hidden_size = model_params["rnn_hidden_size"]
        best_model.classifier = nn.Linear(hidden_size, len(vocabulary))
    else:
        model = DeepSpeech2(num_classes=len(vocabulary), **model_params)
    best_model_state_dict = torch.load(best_model_path, map_location=device)
    best_model.load_state_dict(best_model_state_dict)
    best_model.to(device)

    info_test = run_test(
        model=best_model,
        dataloader=test_dataloader,
        fn_metrics=metrics,
        device=device,
        feature=feature,
        target=target,
        use_voicing=(voicing_filepath is not None),
        save_dir=RESULTS_DIR,
    )

    info_test_filepath = os.path.join(RESULTS_DIR, "info_test.json")
    with open(info_test_filepath, "w") as f:
        ujson.dump(info_test, f)

    mlflow.log_artifact(info_test_filepath)
    mlflow.log_artifact(os.path.join(RESULTS_DIR, "confusion_matrix.pdf"))
    mlflow.log_artifact(os.path.join(RESULTS_DIR, "model_features.pdf"))

    mlflow.log_metrics(
        {
            f"test_{metric}": value
            for metric, value in info_test.items()
        },
        step=epoch
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_filepath")
    parser.add_argument("--experiment", dest="experiment_name", default="phoneme_recognition")
    parser.add_argument("--run", dest="run_name", default=None)
    args = parser.parse_args()

    with open(args.config_filepath) as f:
        cfg = yaml.safe_load(f)

    experiment = mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=args.run_name
    ):
        mlflow.log_dict(cfg, "config.json")
        try:
            main(**cfg)
        finally:
            shutil.rmtree(TMP_DIR)
