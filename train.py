import pdb

import json
import logging
import numpy as np
import os
import torch

from sacred import Experiment
from sacred.observers import FileStorageObserver
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ArtSpeechDataset, pad_sequence_collate_fn
from evaluation import run_test
from helpers import set_seeds
from loss import EuclideanDistanceLoss
from metrics import pearsons_correlation
from model import ArtSpeech

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN = "train"
VALID = "validation"
TEST = "test"

ex = Experiment()
fs_observer = FileStorageObserver.create(os.path.join(BASE_DIR, "results"))
ex.observers.append(fs_observer)


def run_epoch(phase, epoch, model, dataloader, optimizer, criterion, articulators, writer=None, device=None):
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
    for i, (sentence, targets, lengths, _) in enumerate(progress_bar):
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
    loss_tag = f"{phase}/loss"
    if writer is not None:
        writer.add_scalar(loss_tag, mean_loss, epoch)

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


@ex.automain
def main(
    _run, datadir, n_epochs, batch_size, patience, learning_rate, weight_decay,
    train_filepath, valid_filepath, test_filepath, vocab_filepath, articulators,
    p_aug=0., state_dict_fpath=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on '{device.type}'")

    writer = SummaryWriter(os.path.join(fs_observer.dir, f"experiment"))
    best_model_path = os.path.join(fs_observer.dir, "best_model.pt")
    last_model_path = os.path.join(fs_observer.dir, "last_model.pt")

    outputs_dir = os.path.join(fs_observer.dir, "outputs")
    if not os.path.exists(outputs_dir):
        os.mkdir(outputs_dir)

    with open(vocab_filepath) as f:
        tokens = json.load(f)
        vocabulary = {token: i for i, token in enumerate(tokens)}

    n_articulators = len(articulators)

    model = ArtSpeech(len(vocabulary), n_articulators, gru_dropout=0.2)
    if state_dict_fpath is not None:
        state_dict = torch.load(state_dict_fpath, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)

    loss_fn = EuclideanDistanceLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=0.1,
        patience=10,
    )

    train_dataset = ArtSpeechDataset(
        os.path.dirname(datadir),
        train_filepath,
        vocabulary,
        articulators,
        p_aug=p_aug,
        clip_tails=True
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=set_seeds,
        collate_fn=pad_sequence_collate_fn
    )

    valid_dataset = ArtSpeechDataset(
        os.path.dirname(datadir),
        valid_filepath,
        vocabulary,
        articulators,
        p_aug=p_aug,
        clip_tails=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=set_seeds,
        collate_fn=pad_sequence_collate_fn
    )

    info = {}
    epochs = range(1, n_epochs + 1)
    best_metric = np.inf
    epochs_since_best = 0

    for epoch in epochs:
        info[TRAIN] = run_epoch(
            phase=TRAIN,
            epoch=epoch,
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=loss_fn,
            articulators=train_dataset.articulators,
            writer=writer,
            device=device
        )

        info[VALID] = run_epoch(
            phase=VALID,
            epoch=epoch,
            model=model,
            dataloader=valid_dataloader,
            optimizer=optimizer,
            criterion=loss_fn,
            articulators=valid_dataset.articulators,
            writer=writer,
            device=device
        )

        scheduler.step(info[VALID]["loss"])

        if info[VALID]["loss"] < best_metric:
            best_metric = info[VALID]["loss"]
            torch.save(model.state_dict(), best_model_path)
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        torch.save(model.state_dict(), last_model_path)

        if epochs_since_best > patience:
            break

    test_dataset = ArtSpeechDataset(
        os.path.dirname(datadir),
        test_filepath,
        vocabulary,
        articulators,
        clip_tails=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=set_seeds,
        collate_fn=pad_sequence_collate_fn
    )

    best_model = ArtSpeech(len(vocabulary), n_articulators, gru_dropout=0.2)
    state_dict = torch.load(best_model_path, map_location=device)
    best_model.load_state_dict(state_dict)
    best_model.to(device)

    test_outputs_dir = os.path.join(fs_observer.dir, "test_outputs")
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

    test_results_filepath = os.path.join(fs_observer.dir, "test_results.json")
    with open(test_results_filepath, "w") as f:
        json.dump(test_results, f)

    results_item = {
        "exp": _run.id,
        "loss": test_results["loss"],
    }

    for articulator in test_dataset.articulators:
        results_item[f"p2cp_{articulator}"] = test_results[articulator]["p2cp"]
        results_item[f"med_{articulator}"] = test_results[articulator]["med"]
        results_item[f"x_corr_{articulator}"] = test_results[articulator]["x_corr"]
        results_item[f"y_corr_{articulator}"] = test_results[articulator]["y_corr"]

    df = pd.DataFrame([results_item])
    df_filepath = os.path.join(fs_observer.dir, "test_results.csv")
    df.to_csv(df_filepath, index=False)
