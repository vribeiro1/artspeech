import pdb

import logging
import numpy as np
import os
import torch
import torch.nn as nn

from sacred import Experiment
from sacred.observers import FileStorageObserver
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from area_function_estimation.dataset import AreaFunctionDataset2
from area_function_estimation.evaluation import run_test
from articulation_to_melspec.model import ArticulatorsEmbedding
from helpers import set_seeds

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN = "train"
VALID = "validation"
TEST = "test"

ex = Experiment()
fs_observer = FileStorageObserver.create(os.path.join(BASE_DIR, "area_function_estimation", "results"))
ex.observers.append(fs_observer)


def run_epoch(phase, epoch, model, dataloader, optimizer, criterion, writer=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training = phase == TRAIN

    if training:
        model.train()
    else:
        model.eval()

    losses = []
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - {phase}")
    for i, (frame_articulators, _, fx) in enumerate(progress_bar):
        frame_articulators = frame_articulators.to(device)
        fx = fx.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            outputs = model(frame_articulators.unsqueeze(dim=1))
            loss = criterion(outputs.squeeze(dim=1), fx)

            if training:
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
            progress_bar.set_postfix(loss=np.mean(losses))

    mean_loss = np.mean(losses)
    loss_tag = f"{phase}/loss"
    if writer is not None:
        writer.add_scalar(loss_tag, mean_loss, epoch)

    info = {
        "loss": mean_loss
    }

    return info


@ex.automain
def main(
    _run, datadir, n_epochs, batch_size, patience, learning_rate, weight_decay,
    train_seq_dict, valid_seq_dict, test_seq_dict, articulators,
    clip_tails=True, state_dict_fpath=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on '{device.type}'")

    writer = SummaryWriter(os.path.join(fs_observer.dir, f"experiment"))
    best_model_path = os.path.join(fs_observer.dir, "best_model.pt")
    last_model_path = os.path.join(fs_observer.dir, "last_model.pt")

    outputs_dir = os.path.join(fs_observer.dir, "outputs")
    if not os.path.exists(outputs_dir):
        os.mkdir(outputs_dir)

    n_articulators = len(articulators)

    embed = ArticulatorsEmbedding(n_curves=n_articulators, n_samples=50, embed_size=200)
    embed = embed.to(device)

    loss_fn = nn.L1Loss()

    optimizer = Adam(
        embed.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=0.1,
        patience=5,
    )

    train_dataset = AreaFunctionDataset2(datadir, train_seq_dict, clip_tails=clip_tails)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=set_seeds
    )

    valid_dataset = AreaFunctionDataset2(datadir, valid_seq_dict, clip_tails=clip_tails)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=set_seeds
    )

    epochs = range(1, n_epochs + 1)
    best_metric = np.inf
    epochs_since_best = 0
    for epoch in epochs:
        info_train = run_epoch(
            phase=TRAIN,
            epoch=epoch,
            model=embed,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=loss_fn,
            writer=writer,
            device=device
        )

        info_valid = run_epoch(
            phase=VALID,
            epoch=epoch,
            model=embed,
            dataloader=valid_dataloader,
            optimizer=optimizer,
            criterion=loss_fn,
            writer=writer,
            device=device
        )

        scheduler.step(info_valid["loss"])

        if info_valid["loss"] < best_metric:
            best_metric = info_valid["loss"]
            epochs_since_best = 0
            torch.save(embed.state_dict(), best_model_path)
        else:
            epochs_since_best += 1

        torch.save(embed.state_dict(), last_model_path)

        if epochs_since_best > patience:
            break

    test_dataset = AreaFunctionDataset2(datadir, test_seq_dict, clip_tails=clip_tails)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=set_seeds
    )

    best_embed = ArticulatorsEmbedding(
        n_curves=n_articulators,
        n_samples=50,
        embed_size=200
    )
    state_dict = torch.load(best_model_path, map_location=device)
    best_embed.load_state_dict(state_dict)
    embed.to(device)

    test_outputs_dir = os.path.join(fs_observer.dir, "test_outputs")
    if not os.path.exists(test_outputs_dir):
        os.makedirs(test_outputs_dir)

    info_test = run_test(
        epoch=0,
        model=best_embed,
        dataloader=test_dataloader,
        criterion=loss_fn,
        outputs_dir=test_outputs_dir,
        device=device
    )
