import pdb

import json
import logging
import numpy as np
import os
import torch
import torch.nn as nn

from collections import OrderedDict
from sacred import Experiment
from sacred.observers import FileStorageObserver
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from articulation_to_melspec.model import ArticulatorsEmbedding
from helpers import set_seeds
from phoneme_to_articulation.metrics import EuclideanDistance
from phoneme_to_articulation.encoder_decoder.dataset import ArtSpeechDataset, pad_sequence_collate_fn
from phoneme_to_articulation.encoder_decoder.evaluation import save_outputs
from phoneme_to_articulation.encoder_decoder.models import Decoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN = "train"
VALID = "validation"
TEST = "test"

ex = Experiment()
fs_observer = FileStorageObserver.create(os.path.join(BASE_DIR, "articulatory_autoencoder", "results"))
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
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - {phase}")
    for i, (_, _, targets, _, _) in enumerate(progress_bar):
        targets = targets.type(torch.float).to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            outputs = model(targets)
            loss = criterion(outputs, targets)

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


def run_test(epoch, model, dataloader, criterion, outputs_dir, articulators, device=None, regularize_out=False):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch_outputs_dir = os.path.join(outputs_dir, str(epoch))
    if not os.path.exists(epoch_outputs_dir):
        os.makedirs(epoch_outputs_dir)

    model.eval()

    losses = []
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - inference")
    for i_batch, (sentences_ids, _, targets, lengths, phonemes) in enumerate(progress_bar):
        targets = targets.type(torch.float).to(device)

        with torch.set_grad_enabled(False):
            outputs = model(targets)
            loss = criterion(outputs, targets)

            outputs = outputs.detach().cpu()
            targets = targets.detach().cpu()

            losses.append(loss.item())
            progress_bar.set_postfix(loss=np.mean(losses))

            save_outputs(
                sentences_ids,
                outputs,
                targets,
                lengths,
                phonemes,
                articulators,
                epoch_outputs_dir,
                regularize_out
            )

    mean_loss = np.mean(losses)

    info = {
        "loss": mean_loss
    }

    return info


@ex.automain
def main(
    _run, datadir, n_epochs, batch_size, patience, learning_rate, weight_decay,
    train_filepath, valid_filepath, test_filepath, vocab_filepath, articulators,
    clip_tails=True, state_dict_fpath=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on '{device.type}'")

    writer = SummaryWriter(os.path.join(fs_observer.dir, f"experiment"))
    best_encoder_path = os.path.join(fs_observer.dir, "best_encoder.pt")
    best_decoder_path = os.path.join(fs_observer.dir, "best_decoder.pt")
    last_encoder_path = os.path.join(fs_observer.dir, "last_encoder.pt")
    last_decoder_path = os.path.join(fs_observer.dir, "last_decoder.pt")

    outputs_dir = os.path.join(fs_observer.dir, "outputs")
    if not os.path.exists(outputs_dir):
        os.mkdir(outputs_dir)

    with open(vocab_filepath) as f:
        tokens = json.load(f)
        vocabulary = {token: i for i, token in enumerate(tokens)}

    n_articulators = len(articulators)

    embed_size = 192
    encoder = ArticulatorsEmbedding(n_curves=n_articulators, n_samples=50, embed_size=embed_size)
    decoder = Decoder(n_articulators=n_articulators, hidden_size=embed_size, n_samples=50)

    model = nn.Sequential(OrderedDict([
        ("encoder", encoder),
        ("decoder", decoder)
    ]))
    model.to(device)

    loss_fn = EuclideanDistance()
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
        clip_tails=clip_tails
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
        clip_tails=clip_tails
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
            torch.save(model.encoder.state_dict(), best_encoder_path)
            torch.save(model.decoder.state_dict(), best_decoder_path)
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        torch.save(model.encoder.state_dict(), last_encoder_path)
        torch.save(model.decoder.state_dict(), last_decoder_path)

        if epochs_since_best > patience:
            break

    test_dataset = ArtSpeechDataset(
        os.path.dirname(datadir),
        test_filepath,
        vocabulary,
        articulators,
        clip_tails=clip_tails,
        lazy_load=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=set_seeds,
        collate_fn=pad_sequence_collate_fn
    )

    best_encoder = ArticulatorsEmbedding(n_curves=n_articulators, n_samples=50, embed_size=embed_size)
    state_dict = torch.load(best_encoder_path, map_location=device)
    best_encoder.load_state_dict(state_dict)
    best_encoder.to(device)

    best_decoder = Decoder(n_articulators=n_articulators, hidden_size=embed_size, n_samples=50)
    state_dict = torch.load(best_decoder_path, map_location=device)
    best_decoder.load_state_dict(state_dict)
    best_decoder.to(device)

    best_model = nn.Sequential(OrderedDict([
        ("encoder", encoder),
        ("decoder", decoder)
    ]))
    model.to(device)

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
