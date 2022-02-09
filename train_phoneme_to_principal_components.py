import pdb

import logging
import numpy as np
import os
import torch
import torch.nn as nn
import ujson

from sacred import Experiment
from sacred.observers import FileStorageObserver
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from helpers import set_seeds, sequences_from_dict
from phoneme_to_articulation.encoder_decoder.dataset import pad_sequence_collate_fn
from phoneme_to_articulation.principal_components.dataset import PrincipalComponentsPhonemeToArticulationDataset
from phoneme_to_articulation.principal_components.evaluation import run_phoneme_to_PC_test
from phoneme_to_articulation.principal_components.losses import AutoencoderLoss
from phoneme_to_articulation.principal_components.models import PrincipalComponentsArtSpeech

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN = "train"
VALID = "validation"
TEST = "test"

ex = Experiment()
fs_observer = FileStorageObserver.create(os.path.join(BASE_DIR, "phoneme_to_articulation", "principal_components", "results"))
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
    for _, sentence, targets, lengths, _ in progress_bar:
        sentence = sentence.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            outputs = model(sentence, lengths)
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


@ex.automain
def main(
    _run, datadir, n_epochs, batch_size, patience, learning_rate, weight_decay,
    train_seq_dict, valid_seq_dict, test_seq_dict, articulator,
    vocab_fpath, encoder_state_dict_fpath, decoder_state_dict_fpath,
    clip_tails=True, state_dict_fpath=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on '{device.type}'")

    writer = SummaryWriter(os.path.join(fs_observer.dir, f"experiment"))
    best_model_path = os.path.join(fs_observer.dir, "best_model.pt")
    last_model_path = os.path.join(fs_observer.dir, "last_model.pt")

    with open(vocab_fpath) as f:
        tokens = ujson.load(f)
        vocabulary = {token: i for i, token in enumerate(tokens)}

    model = PrincipalComponentsArtSpeech(vocab_size=len(vocabulary), n_components=12, gru_dropout=0.2)
    if state_dict_fpath is not None:
        state_dict = torch.load(state_dict_fpath, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)

    loss_fn = AutoencoderLoss(
        in_features=100,
        n_components=12,
        encoder_state_dict_fpath=encoder_state_dict_fpath,
        decoder_state_dict_fpath=decoder_state_dict_fpath,
        device=device
    )
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

    train_sequences = sequences_from_dict(datadir, train_seq_dict)
    train_dataset = PrincipalComponentsPhonemeToArticulationDataset(
        datadir=datadir,
        sequences=train_sequences,
        vocabulary=vocabulary,
        articulator=articulator,
        sync_shift=0,
        framerate=55,
        clip_tails=clip_tails
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=set_seeds,
        collate_fn=pad_sequence_collate_fn
    )

    valid_sequences = sequences_from_dict(datadir, valid_seq_dict)
    valid_dataset = PrincipalComponentsPhonemeToArticulationDataset(
        datadir=datadir,
        sequences=valid_sequences,
        vocabulary=vocabulary,
        articulator=articulator,
        sync_shift=0,
        framerate=55,
        clip_tails=clip_tails
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=set_seeds,
        collate_fn=pad_sequence_collate_fn
    )

    best_metric = np.inf
    epochs_since_best = 0

    epochs = range(1, n_epochs + 1)
    for epoch in epochs:
        info_train = run_epoch(
            phase=TRAIN,
            epoch=epoch,
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=loss_fn,
            writer=writer,
            device=device
        )

        info_valid = run_epoch(
            phase=VALID,
            epoch=epoch,
            model=model,
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
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_since_best += 1

        torch.save(model.state_dict(), last_model_path)

        if epochs_since_best > patience:
            break

    test_sequences = sequences_from_dict(datadir, test_seq_dict)
    test_dataset = PrincipalComponentsPhonemeToArticulationDataset(
        datadir=datadir,
        sequences=test_sequences,
        vocabulary=vocabulary,
        articulator=articulator,
        sync_shift=0,
        framerate=55,
        clip_tails=clip_tails
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=set_seeds,
        collate_fn=pad_sequence_collate_fn
    )

    best_model = PrincipalComponentsArtSpeech(vocab_size=len(vocabulary), n_components=12, gru_dropout=0.2)
    best_model_state_dict = torch.load(best_model_path, map_location=device)
    best_model.load_state_dict(best_model_state_dict)
    best_model.to(device)

    test_outputs_dir = os.path.join(fs_observer.dir, "test_outputs")
    if not os.path.exists(test_outputs_dir):
        os.makedirs(test_outputs_dir)

    info_test = run_phoneme_to_PC_test(
        epoch=0,
        model=best_model,
        decoder_state_dict_fpath=decoder_state_dict_fpath,
        criterion=loss_fn,
        outputs_dir=test_outputs_dir,
        device=device
    )
