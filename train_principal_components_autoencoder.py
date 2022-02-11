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

from helpers import set_seeds, sequences_from_dict
from phoneme_to_articulation.principal_components.dataset import PrincipalComponentsAutoencoderDataset
from phoneme_to_articulation.principal_components.evaluation import run_autoencoder_test
from phoneme_to_articulation.principal_components.models import Autoencoder

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
    for _, anchor, pos, neg, sample_weigths, _ in progress_bar:
        anchor = anchor.to(device)
        pos = pos.to(device)
        neg = neg.to(device)
        sample_weigths = sample_weigths.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            anchor_outputs, anchor_latents = model(anchor)
            _, pos_latents = model(pos)
            _, neg_latents = model(neg)

            loss = criterion(
                anchor_outputs,
                anchor_latents,
                pos_latents,
                neg_latents,
                anchor,
                sample_weigths
            )

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


class RegularizedLatentsMSELoss(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss(reduction="none")
        self.triplet = nn.TripletMarginLoss()

    def forward(self, anchor_outputs, anchor_latents, pos_latents, neg_latents, anchor_target, sample_weights=None):
        mse = self.mse(anchor_outputs, anchor_target)
        if sample_weights is not None:
            mse = (sample_weights * mse.T).T
        mse = mse.mean()

        triplet = self.triplet(anchor_latents, pos_latents, neg_latents)
        reg_latents = torch.norm(anchor_latents, p=2, dim=1).mean()
        cov_features = torch.cov(anchor_latents.T).square().sum()

        return mse + triplet + self.alpha * reg_latents + self.beta * cov_features


@ex.automain
def main(
    _run, datadir, n_epochs, batch_size, patience, learning_rate, weight_decay,
    train_seq_dict, valid_seq_dict, test_seq_dict, articulator,
    clip_tails=True, state_dict_fpath=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on '{device.type}'")

    writer = SummaryWriter(os.path.join(fs_observer.dir, f"experiment"))
    best_encoder_path = os.path.join(fs_observer.dir, "best_encoder.pt")
    best_decoder_path = os.path.join(fs_observer.dir, "best_decoder.pt")
    last_encoder_path = os.path.join(fs_observer.dir, "last_encoder.pt")
    last_decoder_path = os.path.join(fs_observer.dir, "last_decoder.pt")

    autoencoder = Autoencoder(in_features=100, n_components=12)
    if state_dict_fpath is not None:
        state_dict = torch.load(state_dict_fpath, map_location=device)
        autoencoder.load_state_dict(state_dict)
    autoencoder.to(device)

    loss_fn = RegularizedLatentsMSELoss(alpha=1e-2, beta=1e-2)
    optimizer = Adam(autoencoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

    train_sequences = sequences_from_dict(datadir, train_seq_dict)
    train_dataset = PrincipalComponentsAutoencoderDataset(
        datadir=datadir,
        sequences=train_sequences,
        articulator=articulator,
        sync_shift=0,
        framerate=55,
        clip_tails=clip_tails
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=5,
        worker_init_fn=set_seeds
    )

    valid_sequences = sequences_from_dict(datadir, valid_seq_dict)
    valid_dataset = PrincipalComponentsAutoencoderDataset(
        datadir=datadir,
        sequences=valid_sequences,
        articulator=articulator,
        sync_shift=0,
        framerate=55,
        clip_tails=clip_tails
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=5,
        worker_init_fn=set_seeds
    )

    best_metric = np.inf
    epochs_since_best = 0

    epochs = range(1, n_epochs + 1)
    for epoch in epochs:
        info_train = run_epoch(
            phase=TRAIN,
            epoch=epoch,
            model=autoencoder,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=loss_fn,
            writer=writer,
            device=device
        )

        info_valid = run_epoch(
            phase=VALID,
            epoch=epoch,
            model=autoencoder,
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
            torch.save(autoencoder.encoder.state_dict(), best_encoder_path)
            torch.save(autoencoder.decoder.state_dict(), best_decoder_path)
        else:
            epochs_since_best += 1

        torch.save(autoencoder.encoder.state_dict(), last_encoder_path)
        torch.save(autoencoder.decoder.state_dict(), last_decoder_path)

        if epochs_since_best > patience:
            break

    test_sequences = sequences_from_dict(datadir, test_seq_dict)
    test_dataset = PrincipalComponentsAutoencoderDataset(
        datadir=datadir,
        sequences=test_sequences,
        articulator=articulator,
        sync_shift=0,
        framerate=55,
        clip_tails=clip_tails
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=5,
        worker_init_fn=set_seeds
    )

    best_autoencoder = Autoencoder(in_features=100, n_components=12)
    best_encoder_state_dict = torch.load(best_encoder_path, map_location=device)
    best_autoencoder.encoder.load_state_dict(best_encoder_state_dict)
    best_decoder_state_dict = torch.load(best_decoder_path, map_location=device)
    best_autoencoder.decoder.load_state_dict(best_decoder_state_dict)
    best_autoencoder.to(device)

    test_outputs_dir = os.path.join(fs_observer.dir, "test_outputs")
    if not os.path.exists(test_outputs_dir):
        os.makedirs(test_outputs_dir)

    info_test = run_autoencoder_test(
        epoch=0,
        model=best_autoencoder,
        dataloader=test_dataloader,
        criterion=loss_fn,
        outputs_dir=test_outputs_dir,
        device=device
    )
