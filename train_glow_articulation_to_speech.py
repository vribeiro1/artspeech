import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from glow_tts.commons import Adam, mle_loss, duration_loss, clip_grad_value
from sacred import Experiment
from sacred.observers import FileStorageObserver
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from articulation_to_melspec.dataset import ArticulToMelSpecDataset, pad_sequence_collate_fn
from articulation_to_melspec.evaluation import run_glow_tts_inference
from articulation_to_melspec.model import GlowATS
from loss import Tacotron2Loss
from helpers import set_seeds, sequences_from_dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN = "train"
VALID = "validation"
TEST = "test"

ex = Experiment()
fs_observer = FileStorageObserver.create(os.path.join(BASE_DIR, "articulation_to_melspec", "results"))
ex.observers.append(fs_observer)


def run_epoch(phase, epoch, model, dataloader, optimizer, writer=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training = phase == TRAIN
    if training:
        model.train()
    else:
        model.eval()

    losses = []
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - {phase}")
    for _, sentences, len_sentences, targets, _, len_targets in progress_bar:
        sentences = sentences.to(device)
        len_sentences = len_sentences.to(device)
        targets = targets.to(device)
        len_targets = len_targets.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_) = model(
                sentences, len_sentences, targets, len_targets, gen=False
            )

            loss_mle = mle_loss(z, z_m, z_logs, logdet, z_mask)
            loss_length = duration_loss(logw, logw_, len_sentences)
            loss = loss_mle + loss_length

            if training:
                loss.backward()
                clip_grad_value(model.parameters(), 5)
                optimizer.step()

        losses.append(loss.item())
        progress_bar.set_postfix(loss=np.mean(losses))

    mean_loss = np.mean(loss.item())
    loss_tag = f"{phase}/loss"
    if writer is not None:
        writer.add_scalar(loss_tag, mean_loss, epoch)

    info = {
        "loss": mean_loss
    }

    return info


@ex.automain
def main(
    _run, datadir, batch_size, n_epochs, patience, learning_rate, weight_decay,
    train_seq_dict, valid_seq_dict, test_seq_dict,
    articulators, num_workers=4, state_dict_fpath=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on '{device.type}'")

    writer = SummaryWriter(os.path.join(fs_observer.dir, f"experiment"))
    best_model_path = os.path.join(fs_observer.dir, "best_model.pt")
    last_model_path = os.path.join(fs_observer.dir, "last_model.pt")

    glow_tts_hparams = {
        "n_vocab": 148,
        "out_channels": 80,  # n_mel_channels on the Mel Spectrogram
        "hidden_channels": 192,
        "filter_channels": 768,
        "filter_channels_dp": 256,
        "kernel_size": 3,
        "p_dropout": 0.1,
        "n_blocks_dec": 12,
        "n_layers_enc": 6,
        "n_heads": 2,
        "p_dropout_dec": 0.05,
        "dilation_rate": 1,
        "kernel_size_dec": 5,
        "n_block_layers": 4,
        "n_sqz": 2,
        "prenet": True,
        "mean_only": True,
        "hidden_channels_enc": 192,
        "hidden_channels_dec": 192,
        "window_size": 4
    }

    model = GlowATS(
        n_articulators=len(articulators),
        n_samples=50,
        pretrained=True,
        **glow_tts_hparams
    )
    if state_dict_fpath is not None:
        state_dict = torch.load(state_dict_fpath, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)

    optimizer = Adam(
        model.parameters(),
        scheduler="noam",
        dim_model=glow_tts_hparams["hidden_channels"],
        warmup_steps=4000,
        lr=learning_rate,
        betas=[0.9, 0.98],
        eps=1e-9
    )

    train_sequences = sequences_from_dict(datadir, train_seq_dict)
    train_dataset = ArticulToMelSpecDataset(datadir, train_sequences, articulators)
    assert len(train_dataset) % batch_size > 1
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=set_seeds,
        collate_fn=pad_sequence_collate_fn
    )

    valid_sequences = sequences_from_dict(datadir, valid_seq_dict)
    valid_dataset = ArticulToMelSpecDataset(datadir, valid_sequences, articulators)
    assert len(valid_dataset) % batch_size > 1
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=set_seeds,
        collate_fn=pad_sequence_collate_fn
    )

    epochs = range(1, n_epochs + 1)
    best_metric = np.inf
    epochs_since_best = 0

    for epoch in epochs:
        info_train = run_epoch(
            phase=TRAIN,
            epoch=epoch,
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            writer=writer,
            device=device
        )

        info_valid = run_epoch(
            phase=VALID,
            epoch=epoch,
            model=model,
            dataloader=valid_dataloader,
            optimizer=optimizer,
            writer=writer,
            device=device
        )

        if info_valid["loss"] < best_metric:
            best_metric = info_valid["loss"]
            torch.save(model.state_dict(), best_model_path)
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        torch.save(model.state_dict(), last_model_path)

        if epochs_since_best > patience:
            break

    test_sequences = sequences_from_dict(datadir, test_seq_dict)
    test_dataset = ArticulToMelSpecDataset(datadir, test_sequences, articulators)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=set_seeds,
        collate_fn=pad_sequence_collate_fn
    )

    test_outputs_dir = os.path.join(fs_observer.dir, "test_outputs")
    if not os.path.exists(test_outputs_dir):
        os.makedirs(test_outputs_dir)

    run_glow_tts_inference(
        model=model,
        dataloader=test_dataloader,
        device=device,
        save_to=test_outputs_dir
    )
