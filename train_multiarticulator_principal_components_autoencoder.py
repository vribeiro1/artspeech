import logging
import numpy as np
import os
import torch

from sacred import Experiment
from sacred.observers import FileStorageObserver
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
from torch.utils.data import DataLoader
from vt_tools import *

from helpers import set_seeds, sequences_from_dict
from phoneme_to_articulation.principal_components import run_autoencoder_epoch, TRAIN, VALID
from phoneme_to_articulation.principal_components.dataset import PrincipalComponentsMultiArticulatorAutoencoderDataset
from phoneme_to_articulation.principal_components.evaluation import run_multiart_autoencoder_test
from phoneme_to_articulation.principal_components.losses import MultiArtRegularizedLatentsMSELoss
from phoneme_to_articulation.principal_components.models import MultiArticulatorAutoencoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ex = Experiment()
fs_observer = FileStorageObserver.create(
    os.path.join(
        BASE_DIR,
        "phoneme_to_articulation",
        "principal_components",
        "results",
        "autoencoder"
    )
)
ex.observers.append(fs_observer)


@ex.automain
def main(
    _run, datadir, n_epochs, batch_size, patience, learning_rate, weight_decay,
    train_seq_dict, valid_seq_dict, test_seq_dict, articulators_indices_dict,
    hidden_blocks, hidden_features, clip_tails=True, state_dict_fpath=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on '{device.type}'")

    writer = SummaryWriter(os.path.join(fs_observer.dir, f"experiment"))
    best_encoders_path = os.path.join(fs_observer.dir, "best_encoders.pt")
    best_decoders_path = os.path.join(fs_observer.dir, "best_decoders.pt")
    last_encoders_path = os.path.join(fs_observer.dir, "last_encoders.pt")
    last_decoders_path = os.path.join(fs_observer.dir, "last_decoders.pt")

    articulators = sorted(articulators_indices_dict.keys())

    model_kwargs = dict(
        in_features=100,
        indices_dict=articulators_indices_dict,
        hidden_blocks=hidden_blocks,
        hidden_features=hidden_features
    )

    autoencoder = MultiArticulatorAutoencoder(**model_kwargs)

    if state_dict_fpath is not None:
        state_dict = torch.load(state_dict_fpath, map_location=device)
        autoencoder.load_state_dict(state_dict)
    autoencoder.to(device)

    loss_fn = MultiArtRegularizedLatentsMSELoss(alpha=1e-2, indices_dict=articulators_indices_dict)
    optimizer = Adam(autoencoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10)
    # scheduler = CyclicLR(
    #     optimizer,
    #     base_lr=learning_rate/10,
    #     max_lr=learning_rate,
    #     cycle_momentum=False
    # )

    num_workers = 5
    train_sequences = sequences_from_dict(datadir, train_seq_dict)
    train_dataset = PrincipalComponentsMultiArticulatorAutoencoderDataset(
        datadir=datadir,
        sequences=train_sequences,
        articulators=articulators,
        sync_shift=0,
        framerate=55,
        clip_tails=clip_tails
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=set_seeds
    )

    valid_sequences = sequences_from_dict(datadir, valid_seq_dict)
    valid_dataset = PrincipalComponentsMultiArticulatorAutoencoderDataset(
        datadir=datadir,
        sequences=valid_sequences,
        articulators=articulators,
        sync_shift=0,
        framerate=55,
        clip_tails=clip_tails
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=set_seeds
    )

    best_metric = np.inf
    epochs_since_best = 0

    epochs = range(1, n_epochs + 1)
    for epoch in epochs:
        info_train = run_autoencoder_epoch(
            phase=TRAIN,
            epoch=epoch,
            model=autoencoder,
            dataloader=train_dataloader,
            optimizer=optimizer,
            # scheduler=scheduler,
            criterion=loss_fn,
            writer=writer,
            device=device
        )

        info_valid = run_autoencoder_epoch(
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
            torch.save(autoencoder.encoders.state_dict(), best_encoders_path)
            torch.save(autoencoder.decoders.state_dict(), best_decoders_path)
        else:
            epochs_since_best += 1

        torch.save(autoencoder.encoders.state_dict(), last_encoders_path)
        torch.save(autoencoder.decoders.state_dict(), last_decoders_path)

        if epochs_since_best > patience:
            break

    test_sequences = sequences_from_dict(datadir, test_seq_dict)
    test_dataset = PrincipalComponentsMultiArticulatorAutoencoderDataset(
        datadir=datadir,
        sequences=test_sequences,
        articulators=articulators,
        sync_shift=0,
        framerate=55,
        clip_tails=clip_tails
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=set_seeds
    )

    best_autoencoder = MultiArticulatorAutoencoder(
        **model_kwargs
    )
    best_encoders_state_dict = torch.load(best_encoders_path, map_location=device)
    best_autoencoder.encoders.load_state_dict(best_encoders_state_dict)
    best_decoders_state_dict = torch.load(best_decoders_path, map_location=device)
    best_autoencoder.decoders.load_state_dict(best_decoders_state_dict)
    best_autoencoder.to(device)

    test_outputs_dir = os.path.join(fs_observer.dir, "test_outputs")
    if not os.path.exists(test_outputs_dir):
        os.makedirs(test_outputs_dir)

    info_test = run_multiart_autoencoder_test(
        epoch=0,
        model=best_autoencoder,
        dataloader=test_dataloader,
        criterion=loss_fn,
        outputs_dir=test_outputs_dir,
        device=device
    )
