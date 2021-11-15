import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from sacred import Experiment
from sacred.observers import FileStorageObserver
from scipy.io import wavfile
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchaudio.models import Tacotron2
from tqdm import tqdm

from articul_to_melspec import NVIDIA_TACOTRON2_WEIGHTS_FILEPATH
from articul_to_melspec.dataset import ArticulToMelSpecDataset, pad_sequence_collate_fn
from articul_to_melspec.model import ArticulatorsEmbedding
from articul_to_melspec.waveglow import melspec_to_audio
from loss import Tacotron2Loss
from helpers import set_seeds

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN = "train"
VALID = "validation"
TEST = "test"

ex = Experiment()
fs_observer = FileStorageObserver.create(os.path.join(BASE_DIR, "articul_to_melspec", "results"))
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
    for _, sentences, len_sentences, targets, gate_targets, len_targets in progress_bar:
        sentences = sentences.to(device)
        len_sentences = len_sentences.to(device)
        targets = targets.to(device)
        gate_targets = gate_targets.to(device)
        len_targets = len_targets.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            outputs = model(sentences, len_sentences, targets, len_targets)
            loss = criterion(outputs, (targets, gate_targets))

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


def run_test(model, dataloader, criterion, device=None, save_to=None, sampling_rate=22050):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    losses = []
    progress_bar = tqdm(dataloader, desc=f"Running test")

    for sentences_names, sentences, len_sentences, targets, gate_targets, len_targets in progress_bar:
        sentences = sentences.to(device)
        len_sentences = len_sentences.to(device)
        targets = targets.to(device)
        gate_targets = gate_targets.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(sentences, len_sentences, targets, len_targets)
            loss = criterion(outputs, (targets, gate_targets))

        mel_specs, mel_specs_postnet, gate_outputs, attn_weights = outputs

        losses.append(loss.item())
        progress_bar.set_postfix(loss=np.mean(losses))

        target_audios = melspec_to_audio(targets)
        output_audios = melspec_to_audio(mel_specs_postnet)

        for sentence_name, target_spec, target_audio, output_spec, output_audio in zip(
            sentences_names, targets, target_audios, mel_specs_postnet, output_audios
        ):
            target_spec = target_spec.cpu().detach().numpy()
            spec_save_filepath = os.path.join(save_to, f"{sentence_name}_ground_truth.pt")
            torch.save(target_spec, spec_save_filepath)

            target_audio = target_audio.cpu().numpy()
            target_audio = target_audio.astype("int16")
            audio_save_filepath = os.path.join(save_to, f"{sentence_name}_ground_truth.wav")
            wavfile.write(audio_save_filepath, sampling_rate, target_audio)

            output_spec = output_spec.cpu().detach().numpy()
            spec_save_filepath = os.path.join(save_to, f"{sentence_name}.pt")
            torch.save(output_spec, spec_save_filepath)

            output_audio = output_audio.cpu().numpy()
            output_audio = output_audio.astype("int16")
            audio_save_filepath = os.path.join(save_to, f"{sentence_name}.wav")
            wavfile.write(audio_save_filepath, sampling_rate, output_audio)

            plt.figure(figsize=(20, 20))

            plt.subplot(2, 1, 1)
            ax = plt.imshow(target_spec, origin="lower", aspect="auto", cmap="coolwarm")
            plt.colorbar(ax)

            plt.title("Target MelSpectogram")
            plt.xlabel("Frame")
            plt.ylabel("Frequenct bin")
            plt.grid()

            plt.subplot(2, 1, 2)
            ax = plt.imshow(output_spec, origin="lower", aspect="auto", cmap="coolwarm")
            plt.colorbar(ax)

            plt.title("Predicted MelSpectogram")
            plt.xlabel("Frame")
            plt.ylabel("Frequenct bin")
            plt.grid()

            plt.tight_layout()
            fig_save_filepath = os.path.join(save_to, f"{sentence_name}.jpg")
            plt.savefig(fig_save_filepath)
            plt.close()

    info = {
        "loss": np.mean(losses)
    }

    return info


def sequences_from_dict(datadir, sequences_dict):
    sequences = []
    for subj, seqs in sequences_dict.items():
        use_seqs = seqs
        if len(seqs) == 0:
            # Use all sequences
            use_seqs = filter(
                lambda s: os.path.isdir(os.path.join(datadir, subj, s)),
                os.listdir(os.path.join(datadir, subj))
            )

        sequences.extend([(subj, seq) for seq in use_seqs])

    return sequences


@ex.automain
def main(
    _run, datadir, batch_size, n_epochs, patience, learning_rate, weight_decay,
    train_seq_dict, valid_seq_dict, test_seq_dict,
    articulators, state_dict_fpath=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(os.path.join(fs_observer.dir, f"experiment"))
    best_model_path = os.path.join(fs_observer.dir, "best_model.pt")
    last_model_path = os.path.join(fs_observer.dir, "last_model.pt")

    model = Tacotron2()
    tacotron2_state_dict = torch.load(NVIDIA_TACOTRON2_WEIGHTS_FILEPATH, map_location=device)
    model.load_state_dict(tacotron2_state_dict)

    model.embedding = ArticulatorsEmbedding(n_curves=len(articulators), n_samples=50)
    if state_dict_fpath is not None:
        state_dict = torch.load(state_dict_fpath, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)

    loss_fn = Tacotron2Loss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

    train_sequences = sequences_from_dict(datadir, train_seq_dict)
    train_dataset = ArticulToMelSpecDataset(datadir, train_sequences, articulators)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=set_seeds,
        collate_fn=pad_sequence_collate_fn
    )

    valid_sequences = sequences_from_dict(datadir, valid_seq_dict)
    valid_dataset = ArticulToMelSpecDataset(datadir, valid_sequences, articulators)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
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
        worker_init_fn=set_seeds,
        collate_fn=pad_sequence_collate_fn
    )

    best_model = Tacotron2()
    best_model.embedding = ArticulatorsEmbedding(n_curves=len(articulators), n_samples=50)

    state_dict = torch.load(best_model_path, map_location=device)
    best_model.load_state_dict(state_dict)
    best_model.to(device)

    save_to = os.path.join(fs_observer.dir, "test_outputs")
    if not os.path.exists(save_to):
        os.makedirs(save_to)

    info_test = run_test(
        model=best_model,
        dataloader=test_dataloader,
        criterion=loss_fn,
        device=device,
        save_to=save_to
    )
