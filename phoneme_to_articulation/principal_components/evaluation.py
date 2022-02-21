import pdb

import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from tqdm import tqdm
from vt_tools import COLORS, TONGUE, UPPER_INCISOR
from vt_shape_gen.helpers import load_articulator_array

from phoneme_to_articulation.encoder_decoder.evaluation import save_outputs
from phoneme_to_articulation.principal_components.models import Decoder
from settings import DatasetConfig


def plot_array(output, target, reference, save_to, phoneme=None):
    plt.figure(figsize=(10, 10))

    plt.plot(*reference, color=COLORS[UPPER_INCISOR])
    plt.plot(*output, color=COLORS[TONGUE])
    plt.plot(*target, "r--")

    plt.xlim([0., 1.])
    plt.ylim([1., 0])

    if phoneme is not None:
        fontdict = dict(
            family="serif",
            fontsize=22,
            color="darkblue"
        )
        plt.text(0.5, 0.2, phoneme, fontdict=fontdict)

    plt.grid(which="major")
    plt.grid(which="minor", alpha=0.4)
    plt.minorticks_on()

    plt.tight_layout()
    plt.savefig(save_to)
    plt.close()


def run_autoencoder_test(epoch, model, dataloader, criterion, outputs_dir, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch_outputs_dir = os.path.join(outputs_dir, str(epoch))
    if not os.path.exists(epoch_outputs_dir):
        os.makedirs(epoch_outputs_dir)

    model.eval()

    losses = []
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - test")
    for frame_ids, inputs, sample_weigths, phonemes in progress_bar:
        inputs = inputs.to(device)
        sample_weigths = sample_weigths.to(device)

        with torch.set_grad_enabled(False):
            outputs, anchor_latents = model(inputs)
            loss = criterion(
                outputs, anchor_latents, inputs, sample_weigths
            )

            losses.append(loss.item())
            progress_bar.set_postfix(loss=np.mean(losses))

            for frame_id, output, target, phoneme in zip(frame_ids, outputs, inputs, phonemes):
                subject, sequence, inumber = frame_id.split("_")

                reference_filepath = os.path.join(
                    dataloader.dataset.datadir, subject, sequence, "inference_contours", f"{inumber}_{UPPER_INCISOR}.npy"
                )
                reference = load_articulator_array(reference_filepath, norm_value=DatasetConfig.RES)
                reference = (reference - reference[-1] + 0.3).T

                target = dataloader.dataset.normalize.inverse(target.reshape(2, 50).detach().cpu())
                output = dataloader.dataset.normalize.inverse(output.reshape(2, 50).detach().cpu())

                save_to = os.path.join(epoch_outputs_dir, f"{frame_id}.jpg")
                plot_array(output, target, reference, save_to, phoneme)

    mean_loss = np.mean(losses)
    info = {
        "loss": mean_loss
    }

    return info


def run_phoneme_to_PC_test(epoch, model, decoder_state_dict_fpath, dataloader, criterion, outputs_dir, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch_outputs_dir = os.path.join(outputs_dir, str(epoch))
    if not os.path.exists(epoch_outputs_dir):
        os.makedirs(epoch_outputs_dir)

    model.eval()

    decoder = Decoder(n_components=12, out_features=100)
    decoder_state_dict = torch.load(decoder_state_dict_fpath, map_location=device)
    decoder.load_state_dict(decoder_state_dict)
    decoder.to(device)
    decoder.eval()

    losses = []
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - test")
    for sentence_ids, sentence, targets, lengths, phonemes, references, critical_weights, frames in progress_bar:
        sentence = sentence.to(device)
        targets = targets.to(device)
        references = references.to(device)
        critical_weights = critical_weights.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(sentence, lengths)
            loss = criterion(outputs, targets, references, critical_weights)

            pred_shapes = decoder(outputs)
            bs, seq_len, _ = pred_shapes.shape
            pred_shapes = pred_shapes.reshape(bs, seq_len, 1, 2, 50)

            pred_shapes = pred_shapes.detach().cpu()
            target_shapes = targets.detach().cpu()

        save_outputs(
            sentence_ids,
            pred_shapes,
            target_shapes,
            lengths,
            phonemes,
            [TONGUE],
            epoch_outputs_dir,
            regularize_out=False
        )

        for (
            sentence_id, sentence_pred_shapes, sentence_target_shapes, sentence_references, sentence_length, sentence_phonemes, sentence_frames
        ) in zip(sentence_ids, pred_shapes, target_shapes, references, lengths, phonemes, frames):
            save_to_dir = os.path.join(epoch_outputs_dir, sentence_id, "plots")

            if not os.path.exists(save_to_dir):
                os.makedirs(save_to_dir)

            sentence_pred_shapes = sentence_pred_shapes[:sentence_length]
            sentence_target_shapes = sentence_target_shapes[:sentence_length]
            sentence_references = sentence_references[:sentence_length]
            for pred_shape, target_shape, reference, phoneme, frame_id in zip(sentence_pred_shapes, sentence_target_shapes, sentence_references, sentence_phonemes, sentence_frames):
                pred_shape = dataloader.dataset.normalize.inverse(pred_shape.squeeze(dim=0))
                target_shape = dataloader.dataset.normalize.inverse(target_shape.squeeze(dim=0))
                reference = dataloader.dataset.normalize.inverse(reference.detach().cpu())

                save_to_filepath = os.path.join(save_to_dir, f"{frame_id}.jpg")
                plot_array(pred_shape, target_shape, reference, save_to_filepath, phoneme)

        losses.append(loss.item())
        progress_bar.set_postfix(loss=np.mean(losses))

    mean_loss = np.mean(losses)
    info = {
        "loss": mean_loss,
        "saves_dir": epoch_outputs_dir
    }

    return info
