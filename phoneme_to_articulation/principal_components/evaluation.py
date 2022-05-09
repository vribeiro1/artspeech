import pdb

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch

from tqdm import tqdm
from vt_tools import COLORS, TONGUE, UPPER_INCISOR
from vt_shape_gen.helpers import load_articulator_array

from phoneme_to_articulation.metrics import minimal_distance, MeanP2CPDistance
from phoneme_to_articulation.encoder_decoder.evaluation import save_outputs
from phoneme_to_articulation.principal_components.models import Decoder
from settings import DatasetConfig


def plot_array(outputs, targets, references, save_to, phoneme=None, tag=None):
    plt.figure(figsize=(10, 10))

    for reference in references:
        plt.plot(*reference, color=COLORS[UPPER_INCISOR])

    for output in outputs:
        plt.plot(*output)
    for target in targets:
        plt.plot(*target, "r--")

    plt.xlim([0., 1.])
    plt.ylim([1., 0])

    if phoneme is not None:
        fontdict = dict(
            fontsize=22,
            color="darkblue"
        )
        plt.text(0.5, 0.2, phoneme, fontdict=fontdict)

    if tag is not None:
        fontdict = dict(
            fontsize=22,
            color="red"
        )
        plt.text(0.05, 0.95, tag, fontdict=fontdict)

    plt.grid(which="major")
    plt.grid(which="minor", alpha=0.4)
    plt.minorticks_on()

    plt.tight_layout()
    plt.savefig(save_to)
    plt.close()


def plot_autoencoder_outputs(datadir, frame_ids, outputs, inputs, phonemes, denorm_fn, outputs_dir, multiarticulator=True):
    if not multiarticulator:
        inputs = inputs.unsqueeze(dim=1)
        outputs = outputs.unsqueeze(dim=1)

    for frame_id, output, target, phoneme in zip(frame_ids, outputs, inputs, phonemes):
        subject, sequence, inumber = frame_id.split("_")

        reference_filepath = os.path.join(
            datadir, subject, sequence, "inference_contours", f"{inumber}_{UPPER_INCISOR}.npy"
        )
        reference = load_articulator_array(reference_filepath, norm_value=DatasetConfig.RES)
        reference = torch.from_numpy((reference - reference[-1] + 0.3).T)

        articulators = sorted(denorm_fn.keys())
        denorm_targets = torch.zeros(size=(len(articulators), 2, 50), device=target.device, dtype=target.dtype)
        denorm_outputs = torch.zeros(size=(len(articulators), 2, 50), device=target.device, dtype=target.dtype)
        for i, articulator in enumerate(articulators):
            denorm_targets[i] = denorm_fn[articulator].inverse(target[i].reshape(2, 50))
            denorm_outputs[i] = denorm_fn[articulator].inverse(output[i].reshape(2, 50))

        denorm_targets = denorm_targets.detach().cpu()
        denorm_outputs = denorm_outputs.detach().cpu()

        save_to = os.path.join(outputs_dir, f"{frame_id}.jpg")
        plot_array(denorm_outputs, denorm_targets, reference.unsqueeze(dim=0), save_to, phoneme)


def run_autoencoder_test(epoch, model, dataloader, criterion, outputs_dir=None, fn_metrics=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if fn_metrics is None:
        fn_metrics={}

    plot_outputs = outputs_dir is not None
    if plot_outputs:
        epoch_outputs_dir = os.path.join(outputs_dir, str(epoch))
        if not os.path.exists(epoch_outputs_dir):
            os.makedirs(epoch_outputs_dir)

    model.eval()

    losses = []
    metrics_values = {metric_name: [] for metric_name in fn_metrics}
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - test")
    for frame_ids, inputs, sample_weigths, phonemes in progress_bar:
        inputs = inputs.to(device)
        sample_weigths = sample_weigths.to(device)

        with torch.set_grad_enabled(False):
            outputs, latents = model(inputs)
            loss = criterion(
                outputs, latents, inputs, sample_weigths
            )

            for metric_name, fn_metric in fn_metrics.items():
                metric_val = fn_metric(outputs, inputs)
                metric_val = metric_val.flatten()
                metrics_values[metric_name].extend([val.item() for val in metric_val])

            losses.append(loss.item())
            progress_bar.set_postfix(loss=np.mean(losses))

            if plot_outputs:
                plot_autoencoder_outputs(
                    dataloader.dataset.datadir,
                    frame_ids,
                    outputs,
                    inputs,
                    phonemes,
                    {TONGUE: dataloader.dataset.normalize.inverse},
                    outputs_dir=epoch_outputs_dir,
                    multiarticulator=False
                )

    mean_loss = np.mean(losses)
    info = {
        "loss": mean_loss
    }

    return info


def run_multiart_autoencoder_test(epoch, model, dataloader, criterion, outputs_dir=None, fn_metrics=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if fn_metrics is None:
        fn_metrics={}

    plot_outputs = outputs_dir is not None
    if plot_outputs:
        epoch_outputs_dir = os.path.join(outputs_dir, str(epoch))
        if not os.path.exists(epoch_outputs_dir):
            os.makedirs(epoch_outputs_dir)

    model.eval()

    losses = []
    metrics_values = {metric_name: [] for metric_name in fn_metrics}
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - test")
    for frame_ids, inputs, sample_weigths, phonemes in progress_bar:
        inputs = inputs.to(device)
        sample_weigths = sample_weigths.to(device)

        with torch.set_grad_enabled(False):
            outputs, latents = model(inputs)
            loss = criterion(
                outputs, latents, inputs, sample_weigths
            )

            for metric_name, fn_metric in fn_metrics.items():
                metric_val = fn_metric(outputs, inputs)
                metric_val = metric_val.flatten()
                metrics_values[metric_name].extend([val.item() for val in metric_val])

            losses.append(loss.item())
            progress_bar.set_postfix(loss=np.mean(losses))

            if plot_outputs:
                plot_autoencoder_outputs(
                    dataloader.dataset.datadir,
                    frame_ids,
                    outputs,
                    inputs,
                    phonemes,
                    dataloader.dataset.normalize,
                    outputs_dir=epoch_outputs_dir,
                    multiarticulator=True
                )

    mean_loss = np.mean(losses)
    info = {
        "loss": mean_loss
    }

    return info


def run_phoneme_to_PC_test(
    epoch, model, decoder_state_dict_fpath, n_components, dataloader, criterion, outputs_dir,
    fn_metrics=None, device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if fn_metrics is None:
        fn_metrics = {}

    epoch_outputs_dir = os.path.join(outputs_dir, str(epoch))
    if not os.path.exists(epoch_outputs_dir):
        os.makedirs(epoch_outputs_dir)

    model.eval()
    denorm_fn = dataloader.dataset.normalize.inverse

    decoder = Decoder(n_components=n_components, out_features=100)
    decoder_state_dict = torch.load(decoder_state_dict_fpath, map_location=device)
    decoder.load_state_dict(decoder_state_dict)
    decoder.to(device)
    decoder.eval()

    losses = []
    p2cp_metric_values = []
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - test")
    for sentence_ids, sentence, targets, lengths, phonemes, critical_masks, critical_references, frames in progress_bar:
        sentence = sentence.to(device)
        targets = targets.to(device)
        critical_references = critical_references.to(device)
        critical_masks = critical_masks.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(sentence, lengths)
            loss = criterion(outputs, targets, critical_references, critical_masks)

            pred_shapes = decoder(outputs)
            bs, seq_len, _ = pred_shapes.shape
            pred_shapes = pred_shapes.reshape(bs, seq_len, 1, 2, 50)

            pred_shapes = denorm_fn(pred_shapes)
            pred_shapes = pred_shapes.detach().cpu()

            target_shapes = denorm_fn(targets)
            target_shapes = target_shapes.detach().cpu()

            critical_references = denorm_fn(critical_references)
            critical_references = critical_references.detach().cpu()

        ############################################################################################
        # Compute the TVs associated with the tongue and save to a csv file
        # TODO: Move this code to a separate function

        TV_target_shapes = torch.transpose(target_shapes[..., 0, :, :], -2, -1)
        TV_pred_shapes = torch.transpose(pred_shapes[..., 0, :, :], -2, -1)
        TBCD_reference = torch.transpose(critical_references[..., 0, :, :], -2, -1)
        TTCD_reference = torch.transpose(critical_references[..., 1, :, :], -2, -1)

        TBCD_true= minimal_distance(TV_target_shapes, TBCD_reference)
        TBCD_pred = minimal_distance(TV_pred_shapes, TBCD_reference)
        TTCD_true= minimal_distance(TV_target_shapes, TTCD_reference)
        TTCD_pred = minimal_distance(TV_pred_shapes, TTCD_reference)

        for (
            sentence_id,
            sentence_TBCD_true,
            sentence_TBCD_pred,
            sentence_TTCD_true,
            sentence_TTCD_pred,
            length,
            sentence_phonemes
        ) in zip(sentence_ids, TBCD_true, TBCD_pred, TTCD_true, TTCD_pred, lengths, phonemes):
            TV_frames = list(range(len(sentence_phonemes)))

            sentence_TBCD_true = sentence_TBCD_true[:length, ...].numpy()
            sentence_TBCD_pred = sentence_TBCD_pred[:length, ...].numpy()
            sentence_TTCD_true = sentence_TTCD_true[:length, ...].numpy()
            sentence_TTCD_pred = sentence_TTCD_pred[:length, ...].numpy()

            df = pd.DataFrame({
                "sentence": sentence_id,
                "frame": TV_frames,
                "phoneme": sentence_phonemes,
                "TBCD_target": sentence_TBCD_true,
                "TBCD_pred": sentence_TBCD_pred,
                "TTCD_target": sentence_TTCD_true,
                "TTCD_pred": sentence_TTCD_pred
            })

            sentence_id_dir = os.path.join(epoch_outputs_dir, sentence_id)
            if not os.path.exists(sentence_id_dir):
                os.makedirs(sentence_id_dir)

            csv_filepath = os.path.join(sentence_id_dir, "tract_variables.csv")
            df.to_csv(csv_filepath, index=False)

        ############################################################################################

        save_outputs(
            sentence_ids,
            frames,
            pred_shapes,
            target_shapes,
            lengths,
            phonemes,
            [TONGUE],
            epoch_outputs_dir,
            regularize_out=False
        )

        ############################################################################################
        # Plot the results
        # TODO: Move this code to a separate function

        p2cp_dist = MeanP2CPDistance(reduction="none")
        for (
            sentence_id, sentence_pred_shapes, sentence_target_shapes, sentence_references, length, sentence_phonemes, sentence_frames
        ) in zip(sentence_ids, pred_shapes, target_shapes, critical_references, lengths, phonemes, frames):
            save_to_dir = os.path.join(epoch_outputs_dir, sentence_id, "plots")

            if not os.path.exists(save_to_dir):
                os.makedirs(save_to_dir)

            sentence_pred_shapes = sentence_pred_shapes[:length]
            sentence_target_shapes = sentence_target_shapes[:length]

            sentence_p2cp = p2cp_dist(
                torch.transpose(sentence_pred_shapes, -1, -2),
                torch.transpose(sentence_target_shapes, -1, -2)
            )
            p2cp_metric_values.extend([d.item() for d in sentence_p2cp])

            sentence_references = sentence_references[:length]
            for (
                pred_shape, target_shape, reference, phoneme, frame_id, p2cp
            ) in zip(
                sentence_pred_shapes, sentence_target_shapes, sentence_references, sentence_phonemes, sentence_frames, sentence_p2cp
            ):
                save_to_filepath = os.path.join(save_to_dir, f"{frame_id}.jpg")
                pred_shape = pred_shape.squeeze(dim=0)
                target_shape = target_shape.squeeze(dim=0)

                p2cp_mm = p2cp * DatasetConfig.PIXEL_SPACING * DatasetConfig.RES
                p2cp__mm_str ="%0.3f mm" % p2cp_mm

                plot_array(pred_shape, target_shape, reference, save_to_filepath, phoneme, p2cp__mm_str)

        losses.append(loss.item())
        progress_bar.set_postfix(loss=np.mean(losses))

        ############################################################################################

    mean_loss = np.mean(losses)
    info = {
        "loss": mean_loss,
        "p2cp_mean": np.mean(p2cp_metric_values),
        "p2cp_std": np.std(p2cp_metric_values),
        "p2cp_mean_mm": np.mean(p2cp_metric_values) * DatasetConfig.PIXEL_SPACING * DatasetConfig.RES,
        "p2cp_std_mm": np.std(p2cp_metric_values) * DatasetConfig.PIXEL_SPACING * DatasetConfig.RES,
        "saves_dir": epoch_outputs_dir
    }

    return info
