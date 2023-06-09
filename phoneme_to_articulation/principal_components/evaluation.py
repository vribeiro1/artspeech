import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import torch

from collections import OrderedDict
from tqdm import tqdm
from vt_tools import COLORS, TONGUE, UPPER_INCISOR
from vt_shape_gen.helpers import load_articulator_array

from phoneme_to_articulation import (
    save_outputs,
    tract_variables,
    REQUIRED_ARTICULATORS_FOR_TVS
)


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


def plot_autoencoder_outputs(datadir, frame_ids, outputs, inputs, phonemes, denorm_fn, res, outputs_dir):
    for frame_id, output, target, phoneme in zip(frame_ids, outputs, inputs, phonemes):
        subject, sequence, inumber = frame_id.split("_")

        reference_filepath = os.path.join(
            datadir, subject, sequence, "inference_contours", f"{inumber}_{UPPER_INCISOR}.npy"
        )
        reference = load_articulator_array(reference_filepath, norm_value=res)
        reference = torch.from_numpy((reference - reference[-1] + 0.3).T)

        articulators = sorted(denorm_fn.keys())
        denorm_targets = torch.zeros(size=(len(articulators), 2, 50), device=target.device, dtype=target.dtype)
        denorm_outputs = torch.zeros(size=(len(articulators), 2, 50), device=target.device, dtype=target.dtype)
        for i, articulator in enumerate(articulators):
            denorm_targets[i] = denorm_fn[articulator](target[i].reshape(2, 50))
            denorm_outputs[i] = denorm_fn[articulator](output[i].reshape(2, 50))

        denorm_targets = denorm_targets.detach().cpu()
        denorm_outputs = denorm_outputs.detach().cpu()

        save_to = os.path.join(outputs_dir, f"{frame_id}.jpg")
        plot_array(denorm_outputs, denorm_targets, reference.unsqueeze(dim=0), save_to, phoneme)


def plot_cov_matrix(cov_matrix, saves_dir, suffix=""):
    n_components, _ = cov_matrix.shape

    plt.figure(figsize=(10, 10))

    sns.heatmap(
        cov_matrix,
        cmap="BuPu",
        linewidths=.5,
        annot=True,
        cbar=False,
        xticklabels=[i + 1 for i in range(n_components)],
        yticklabels=[i + 1 for i in range(n_components)],
        fmt=".3f",
        annot_kws={
            "fontsize": 16
        }
    )

    plt.tick_params(axis="both", which="major", labelsize=18)
    plt.tight_layout()
    filename = "covariance_matrix" + suffix
    plt.savefig(os.path.join(saves_dir, f"{filename}.pdf"))
    plt.savefig(os.path.join(saves_dir, f"{filename}.png"))


def run_autoencoder_test(
    epoch,
    model,
    dataloader,
    criterion,
    outputs_dir=None,
    plots_dir=None,
    fn_metrics=None,
    device=None
):
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

    all_latents = torch.zeros(size=(0, model.latent_size))
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
                    outputs.unsqueeze(dim=1),
                    inputs.unsqueeze(dim=1),
                    phonemes,
                    {TONGUE: dataloader.dataset.normalize.inverse},
                    res=dataloader.dataset.dataset_config.RES,
                    outputs_dir=epoch_outputs_dir,
                )

            latents = latents.detach().cpu()
            all_latents = torch.concat([all_latents, latents], dim=0)

    cov_latents = torch.cov(all_latents.T)
    cov_latents = cov_latents.detach().cpu()
    if plots_dir:
        plot_cov_matrix(cov_latents, plots_dir)

    mean_loss = np.mean(losses)
    info = {
        "loss": mean_loss,
    }
    info.update({
        metric_name: np.mean(values)
        for metric_name, values in  metrics_values.items()
    })

    return info


def run_multiart_autoencoder_test(
    epoch,
    model,
    dataloader,
    criterion,
    dataset_config,
    outputs_dir=None,
    plots_dir=None,
    indices_dict=None,
    fn_metrics=None,
    device=None,
):
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

    all_latents = torch.zeros(size=(0, model.latent_size), device=device)
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
                denorm_fn = {
                    articulator: norm_fn.inverse
                    for articulator, norm_fn
                    in dataloader.dataset.normalize.items()
                }

                plot_autoencoder_outputs(
                    dataloader.dataset.datadir,
                    frame_ids,
                    outputs,
                    inputs,
                    phonemes,
                    denorm_fn,
                    res=dataset_config.RES,
                    outputs_dir=epoch_outputs_dir,
                )
            all_latents = torch.concat([all_latents, latents], dim=0)

    cov_latents = torch.cov(all_latents.T)
    cov_latents = cov_latents.detach().cpu()
    if plots_dir:
        if indices_dict is None:
            plot_cov_matrix(
                cov_latents,
                plots_dir
            )

            np.save(
                os.path.join(plots_dir, f"covariance_matrix.npy"),
                cov_latents
            )
        else:
            for articulator, indices in indices_dict.items():
                plot_cov_matrix(
                    cov_latents[indices][:, indices],
                    plots_dir,
                    f"_{articulator}"
                )

                np.save(
                    os.path.join(plots_dir, f"covariance_matrix_{articulator}.npy"),
                    cov_latents[indices][:, indices]
                )

    mean_loss = np.mean(losses)
    info = {
        "loss": mean_loss
    }

    return info


def run_phoneme_to_principal_components_test(
    epoch,
    model,
    dataloader,
    criterion,
    fn_metrics=None,
    outputs_dir=None,
    decode_transform=None,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if fn_metrics is None:
        fn_metrics = {}

    model.eval()
    articulators = dataloader.dataset.articulators
    normalize_dict = dataloader.dataset.normalize

    losses = []
    metrics_values = {
        metric_name: []
        for metric_name in fn_metrics
    }
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - inference")
    for (
        sentence_names,
        sentence_inputs,
        sentence_targets,
        sentence_lengths,
        sentence_phonemes,
        critical_masks,
        reference_arrays,
        sentence_frames
    ) in progress_bar:
        sentence_inputs = sentence_inputs.to(device)
        sentence_targets = sentence_targets.to(device)

        with torch.set_grad_enabled(False):
            sentence_outputs = model(sentence_inputs, sentence_lengths)
            loss = criterion(
                sentence_outputs,
                sentence_targets,
                sentence_lengths,
                critical_masks,
            )
            losses.append(loss.item())

            for metric_name, fn_metric in fn_metrics.items():
                metric_val = fn_metric(
                    sentence_outputs,
                    sentence_targets,
                    sentence_lengths,
                )
                metrics_values[metric_name].append(metric_val.item())

        postfixes = {
            "loss": np.mean(losses)
        }
        postfixes.update({
            metric_name: np.mean(metric_vals)
            for metric_name, metric_vals
            in metrics_values.items()
        })
        progress_bar.set_postfix(OrderedDict(postfixes))

        if outputs_dir is not None:
            epoch_outputs_dir = os.path.join(outputs_dir, str(epoch))
            if not os.path.exists(epoch_outputs_dir):
                os.makedirs(epoch_outputs_dir)

            sentence_pred_shapes = decode_transform(sentence_outputs)  # (B, N_art, T, 2 * D)
            sentence_pred_shapes = sentence_pred_shapes.permute(0, 2, 1, 3)
            bs, seq_len, num_articulators, features = sentence_pred_shapes.shape
            sentence_pred_shapes = sentence_pred_shapes.reshape(
                bs,
                seq_len,
                num_articulators,
                2,
                features // 2
            )  # (B, T, N_art, 2, D)

            sentence_pred_shapes = sentence_pred_shapes.detach().cpu()
            sentence_targets = sentence_targets.detach().cpu()

            for i, articulator in enumerate(articulators):
                articulator_denorm_fn = normalize_dict[articulator].inverse

                articulator_pred_shapes = sentence_pred_shapes[:, :, i, :, :]
                sentence_pred_shapes[:, :, i, :, :] = articulator_denorm_fn(articulator_pred_shapes)

                articulator_targets = sentence_targets[:, :, i, :, :]
                sentence_targets[:, :, i, :, :] = articulator_denorm_fn(articulator_targets)

            # The upper incisor is the reference of the coordinate system and since it has a fixed
            # shape, it is non-sense to include it in the prediction. However, it is important for
            # tract variables and visualization. Therefore, we inject it in the arrays in order to
            # have it available for the next steps.
            if UPPER_INCISOR not in articulators:
                tv_articulators = sorted(articulators + [UPPER_INCISOR])
                ref_idx = tv_articulators.index(UPPER_INCISOR)

                sentence_pred_shapes = torch.concat([
                    sentence_pred_shapes[:, :, :ref_idx, :, :],
                    reference_arrays,
                    sentence_pred_shapes[:, :, ref_idx:, :, :],
                ], dim=2)

                sentence_targets = torch.concat([
                    sentence_targets[:, :, :ref_idx, :, :],
                    reference_arrays,
                    sentence_targets[:, :, ref_idx:, :, :],
                ], dim=2)
            else:
                tv_articulators = articulators

            # Only calculate the tract variables if all of the required articulators are included
            # in the test
            if all(
                [
                    articulator in tv_articulators
                    for articulator in REQUIRED_ARTICULATORS_FOR_TVS
                ]
            ):
                tract_variables(
                    sentence_names,
                    sentence_frames,
                    sentence_pred_shapes,
                    sentence_targets,
                    sentence_lengths,
                    sentence_phonemes,
                    tv_articulators,
                    epoch_outputs_dir
                )

            save_outputs(
                sentence_names,
                sentence_frames,
                sentence_pred_shapes,
                sentence_targets,
                sentence_lengths,
                sentence_phonemes,
                articulators,
                epoch_outputs_dir,
                regularize_out=False
            )

    info = {
        "loss": np.mean(losses)
    }
    info.update({
        metric_name: np.mean(metric_vals)
        for metric_name, metric_vals
        in metrics_values.items()
    })
    return info
