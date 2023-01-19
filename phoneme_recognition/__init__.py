import pdb
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn

from enum import Enum
from tqdm import tqdm
from sklearn.manifold import TSNE

from phoneme_recognition.metrics import CrossEntropyLoss
from settings import TRAIN

SIL = "#"
UNKNOWN = "<unk>"
BLANK = "<blank>"

class Criterion(Enum):
    CE = CrossEntropyLoss
    CTC = nn.CTCLoss


class Feature(Enum):
    MELSPEC = "melspec"
    VOCAL_TRACT = "vocal_tract"
    AIR_COLUMN = "air_column"


class Target(Enum):
    CTC = "ctc_target"
    ACOUSTIC = "acoustic_target"
    ARTICULATORY = "articulatory_target"


def run_epoch(
    phase,
    epoch,
    model,
    dataloader,
    optimizer,
    criterion,
    use_log_prob: bool,
    target: Target,
    feature: Feature = Feature.MELSPEC,
    logits_large_margins=0.0,
    scheduler=None,
    fn_metrics=None,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if fn_metrics is None:
        fn_metrics={}
    training = phase == TRAIN

    if training:
        model.train()
    else:
        model.eval()

    losses = []
    metrics_values = {metric_name: [] for metric_name in fn_metrics}
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - {phase}")
    for batch in progress_bar:
        inputs = batch[feature.value]
        input_lengths = batch[f"{feature.value}_length"]
        targets = batch[target.value]
        target_lengths = batch[f"{target.value}_length"]

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            outputs = model(inputs, input_lengths)
            if training and logits_large_margins > 0.0:
                outputs = model.get_noise_logits(outputs, logits_large_margins)
            norm_outputs = model.get_normalized_outputs(outputs, use_log_prob=use_log_prob)
            loss = criterion(
                outputs.permute(1, 0, 2),
                targets,
                input_lengths,
                target_lengths
            )

            if training:
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            losses.append(loss.item())

        for metric_name, fn_metric in fn_metrics.items():
            metric_val = fn_metric(norm_outputs, targets, input_lengths, target_lengths)
            metrics_values[metric_name].append(metric_val.item())

        display_metrics = {
            "loss": np.mean(losses)
        }
        display_metrics.update({
            metric_name: np.mean(values)
            for metric_name, values in  metrics_values.items()
        })
        progress_bar.set_postfix(**display_metrics)

    mean_loss = np.mean(losses)
    info = {
        "loss": mean_loss
    }
    info.update({
        metric_name: np.mean(values)
        for metric_name, values in  metrics_values.items()
    })

    return info


def run_test(
    model,
    dataloader,
    fn_metrics,
    target: Target,
    feature: Feature = Feature.MELSPEC,
    device=None,
    save_dir=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_features = torch.zeros(size=(0, 128))
    model_targets = torch.zeros(size=(0,))
    metrics_values = {metric_name: [] for metric_name in fn_metrics}
    progress_bar = tqdm(dataloader, desc=f"Running test")
    for batch in progress_bar:
        inputs = batch[feature.value]
        input_lengths = batch[f"{feature.value}_length"]
        targets = batch[target.value]
        target_lengths = batch[f"{target.value}_length"]

        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(False):
            outputs, features = model(inputs, input_lengths, return_features=True)

        outputs = model.get_normalized_outputs(outputs, use_log_prob=False)
        for metric_name, fn_metric in fn_metrics.items():
            metric_val = fn_metric(outputs, targets, input_lengths, target_lengths)
            metrics_values[metric_name].append(metric_val.item())

        bs, time, dim = features.shape
        features = features.detach().cpu()
        features = [feat[:length] for feat, length in zip(features, input_lengths)]
        targets = targets.detach().cpu()
        targets = [tgt[:length] for tgt, length in zip(targets, target_lengths)]
        model_features = torch.cat([model_features] + features, dim=0)
        model_targets = torch.cat([model_targets] + targets, dim=0)

    if save_dir is not None:
        save_filepath = os.path.join(save_dir, "model_features.pdf")
        plot_features(
            features=model_features.numpy(),
            targets=model_targets.numpy(),
            vocabulary=dataloader.dataset.vocabulary,
            max_items_per_class=100,
            save_filepath=save_filepath,
        )

    info = {}
    info.update({
        metric_name: np.mean(values)
        for metric_name, values in  metrics_values.items()
    })

    return info


def plot_features(
    features,
    targets,
    class_map,
    max_items_per_class,
    save_filepath
):
    plot_features = np.zeros(shape=(0, 128))
    plot_targets = np.zeros(shape=(0,))
    for model_class, i in class_map.items():
        class_indices = np.argwhere(targets == i).squeeze()
        if len(class_indices) == 0:
            continue
        size = min(max_items_per_class, len(class_indices))
        class_sample_indices = np.random.choice(class_indices, size=size)
        class_features = features[class_sample_indices]
        class_targets = targets[class_sample_indices]

        plot_features = np.concatenate([plot_features, class_features])
        plot_targets = np.concatenate([plot_targets, class_targets])

    tsne = TSNE(n_components=2)
    tsne_features = tsne.fit_transform(plot_features)  # (N, 2)

    cmap = plt.get_cmap("seismic")
    colors = [c / len(class_map) for c in plot_targets]

    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(*tsne_features.T, alpha=0.7, c=colors, cmap="seismic")
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

    plt.tight_layout()
    plt.savefig(save_filepath)
