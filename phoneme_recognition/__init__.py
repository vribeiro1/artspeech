import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import torch
import torch.nn as nn

from enum import Enum
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from phoneme_recognition.metrics import CrossEntropyLoss
from settings import TRAIN

SIL = "#"
UNKNOWN = "<unk>"
BLANK = "<blank>"

PHONETIC_FAMILIES = {
    "fricatives": [
        "f",
        "v",
        "S",
        "Z",
        "s",
        "z"
    ],
    "plosives": [
        "p",
        "b",
        "t",
        "d",
        "k",
        "g"
    ],
    "laterals": [
        "l"
    ],
    "nasals": [
        "m",
        "n",
        "U~/",
        "a~",
        "o~"
    ],
    "vowels": [
        "a",
        "e",
        "i",
        "o",
        "u",
        "y",
        "E",
        "E/",
        "O",
        "O/",
        "2",
        "9",
        "@",
    ],
    "semi vowels": [
        "w",
        "j",
        "H"
    ],
}

PHONETIC_PLACES_OF_ARTICULATION = {
    0: ["i", "j", "e", "E", "E/", "a", "O", "O/", "o", "u", "w"],
    1: ["y", "H", "2", "9", "@"],
    2: ["U~/", "o~", "a~", "m", "n"],
    3: ["p", "b", "t", "d", "l", "k", "g"],
    4: ["f", "v", "s", "z", "S", "Z"],
}


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
            outputs = model(inputs)
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

    vocabulary = dataloader.dataset.vocabulary
    model_predictions = torch.zeros(size=(0,))
    model_features = torch.zeros(size=(0, 128))
    model_targets = torch.zeros(size=(0,))

    averaged_metrics = [
        metric_name for metric_name in fn_metrics
        if "_per_class" not in metric_name
    ]
    per_class_metrics = [
        metric_name for metric_name in fn_metrics
        if "_per_class" in metric_name
    ]

    metrics_values = {metric_name: [] for metric_name in averaged_metrics}
    for metric_name in per_class_metrics:
        metrics_values[metric_name] = {}
        for i in range(len(vocabulary)):
            metrics_values[metric_name][i] = []

    progress_bar = tqdm(dataloader, desc=f"Running test")
    for batch in progress_bar:
        inputs = batch[feature.value]
        input_lengths = batch[f"{feature.value}_length"]
        targets = batch[target.value]
        target_lengths = batch[f"{target.value}_length"]

        inputs = inputs.to(device)
        targets = targets.to(device)
        with torch.set_grad_enabled(False):
            outputs, features = model(inputs, return_features=True)
        outputs = model.get_normalized_outputs(outputs, use_log_prob=False)

        for metric_name in averaged_metrics:
            fn_metric = fn_metrics[metric_name]
            metric_val = fn_metric(outputs, targets, input_lengths, target_lengths)
            metrics_values[metric_name].append(metric_val.item())

        for metric_name in per_class_metrics:
            fn_metric = fn_metrics[metric_name]
            metric_val = fn_metric(outputs, targets, input_lengths, target_lengths)
            for i in range(len(vocabulary)):
                metrics_values[metric_name][i].append(metric_val[i].item())

        features = features.detach().cpu()
        features = [feat[:length] for feat, length in zip(features, input_lengths)]
        model_features = torch.cat([model_features] + features, dim=0)

        targets = targets.detach().cpu()
        targets = [tgt[:length] for tgt, length in zip(targets, target_lengths)]
        model_targets = torch.cat([model_targets] + targets, dim=0)

        outputs = outputs.detach().cpu()
        predictions = torch.topk(outputs, k=1, dim=-1).indices
        predictions = [pred[:length] for pred, length in zip(predictions, input_lengths)]
        model_predictions = torch.cat([model_predictions] + predictions, dim=0)

    if save_dir is not None:
        save_filepath = os.path.join(save_dir, "model_features.pdf")

        class_map = {}
        for family, tokens in PHONETIC_FAMILIES.items():
            for token in tokens:
                class_map[token] = vocabulary[token]

        plot_features(
            features=model_features.numpy(),
            targets=model_targets.numpy(),
            class_map=class_map,
            max_items_per_class=100,
            save_filepath=save_filepath,
            plot_groups=PHONETIC_PLACES_OF_ARTICULATION,
        )

        save_filepath = os.path.join(save_dir, "confusion_matrix.pdf")
        plot_confusion_matrix(
            predictions=model_predictions.numpy(),
            targets=model_targets.numpy(),
            save_filepath=save_filepath,
            vocabulary=vocabulary,
            groups=PHONETIC_FAMILIES,
            normalize="true",
        )

    info = {}
    info.update({
        metric_name: np.mean(metrics_values[metric_name])
        for metric_name in averaged_metrics
    })
    for metric_name in per_class_metrics:
        info[metric_name] = {}
        for i in range(len(vocabulary)):
            info[metric_name][i] = np.mean(metrics_values[metric_name][i])

    return info


def plot_features(
    features,
    targets,
    class_map,
    max_items_per_class,
    save_filepath,
    plot_groups
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

    cmap = plt.get_cmap("hsv")
    fig, ax = plt.subplots(figsize=(10, 10))
    num_groups = len(plot_groups)
    for group, classes in plot_groups.items():
        group_features = np.zeros(shape=(0, 2))
        for class_ in classes:
            i = class_map[class_]
            group_features = np.concatenate([
                group_features,
                tsne_features[plot_targets == i]
            ])
        if len(group_features) == 0:
            continue

        color = cmap(group / num_groups)
        label = " ".join(classes)
        scatter = ax.scatter(
            *group_features.T,
            alpha=0.7,
            c=[color] * len(group_features),
            label=label
        )

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    handles, labels = ax.get_legend_handles_labels()
    plt.tight_layout()
    plt.savefig(save_filepath)

    fig_legend = plt.figure(figsize=(5, 3))
    axi = fig_legend.add_subplot(111)

    axi.legend(
        handles,
        labels,
        loc="center",
        ncol=1,
        fontsize=22,
        markerscale=2,
    )
    axi.axis("off")
    fig_legend.canvas.draw()

    filename, _ = os.path.basename(save_filepath).split(".")
    fig_filepath = os.path.join(
        os.path.dirname(save_filepath),
        f"{filename}_legend.pdf"
    )
    plt.tight_layout()
    fig_legend.savefig(fig_filepath)


def plot_confusion_matrix(
    predictions,
    targets,
    save_filepath,
    vocabulary,
    groups=None,
    normalize=None
):
    vocabulary_transposed = {i: token for token, i in vocabulary.items()}
    target_tokens = [vocabulary_transposed.get(i.item(), UNKNOWN) for i in targets.astype(np.int)]
    predicted_tokens = [vocabulary_transposed.get(i.item(), UNKNOWN) for i in predictions.astype(np.int)]

    if groups is not None:
        groups_transposed = {}
        for group_name, symbols in groups.items():
            for symbol in symbols:
                groups_transposed[symbol] = group_name

        target_tokens = [groups_transposed.get(symbol, "other") for symbol in target_tokens]
        predicted_tokens = [groups_transposed.get(symbol, "other") for symbol in predicted_tokens]

    labels = sorted(groups.keys()) + ["other"]
    conf_mtx = confusion_matrix(target_tokens, predicted_tokens, normalize=normalize, labels=labels)

    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(
        conf_mtx,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        cbar=False,
        square=True,
        ax=ax,
        fmt=".3f",
        annot_kws={"fontsize": 22},
        cmap="coolwarm",
    )

    ax.set_xlabel("Predicted Phonemes", fontsize=28)
    ax.set_ylabel("True Phonemes", fontsize=28)
    ax.tick_params(axis="both", which="major", labelsize=24)


    plt.tight_layout()
    plt.savefig(save_filepath)
