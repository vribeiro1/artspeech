####################################################################################################
#
# Code base for the phoneme recognition experiment
#
####################################################################################################
import funcy
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import torch
import torch.nn as nn

from enum import Enum
from matplotlib.axes import Axes
from tqdm import tqdm
from typing import Union, Tuple, List
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from phoneme_recognition.metrics import CrossEntropyLoss, MetricsMixin, substitution_matrix
from settings import TRAIN, UNKNOWN

CLASSES_NAMES = {
    0: "dental",
    1: "labial",
    2: "palatal",
    3: "front vowels",
    4: "back vowels",
    5: "open vowels",
    6: "rounded vowels",
    7: "other",
}

PHONETIC_CLASSES = {
    0: ["t", "d", "n", "l", "z", "s"],
    1: ["p", "b", "m", "f", "v"],
    2: ["k", "g", "Z", "S"],
    3: ["i", "e", "E", "E/", "U~/", "j"],
    4: ["u", "o", "O", "O/", "o~", "w"],
    5: ["a", "a~"],
    6: ["y", "2", "9", "H"],
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
    normalize_outputs: bool,
    use_log_prob: bool,
    target: Target,
    feature: Feature = Feature.MELSPEC,
    use_voicing: bool = False,
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
        if use_voicing:
            voicing = batch["voicing"]
            voicing = voicing.to(device)
        else:
            voicing = None

        optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            outputs = model(inputs, voicing)
            if training and logits_large_margins > 0.0:
                outputs = model.get_noise_logits(outputs, logits_large_margins)
            if normalize_outputs:
                outputs = model.get_normalized_outputs(outputs, use_log_prob=use_log_prob)
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

        norm_outputs = model.get_normalized_outputs(outputs)
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
    decoder,
    target: Target,
    plot_target: Target,
    feature: Feature = Feature.MELSPEC,
    use_voicing: bool = False,
    device=None,
    save_dir=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocabulary = dataloader.dataset.vocabulary
    model_predictions = torch.zeros(size=(0,))  # Used for plotting the confusion matrix with CE
    model_features = torch.zeros(size=(0, model.classifier.in_features)) # Used for plotting model features
    model_targets = torch.zeros(size=(0,))  # Used for plotting the confusion matrix with CE

    all_outputs = []  # Used for computing the substitution matrix with CTC
    all_targets = []  # Used for computing the substitution matrix with CTC
    all_inputs_lengths = []  # Used for computing the substitution matrix with CTC
    all_target_lengths = []  # Used for computing the substitution matrix with CE

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
        plot_targets = batch[plot_target.value]
        plot_target_lengths = batch[f"{plot_target.value}_length"]

        inputs = inputs.to(device)
        targets = targets.to(device)
        if use_voicing:
            voicing = batch["voicing"]
            voicing = voicing.to(device)
        else:
            voicing = None

        with torch.set_grad_enabled(False):
            outputs, features = model(inputs, voicing, return_features=True)
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

        plot_targets = [tgt[:length] for tgt, length in zip(plot_targets, plot_target_lengths)]
        model_targets = torch.cat([model_targets] + plot_targets, dim=0)

        outputs = outputs.detach().cpu()
        predictions = torch.topk(outputs, k=1, dim=-1).indices
        predictions = [pred[:length] for pred, length in zip(predictions, input_lengths)]
        model_predictions = torch.cat([model_predictions] + predictions, dim=0)

        targets = targets.detach().cpu()
        all_outputs.extend(outputs)
        all_targets.extend(targets)
        all_inputs_lengths.extend(input_lengths)
        all_target_lengths.extend(target_lengths)

    if save_dir is not None:
        save_filepath = os.path.join(save_dir, "model_features.pdf")

        class_map = {}
        for _, tokens in PHONETIC_CLASSES.items():
            for token in tokens:
                class_map[token] = vocabulary[token]

        plot_features(
            features=model_features.numpy(),
            targets=model_targets.numpy(),
            class_map=class_map,
            max_items_per_class=100,
            save_filepath=save_filepath,
            plot_groups=PHONETIC_CLASSES,
        )

        conf_mtx = compute_confusion_matrix(
            predictions=model_predictions.numpy(),
            targets=model_targets.numpy(),
            vocabulary=vocabulary,
            groups=PHONETIC_CLASSES,
            normalize="true",
        )
        np.save(
            os.path.join(save_dir, "confusion_matrix.npy"),
            conf_mtx
        )

        plot_confusion_matrix(
            conf_mtx,
            save_filepath=os.path.join(save_dir, "confusion_matrix.png"),
        )
        plot_confusion_matrix(
            conf_mtx,
            save_filepath=os.path.join(save_dir, "confusion_matrix.pdf"),
        )

        subs_mtx = compute_substitution_matrix(
            emissions=all_outputs,
            targets=all_targets,
            groups=PHONETIC_CLASSES,
            input_lengths=all_inputs_lengths,
            target_lengths=all_target_lengths,
            decoder=decoder,
            vocabulary=vocabulary,
        )
        np.save(
            os.path.join(save_dir, "substitution_matrix.npy"),
            subs_mtx
        )

        plot_substitution_matrix(
            subs_mtx,
            include_deletions=True,
            include_insertions=True,
            vocab=list(CLASSES_NAMES.values()),
            figsize=(15, 15),
            width_ratios=(8, 1),
            height_ratios=(8, 1),
            save_filepath=os.path.join(save_dir, "substitution_matrix.png"),
        )
        plot_substitution_matrix(
            subs_mtx,
            include_deletions=True,
            include_insertions=True,
            vocab=list(CLASSES_NAMES.values()),
            figsize=(15, 15),
            width_ratios=(8, 1),
            height_ratios=(8, 1),
            save_filepath=os.path.join(save_dir, "substitution_matrix.pdf"),
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
    _, num_features = features.shape
    plot_features = np.zeros(shape=(0, num_features))
    plot_targets = np.zeros(shape=(0,))
    for _, i in class_map.items():
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
    _, ax = plt.subplots(figsize=(10, 10))
    num_groups = len(plot_groups)
    for group_i, (_, classes) in enumerate(plot_groups.items()):
        group_features = np.zeros(shape=(0, 2))
        for class_ in classes:
            class_i = class_map[class_]
            group_features = np.concatenate([
                group_features,
                tsne_features[plot_targets == class_i]
            ])
        if len(group_features) == 0:
            continue

        color = cmap(group_i / num_groups)
        label = " ".join(classes)
        ax.scatter(
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


def compute_confusion_matrix(
    predictions,
    targets,
    vocabulary,
    groups=None,
    normalize=None,
):
    vocabulary_transposed = {i: token for token, i in vocabulary.items()}
    target_tokens = [vocabulary_transposed.get(i.item(), UNKNOWN) for i in targets.astype(np.int)]
    predicted_tokens = [vocabulary_transposed.get(i.item(), UNKNOWN) for i in predictions.astype(np.int)]

    if groups is not None:
        groups_transposed = {}
        for group_name, symbols in groups.items():
            for symbol in symbols:
                groups_transposed[symbol] = group_name

        other = max(groups_transposed.keys())
        target_tokens = [groups_transposed.get(symbol, other) for symbol in target_tokens]
        predicted_tokens = [groups_transposed.get(symbol, other) for symbol in predicted_tokens]

    conf_mtx = confusion_matrix(target_tokens, predicted_tokens, normalize=normalize)
    return conf_mtx


def _make_tokenized_sequence(
    sequence,
    vocabulary_T,
    groups_T=None,
):
    """
    Args:
        sequence (List[str])
        vocabulary_T (Dict[int, str])
        groups (Dict[str, int])
    """
    # sequence: List[str]
    numerized_seq = funcy.lmap(lambda s: funcy.lmap(int, s.split()), sequence)

    # numerized_seq: List[List[int]]
    tokenized_seq = funcy.lmap(
        lambda tokens: [vocabulary_T[i] for i in tokens],
        numerized_seq
    )

    if groups_T:
        other = max(groups_T.values()) + 1
        tokenized_seq = funcy.lmap(
            lambda tokens: [groups_T.get(i, other) for i in tokens],
            tokenized_seq
        )

    tokenized_seq = funcy.lmap(
        lambda tokens: " ".join(map(str, tokens)),
        tokenized_seq
    )

    return tokenized_seq


def compute_substitution_matrix(
    emissions,
    targets,
    input_lengths,
    target_lengths,
    decoder,
    vocabulary,
    groups=None,
):
    pred_sequences = []
    target_sequences = []

    vocabulary_T = {i: token for token, i in vocabulary.items()}
    subs_tokens = [token for token in vocabulary]
    groups_T = None
    if groups is not None:
        subs_tokens = funcy.lmap(str, list(groups.keys()) + [max(groups.keys()) + 1])
        groups_T = {}
        for group_name, symbols in groups.items():
            for symbol in symbols:
                groups_T[symbol] = group_name

    for (
        emission,
        target,
        input_length,
        target_length
    ) in zip(
        emissions,
        targets,
        input_lengths,
        target_lengths
    ):
        emission = emission.unsqueeze(dim=0)
        target = target.unsqueeze(dim=0)
        input_length = torch.tensor([input_length])
        target_length = torch.tensor([target_length])
        pred, tgt = MetricsMixin.make_pred_and_target_sentences(
            decoder,
            emission,
            target,
            input_length,
            target_length,
        )

        tokenized_pred = _make_tokenized_sequence(pred, vocabulary_T, groups_T)
        tokenized_target = _make_tokenized_sequence(tgt, vocabulary_T, groups_T)
        pred_sequences.extend(tokenized_pred)
        target_sequences.extend(tokenized_target)

    return substitution_matrix(
        pred_sequences,
        target_sequences,
        subs_tokens,
        insertions_and_deletions="both",
        normalize="true",
    )


def plot_confusion_matrix(
    conf_mtx,
    save_filepath,
):
    _, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(
        conf_mtx,
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
    plt.close()


def plot_substitution_matrix(
    subs_mtx: np.ndarray,
    include_insertions: bool,
    include_deletions: bool,
    vocab: List[str],
    figsize: Union[int, Tuple[int, int]],
    width_ratios: Tuple,
    height_ratios: Tuple,
    save_filepath: str,
    cmap: str = "PuRd",
):
    ncols = 2 if include_deletions else 1
    nrows = 2 if include_insertions else 1

    if isinstance(figsize, int):
        figsize=(figsize, figsize)

    _, ax = plt.subplots(
        nrows, ncols,
        sharex=False,
        sharey=False,
        figsize=figsize,
        width_ratios=width_ratios,
        height_ratios=height_ratios
    )

    if isinstance(ax, Axes):
        ax = [ax]
    ax = funcy.lflatten(ax)
    try:
        ax = list(np.concatenate(ax).ravel())
    except ValueError:
        pass

    ax0 = ax.pop(0)

    plot_subs_mtx = subs_mtx
    if include_insertions:
        plot_subs_mtx = plot_subs_mtx[:-1, :]
    if include_deletions:
        plot_subs_mtx = plot_subs_mtx[:, :-1]

    sns.heatmap(
        plot_subs_mtx,
        annot=True,
        cbar=False,
        square=True,
        ax=ax0,
        xticklabels=vocab,
        yticklabels=vocab,
        fmt=".3f",
        annot_kws={"fontsize": 22},
        cmap=cmap,
    )

    ax0.tick_params(axis="both", which="major", labelsize=24, labelrotation=45)

    if include_deletions:
        ax1 = ax.pop(0)

        sns.heatmap(
            np.expand_dims(subs_mtx[:-1, -1], axis=1),
            annot=True,
            cbar=False,
            square=False,
            ax=ax1,
            xticklabels=[""],
            yticklabels=vocab,
            fmt=".3f",
            annot_kws={"fontsize": 22},
            cmap=cmap,
        )

        ax1.tick_params(axis="both", which="major", labelsize=24, labelrotation=45)

        ax1.axes.get_yaxis().set_visible(False)

    if include_insertions:
        ax0.axes.get_xaxis().set_visible(False)

        ax2 = ax.pop(0)

        sns.heatmap(
            np.expand_dims(subs_mtx[-1, :-1], axis=0),
            annot=True,
            cbar=False,
            square=False,
            ax=ax2,
            xticklabels=vocab,
            yticklabels=["insertions"],
            fmt=".3f",
            annot_kws={"fontsize": 22},
            cmap=cmap,
        )

        ax2.tick_params(axis="both", which="major", labelsize=24, labelrotation=45)

        if include_deletions:
            ax3 = ax.pop(0)
            ax3.set_xlabel("deletions", fontsize=24, rotation=45)

            ax3.set_yticklabels([])
            ax3.set_xticklabels([])

            ax3.tick_params(axis="both", color="white")

            ax3.spines["top"].set_color("white")
            ax3.spines["bottom"].set_color("white")
            ax3.spines["left"].set_color("white")
            ax3.spines["right"].set_color("white")

            # ax3.plot([0.5, 0.5], [0., 1.], "--", color="black", lw=0.7)

    plt.tight_layout()
    plt.savefig(save_filepath)
    plt.close()
