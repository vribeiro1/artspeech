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
from settings import BASE_DIR, TRAIN

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
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            outputs = model(inputs, input_lengths)
        outputs = model.get_normalized_outputs(outputs, use_log_prob=False)
        for metric_name, fn_metric in fn_metrics.items():
            metric_val = fn_metric(outputs, targets, input_lengths, target_lengths)
            metrics_values[metric_name].append(metric_val.item())

    info = {}
    info.update({
        metric_name: np.mean(values)
        for metric_name, values in  metrics_values.items()
    })

    return info