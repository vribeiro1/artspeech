import numpy as np
import os
import pandas as pd
import torch

from tqdm import tqdm

from helpers import make_padding_mask
from metrics import pearsons_correlation, p2cp_distance, euclidean_distance
from phoneme_to_articulation import (
    save_outputs,
    tract_variables,
    REQUIRED_ARTICULATORS_FOR_TVS
)


def run_test(
    epoch,
    model,
    dataloader,
    criterion,
    outputs_dir,
    articulators,
    device=None,
    regularize_out=False
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch_outputs_dir = os.path.join(outputs_dir, str(epoch))
    if not os.path.exists(epoch_outputs_dir):
        os.makedirs(epoch_outputs_dir)

    model.eval()
    dataset_config = dataloader.dataset.dataset_config

    losses = []
    euclidean_per_articulator = [[] for _ in articulators]
    p2cp_per_articulator = [[] for _ in articulators]
    x_corrs = [[] for _ in articulators]
    y_corrs = [[] for _ in articulators]
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - inference")
    for (
        sentences_ids,
        sentences,
        targets,
        lengths,
        phonemes,
        sentence_frames
    ) in progress_bar:
        sentences = sentences.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(sentences, lengths)
            loss = criterion(outputs, targets)
            padding_mask = make_padding_mask(lengths)
            bs, max_len, num_articulators, features = loss.shape
            loss = loss.view(bs * max_len, num_articulators, features)
            loss = loss[padding_mask.view(bs * max_len)].mean()

        outputs = outputs.detach().cpu()
        targets = targets.detach().cpu()

        for sentence_outputs, sentence_targets, length in zip(outputs, targets, lengths):
            sentence_outputs = sentence_outputs[:length].unsqueeze(dim=0)
            sentence_targets = sentence_targets[:length].unsqueeze(dim=0)

            p2cp = p2cp_distance(sentence_outputs, sentence_targets).mean(dim=1)  # (bs, n_articulators)
            euclidean = euclidean_distance(sentence_outputs, sentence_targets).mean(dim=1)  # (bs, n_articulators)

            x_corr, y_corr = pearsons_correlation(sentence_outputs, sentence_targets)
            x_corr = x_corr.mean(dim=-1)[0]
            y_corr = y_corr.mean(dim=-1)[0]

            for i, _ in enumerate(articulators):
                x_corrs[i].append(x_corr[i].item())
                y_corrs[i].append(y_corr[i].item())

                p2cp_per_articulator[i].extend([dist.item() for dist in p2cp[:, i]])
                euclidean_per_articulator[i].extend([dist.item() for dist in euclidean[:, i]])

        losses.append(loss.item())
        progress_bar.set_postfix(loss=np.mean(losses))

        # Only calculate the tract variables if all of the required articulators are included
        # in the test
        if all(
            [
                articulator in articulators
                for articulator in REQUIRED_ARTICULATORS_FOR_TVS
            ]
        ):
            tract_variables(
                sentences_ids,
                sentence_frames,
                outputs,
                targets,
                lengths,
                phonemes,
                articulators,
                epoch_outputs_dir
            )

        save_outputs(
            sentences_ids,
            sentence_frames,
            outputs,
            targets,
            lengths,
            phonemes,
            articulators,
            epoch_outputs_dir,
            regularize_out
        )

    mean_loss = np.mean(losses)

    info = {
        "loss": mean_loss
    }

    to_mm = dataset_config.RES * dataset_config.PIXEL_SPACING
    info.update({
        art: {
            "x_corr": np.mean(x_corrs[i_art]),
            "y_corr": np.mean(y_corrs[i_art]),
            "p2cp": np.mean(p2cp_per_articulator[i_art]),
            "p2cp_mm": np.mean(p2cp_per_articulator[i_art]) * to_mm,
            "med": np.mean(euclidean_per_articulator[i_art]),
            "med_mm": np.mean(euclidean_per_articulator[i_art]) * to_mm,
        }
        for i_art, art in enumerate(articulators)
    })

    return info
