import pdb

import numpy as np
import os
import torch

from helpers import make_padding_mask
from metrics import pearsons_correlation, p2cp_distance, euclidean_distance
from tqdm import tqdm

from phoneme_to_articulation import (
    save_outputs,
    tract_variables,
    REQUIRED_ARTICULATORS_FOR_TVS
)
from vt_tools import UPPER_INCISOR


def run_transformer_test(
    epoch,
    model,
    dataloader,
    criterion,
    outputs_dir,
    articulators,
    device=None,
    regularize_out=False,
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
        reference_arrays,
        sentence_frames,
        _,  # voicing
        src_key_padding_mask,  # src_key_padding_mask
        tgt_key_padding_mask,  # tgt_key_padding_mask
        src_attn_mask,  # src_attn_mask
        tgt_attn_mask,  # tgt_attn_mask
    ) in progress_bar:
        sentences = sentences.to(device)
        targets = targets.to(device)
        src_key_padding_mask = src_key_padding_mask.to(device)

        with torch.set_grad_enabled(False):
            outputs = model.generate(
                sentences,
                src_key_padding_mask=src_key_padding_mask,
            )

            # For some inputs, the Transformer's encoder generates only NaNs, and the output
            # prediction is invalid. We need to filter out these cases to ba able to calculate the
            # metrics.
            # TODO: Find out why these (apparently valid) inputs produce invalid outputs in the
            # Transformer's encoder.
            nan_indices = [i for i, out in enumerate(outputs) if out.isnan().any()]
            notnan_indices = [i for i, out in enumerate(outputs) if not out.isnan().any()]
            if len(notnan_indices) == 0:
                continue

            if len(nan_indices) > 0:
                nan_sentence_ids = [sentences_ids[i] for i in nan_indices]
                invalid_sentences = "\n".join(nan_sentence_ids)
                print(f"Invalid outputs produced for sentences:\n{invalid_sentences}\n")

            outputs = outputs[notnan_indices]
            targets = targets[notnan_indices]
            reference_arrays = reference_arrays[notnan_indices]

            loss = criterion(outputs, targets)
            padding_mask = make_padding_mask(lengths)
            padding_mask = padding_mask[notnan_indices]
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

        # The upper incisor is the reference of the coordinate system and since it has a fixed
        # shape, it is non-sense to include it in the prediction. However, it is important for
        # tract variables and visualization. Therefore, we inject it in the arrays in order to
        # have it available for the next steps.
        if UPPER_INCISOR not in articulators:
            tv_articulators = sorted(articulators + [UPPER_INCISOR])
            ref_idx = tv_articulators.index(UPPER_INCISOR)

            outputs = torch.concat([
                outputs[:, :, :ref_idx, :, :],
                reference_arrays,
                outputs[:, :, ref_idx:, :, :],
            ], dim=2)

            targets = torch.concat([
                targets[:, :, :ref_idx, :, :],
                reference_arrays,
                targets[:, :, ref_idx:, :, :],
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
                sentences_ids,
                sentence_frames,
                outputs,
                targets,
                lengths,
                phonemes,
                tv_articulators,
                epoch_outputs_dir
            )

        save_outputs(
            sentences_ids,
            sentence_frames,
            outputs,
            targets,
            lengths,
            phonemes,
            tv_articulators,
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
