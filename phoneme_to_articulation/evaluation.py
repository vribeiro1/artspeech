import pdb

import numpy as np
import os
import pandas as pd
import torch

from tqdm import tqdm
from vt_tracker import (
    LOWER_LIP,
    PHARYNX,
    SOFT_PALATE,
    TONGUE,
    UPPER_LIP,
    UPPER_INCISOR,
)

from bs_regularization import regularize_Bsplines
from metrics import pearsons_correlation, p2cp_distance, euclidean_distance
from tract_variables import calculate_vocal_tract_variables

required_articulators_for_TVs = [
    LOWER_LIP,
    PHARYNX,
    SOFT_PALATE,
    TONGUE,
    UPPER_LIP,
    UPPER_INCISOR,
]


def save_outputs(sentences_ids, outputs, targets, lengths, phonemes, articulators, save_to, regularize_out):
    """
    Args:
    sentences_ids (str): Unique id of each sentence to save the results.
    outputs (torch.tensor): Tensor with shape (bs, seq_len, n_articulators, 2, n_samples).
    targets (torch.tensor): Tensor with shape (bs, seq_len, n_articulators, 2, n_samples).
    lengths (List): List with the length of each sentence in the batch.
    phonemes (List): List with the sequence of phonemes for each sentence in the batch.
    articulators (List[str]): List of articulators.
    save_to (str): Path to the directory to save the results.
    regularize_out (bool): If should apply bspline regularization or not.
    """
    for i_sentence, (
        sentence_id, sentence_outs, sentence_targets, length, sentence_phonemes
    ) in enumerate(zip(
        sentences_ids, outputs, targets, lengths, phonemes
    )):
        saves_i_dir = os.path.join(save_to, sentence_id)
        if not os.path.exists(os.path.join(saves_i_dir, "contours")):
            os.makedirs(os.path.join(saves_i_dir, "contours"))

        for i_frame, (out, target, _) in enumerate(zip(sentence_outs[:length], sentence_targets[:length], sentence_phonemes)):
            for i_art, art in enumerate(sorted(articulators)):
                pred_art_arr = out[i_art].numpy()
                true_art_arr = target[i_art].numpy()

                if regularize_out:
                    resX, resY = regularize_Bsplines(pred_art_arr.transpose(1, 0), 3)
                    pred_art_arr = np.array([resX, resY])

                pred_npy_filepath = os.path.join(saves_i_dir, "contours", f"{'%04d' % i_frame}_{art}.npy")
                with open(pred_npy_filepath, "wb") as f:
                    np.save(f, pred_art_arr)

                true_npy_filepath = os.path.join(saves_i_dir, "contours", f"{'%04d' % i_frame}_{art}_true.npy")
                with open(true_npy_filepath, "wb") as f:
                    np.save(f, true_art_arr)


def tract_variables(sentences_ids, outputs, targets, lengths, phonemes, articulators, save_to):
    """
    Args:
    sentences_ids (str): Unique id of each sentence to save the results.
    outputs (torch.tensor): Tensor with shape (bs, seq_len, n_articulators, 2, n_samples).
    targets (torch.tensor): Tensor with shape (bs, seq_len, n_articulators, 2, n_samples).
    lengths (List): List with the length of each sentence in the batch.
    phonemes (List): List with the sequence of phonemes for each sentence in the batch.
    articulators (List[str]): List of articulators.
    save_to (str): Path to the directory to save the results.
    """
    for i_sentence, (
        sentence_id, sentence_outs, sentence_targets, length, sentence_phonemes
    ) in enumerate(zip(
        sentences_ids, outputs, targets, lengths, phonemes
    )):
        saves_i_dir = os.path.join(save_to, sentence_id)
        if not os.path.exists(saves_i_dir):
            os.makedirs(saves_i_dir)

        TVs_data = []
        for i_frame, (out, target, phoneme) in enumerate(zip(sentence_outs[:length], sentence_targets[:length], sentence_phonemes)):
            pred_input_dict = {
                art: tensor.T for art, tensor in zip(articulators, out)
            }
            pred_TVs = calculate_vocal_tract_variables(pred_input_dict)

            target_input_dict = {
                art: tensor.T for art, tensor in zip(articulators, target)
            }
            target_TVs = calculate_vocal_tract_variables(target_input_dict)

            item = {
                "sentence": sentence_id,
                "frame": '%04d' % i_frame,
                "phoneme": phoneme
            }

            for TV, TV_dict in target_TVs.items():
                if TV_dict is None:
                    continue

                item.update({
                    f"{TV}_target": TV_dict["value"],
                    f"{TV}_target_poc_1_x": TV_dict["poc_1"][0].item(),
                    f"{TV}_target_poc_1_y": TV_dict["poc_1"][1].item(),
                    f"{TV}_target_poc_2_x": TV_dict["poc_2"][0].item(),
                    f"{TV}_target_poc_2_y": TV_dict["poc_2"][1].item()
                })

            for TV, TV_dict in pred_TVs.items():
                if TV_dict is None:
                    continue

                item.update({
                    f"{TV}_pred": TV_dict["value"],
                    f"{TV}_pred_poc_1_x": TV_dict["poc_1"][0].item(),
                    f"{TV}_pred_poc_1_y": TV_dict["poc_1"][1].item(),
                    f"{TV}_pred_poc_2_x": TV_dict["poc_2"][0].item(),
                    f"{TV}_pred_poc_2_y": TV_dict["poc_2"][1].item()
                })

            TVs_data.append(item)

        filepath = os.path.join(saves_i_dir, "tract_variables.csv")
        df = pd.DataFrame(TVs_data)
        df.to_csv(filepath, index=False)


def run_test(epoch, model, dataloader, criterion, outputs_dir, articulators,
             device=None, regularize_out=False):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch_outputs_dir = os.path.join(outputs_dir, str(epoch))
    if not os.path.exists(epoch_outputs_dir):
        os.makedirs(epoch_outputs_dir)

    model.eval()

    losses = []
    euclidean_per_articulator = [[] for _ in articulators]
    p2cp_per_articulator = [[] for _ in articulators]
    x_corrs = [[] for _ in articulators]
    y_corrs = [[] for _ in articulators]
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - inference")
    for i_batch, (sentences_ids, sentences, targets, lengths, phonemes) in enumerate(progress_bar):
        sentences = sentences.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(sentences, lengths)
            loss = criterion(outputs, targets)

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
            if [articulator in articulators for articulator in required_articulators_for_TVs]:
                tract_variables(
                    sentences_ids,
                    outputs,
                    targets,
                    lengths,
                    phonemes,
                    articulators,
                    epoch_outputs_dir
                )

            save_outputs(
                sentences_ids,
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

    info.update({
        art: {
            "x_corr": np.mean(x_corrs[i_art]),
            "y_corr": np.mean(y_corrs[i_art]),
            "p2cp": np.mean(p2cp_per_articulator[i_art]),
            "med": np.mean(euclidean_per_articulator[i_art])
        }
        for i_art, art in enumerate(articulators)
    })

    return info
