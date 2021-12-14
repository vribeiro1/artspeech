import pdb

import numpy as np
import os
import pandas as pd
import torch

from tqdm import tqdm

from bs_regularization import regularize_Bsplines
from metrics import pearsons_correlation, p2cp_distance, euclidean_distance
from tract_variables import calculate_vocal_tract_variables

RES = 136
# PIXEL_SPACING = 1.62
PIXEL_SPACING = 1.4117647409439


def save_outputs(outputs, targets, lengths, phonemes, articulators, save_to, regularize_out, offset=0):
    """
    Args:
    outputs (torch.tensor): Tensor with shape (bs, seq_len, n_articulators, 2, n_samples).
    targets (torch.tensor): Tensor with shape (bs, seq_len, n_articulators, 2, n_samples).
    lengths (List): List with the length of each sentence in the batch.
    phonemes (List): List with the sequence of phonemes for each sentence in the batch.
    articulators (List[str]): List of articulators.
    save_to (str): Path to the directory to save the results.
    regularize_out (bool): If should apply bspline regularization or not.
    offset (int): Offset from previous processed batch.
    """
    for i_sentence, (sentence_outs, sentence_targets, length, sentence_phonemes) in enumerate(zip(outputs, targets, lengths, phonemes)):
        sentence_id = str(i_sentence + offset)
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

                pred_npy_filepath = os.path.join(saves_i_dir, "contours", f"{i_frame}_{art}.npy")
                with open(pred_npy_filepath, "wb") as f:
                    np.save(f, pred_art_arr)

                true_npy_filepath = os.path.join(saves_i_dir, "contours", f"{i_frame}_{art}_true.npy")
                with open(true_npy_filepath, "wb") as f:
                    np.save(f, true_art_arr)


def tract_variables(outputs, targets, lengths, phonemes, articulators, save_to, offset=0):
    """
    Args:
    outputs (torch.tensor): Tensor with shape (bs, seq_len, n_articulators, 2, n_samples).
    targets (torch.tensor): Tensor with shape (bs, seq_len, n_articulators, 2, n_samples).
    lengths (List): List with the length of each sentence in the batch.
    phonemes (List): List with the sequence of phonemes for each sentence in the batch.
    articulators (List[str]): List of articulators.
    save_to (str): Path to the directory to save the results.
    offset (int): Offset from previous processed batch.
    """
    for i_sentence, (sentence_outs, sentence_targets, length, sentence_phonemes) in enumerate(zip(outputs, targets, lengths, phonemes)):
        sentence_id = str(i_sentence + offset)
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
                "frame": i_frame,
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
    for i_batch, (sentence, targets, lengths, phonemes) in enumerate(progress_bar):
        sentence = sentence.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(sentence, lengths)
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

            tract_variables(
                outputs,
                targets,
                lengths,
                phonemes,
                articulators,
                epoch_outputs_dir,
                i_batch * dataloader.batch_size,
            )

            save_outputs(
                outputs,
                targets,
                lengths,
                phonemes,
                articulators,
                epoch_outputs_dir,
                regularize_out,
                i_batch * dataloader.batch_size,
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
