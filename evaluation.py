import pdb

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from tqdm import tqdm

from bs_regularization import regularize_Bsplines
from loss import pearsons_correlation
from tract_variables import calculate_vocal_tract_variables

RES = 136
# PIXEL_SPACING = 1.62
PIXEL_SPACING = 1.4117647409439


def save_outputs(outputs, targets, phonemes, save_to, articulators, regularize_out):
    for j, (out, target, phoneme) in enumerate(zip(outputs, targets, phonemes)):
        for i_art, art in enumerate(sorted(articulators)):
            pred_art_arr = out[i_art].numpy()
            true_art_arr = target[i_art].numpy()

            if regularize_out:
                resX, resY = regularize_Bsplines(pred_art_arr.transpose(1, 0), 3)
                pred_art_arr = np.array([resX, resY]).T

            pred_npy_filepath = os.path.join(save_to, f"{j}_{art}.npy")
            with open(pred_npy_filepath, "wb") as f:
                np.save(f, pred_art_arr)

            true_npy_filepath = os.path.join(save_to, f"{j}_{art}_true.npy")
            with open(true_npy_filepath, "wb") as f:
                np.save(f, true_art_arr)


def prepare_for_serialization(TVs_out):
    TVs_out_serializable = {}
    for TV, TV_dict in TVs_out.items():
        TV_dict_serializable = {
            "value": TV_dict["value"],
            "poc_1": TV_dict["poc_1"].tolist(),
            "poc_2": TV_dict["poc_2"].tolist()
        } if TV_dict is not None else None

        TVs_out_serializable[TV] = TV_dict_serializable

    return TVs_out_serializable


def tract_variables(outputs, targets, phonemes, articulators, save_to=None):
    TVs_data = []
    for j, (out, target, phoneme) in enumerate(zip(outputs, targets, phonemes)):
        pred_input_dict = {
            art: tensor.T for art, tensor in zip(articulators, out)
        }
        pred_TVs = calculate_vocal_tract_variables(pred_input_dict)

        target_input_dict = {
            art: tensor.T for art, tensor in zip(articulators, target)
        }
        target_TVs = calculate_vocal_tract_variables(target_input_dict)

        phone, = phoneme
        TVs_data.append(
            {
                "frame": j,
                "phoneme": phone,
                "tract_variables": {
                    "target": target_TVs,
                    "predicted": pred_TVs
                }
            }
        )

    if save_to is not None:
        serializable_data = [
            {
                "frame": item["frame"],
                "phoneme": item["phoneme"],
                "tract_variables": {
                    "target": prepare_for_serialization(item["tract_variables"]["target"]),
                    "predicted": prepare_for_serialization(item["tract_variables"]["predicted"])
                }
            } for item in TVs_data
        ]

        json_filepath = os.path.join(save_to, "tract_variables_data.json")
        with open(json_filepath, "w") as f:
            json.dump(serializable_data, f)

    return TVs_data


def run_test(epoch, model, dataloader, criterion, outputs_dir, articulators, device=None,
             regularize_out=False):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch_outputs_dir = os.path.join(outputs_dir, str(epoch))
    if not os.path.exists(epoch_outputs_dir):
        os.makedirs(epoch_outputs_dir)

    model.eval()

    losses = []
    x_corrs = [[] for _ in articulators]
    y_corrs = [[] for _ in articulators]
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - inference")
    for i, (sentence, targets, phonemes) in enumerate(progress_bar):
        saves_i_dir = os.path.join(epoch_outputs_dir, str(i))
        if not os.path.exists(os.path.join(saves_i_dir, "contours")):
            os.makedirs(os.path.join(saves_i_dir, "contours"))

        sentence = sentence.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(sentence)
            loss = criterion(outputs, targets)

            x_corr, y_corr = pearsons_correlation(outputs, targets)
            x_corr = x_corr.mean(dim=-1)[0]
            y_corr = y_corr.mean(dim=-1)[0]

            for i, _ in enumerate(articulators):
                x_corrs[i].append(x_corr[i].item())
                y_corrs[i].append(y_corr[i].item())

            losses.append(loss.item())
            progress_bar.set_postfix(loss=np.mean(losses))

            outputs = outputs.squeeze(dim=0)
            targets = targets.squeeze(dim=0)

            tract_variables(
                outputs.detach().cpu(),
                targets.detach().cpu(),
                phonemes,
                articulators,
                saves_i_dir
            )

            save_outputs(
                outputs.detach().cpu(),
                targets.detach().cpu(),
                phonemes,
                os.path.join(saves_i_dir, "contours"),
                articulators,
                regularize_out
            )

    mean_loss = np.mean(losses)

    info = {
        "loss": mean_loss
    }

    info.update({
        art: {
            "x_corr": np.mean(x_corrs[i_art]),
            "y_corr": np.mean(y_corrs[i_art])
        }
        for i_art, art in enumerate(articulators)
    })

    return info
