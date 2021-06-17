import pdb

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from tqdm import tqdm

from bs_regularization import regularize_Bsplines
from loss import pearsons_correlation
from reconstruct_snail import reconstruct_snail_from_midline
from tract_variables import calculate_vocal_tract_variables

RES = 136
PIXEL_SPACING = 1.62

COLORS = {
    "arytenoid-muscle": "blueviolet",
    "epiglottis": "turquoise",
    "hyoid-bone": "slategray",
    "lower-incisor": "cyan",
    "lower-lip": "lime",
    "pharynx": "goldenrod",
    "soft-palate": "dodgerblue",
    "thyroid-cartilage": "saddlebrown",
    "tongue": "darkorange",
    "upper-incisor": "yellow",
    "upper-lip": "magenta",
    "vocal-folds": "hotpink"
}

CLOSED = [
    "hyoid-bone",
    "thyroid-cartilage"
]

SNAIL = {
    "epiglottis": {
        "width_int": 2 / (RES * PIXEL_SPACING),
        "width_ext": 2 / (RES * PIXEL_SPACING),
        "width_apex_int": 1 / (RES * PIXEL_SPACING),
        "width_apex_ext": 1 / (RES * PIXEL_SPACING)
    },
    "soft-palate": {
        "width_int": 2 / (RES * PIXEL_SPACING),
        "width_ext": 6 / (RES * PIXEL_SPACING),
        "width_apex_int": 1 / (RES * PIXEL_SPACING),
        "width_apex_ext": 3 / (RES * PIXEL_SPACING)
    }
}


def save_outputs(outputs, targets, phonemes, save_to, articulators, regularize_out, reconstruct_snail=False):
    for j, (out, target, phoneme) in enumerate(zip(outputs, targets, phonemes)):
        plt.figure(figsize=(10, 10))

        lw = 5
        for i_art, art in enumerate(sorted(articulators)):
            art_arr = out[i_art].numpy()

            if regularize_out:
                resX, resY = regularize_Bsplines(art_arr.transpose(1, 0), 3)
                art_arr = np.array([resX, resY])

            npy_filepath = os.path.join(save_to, "contours", f"{j}_{art}.npy")
            with open(npy_filepath, "wb") as f:
                np.save(f, art_arr)

            art_arr = art_arr.transpose(1, 0)

            if reconstruct_snail and art in SNAIL:
                snail_params = SNAIL[art]
                w_int = snail_params["width_int"]
                w_ext = snail_params["width_ext"]
                w_apex_int = snail_params["width_apex_int"]
                w_apex_ext = snail_params["width_apex_ext"]

                art_arr = reconstruct_snail_from_midline(
                    art_arr,
                    w_int, w_ext,
                    w_apex_int, w_apex_ext
                )

            if art in CLOSED:
                art_arr = np.append(art_arr, [art_arr[0]], axis=0)

            reg_x, reg_y = (art_arr * RES).transpose(1, 0)
            color = COLORS.get(art, "black")
            plt.plot(reg_x, RES - reg_y, linewidth=lw, c=color)

        for i_art, art in enumerate(sorted(articulators)):
            art_arr = target[i_art].numpy()

            npy_filepath = os.path.join(save_to, "contours", f"{j}_{art}_true.npy")
            with open(npy_filepath, "wb") as f:
                np.save(f, art_arr)

            if art in CLOSED:
                art_arr = art_arr.transpose(1, 0)
                art_arr = np.append(art_arr, [art_arr[0]], axis=0)
                art_arr = art_arr.transpose(1, 0)

            x, y = art_arr * RES
            plt.plot(x, RES - y, "--r", linewidth=lw, alpha=0.5)

        phone, = phoneme
        phone = f"/{phone}/"
        plt.text(64, 10, phone, c="blue", fontsize=56)
        plt.xlim([0, 136])
        plt.ylim([0, 136])

        plt.axis("off")
        plt.tight_layout()

        filepath = os.path.join(save_to, f"{j}.jpg")
        plt.savefig(filepath)
        filepath = os.path.join(save_to, f"{j}.pdf")
        plt.savefig(filepath)
        plt.close()


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
             regularize_out=False, reconstruct_snail=False):
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

            # save_outputs(
            #     outputs.detach().cpu(),
            #     targets.detach().cpu(),
            #     phonemes,
            #     saves_i_dir,
            #     articulators,
            #     regularize_out,
            #     reconstruct_snail
            # )

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
