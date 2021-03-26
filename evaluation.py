import pdb

import funcy
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from tqdm import tqdm

from bs_regularization import regularize_Bsplines
from loss import pearsons_correlation

COLORS = {
    "lower-lip": "lime",
    "pharynx": "goldenrod",
    "upper-lip": "magenta",
    "tongue": "darkorange",
    "soft-palate": "dodgerblue"
}


def save_outputs(outputs, targets, phonemes, save_to, regularize_out, articulators):
    for j, (out, target, phoneme) in enumerate(zip(outputs, targets, phonemes)):
        plt.figure(figsize=(10, 10))

        lw = 5
        for i_art, art in enumerate(sorted(articulators)):
            art_arr = out[i_art].detach().numpy()
            if regularize_out:
                resX, resY = regularize_Bsplines(art_arr.transpose(1, 0), 3)
                reg_art_arr = np.array([resX, resY])

            npy_filepath = os.path.join(save_to, "contours", f"{j}_{art}.npy")
            with open(npy_filepath, "wb") as f:
                np.save(f, reg_art_arr)

            reg_x, reg_y = reg_art_arr * 136
            color = COLORS.get(art, "black")
            plt.plot(reg_x, 136 - reg_y, linewidth=lw, c=color)

        for i_art, art in enumerate(sorted(articulators)):
            art_arr = target[i_art].detach().numpy()

            npy_filepath = os.path.join(save_to, "contours", f"{j}_{art}_true.npy")
            with open(npy_filepath, "wb") as f:
                np.save(f, art_arr)

            x, y = art_arr * 136
            plt.plot(x, 136 - y, "--r", linewidth=lw)

        phone, = phoneme
        phone = f"/{phone}/"
        plt.text(40, 90, phone, c="blue", fontsize=56)
        plt.xlim([0, 80])
        plt.ylim([20, 100])

        plt.axis("off")
        plt.tight_layout()

        filepath = os.path.join(save_to, f"{j}.jpg")
        plt.savefig(filepath)
        filepath = os.path.join(save_to, f"{j}.pdf")
        plt.savefig(filepath)
        plt.close()



def run_test(epoch, model, dataloader, criterion, outputs_dir, articulators, device=None, regularize_out=False):
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

            save_outputs(outputs, targets, phonemes, saves_i_dir, regularize_out, articulators)

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
