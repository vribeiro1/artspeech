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
    "upper-lip": "fuchsia",
    "tongue": "darkorange",
    "soft-palate": "dodgerblue"
}


def save_outputs(outputs, targets, phonemes, save_to, regularize_out):
    for j, (out, target, phoneme) in enumerate(zip(outputs, targets, phonemes)):
        lower_lip, soft_palate, tongue, upper_lip = out
        lower_lip_true, soft_palate_true, tongue_true, upper_lip_true = target

        plt.figure(figsize=(10, 10))

        lw = 5
        for art, art_arr in [
            ("lower-lip", lower_lip),
            ("soft-palate", soft_palate),
            ("tongue", tongue),
            ("upper-lip", upper_lip)
        ]:
            art_arr = art_arr.detach().numpy()
            if regularize_out:
                resX, resY = regularize_Bsplines(art_arr.transpose(1, 0), 3)
                art_arr = np.array([resX, resY])

            npy_filepath = os.path.join(save_to, "contours", f"{j}_{art}.npy")
            with open(npy_filepath, "wb") as f:
                np.save(f, art_arr)

            x, y = art_arr * 136
            plt.plot(x, 136 - y, linewidth=lw, c=COLORS[art])

        for art, art_arr in [
            ("lower-lip", lower_lip_true),
            ("soft-palate", soft_palate_true),
            ("tongue", tongue_true),
            ("upper-lip", upper_lip_true)
        ]:

            art_arr = art_arr.detach().numpy()

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

        # plt.title("".join(funcy.lflatten(phonemes)))
        plt.axis("off")
        plt.tight_layout()

        filepath = os.path.join(save_to, f"{j}.jpg")
        plt.savefig(filepath)
        filepath = os.path.join(save_to, f"{j}.pdf")
        plt.savefig(filepath)
        plt.close()



def run_test(epoch, model, dataloader, criterion, outputs_dir, device=None, regularize_out=False):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch_outputs_dir = os.path.join(outputs_dir, str(epoch))
    if not os.path.exists(epoch_outputs_dir):
        os.makedirs(epoch_outputs_dir)

    model.eval()

    losses = []
    x_corrs = [[], [], [], []]
    y_corrs = [[], [], [], []]
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
            llip_x_corr, sp_x_corr, tongue_x_corr, ulip_x_corr = x_corr.mean(dim=-1)[0]
            llip_y_corr, sp_y_corr, tongue_y_corr, ulip_y_corr = y_corr.mean(dim=-1)[0]

            x_corrs[0].append(llip_x_corr.item())
            x_corrs[1].append(sp_x_corr.item())
            x_corrs[2].append(tongue_x_corr.item())
            x_corrs[3].append(ulip_x_corr.item())

            y_corrs[0].append(llip_y_corr.item())
            y_corrs[1].append(sp_y_corr.item())
            y_corrs[2].append(tongue_y_corr.item())
            y_corrs[3].append(ulip_y_corr.item())

            losses.append(loss.item())
            progress_bar.set_postfix(loss=np.mean(losses))

            outputs = outputs.squeeze(dim=0)
            targets = targets.squeeze(dim=0)

            save_outputs(outputs, targets, phonemes, saves_i_dir, regularize_out)

    mean_loss = np.mean(losses)

    mean_x_corr_llip = np.mean(x_corrs[0])
    mean_y_corr_llip = np.mean(y_corrs[0])

    mean_x_corr_sp = np.mean(x_corrs[1])
    mean_y_corr_sp = np.mean(y_corrs[1])

    mean_x_corr_tongue = np.mean(x_corrs[2])
    mean_y_corr_tongue = np.mean(y_corrs[2])

    mean_x_corr_ulip = np.mean(x_corrs[3])
    mean_y_corr_ulip = np.mean(y_corrs[3])

    info = {
        "loss": mean_loss,
        "lower-lip": {
            "x_corr": mean_x_corr_llip,
            "y_corr": mean_y_corr_llip
        },
        "soft-palate": {
            "x_corr": mean_x_corr_sp,
            "y_corr": mean_y_corr_sp
        },
        "tongue": {
            "x_corr": mean_x_corr_tongue,
            "y_corr": mean_y_corr_tongue
        },
        "upper-lip": {
            "x_corr": mean_x_corr_ulip,
            "y_corr": mean_y_corr_ulip
        },
    }

    return info
