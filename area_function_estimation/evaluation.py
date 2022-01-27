import pdb

import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from tqdm import tqdm


def run_test(epoch, model, dataloader, criterion, outputs_dir, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch_outputs_dir = os.path.join(outputs_dir, str(epoch))
    if not os.path.exists(epoch_outputs_dir):
        os.makedirs(epoch_outputs_dir)

    model.eval()

    losses = []
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - inference")
    for i_batch, (frame_articulators, x, fx_targets) in enumerate(progress_bar):
        frame_articulators = frame_articulators.to(device)
        fx_targets = fx_targets.to(device)

        with torch.set_grad_enabled(False):
            fx_outputs = model(frame_articulators.unsqueeze(dim=1))
            loss = criterion(fx_outputs.squeeze(dim=1), fx_targets)

            losses.append(loss.item())

        progress_bar.set_postfix(loss=np.mean(losses))

        fx_outputs = fx_outputs.detach().cpu()
        x = x.detach().cpu()
        fx_targets = fx_targets.detach().cpu()

        for i_frame, (fx_output, fx_target, x_) in enumerate(zip(fx_outputs, fx_targets, x)):
            plt.figure(figsize=(15, 10))

            plt.plot(x_, fx_output.squeeze(dim=0), c="blue", lw=3)
            plt.plot(x_, fx_target, "r--", lw=3)

            plt.grid(which="major")
            plt.grid(which="minor", linestyle='--', alpha=0.4)
            plt.minorticks_on()
            plt.tight_layout()

            save_filepath = os.path.join(epoch_outputs_dir, f"{i_batch + i_frame}.jpg")
            plt.savefig(save_filepath)
            plt.close()

    mean_loss = np.mean(losses)
    info = {
        "loss": mean_loss
    }

    return info
