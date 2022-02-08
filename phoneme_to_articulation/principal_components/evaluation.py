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
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - test")
    for i, (_, inputs) in enumerate(progress_bar):
        inputs = inputs.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            losses.append(loss.item())
            progress_bar.set_postfix(loss=np.mean(losses))

    mean_loss = np.mean(losses)
    info = {
        "loss": mean_loss
    }

    return info
