import pdb

import numpy as np
import os
import torch

from tqdm import tqdm

from phoneme_to_articulation.encoder_decoder.evaluation import save_outputs
from phoneme_to_articulation.principal_components.models import Decoder

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


def run_phoneme_to_PC_test(epoch, model, decoder_state_dict_fpath, dataloader, criterion, outputs_dir, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch_outputs_dir = os.path.join(outputs_dir, str(epoch))
    if not os.path.exists(epoch_outputs_dir):
        os.makedirs(epoch_outputs_dir)

    model.eval()

    decoder = Decoder(n_components=12, out_features=100)
    decoder_state_dict = torch.load(decoder_state_dict_fpath, map_location=device)
    decoder.load_state_dict(decoder_state_dict)
    decoder.to(device)
    decoder.eval()

    losses = []
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - test")
    for sentence_ids, sentence, targets, lengths, phonemes in progress_bar:
        sentence = sentence.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(sentence, lengths)
            loss = criterion(outputs, targets)

            pred_shapes = decoder(outputs)
            bs, seq_len, _ = pred_shapes.shape
            pred_shapes = pred_shapes.reshape(bs, seq_len, 1, 2, 50)

            pred_shapes = pred_shapes.detach().cpu()
            target_shapes = targets.detach().cpu()

            save_outputs(
                sentence_ids,
                pred_shapes,
                target_shapes,
                lengths,
                phonemes,
                ["tongue"],
                epoch_outputs_dir,
                regularize_out=False
            )

            losses.append(loss.item())
            progress_bar.set_postfix(loss=np.mean(losses))

    mean_loss = np.mean(losses)
    info = {
        "loss": mean_loss,
        "saves_dir": epoch_outputs_dir
    }

    return info
