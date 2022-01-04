import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import yaml

from torch.utils.data import DataLoader
from tqdm import tqdm

from articulation_to_melspec.dataset import ArticulToMelSpecDataset, pad_sequence_collate_fn
from articulation_to_melspec.model import GlowATS
from helpers import set_seeds, sequences_from_dict


def run_inference(model, dataloader, device=None, save_to=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if save_to is not None:
        plots_save_to = os.path.join(save_to, "melspecs_plots")
        if not os.path.exists(plots_save_to):
            os.makedirs(plots_save_to)

        npy_save_to = os.path.join(save_to, "melspecs")
        if not os.path.exists(npy_save_to):
            os.makedirs(npy_save_to)

    model.decoder.store_inverse()
    model.eval()

    progress_bar = tqdm(dataloader, desc=f"Running inference")
    for sentences_names, sentences, len_sentences, targets, _, len_targets in progress_bar:
        sentences = sentences.to(device)
        len_sentences = len_sentences.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(False):
            noise_scale = .667
            length_scale = 1.0

            (melspecs, *_), *_, (attn_weights, *_) = model(
                sentences,
                len_sentences,
                gen=True,
                noise_scale=noise_scale,
                length_scale=length_scale
            )

            for sentence_name, melspec, target, len_target, attn_weights_ in zip(
                sentences_names, melspecs, targets, len_targets, attn_weights
            ):
                melspec = melspec.cpu().detach().numpy()
                target_melspec = target[:, :len_target].cpu().detach().numpy()
                attn_weights_ = attn_weights_.cpu().detach().numpy()

                plt.figure(figsize=(10, 10))

                plt.imshow(attn_weights_.T, origin="lower", aspect="auto")

                plt.title("Attention weights", fontsize=22)
                plt.xlabel("Spectogram Frames", fontsize=22)
                plt.ylabel("VT shape frames", fontsize=22)

                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)

                plt.grid(which="major")
                plt.grid(which="minor", linestyle="--", alpha=0.4)
                plt.minorticks_on()

                plt.tight_layout()
                if save_to is not None:
                    fig_save_filepath = os.path.join(plots_save_to, f"{sentence_name}_attn_weights.jpg")
                    plt.savefig(fig_save_filepath)
                plt.close()

                plt.figure(figsize=(20, 10))

                plt.subplot(2, 1, 1)

                plt.imshow(target_melspec, origin="lower", aspect="auto", cmap="coolwarm")
                plt.title("Target Melspectogram", fontsize=26)
                plt.xlabel("Frame", fontsize=26)
                plt.ylabel("Frequenct bin", fontsize=26)
                plt.grid()

                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)

                plt.subplot(2, 1, 2)

                plt.imshow(melspec, origin="lower", aspect="auto", cmap="coolwarm")
                plt.title("Predicted Melspectogram", fontsize=26)
                plt.xlabel("Frame", fontsize=26)
                plt.ylabel("Frequenct bin", fontsize=26)
                plt.grid()

                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)

                plt.tight_layout()
                if save_to is not None:
                    fig_save_filepath = os.path.join(plots_save_to, f"{sentence_name}.jpg")
                    plt.savefig(fig_save_filepath)
                plt.close()

                np.save(os.path.join(npy_save_to, f"{sentence_name}.npy"), melspec)
                np.save(os.path.join(npy_save_to, f"{sentence_name}_true.npy"), target_melspec)


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_sequences = sequences_from_dict(cfg["datadir"], cfg["test_seq_dict"])
    test_dataset = ArticulToMelSpecDataset(cfg["datadir"], test_sequences, cfg["articulators"])
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=4,
        worker_init_fn=set_seeds,
        collate_fn=pad_sequence_collate_fn
    )

    glow_tts_hparams = {
        "n_vocab": 148,
        "out_channels": 80,  # n_mel_channels on the Mel Spectrogram
        "hidden_channels": 192,
        "filter_channels": 768,
        "filter_channels_dp": 256,
        "kernel_size": 3,
        "p_dropout": 0.1,
        "n_blocks_dec": 12,
        "n_layers_enc": 6,
        "n_heads": 2,
        "p_dropout_dec": 0.05,
        "dilation_rate": 1,
        "kernel_size_dec": 5,
        "n_block_layers": 4,
        "n_sqz": 2,
        "prenet": True,
        "mean_only": True,
        "hidden_channels_enc": 192,
        "hidden_channels_dec": 192,
        "window_size": 4
    }

    best_model = GlowATS(
        n_articulators=len(cfg["articulators"]),
        n_samples=50,
        pretrained=True,
        **glow_tts_hparams
    )
    if cfg["state_dict_fpath"] is not None:
        state_dict = torch.load(cfg["state_dict_fpath"], map_location=device)
        best_model.load_state_dict(state_dict)
    best_model.to(device)

    test_outputs_dir = os.path.join(cfg["save_to"], "test_outputs")
    if not os.path.exists(test_outputs_dir):
        os.makedirs(test_outputs_dir)

    run_inference(
        model=best_model,
        dataloader=test_dataloader,
        device=device,
        save_to=test_outputs_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="cfg_filepath")
    args = parser.parse_args()

    with open(args.cfg_filepath) as f:
        cfg = yaml.safe_load(f.read())

    main(cfg)
