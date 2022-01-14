import pdb

import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from scipy.io import wavfile
from tqdm import tqdm

from articulation_to_melspec.waveglow import melspec_to_audio


def run_tacotron2_inference(model, dataloader, device=None, save_to=None, sampling_rate=22050):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    progress_bar = tqdm(dataloader, desc=f"Running inference")
    for sentences_names, sentences, len_sentences, targets, _, len_targets in progress_bar:
        sentences = sentences.to(device)
        len_sentences = len_sentences.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(False):
            melspecs, len_melspecs, attn_weights = model.infer(sentences, len_sentences)

            for (
                sentence_name, melspec, len_melspec, target, len_target, attn_weights_
            ) in zip(
                sentences_names, melspecs, len_melspecs, targets, len_targets, attn_weights
            ):
                output_melspec = melspec[:, :len_melspec]
                target_melspec = target[:, :len_target]

                if device.type == "cuda":
                    target_audio = melspec_to_audio(target_melspec.unsqueeze(dim=0))
                    output_audio = melspec_to_audio(output_melspec.unsqueeze(dim=0))

                    target_audio = target_audio.squeeze(dim=0).cpu().numpy()
                    target_audio = target_audio.astype("int16")
                    audio_save_filepath = os.path.join(save_to, f"{sentence_name}_ground_truth.wav")
                    wavfile.write(audio_save_filepath, sampling_rate, target_audio)

                    output_audio = output_audio.squeeze(dim=0).cpu().numpy()
                    output_audio = output_audio.astype("int16")
                    audio_save_filepath = os.path.join(save_to, f"{sentence_name}.wav")
                    wavfile.write(audio_save_filepath, sampling_rate, output_audio)

                output_melspec = output_melspec.cpu().detach().numpy()
                target_melspec = target_melspec.cpu().detach().numpy()
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
                    fig_save_filepath = os.path.join(save_to, f"{sentence_name}_attn_weights.jpg")
                    plt.savefig(fig_save_filepath)
                plt.close()

                plt.figure(figsize=(20, 10))

                plt.subplot(2, 1, 1)

                plt.imshow(target_melspec, origin="lower", aspect="auto", cmap="coolwarm")
                plt.title("Target Melspectogram", fontsize=26)
                plt.xlabel("Frame", fontsize=26)
                plt.ylabel("Frequency bin", fontsize=26)
                plt.grid()

                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)

                plt.subplot(2, 1, 2)

                plt.imshow(output_melspec, origin="lower", aspect="auto", cmap="coolwarm")
                plt.title("Predicted Melspectogram", fontsize=26)
                plt.xlabel("Frame", fontsize=26)
                plt.ylabel("Frequency bin", fontsize=26)
                plt.grid()

                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)

                plt.tight_layout()
                if save_to is not None:
                    fig_save_filepath = os.path.join(save_to, f"{sentence_name}.jpg")
                    plt.savefig(fig_save_filepath)
                plt.close()


def run_glow_tts_inference(model, dataloader, device=None, save_to=None):
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
                plt.ylabel("Frequency bin", fontsize=26)
                plt.grid()

                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)

                plt.subplot(2, 1, 2)

                plt.imshow(melspec, origin="lower", aspect="auto", cmap="coolwarm")
                plt.title("Predicted Melspectogram", fontsize=26)
                plt.xlabel("Frame", fontsize=26)
                plt.ylabel("Frequency bin", fontsize=26)
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
