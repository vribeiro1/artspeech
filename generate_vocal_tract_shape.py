import pdb

import argparse
import cv2
import funcy
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from tgt import read_textgrid
from torch.utils.data import DataLoader

from bs_regularization import regularize_Bsplines
from connect_vocal_tract_articulators import connect_articulators
from dataset import TailClipper
from helpers import npy_to_xarticul, set_seeds
from model import ArtSpeech
from vt_tracker.visualization import COLORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOCABULARY_FILEPATH = os.path.join(BASE_DIR, "data", "vocabulary_gottingen.json")


def validate_textgrid(textgrid_filepath, encoding="utf-8"):
    textgrid = read_textgrid(textgrid_filepath, encoding=encoding)
    textgrid_tier_names = [tier.name for tier in textgrid.tiers]

    required_tiers = ["PhonTier", "SentenceTier"]
    missing_tiers = funcy.lfilter(lambda tier: tier not in textgrid_tier_names, required_tiers)

    if any(missing_tiers):
        raise Exception(f"Textgrid file is missing the tiers '{missing_tiers}'")


def get_repeated_phoneme(phone, framerate):
    period = 1 / framerate
    phone_duration = phone.end_time - phone.start_time

    return [phone.text] * int("%.0f" % (phone_duration / period))


def get_phonetic_sequences(textgrid, framerate=55):
    phone_tier = textgrid.get_tier_by_name("PhonTier")
    sentence_tier = textgrid.get_tier_by_name("SentenceTier")

    sentences = []
    for sentence_interval in sentence_tier:
        sentence_start = sentence_interval.start_time
        sentence_end = sentence_interval.end_time

        in_interval = lambda interval: interval.start_time >= sentence_start and interval.end_time <= sentence_end
        phonemes = funcy.lfilter(in_interval, phone_tier)
        repeat_phoneme = funcy.partial(get_repeated_phoneme, framerate=framerate)
        phonetic_sequence = funcy.lflatten(map(repeat_phoneme, phonemes))

        sentences.append(phonetic_sequence)

    return sentences


def make_frame(outputs, phoneme, regularize_outputs=True):
    lw = 5
    fig = Figure(figsize=(10, 10))
    canvas = FigureCanvas(fig)

    ax = fig.gca()

    for j, output in enumerate(outputs):
        articulator = list(COLORS.keys())[j]
        output = output.copy().T

        if regularize_outputs:
            reg_X, reg_Y = regularize_Bsplines(output, 3)
            output = np.array([reg_X, reg_Y]).T

        ax.plot(*zip(*output), c=COLORS[articulator], linewidth=lw)

    ax.text(0.475, 0.15, f"/{phoneme[0]}/", fontsize=56, color="blue")

    ax.set_ylim([1., 0.])
    ax.set_xlim([0., 1.])
    ax.axis("off")

    canvas.draw()
    frame = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    frame = np.reshape(frame, (1000, 1000, 3))

    return frame


def make_vocal_tract_shape_video(outputs, phonemes, save_to, regularize_outputs=True, framerate=55):
    video_filepath = os.path.join(save_to, f"{i}.avi")
    video_writer = cv2.VideoWriter(
        video_filepath,
        cv2.VideoWriter_fourcc(*"DIVX"),
        framerate,
        (1000, 1000)
    )

    np_outputs = outputs.detach().cpu().numpy()
    for i_frame, (frame_outputs, phoneme) in enumerate(zip(np_outputs, phonemes)):
        frame = make_frame(frame_outputs, phoneme, regularize_outputs)
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video_writer.release()


def save_vocal_tract_shape(outputs, phonemes, save_to, regularize_outputs=True):
    lw = 5

    np_outputs = outputs.detach().cpu().numpy()
    for i_frame, (frame_outputs, phoneme) in enumerate(zip(np_outputs, phonemes)):
        plt.figure(figsize=(10, 10))

        frame = make_frame(frame_outputs, phoneme, regularize_outputs)
        plt.imshow(frame)
        plt.axis("off")

        jpg_filepath = os.path.join(save_to, f"{'%04d' % i_frame}.jpg")
        plt.savefig(jpg_filepath, format="jpg")

        pdf_filepath = os.path.join(save_to, f"{'%04d' % i_frame}.pdf")
        plt.savefig(pdf_filepath, format="pdf")

        plt.close()


def save_contours(outputs, save_to, regularize_outputs=True):
    filepaths = []
    np_outputs = outputs.detach().cpu().numpy()
    for i_frame, frame_outputs in enumerate(np_outputs):
        articul_dicts = {}
        for i_articul, articul_arr in enumerate(frame_outputs):
            articulator = list(COLORS.keys())[i_articul]

            if regularize_outputs:
                reg_X, reg_Y = regularize_Bsplines(articul_arr.T, 3)
                articul_arr = np.array([reg_X, reg_Y])

            filepath = os.path.join(save_to, f"{i_frame}_{articulator}.npy")
            np.save(filepath, articul_arr)

            articul_dicts[articulator] = filepath

        filepaths.append(articul_dicts)

    return filepaths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--textgrid", dest="textgrid_filepath", required=True)
    parser.add_argument("--encoding", dest="textgrid_encoding", default="utf-8")
    parser.add_argument("--state-dict", dest="state_dict_filepath", required=True)
    parser.add_argument("--save-to", dest="save_to", required=True)
    args = parser.parse_args()

    validate_textgrid(args.textgrid_filepath, args.textgrid_encoding)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(VOCABULARY_FILEPATH) as f:
        tokens = json.load(f)
        vocabulary = {token: i for i, token in enumerate(tokens)}

    textgrid = read_textgrid(args.textgrid_filepath, args.textgrid_encoding)
    phonetic_sequences = get_phonetic_sequences(textgrid)

    sentences_numerized = [
        torch.tensor([
            vocabulary[token] for token in sentence_tokens
        ], dtype=torch.long)
        for sentence_tokens in phonetic_sequences
    ]

    dataloader = DataLoader(
        list(zip(sentences_numerized, phonetic_sequences)),
        batch_size=1,
        shuffle=False,
        worker_init_fn=set_seeds
    )

    model = ArtSpeech(len(vocabulary), 11)
    state_dict = torch.load(args.state_dict_filepath, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    for i, (inputs, phonemes) in enumerate(dataloader):
        inputs = inputs.to(device)
        _, seq_len = inputs.shape
        lengths = torch.tensor([seq_len], dtype=torch.long).cpu()
        outputs = model(inputs, lengths)

        save_sentence_dir = os.path.join(args.save_to, str(i))
        if not os.path.exists(save_sentence_dir):
            os.makedirs(save_sentence_dir)

        save_contours_dir = os.path.join(save_sentence_dir, "contours")
        if not os.path.exists(save_contours_dir):
            os.makedirs(save_contours_dir)

        save_vocal_tract_shape(outputs.squeeze(dim=0), phonemes, save_sentence_dir)
        make_vocal_tract_shape_video(outputs.squeeze(dim=0), phonemes, save_sentence_dir)

        xarticul_dir = os.path.join(save_sentence_dir, "xarticul")
        if not os.path.exists(xarticul_dir):
            os.makedirs(xarticul_dir)

        articulators_dicts = save_contours(outputs.squeeze(dim=0), save_contours_dir)
        for i_frame, articuls_dict in enumerate(articulators_dicts):
            internal_wall, external_wall = connect_articulators(articuls_dict)

            xarticul_int = npy_to_xarticul(internal_wall * TailClipper.RES)
            xarticul_ext = npy_to_xarticul(external_wall * TailClipper.RES)

            plt.figure(figsize=(10, 10))

            plt.plot(*zip(*internal_wall), lw=3, color="red")
            plt.plot(*zip(*external_wall), lw=3, color="blue")

            plt.ylim([1., 0.])
            plt.xlim([0., 1.])
            plt.axis("off")

            plt.savefig(os.path.join(save_sentence_dir, f"{'%04d' % i_frame}_shape.jpg"))
            plt.close()

            xarticul_array = xarticul_int + xarticul_ext
            xarticul_filepath = os.path.join(xarticul_dir, f"{'%04d' % i_frame}.txt")
            with open(xarticul_filepath, "w") as f:
                f.write("\n".join(xarticul_array))
