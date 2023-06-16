import argparse
import cv2
import funcy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
import ujson
import yaml

from functools import partial
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from tgt import read_textgrid
from tqdm import tqdm
from vt_tools import (
    COLORS,
    UPPER_INCISOR,
)
from vt_tools.bs_regularization import regularize_Bsplines

from vt_shape_gen.vocal_tract_tube import generate_vocal_tract_tube
from helpers import npy_to_xarticul, sequences_from_dict
from phoneme_to_articulation.phoneme_wise_mean_contour import forward_mean_contour
from phoneme_to_articulation.encoder_decoder.dataset import ArtSpeechDataset
from phoneme_to_articulation.encoder_decoder.models import ArtSpeech
from settings import DATASET_CONFIG, BLANK, UNKNOWN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


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


def make_vocal_tract_shape_video(outputs, phonemes, video_filepath, regularize_outputs=True, framerate=55):
    video_writer = cv2.VideoWriter(
        video_filepath,
        cv2.VideoWriter_fourcc(*"DIVX"),
        framerate,
        (1000, 1000)
    )

    np_outputs = outputs.detach().cpu().numpy()
    for _, (frame_outputs, phoneme) in enumerate(zip(np_outputs, phonemes)):
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


def save_contours(outputs, reference_arrays, save_to, articulators, regularize_outputs=True):
    filepaths = []
    np_outputs = outputs.detach().cpu().numpy()

    for i_frame, (frame_outputs, frame_reference) in enumerate(zip(np_outputs, reference_arrays), start=1):
        articul_dicts = {}
        for articulator, articul_arr in zip(articulators, frame_outputs):
            if regularize_outputs:
                reg_X, reg_Y = regularize_Bsplines(articul_arr.T, 3)
                articul_arr = np.array([reg_X, reg_Y])

            filepath = os.path.join(save_to, f"{'%04d' % i_frame}_{articulator}.npy")
            np.save(filepath, articul_arr)
            articul_dicts[articulator] = filepath

        if UPPER_INCISOR not in articulators:
            filepath = os.path.join(save_to, f"{'%04d' % i_frame}_{UPPER_INCISOR}.npy")
            articul_arr = frame_reference.squeeze(dim=0).numpy()
            np.save(filepath, articul_arr)
            articul_dicts[UPPER_INCISOR] = filepath

        filepaths.append(articul_dicts)

    return filepaths


def main(
    database_name,
    datadir,
    seq_dict,
    method,
    state_dict_filepath,
    vocab_filepath,
    articulators,
    save_to,
):
    dataset_config = DATASET_CONFIG[database_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    default_tokens = [BLANK, UNKNOWN]
    vocabulary = {token: i for i, token in enumerate(default_tokens)}
    with open(vocab_filepath) as f:
        tokens = ujson.load(f)
        for i, token in enumerate(tokens, start=len(vocabulary)):
            vocabulary[token] = i

    if method == "neural-network":
        model = ArtSpeech(len(vocabulary), 10)
        state_dict = torch.load(state_dict_filepath, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
    elif method == "mean-contour":
        articulators = sorted(COLORS.keys())

        df = pd.read_csv(state_dict_filepath)
        for articulator in articulators:
            df[articulator] = df[articulator].apply(eval)

        model = partial(forward_mean_contour, df=df, articulators=articulators)
    else:
        raise Exception(f"Unavailable method '{method}'")

    sequences = sequences_from_dict(datadir, seq_dict)
    dataset = ArtSpeechDataset(
        datadir,
        database_name,
        sequences,
        vocabulary,
        articulators,
        clip_tails=True,
    )
    progress_bar = tqdm(dataset, desc="Synthesizing vocal tract")

    for (
        sentence_name,
        sentence_numerized,
        _,
        sentence_tokens,
        reference_arrays,
        _,
        _,
        _
    ) in progress_bar:
        subject_sequence, _ = sentence_name.split("-")
        subject, _ = subject_sequence.split("_")

        save_sentence_dir = os.path.join(save_to, subject, sentence_name)
        if not os.path.exists(save_sentence_dir):
            os.makedirs(save_sentence_dir)
        save_contours_dir = os.path.join(save_sentence_dir, "inference_contours")
        if not os.path.exists(save_contours_dir):
            os.makedirs(save_contours_dir)
        save_air_column_dir = os.path.join(save_sentence_dir, "air_column")
        if not os.path.exists(save_air_column_dir):
            os.makedirs(save_air_column_dir)
        save_plots_dir = os.path.join(save_sentence_dir, "vocal_tract_shapes")
        if not os.path.exists(save_plots_dir):
            os.makedirs(save_plots_dir)
        xarticul_dir = os.path.join(save_sentence_dir, "xarticul")
        if not os.path.exists(xarticul_dir):
            os.makedirs(xarticul_dir)

        with open(os.path.join(save_sentence_dir, "target_sequence.txt"), "w") as f:
            f.write(" ".join(sentence_tokens))

        sentence_numerized = sentence_numerized.to(device)
        seq_len = len(sentence_tokens)
        lengths = torch.tensor([seq_len], dtype=torch.long).cpu()

        if method == "neural-network":
            sentence_numerized = sentence_numerized.unsqueeze(dim=0)
            outputs = model(sentence_numerized, lengths)
        elif method == "mean-contour":
            outputs = model(sentence_tokens)

        video_filepath = os.path.join(save_sentence_dir, f"{sentence_name}.avi")
        save_vocal_tract_shape(outputs.squeeze(dim=0), sentence_tokens, save_plots_dir)
        make_vocal_tract_shape_video(outputs.squeeze(dim=0), sentence_tokens, video_filepath)

        articulators_dicts = save_contours(
            outputs.squeeze(dim=0),
            reference_arrays,
            save_contours_dir,
            articulators
        )
        for i_frame, articuls_dict in enumerate(articulators_dicts, start=1):
            internal_wall, external_wall = generate_vocal_tract_tube(articuls_dict)
            air_column = np.array([internal_wall, external_wall])

            frame_id = "%04d" % i_frame
            air_column_filepath = os.path.join(save_air_column_dir, f"{frame_id}.npy")
            np.save(air_column_filepath, air_column)

            xarticul_int = npy_to_xarticul(internal_wall * dataset_config.RES)
            xarticul_ext = npy_to_xarticul(external_wall * dataset_config.RES)

            xarticul_array = xarticul_int + xarticul_ext
            xarticul_filepath = os.path.join(xarticul_dir, f"{'%04d' % i_frame}.txt")
            with open(xarticul_filepath, "w") as f:
                f.write("\n".join(xarticul_array))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_filepath", required=True)
    args = parser.parse_args()

    with open(args.config_filepath) as f:
        cfg = yaml.safe_load(f)

    main(**cfg)
