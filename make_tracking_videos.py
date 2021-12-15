import pdb

import argparse
import cv2
import funcy
import numpy as np
import os
import pydicom

from glob import glob
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.io import wavfile
from tgt import read_textgrid
from vt_tracker import (
    ARYTENOID_MUSCLE,
    EPIGLOTTIS,
    LOWER_INCISOR,
    LOWER_LIP,
    PHARYNX,
    SOFT_PALATE,
    THYROID_CARTILAGE,
    TONGUE,
    UPPER_INCISOR,
    UPPER_LIP,
    VOCAL_FOLDS
)
from vt_tracker.visualization import COLORS, uint16_to_uint8

from phoneme_to_articulation.dataset import ArtSpeechDataset, TailClipper
from video import Video

ARTICULATORS = sorted([
    ARYTENOID_MUSCLE,
    EPIGLOTTIS,
    LOWER_INCISOR,
    LOWER_LIP,
    PHARYNX,
    SOFT_PALATE,
    THYROID_CARTILAGE,
    TONGUE,
    UPPER_INCISOR,
    UPPER_LIP,
    VOCAL_FOLDS
])


def load_input_image(filepath):
    pixel_arr = pydicom.dcmread(filepath).pixel_array
    uint8_pixel_arr = uint16_to_uint8(pixel_arr)
    return uint8_pixel_arr


def exists_or_none(filepath):
    return filepath if os.path.isfile(filepath) else None


def get_instance_number_from_filepath(filepath):
    basename = os.path.basename(filepath)
    inumber_str, _ = basename.split(".")
    inumber = int(inumber_str)

    return inumber


def get_contours_for_frames(sequence_dirname, instance_numbers):
    contours = {}
    for inumber in instance_numbers:
        articulator_filepath = lambda art: os.path.join(
            sequence_dirname,
            "inference_contours",
            f"{inumber}_{art}.npy"
        )

        contours[inumber] = {
            art: exists_or_none(articulator_filepath(art))
            for art in ARTICULATORS
        }

    return contours


def make_frame(frame_filepath, frame_contours_filepaths, phoneme):
    sequence_dirname = os.path.dirname(os.path.dirname(frame_filepath))
    subject = os.path.basename(os.path.dirname(sequence_dirname))
    sequence = os.path.basename(sequence_dirname)
    filename, _ = os.path.basename(frame_filepath).split(".")

    fig = Figure(figsize=(10, 10))
    canvas = FigureCanvas(fig)

    ax = fig.gca()

    frame = load_input_image(frame_filepath)
    ax.imshow(frame, cmap="gray")

    # References for tail clipping
    lower_incisor_fp = frame_contours_filepaths["lower-incisor"]
    lower_incisor = ArtSpeechDataset.load_target_array(lower_incisor_fp)

    upper_incisor_fp = frame_contours_filepaths["upper-incisor"]
    upper_incisor = ArtSpeechDataset.load_target_array(upper_incisor_fp)

    epiglottis_fp = frame_contours_filepaths["epiglottis"]
    epiglottis = ArtSpeechDataset.load_target_array(epiglottis_fp)

    for articulator, articulator_filepath in frame_contours_filepaths.items():
        articulator_arr = ArtSpeechDataset.load_target_array(articulator_filepath)

        tail_clip_method = getattr(
            TailClipper, f"clip_{articulator.replace('-', '_')}_tails", None
        )

        if tail_clip_method:
            articulator_arr = tail_clip_method(
                articulator_arr,
                lower_incisor=lower_incisor,
                upper_incisor=upper_incisor,
                epiglottis=epiglottis
            )

        color = COLORS[articulator]
        ax.plot(*zip(*articulator_arr), lw=5, c=color)

    ax.text(65, 20, f"/{phoneme}/", fontsize=24, color="yellow")
    ax.text(10, 130, f"{subject} - {sequence} - {filename}", fontsize=18, color="yellow")

    ax.axis("off")

    canvas.draw()
    frame = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    frame = np.reshape(frame, (1000, 1000, 3))

    return frame


def make_video(sequence_dirname, save_to):
    subject = os.path.basename(os.path.dirname(sequence_dirname))
    sequence = os.path.basename(sequence_dirname)

    if not os.path.exists(save_to):
        os.makedirs(save_to)

    frames_filepaths = sorted(glob(os.path.join(sequence_dirname, "dicoms", "*.dcm")))
    audio_filepath = os.path.join(sequence_dirname, f"vol_{subject}_{sequence}.wav")
    textgrid_filepath = os.path.join(sequence_dirname, f"vol_{subject}_{sequence}.textgrid")

    video = Video(frames_filepaths, audio_filepath, framerate=55)

    textgrid = read_textgrid(textgrid_filepath)
    phone_tier = textgrid.get_tier_by_name("PhonTier")
    word_tier = textgrid.get_tier_by_name("WordTier")
    sentence_tier = textgrid.get_tier_by_name("SentenceTier")

    for i, sentence_interval in enumerate(sentence_tier):
        sentence_start = sentence_interval.start_time
        sentence_end = sentence_interval.end_time

        in_interval = lambda interval: (interval.start_time >= sentence_start
                                        and interval.end_time <= sentence_end)

        sentence_phonemes = [{
            "text": phone.text,
            "start_time": phone.start_time,
            "end_time": phone.end_time,
            "n_frames": len(video.get_frames_interval(phone.start_time, phone.end_time, load_frames=False))
        } for phone in filter(in_interval, phone_tier.intervals)]
        phonetic_sequence = funcy.lflatten(map(lambda item: [item["text"]] * item["n_frames"], sentence_phonemes))

        sentence_frames = video.get_frames_interval(sentence_start, sentence_end)
        instance_numbers = funcy.lmap(lambda filepath: os.path.basename(filepath).split(".")[0], sentence_frames)
        contours = get_contours_for_frames(sequence_dirname, instance_numbers)

        audio_interval = video.get_audio_interval(sentence_start, sentence_end)
        audio_filepath = os.path.join(save_to, f"{i}.wav")
        wavfile.write(audio_filepath, video.sample_rate, audio_interval.numpy())

        video_filepath = os.path.join(save_to, f"{i}.avi")
        video_writer = cv2.VideoWriter(
            video_filepath,
            cv2.VideoWriter_fourcc(*"DIVX"),
            video.framerate,
            (1000, 1000)
        )

        for phoneme, frame_fpath, (iframe, frame_contours_fpaths) in zip(phonetic_sequence, sentence_frames, contours.items()):
            frame = make_frame(frame_fpath, frame_contours_fpaths, phoneme)
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        video_writer.release()

        video_audio_filepath = os.path.join(save_to, f"{i}_with_audio.avi")
        join_video_audio_cmd = f"ffmpeg -i {video_filepath} -i {audio_filepath} -c:v copy -c:a aac {video_audio_filepath} -y"
        os.system(join_video_audio_cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", dest="datadir")
    parser.add_argument("--subject", dest="subject", type=str)
    parser.add_argument("--sequences", dest="sequences", type=str, nargs="+", default=[])
    parser.add_argument("--save-to", dest="save_to")
    args = parser.parse_args()

    if len(args.sequences) == 0:
        sequences = sorted(os.listdir(os.path.join(args.datadir, args.subject)))
    else:
        sequences = sorted(args.sequences)

    for sequence in sequences:
        sequence_dirname = os.path.join(args.datadir, args.subject, sequence)
        make_video(sequence_dirname, os.path.join(args.save_to, args.subject, sequence))
