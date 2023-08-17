"""
Create videos from the datasets, plotting together the MR images and the articulator contours,
together with phonemes, subject id, sequence id and frame number.
"""
import argparse
import cv2
import os
import numpy as np
import shutil
import tempfile
import yaml

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from tqdm import tqdm
from vt_tools import COLORS
from vt_tracker.visualization import uint16_to_uint8

from database_collector import DATABASE_COLLECTORS
from helpers import sequences_from_dict
from vocal_tract_loader import VocalTractShapeLoader
from settings import DATASET_CONFIG, BASE_DIR

TMPFILES = os.path.join(BASE_DIR, "tmp")


def make_frame(
    subject,
    sequence,
    frame_id,
    phoneme,
    datadir,
    dataset_config,
    articulators,
):
    image_filepath = os.path.join(
        datadir,
        subject,
        sequence,
        "NPY_MR",
        f"{frame_id}.npy"
    )
    image = np.load(image_filepath)
    image = uint16_to_uint8(image, norm_hist=False)

    contours = {}
    for articulator in articulators:
        contour = VocalTractShapeLoader.prepare_articulator_array(
            datadir,
            subject,
            sequence,
            frame_id,
            articulator,
            dataset_config,
        )
        contours[articulator] = contour * dataset_config.RES

    fig = Figure(figsize=(10, 10))
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    ax.imshow(image, cmap="gray")
    for articulator in articulators:
        ax.plot(*contours[articulator], lw=3, color=COLORS[articulator])
    ax.text(65, 20, f"/{phoneme}/", color="yellow", fontsize=24)
    ax.text(10, 130, f"{subject} {sequence} {frame_id}", color="yellow", fontsize=18)

    ax.axis("off")
    ax.set_xlim([0, 136])
    ax.set_ylim([136, 0])
    canvas.draw()
    frame = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    frame = np.reshape(frame, (1000, 1000, 3))

    return frame


def main(
    database_name,
    datadir,
    seq_dict,
    articulators,
    save_to,
    tmp_dir,
):
    save_audio_dir = os.path.join(tmp_dir, "audios")
    if not os.path.exists(save_audio_dir):
        os.makedirs(save_audio_dir)

    dataset_config = DATASET_CONFIG[database_name]
    sequences = sequences_from_dict(datadir, seq_dict)
    collector = DATABASE_COLLECTORS[database_name](datadir, save_audio_dir)
    data = collector.collect_data(sequences)

    for sentence_item in tqdm(data, desc="Making videos"):
        subject = sentence_item["subject"]
        sequence = sentence_item["sequence"]
        audio_filepath = sentence_item["wav_filepath"]
        sentence_name = sentence_item["sentence_name"]
        frame_ids = sentence_item["frame_ids"]
        phonemes = sentence_item["phonemes"]

        video_filepath = os.path.join(save_to, f"{sentence_name}.avi")
        video_writer = cv2.VideoWriter(
            video_filepath,
            cv2.VideoWriter_fourcc(*"DIVX"),
            dataset_config.FRAMERATE,
            (1000, 1000),
        )

        for frame_id, phoneme in zip(frame_ids, phonemes):
            frame = make_frame(
                subject,
                sequence,
                frame_id,
                phoneme,
                datadir,
                dataset_config,
                articulators,
            )
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        video_writer.release()
        video_audio_filepath = os.path.join(save_to, f"{sentence_name}_with_audio.avi")
        join_video_audio_cmd = f"ffmpeg -i {video_filepath} -i {audio_filepath} -c:v copy -c:a aac {video_audio_filepath} -y"
        os.system(join_video_audio_cmd)
        os.remove(video_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_filepath")
    args = parser.parse_args()

    with open(args.config_filepath) as f:
        cfg = yaml.safe_load(f)

    tmp_dir = tempfile.mkdtemp(dir=TMPFILES)
    try:
        main(**cfg, tmp_dir=tmp_dir)
    finally:
        shutil.rmtree(tmp_dir)
