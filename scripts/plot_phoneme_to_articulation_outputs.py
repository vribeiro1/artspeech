import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from tqdm import tqdm
from vt_tools import (
    COLORS,
    ARYTENOID_CARTILAGE,
    EPIGLOTTIS,
    LOWER_INCISOR,
    LOWER_LIP,
    PHARYNX,
    SOFT_PALATE_MIDLINE,
    THYROID_CARTILAGE,
    TONGUE,
    UPPER_INCISOR,
    UPPER_LIP,
    VOCAL_FOLDS
)

ARTICULATORS = [
    ARYTENOID_CARTILAGE,
    EPIGLOTTIS,
    LOWER_INCISOR,
    LOWER_LIP,
    PHARYNX,
    SOFT_PALATE_MIDLINE,
    THYROID_CARTILAGE,
    TONGUE,
    UPPER_INCISOR,
    UPPER_LIP,
    VOCAL_FOLDS
]


def plot_sentence(df_phonemes, contours_dir, save_dir):
    lw = 4

    for i, row in df_phonemes.iterrows():
        frame = "%04d" % row["frame"]
        phoneme = row["phoneme"]

        plt.figure(figsize=(10, 10))
        for articulator in ARTICULATORS:
            pred_filepath = os.path.join(contours_dir, f"{frame}_{articulator}.npy")
            true_filepath = os.path.join(contours_dir, f"{frame}_{articulator}_true.npy")

            pred_array = np.load(pred_filepath)
            true_array = np.load(true_filepath)

            plt.plot(*pred_array, color=COLORS[articulator], lw=lw)
            plt.plot(*true_array, color="red", linestyle="--", lw=lw)

        plt.ylim([1, 0])
        plt.xlim([0, 1])
        plt.text(0.5, 0.1, f"/{phoneme}/", color="blue", fontsize=22)

        plt.axis("off")
        plt.tight_layout()
        save_filepath = os.path.join(save_dir, f"{frame}.jpg")
        plt.savefig(save_filepath)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", dest="dirname")
    args = parser.parse_args()

    sentences = sorted(os.listdir(args.dirname))
    for sentence in tqdm(sentences, desc="Plotting sentences"):
        csv_filepath = os.path.join(args.dirname, sentence, "phonemes.csv")
        df_phonemes = pd.read_csv(csv_filepath)
        contours_dir = os.path.join(args.dirname, sentence, "contours")
        save_dir = os.path.join(args.dirname, sentence, "plots")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plot_sentence(df_phonemes, contours_dir, save_dir)

