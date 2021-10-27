import json
import matplotlib.pyplot as plt
import numpy as np
import os

from tqdm import tqdm

COLORS = {
    "arytenoid-muscle": "blueviolet",
    "epiglottis": "turquoise",
    # "hyoid-bone": "slategray",
    "lower-incisor": "cyan",
    "lower-lip": "lime",
    "pharynx": "goldenrod",
    "soft-palate": "dodgerblue",
    "thyroid-cartilage": "saddlebrown",
    "tongue": "darkorange",
    "upper-incisor": "yellow",
    "upper-lip": "magenta",
    "vocal-folds": "hotpink"
}

CLOSED = [
    "hyoid-bone",
    "thyroid-cartilage"
]


def plot_complete_vocal_tract(pred_curves, true_curves, save_dir, frame, phoneme):
    plt.figure(figsize=(10, 10))

    for articulator, pred_curve in pred_curves.items():
        plt.plot(*zip(*pred_curve), c=COLORS[articulator], linewidth=lw)

    for articulator, true_curve in true_curves.items():
        plt.plot(*zip(*true_curve), "r--", linewidth=lw, alpha=alpha)

    plt.ylim([1., 0.])
    plt.xlim([0., 1.])
    plt.axis("off")
    plt.tight_layout()

    plt.text(0.5, 0.15, f"/{phoneme}/", fontsize=56, color="blue")

    jpg_filepath = os.path.join(save_dir, f"{frame}.jpg")
    plt.savefig(jpg_filepath, format="jpg")

    pdf_filepath = os.path.join(save_dir, f"{frame}.pdf")
    plt.savefig(pdf_filepath, format="pdf")

    plt.close()


def plot_sentence(outputs_dir, save_to):
    sentences = sorted(map(int, os.listdir(outputs_dir)))

    for sentence_number in tqdm(sentences):
        json_filepath = os.path.join(outputs_dir, str(sentence_number), "tract_variables_data.json")
        with open(json_filepath) as f:
            phonemes = {d["frame"]: d["phoneme"] for d in json.load(f)}

        sentence_contours_dir = os.path.join(outputs_dir, str(sentence_number), "contours")
        sentence_frames = set(map(
            int,
            [
                fname.split(".")[0].split("_")[0]
                for fname in os.listdir(sentence_contours_dir)
            ]
        ))

        for frame in sentence_frames:
            pred_curves = {}
            true_curves = {}
            for articulator, _ in COLORS.items():
                if articulator == "hyoid-bone":
                    continue

                pred_filepath = os.path.join(sentence_contours_dir, f"{frame}_{articulator}.npy")
                pred_curve = np.load(pred_filepath).T
                pred_curves[articulator] = pred_curve

                true_filepath = os.path.join(sentence_contours_dir, f"{frame}_{articulator}_true.npy")
                true_curve = np.load(true_filepath).T
                true_curves[articulator] = true_curve

            save_dir = os.path.join(save_to, str(sentence_number))
            plot_complete_vocal_tract(
                pred_curves,
                true_curves,
                save_dir,
                frame,
                phonemes[frame]
            )
