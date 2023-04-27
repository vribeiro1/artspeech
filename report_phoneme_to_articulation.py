import argparse
import funcy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import torch
import yaml

from glob import glob
from tqdm import tqdm

from metrics import p2cp_distance, euclidean_distance
from settings import DATASET_CONFIG

color_palette = sns.color_palette("colorblind", 6)
TV_COLORS = {
    "LA": color_palette[0],
    "TTCD": color_palette[1],
    "TBCD": color_palette[2],
    "VEL": color_palette[3],
}
c_even = color_palette[4]
c_odd = color_palette[5]


def plot_tract_variables_for_sentence(
    df,
    sentence_name,
    plots_dir,
    which="both"
):
    plot_pred = which in ["pred", "both"]
    plot_target = which in ["target", "both"]

    intervals = []
    start_frame = end_frame = None
    current_phoneme = None
    phonemes = list(df.phoneme) + [None]
    frames = list(df.frame)
    frames += [frames[-1] + 1]
    for phoneme, frame in zip(phonemes, frames):
        if phoneme == current_phoneme:
            continue

        end_frame = int(frame) - 1
        interval = (current_phoneme, start_frame, end_frame)
        if current_phoneme is not None:
            intervals.append(interval)

        start_frame = int(frame)
        current_phoneme = phoneme

    plt.figure(figsize=(20, 5))

    y_margin = 9
    y_rule_margin = 4
    y_min = -2
    y_max = max([
        df[f"{TV}_pred"].max() for TV in TV_COLORS
    ] + [
        df[f"{TV}_target"].max() for TV in TV_COLORS
    ])

    for TV in TV_COLORS:
        if plot_pred:
            plt.plot(
                df.frame,
                df[f"{TV}_pred"],
                color=TV_COLORS[TV],
            )
        if plot_target:
            linestyle = "--" if plot_pred else "-"
            plt.plot(
                df.frame,
                df[f"{TV}_target"],
                linestyle=linestyle,
                color=TV_COLORS[TV],
            )

    for i, (phoneme, start_frame, end_frame) in enumerate(intervals):
        even = i % 2 == 0

        x_fill_between = np.arange(start_frame, end_frame + 1)
        y1_fill_between = [y_min for _ in x_fill_between]
        y2_fill_between = [y_max + y_margin for _ in x_fill_between]
        color = c_even if even else c_odd
        plt.fill_between(
            x_fill_between,
            y1_fill_between,
            y2_fill_between,
            alpha=0.3,
            color=color
        )

        # Alternate phonetic labels below and above the rule
        y_rule = y_max + y_rule_margin
        p = 1 if even else -1
        plt.text(
            start_frame,
            y_rule + p,
            phoneme
        )

    plt.xlabel("Frame Number")
    plt.ylabel("TV value (mm)")

    plt.grid(True, "major")
    plt.ylim(y_min, y_max + y_margin)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"TVs_{sentence_name}.pdf"), format="pdf")
    plt.savefig(os.path.join(plots_dir, f"TVs_{sentence_name}.jpg"), format="jpg")
    plt.close()


def main(
    database_name,
    results_dir,
    articulators,
):
    dataset_config = DATASET_CONFIG[database_name]

    sentences_basedir = os.path.join(results_dir, "test_outputs", "0")
    sentences_names = sorted(os.listdir(sentences_basedir))
    sentences_dirs = funcy.lfilter(
        os.path.isdir,
        map(lambda fp: os.path.join(sentences_basedir, fp), sentences_names)
    )

    # If it wasn't done yet, merge all tract_variables.csv files together
    df_TVs_filepath = os.path.join(results_dir, "tract_variables.csv")
    if not os.path.exists(df_TVs_filepath):
        df_TVs = pd.concat([
            pd.read_csv(os.path.join(sentence_dir, "tract_variables.csv"))
            for sentence_dir in sentences_dirs
        ])
        df_TVs.to_csv(df_TVs_filepath, index=False)
    else:
        df_TVs = pd.read_csv(df_TVs_filepath)
    df_TVs = df_TVs.sort_values(["sentence", "frame"])

    to_mm = dataset_config.RES * dataset_config.PIXEL_SPACING
    for TV in TV_COLORS:
        df_TVs[f"{TV}_pred"] = df_TVs[f"{TV}_pred"] * to_mm
        df_TVs[f"{TV}_target"] = df_TVs[f"{TV}_target"] * to_mm
        df_TVs[f"{TV}_abs_error"] = (df_TVs[f"{TV}_target"] - df_TVs[f"{TV}_pred"]).apply(np.abs)

    # Plot the tract variables for each sentence and calculate the errors based on the data stored
    # in the contours folder
    metrics_data = []
    for sentence_dir in tqdm(sentences_dirs, desc="Processing sentences"):
        sentence_name = os.path.basename(sentence_dir)
        df_sentence = df_TVs[df_TVs.sentence == sentence_name]
        plots_dir = os.path.join(sentence_dir, "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        plot_tract_variables_for_sentence(
            df_sentence,
            sentence_name,
            plots_dir
        )

        contours_dir = os.path.join(sentence_dir, "contours")
        frames = list(df_sentence.frame)
        phonemes = list(df_sentence.phoneme)
        for frame, phoneme in zip(frames, phonemes):
            frame_str = "%04d" % int(frame)

            frame_pred_contours = torch.zeros([0, 2, 50])
            frame_true_contours = torch.zeros([0, 2, 50])
            for articulator in articulators:
                pred_filepath = os.path.join(contours_dir, f"{frame_str}_{articulator}.npy")
                true_filepath = os.path.join(contours_dir, f"{frame_str}_{articulator}_true.npy")

                pred_contour = torch.from_numpy(np.load(pred_filepath)).unsqueeze(dim=0)
                true_contour = torch.from_numpy(np.load(true_filepath)).unsqueeze(dim=0)

                frame_pred_contours = torch.cat([frame_pred_contours, pred_contour], dim=0)
                frame_true_contours = torch.cat([frame_true_contours, true_contour], dim=0)

            frame_pred_contours = frame_pred_contours.unsqueeze(dim=0).unsqueeze(dim=0)
            frame_true_contours = frame_true_contours.unsqueeze(dim=0).unsqueeze(dim=0)

            frame_pred_contours = frame_pred_contours.type(torch.float)
            frame_true_contours = frame_true_contours.type(torch.float)

            p2cp = p2cp_distance(frame_pred_contours, frame_true_contours)
            euclidean = euclidean_distance(frame_pred_contours, frame_true_contours)

            p2cp = p2cp.squeeze(dim=0).squeeze(dim=0)
            euclidean = euclidean.squeeze(dim=0).squeeze(dim=0)

            for i, articulator in enumerate(articulators):
                item = {
                    "sentence_name": sentence_name,
                    "frame": frame,
                    "phoneme": phoneme,
                    "articulator": articulator,
                    "p2cp": p2cp[i].item(),
                    "p2cp_mm": p2cp[i].item() * to_mm,
                    "euclidean": euclidean[i].item(),
                    "euclidean_mm": euclidean[i].item() * to_mm
                }
                metrics_data.append(item)

    df_error_report = pd.DataFrame(metrics_data)
    error_report_filepath = os.path.join(results_dir, "error_report_full.csv")
    df_error_report.to_csv(error_report_filepath, index=False)

    # Aggregate the metrics calculating mean, std, median, min, max
    df_error_report_agg = df_error_report.groupby("articulator").agg({
        "p2cp": ["mean", "std", "min", "max"],
        "p2cp_mm": ["mean", "std", "min", "max"],
        "euclidean": ["mean", "std", "min", "max"],
        "euclidean_mm": ["mean", "std", "min", "max"],
    }).reset_index()
    error_report_agg_filepath = os.path.join(results_dir, "error_report_agg.csv")
    df_error_report_agg.to_csv(error_report_agg_filepath, index=False)
    print(df_error_report_agg)

    # Compute the TV correlations for each tract variable
    data = []
    for TV in TV_COLORS:
        grouped = df_TVs.groupby("sentence")
        TV_corr = grouped[[
            f"{TV}_target",
            f"{TV}_pred"
        ]].corr().reset_index()
        TV_corr = TV_corr[TV_corr.level_1 == f"{TV}_target"][[
            "sentence",
            f"{TV}_pred"
        ]]

        TV_mean_corr = TV_corr[f"{TV}_pred"].mean()
        TV_std_corr = TV_corr[f"{TV}_pred"].std()
        TV_min_corr = TV_corr[f"{TV}_pred"].min()
        TV_max_corr = TV_corr[f"{TV}_pred"].max()

        data.append({
            "TV": TV,
            "mean": TV_mean_corr,
            "std": TV_std_corr,
            "min": TV_min_corr,
            "max": TV_max_corr,
        })

    df_TV_corr_report = pd.DataFrame(data)
    TV_corr_report_filepath = os.path.join(results_dir, "TV_corr_report.csv")
    df_TV_corr_report.to_csv(TV_corr_report_filepath)
    print(df_TV_corr_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_filepath")
    args = parser.parse_args()

    with open(args.config_filepath) as f:
        cfg = yaml.safe_load(f)

    main(**cfg)
