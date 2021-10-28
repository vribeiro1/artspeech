import pdb

import argparse
import json
import numpy as np
import os
import pandas as pd
import torch

from glob import glob
from time import time
from tqdm import tqdm
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

from loss import EuclideanDistanceLoss
from metrics import p2cp_distance

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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


def load_art_tensors(zipped):
    preds = torch.zeros(size=(0, 2, 50), dtype=torch.float32)
    targets = torch.zeros(size=(0, 2, 50), dtype=torch.float32)
    for fp_pred, fp_target in zipped:
        pred = torch.tensor(np.load(fp_pred), dtype=torch.float32)
        target = torch.tensor(np.load(fp_target), dtype=torch.float32)

        preds = torch.cat([preds, pred.unsqueeze(dim=0)])
        targets = torch.cat([targets, target.unsqueeze(dim=0)])

    preds = preds.unsqueeze(1)
    targets = targets.unsqueeze(1)

    return preds, targets


def calculate_distances(zips, distance_fn):
    preds = torch.zeros(size=(len(zips[0]), 0, 2, 50))
    targets = torch.zeros(size=(len(zips[0]), 0, 2, 50))
    for zipped in zips:
        art_pred, art_target = load_art_tensors(zipped)

        preds = torch.cat([preds, art_pred], dim=1)
        targets = torch.cat([targets, art_target], dim=1)

    preds = preds.unsqueeze(dim=0)
    targets = targets.unsqueeze(dim=0)

    distances = distance_fn(preds, targets)
    distances = distances.mean(dim=1)

    return distances


def distances_per_articulator(inferences_dir, distance_fn):
    sentences = os.listdir(inferences_dir)

    distances = torch.zeros(size=(0, len(ARTICULATORS)))
    for sentence in tqdm(sorted(sentences)):
        saved_outputs_dir = os.path.join(inferences_dir, sentence, "contours")

        zipped = []
        for articulator in ARTICULATORS:
            filepaths_preds = sorted(glob(os.path.join(saved_outputs_dir, f"*_{articulator}.npy")))
            filepaths_targets = sorted(glob(os.path.join(saved_outputs_dir, f"*_{articulator}_true.npy")))

            zip_list = list(zip(filepaths_preds, filepaths_targets))
            zipped.append(zip_list)

        if len(zipped[0]) == 0:
            continue

        sentences_dists = calculate_distances(zipped, distance_fn)
        distances = torch.cat([distances, sentences_dists], dim=0)

    dist_means = distances.mean(dim=0)
    dist_stds = distances.std(dim=0)

    return {
        articulator: {
            "mean": dist_means[i].item(),
            "std": dist_stds[i].item()
        } for i, articulator in enumerate(ARTICULATORS)
    }


def euclidean_distance(outputs, targets):
    # outputs: torch.Size([bs, seq_len, N_art, 2, N_samples])
    # targets: torch.Size([bs, seq_len, N_art, 2, N_samples])
    euclidean_distance_fn = EuclideanDistanceLoss(reduction="none")
    return euclidean_distance_fn( outputs, targets).mean(dim=-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", dest="basedir")
    parser.add_argument("--exps", dest="exps", type=int, nargs="+")
    parser.add_argument("--save-to", dest="save_to")
    args = parser.parse_args()

    results = []
    for exp in args.exps:
        start_time = time()
        print(f"Starting to process experiment #{exp}")
        exp_dir = os.path.join(args.basedir, str(exp))
        with open(os.path.join(exp_dir, "test_results.json")) as f:
            test_results = json.load(f)

        inferences_dir = os.path.join(exp_dir, "test_outputs", "0")

        p2cp_per_art = distances_per_articulator(inferences_dir, p2cp_distance)
        dist_per_art = distances_per_articulator(inferences_dir, euclidean_distance)

        for articulator in ARTICULATORS:
            test_results[articulator]["med"] = dist_per_art[articulator]["mean"]
            test_results[articulator]["p2cp"] = p2cp_per_art[articulator]["mean"]

        with open(os.path.join(exp_dir, "test_results_update.json"), "w") as f:
            json.dump(test_results, f)

        results_item = {
            "exp": exp,
            "Loss": test_results["loss"]
        }
        for articulator in ARTICULATORS:
            results_item[f"P2CP_{articulator}"] = test_results[articulator]["p2cp"]
            results_item[f"MED_{articulator}"] = test_results[articulator]["med"]
            results_item[f"X_corr_{articulator}"] = test_results[articulator]["x_corr"]
            results_item[f"Y_corr_{articulator}"] = test_results[articulator]["y_corr"]

        results.append(results_item)
        end_time = time()
        enlapsed = end_time - start_time
        print(f"Finished processing experiment #{exp}. Enlapsed time was {enlapsed} seconds.")

    df = pd.DataFrame(results)
    df_filepath = os.path.join(args.save_to, "results.csv")
    df.to_csv(df_filepath, index=False)
