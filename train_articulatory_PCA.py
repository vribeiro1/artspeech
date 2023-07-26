import argparse
import logging
import mlflow
import numpy as np
import os
import pandas as pd
import pickle as pkl
import random
import shutil
import tempfile
import torch
import yaml

from collections import OrderedDict
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA

from helpers import set_seeds, sequences_from_dict
from phoneme_to_articulation.principal_components.metrics import MeanP2CPDistance
from phoneme_to_articulation.principal_components.dataset import PrincipalComponentsAutoencoderDataset2
from settings import BASE_DIR, DATASET_CONFIG

TMPFILES = os.path.join(BASE_DIR, "tmp")
TMP_DIR = tempfile.mkdtemp(dir=TMPFILES)
RESULTS_DIR = os.path.join(TMP_DIR, "results")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


def make_multiarticulator_autoencoder(transformers):
    encoder_dict = OrderedDict()
    decoder_dict = OrderedDict()
    for articulator, transformer in transformers.items():
        eigenvalues = transformer.explained_variance_
        eigenvectors = transformer.components_

        encoder_dict[f"encoders.{articulator}.eigenvalues"] = torch.from_numpy(eigenvalues).type(torch.float)
        encoder_dict[f"encoders.{articulator}.eigenvectors"] = torch.from_numpy(eigenvectors).type(torch.float)

        decoder_dict[f"decoders.{articulator}.eigenvalues"] = torch.from_numpy(eigenvalues).unsqueeze(dim=-1).type(torch.float)
        decoder_dict[f"decoders.{articulator}.eigenvectors"] = torch.from_numpy(eigenvectors).type(torch.float)

    return encoder_dict, decoder_dict


def main(
    database_name,
    datadir,
    batch_size,
    train_seq_dict,
    test_seq_dict,
    model_params,
    num_workers=0,
    clip_tails=True,
    seed=0,
    **kwargs,
):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    articulators_indices_dict = model_params["indices_dict"]
    articulators = sorted(articulators_indices_dict.keys())
    n_articulators = len(articulators)

    dataset_config = DATASET_CONFIG[database_name]
    train_sequences = sequences_from_dict(datadir, train_seq_dict)
    train_dataset = PrincipalComponentsAutoencoderDataset2(
        database_name=database_name,
        datadir=datadir,
        sequences=train_sequences,
        articulators=articulators,
        clip_tails=clip_tails,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=set_seeds,
        generator=gen,
    )

    transformers = {}
    for articulator, indices in articulators_indices_dict.items():
        if isinstance(indices, list):
            num_indices = len(indices)
        else:
            num_indices = indices

        transformers[articulator] = IncrementalPCA(
            n_components=num_indices,
            batch_size=batch_size
        )

    progress_bar = tqdm(train_dataloader, desc=f"Training")
    for _, inputs, _, _ in progress_bar:
        inputs = inputs.numpy()

        for i, articulator in enumerate(articulators):
            transformers[articulator].partial_fit(inputs[:, i, :])

    test_sequences = sequences_from_dict(datadir, test_seq_dict)
    test_dataset = PrincipalComponentsAutoencoderDataset2(
        database_name=database_name,
        datadir=datadir,
        sequences=test_sequences,
        articulators=articulators,
        clip_tails=clip_tails,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=set_seeds,
        generator=gen,
    )

    p2cp_fn = MeanP2CPDistance(reduction="none")
    data_p2cp = torch.zeros(size=(0, n_articulators))
    data_frame_names = []

    progress_bar = tqdm(test_dataloader, desc=f"Testing")
    for frame_names, inputs, _, phonemes in progress_bar:
        inputs = inputs.numpy()
        bs, _, features = inputs.shape

        outputs = []
        for i, articulator in enumerate(articulators):
            latents = transformers[articulator].transform(inputs[:, i, :])
            articulator_outputs = transformers[articulator].inverse_transform(latents)
            outputs.append(articulator_outputs)
        outputs = np.stack(outputs, axis=1)

        targets = torch.from_numpy(inputs.reshape(
            bs,
            n_articulators,
            2, features
            // 2
        )).type(torch.float)
        outputs = torch.from_numpy(outputs.reshape(
            bs,
            n_articulators,
            2,
            features // 2
        )).type(torch.float)

        denorm_fns = {
            articulator: norm_fn.inverse
            for articulator, norm_fn
            in test_dataset.normalize.items()
        }

        for i, articulator in enumerate(articulators):
            outputs[:, i, :, :] = denorm_fns[articulator](outputs[:, i, :, :])
            targets[:, i, :, :] = denorm_fns[articulator](targets[:, i, :, :])

        p2cp = p2cp_fn(
            outputs.permute(0, 1, 3, 2),
            targets.permute(0, 1, 3, 2)
        ) * dataset_config.PIXEL_SPACING * dataset_config.RES
        data_p2cp = torch.cat([data_p2cp, p2cp])
        data_frame_names.extend([frame_name.split("_") for frame_name in frame_names])

    for articulator in articulators:
        articulator_transformer = transformers[articulator]

        pkl_filepath = os.path.join(RESULTS_DIR, f"{articulator}_pca.pkl")
        with open(pkl_filepath, "wb") as f:
            pkl.dump(articulator_transformer, f)
        mlflow.log_artifact(pkl_filepath)
    encoders_dict, decoders_dict = make_multiarticulator_autoencoder(transformers)

    best_encoders_path = os.path.join(RESULTS_DIR, "best_encoders.pt")
    torch.save(encoders_dict, best_encoders_path)
    mlflow.log_artifact(best_encoders_path)

    best_decoders_path = os.path.join(RESULTS_DIR, "best_decoders.pt")
    torch.save(decoders_dict, best_decoders_path)
    mlflow.log_artifact(best_decoders_path)

    df_errors_filepath = os.path.join(RESULTS_DIR, "reconstruction_errors.csv")
    df_errors = pd.DataFrame(data_frame_names, columns=["subject", "sequence", "frame"])
    df_errors[articulators] = data_p2cp
    df_errors.to_csv(df_errors_filepath, index=False)
    mlflow.log_artifact(df_errors_filepath)

    df_errors_agg_filepath = os.path.join(RESULTS_DIR, "reconstruction_errors_agg.csv")
    df_errors_agg = df_errors.agg({
        articulator: ["mean", "std", "median", "min", "max"]
        for articulator in articulators
    }).reset_index()
    df_errors_agg.to_csv(df_errors_agg_filepath, index=False)
    mlflow.log_artifact(df_errors_agg_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_filepath")
    parser.add_argument("--mlflow", dest="mlflow_tracking_uri", default=None)
    parser.add_argument("--experiment", dest="experiment_name", default="articulatory_pca")
    parser.add_argument("--run_id", dest="run_id", default=None)
    parser.add_argument("--run_name", dest="run_name", default=None)
    parser.add_argument("--checkpoint", dest="checkpoint_filepath", default=None)
    args = parser.parse_args()

    seed = 0
    rs = RandomState(MT19937(SeedSequence(seed)))
    random.seed(seed)
    torch.manual_seed(seed)

    if args.mlflow_tracking_uri is not None:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    with open(args.config_filepath) as f:
        cfg = yaml.safe_load(f)

    experiment = mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(
        run_id=args.run_id,
        experiment_id=experiment.experiment_id,
        run_name=args.run_name
    ) as run:
        print(f"Experiment ID: {experiment.experiment_id}\nRun ID: {run.info.run_id}")
        try:
            mlflow.log_artifact(args.config_filepath)
        except shutil.SameFileError:
            logging.info("Skipping logging config file since it already exists.")
        try:
            main(
                **cfg,
                checkpoint_filepath=args.checkpoint_filepath,
                seed=seed,
            )
        finally:
            shutil.rmtree(TMP_DIR)
