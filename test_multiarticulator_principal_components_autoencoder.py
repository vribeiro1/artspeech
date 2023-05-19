import pdb

import argparse
import funcy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import torch
import ujson
import yaml

from torch.utils.data import DataLoader
from tqdm import tqdm

from helpers import set_seeds, sequences_from_dict
from phoneme_to_articulation.principal_components.metrics import MeanP2CPDistance
from phoneme_to_articulation.principal_components.dataset import PrincipalComponentsAutoencoderDataset2
from phoneme_to_articulation.principal_components.evaluation import run_multiart_autoencoder_test
from phoneme_to_articulation.principal_components.losses import RegularizedLatentsMSELoss2
from phoneme_to_articulation.principal_components.models.autoencoder import MultiArticulatorAutoencoder
from settings import DATASET_CONFIG

PINK = np.array([255, 0, 85, 255]) / 255
BLUE = np.array([0, 139, 231, 255]) / 255


def plot_nomograms(
    orig_shape,
    orig_reconstruction,
    orig_latents,
    autoencoder,
    normalize,
    plots_dir,
    device=None,
):
    if device is None:
        device = orig_latents.device
    articulators = autoencoder.sorted_articulators
    num_articulators = len(articulators)
    latent_size = len(orig_latents)
    for i_PC in range(latent_size):
        plt.figure(figsize=(10, 10))

        PC_range = np.arange(-1, 1.01, 0.1)
        for v in PC_range:
            latents = orig_latents.clone()
            latents[i_PC] = v
            latents = latents.unsqueeze(dim=0).to(device)

            reconstruction = torch.concat([
                autoencoder.decoders.decoders[articulator](
                    latents[:, autoencoder.indices_dict[articulator]]
                ).unsqueeze(dim=1)
                for articulator in autoencoder.sorted_articulators
            ], dim=1)
            reconstruction = reconstruction.detach().cpu()
            reconstruction = reconstruction.reshape(1, num_articulators, 2, 50)

            for i, articulator in enumerate(articulators):
                denorm_fn = normalize[articulator].inverse
                reconstruction[:, i, :, :] = denorm_fn(reconstruction[:, i, :, :])
            reconstruction = reconstruction.squeeze(dim=0)

            color = PINK if v <= 0 else BLUE
            for rec in reconstruction:
                plt.plot(*rec, color=color, lw=3, alpha=0.2)

        for shape in orig_shape:
            plt.plot(*shape, "--", color="red", lw=5, alpha=0.5)
        for rec in orig_reconstruction:
            plt.plot(*rec, color="limegreen", lw=5, alpha=0.5)

        plt.xlim([0., 1.])
        plt.ylim([1., 0.])
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"C{i_PC + 1}.pdf"))
        plt.savefig(os.path.join(plots_dir, f"C{i_PC + 1}.png"))
        plt.close()


def plot_latent_space_distribution(df_latent, plots_dir):
    for col in df_latent.columns:
        plt.figure(figsize=(10, 10))
        sns.histplot(df_latent[col])
        plt.savefig(os.path.join(plots_dir, f"C{col}_distribution.pdf"))
        plt.savefig(os.path.join(plots_dir, f"C{col}_distribution.png"))
        plt.close()


def evaluate_autoencoder(
    database_name,
    datadir,
    dataset_config,
    batch_size,
    sequences_dict,
    model_params,
    encoders_filepath,
    decoders_filepath,
    save_to,
    num_workers=0,
):
    plots_dir = os.path.join(save_to, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    articulators_indices_dict = model_params["indices_dict"]
    articulators = sorted(articulators_indices_dict.keys())
    n_articulators = len(articulators)
    latent_size = max(funcy.flatten(articulators_indices_dict.values())) + 1

    sequences = sequences_from_dict(datadir, sequences_dict)
    dataset = PrincipalComponentsAutoencoderDataset2(
        database_name=database_name,
        datadir=datadir,
        sequences=sequences,
        articulators=articulators,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=set_seeds,
        num_workers=num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = MultiArticulatorAutoencoder(**model_params)
    encoders_state_dict = torch.load(encoders_filepath, map_location=device)
    autoencoder.encoders.load_state_dict(encoders_state_dict)
    decoders_state_dict = torch.load(decoders_filepath, map_location=device)
    autoencoder.decoders.load_state_dict(decoders_state_dict)

    autoencoder.to(device)
    autoencoder.eval()

    p2cp_fn = MeanP2CPDistance(reduction="none")
    data_frame_names = []
    data_p2cp = torch.zeros(size=(0, n_articulators))
    data_latents = torch.zeros(size=(0, latent_size))
    data_reconstructions = torch.zeros(size=(0, n_articulators, 2, 50))
    data_targets = torch.zeros(size=(0, n_articulators, 2, 50))
    for frame_names, inputs, _, _ in tqdm(dataloader):
        bs, _, _ = inputs.shape

        inputs = inputs.to(device)
        reconstructions, latents = autoencoder(inputs)
        reconstructions = reconstructions.reshape(bs, n_articulators, 2, 50)
        targets = inputs.reshape(bs, n_articulators, 2, 50)

        for i, articulator in enumerate(articulators):
            denorm_fn = dataset.normalize[articulator].inverse

            reconstructions[:, i, :, :] = denorm_fn(reconstructions[:, i, :, :])
            targets[:, i, :, :] = denorm_fn(targets[:, i, :, :])

        reconstructions = reconstructions.detach().cpu()
        latents = latents.detach().cpu()
        targets = targets.detach().cpu()

        p2cp = p2cp_fn(
            reconstructions.permute(0, 1, 3, 2),
            targets.permute(0, 1, 3, 2)
        ) * dataset_config.PIXEL_SPACING * dataset_config.RES

        data_reconstructions = torch.cat([data_reconstructions, reconstructions])
        data_latents = torch.cat([data_latents, latents])
        data_targets = torch.cat([data_targets, targets])
        data_p2cp = torch.cat([data_p2cp, p2cp])
        data_frame_names.extend([frame_name.split("_") for frame_name in frame_names])

    latent_space_filepath = os.path.join(save_to, "latent_space.csv")
    df_latent = pd.DataFrame(data_latents, columns=[str(i) for i in range(1, latent_size + 1)])
    df_latent.to_csv(latent_space_filepath, index=False)

    errors_filepath = os.path.join(save_to, "reconstruction_errors.npy")
    np.save(errors_filepath, data_p2cp.numpy())

    df_errors_filepath = os.path.join(save_to, "reconstruction_errors.csv")
    df_errors = pd.DataFrame(data_frame_names, columns=["subject", "sequence", "frame"])
    df_errors[articulators] = data_p2cp
    df_errors.to_csv(df_errors_filepath, index=False)

    df_errors_agg_filepath = os.path.join(save_to, "reconstruction_errors_agg.csv")
    df_errors_agg = df_errors.agg({
        articulator: ["mean", "std", "median", "min", "max"]
        for articulator in articulators
    }).reset_index()
    df_errors_agg.to_csv(df_errors_agg_filepath, index=False)

    # Nomogram plots
    idx = 100
    orig_shape = data_targets[idx]
    orig_reconstruction = data_reconstructions[idx]
    orig_latents = data_latents[idx]
    plot_nomograms(
        orig_shape=orig_shape,
        orig_reconstruction=orig_reconstruction,
        orig_latents=orig_latents,
        autoencoder=autoencoder,
        normalize=dataset.normalize,
        plots_dir=plots_dir,
        device=device,
    )

    # Latent space distribution plots
    plot_latent_space_distribution(
        df_latent=df_latent,
        plots_dir=plots_dir,
    )


def main(
    database_name,
    datadir,
    encoders_filepath,
    decoders_filepath,
    batch_size,
    model_params,
    seq_dict,
    save_to,
    num_workers=0,
    clip_tails=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_config = DATASET_CONFIG[database_name]
    articulators_indices_dict = model_params["indices_dict"]
    test_sequences = sequences_from_dict(datadir, seq_dict)
    articulators = sorted(articulators_indices_dict.keys())
    test_dataset = PrincipalComponentsAutoencoderDataset2(
        database_name=database_name,
        datadir=datadir,
        sequences=test_sequences,
        articulators=articulators,
        clip_tails=clip_tails
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=set_seeds,
        num_workers=num_workers,
    )

    best_autoencoder = MultiArticulatorAutoencoder(**model_params)
    best_encoders_state_dict = torch.load(encoders_filepath, map_location=device)
    best_autoencoder.encoders.load_state_dict(best_encoders_state_dict)

    best_decoders_state_dict = torch.load(decoders_filepath, map_location=device)
    best_autoencoder.decoders.load_state_dict(best_decoders_state_dict)
    best_autoencoder.to(device)

    test_outputs_dir = os.path.join(save_to, "test_outputs")
    if not os.path.exists(test_outputs_dir):
        os.makedirs(test_outputs_dir)
    plots_dir = os.path.join(save_to, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    loss_fn = RegularizedLatentsMSELoss2(
        indices_dict=articulators_indices_dict,
        alpha=0.1,
    )

    info_test = run_multiart_autoencoder_test(
        epoch=0,
        model=best_autoencoder,
        dataloader=test_dataloader,
        criterion=loss_fn,
        dataset_config=dataset_config,
        # outputs_dir=test_outputs_dir,
        plots_dir=plots_dir,
        indices_dict=articulators_indices_dict,
        device=device,
    )

    with open(os.path.join(save_to, "test_results.json"), "w") as f:
        ujson.dump(info_test, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="cfg_filepath")
    args = parser.parse_args()

    with open(args.cfg_filepath) as f:
        cfg = yaml.safe_load(f.read())
    main(**cfg)

    database_name = cfg["database_name"]
    dataset_config = DATASET_CONFIG[database_name]
    sequences_dict = cfg["seq_dict"]
    model_params = cfg["model_params"]
    batch_size = cfg["batch_size"]
    num_workers = cfg.get("num_workers", 0)
    datadir = cfg["datadir"]
    save_to = cfg["save_to"]
    encoders_filepath = cfg["encoders_filepath"]
    decoders_filepath = cfg["decoders_filepath"]
    evaluate_autoencoder(
        database_name,
        datadir,
        dataset_config,
        batch_size,
        sequences_dict,
        model_params,
        encoders_filepath,
        decoders_filepath,
        save_to,
        num_workers=num_workers,
    )
