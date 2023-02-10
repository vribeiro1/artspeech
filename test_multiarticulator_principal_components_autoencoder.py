import pdb

import argparse
import funcy
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import ujson
import yaml

from torch.utils.data import DataLoader
from tqdm import tqdm

from helpers import set_seeds, sequences_from_dict
from phoneme_to_articulation.principal_components.metrics import MeanP2CPDistance
from phoneme_to_articulation.principal_components.dataset import PrincipalComponentsMultiArticulatorAutoencoderDataset
from phoneme_to_articulation.principal_components.evaluation import run_multiart_autoencoder_test
from phoneme_to_articulation.principal_components.losses import MultiArtRegularizedLatentsMSELoss
from phoneme_to_articulation.principal_components.models import MultiArticulatorAutoencoder
from settings import DatasetConfig

PINK = np.array([255, 0, 85, 255]) / 255
BLUE = np.array([0, 139, 231, 255]) / 255
ORDER_STR = {
    1: "st",
    2: "nd"
}


def evaluate_autoencoder(datadir, exp_dir):
    config_filepath = os.path.join(exp_dir, "config.json")
    encoders_filepath = os.path.join(exp_dir, "best_encoders.pt")
    decoders_filepath = os.path.join(exp_dir, "best_decoders.pt")

    saves_dir = os.path.join(exp_dir, "plots")
    if not os.path.exists(saves_dir):
        os.makedirs(saves_dir)

    with open(config_filepath) as f:
        config = ujson.load(f)
    sequences_dict = config["test_seq_dict"]

    model_params = config["model_params"]
    articulators_indices_dict = model_params["indices_dict"]
    articulators = sorted(articulators_indices_dict.keys())
    n_articulators = len(articulators)
    latent_size = max(funcy.flatten(articulators_indices_dict.values())) + 1

    sequences = sequences_from_dict(datadir, sequences_dict)
    dataset = PrincipalComponentsMultiArticulatorAutoencoderDataset(
        datadir=datadir,
        dataset_config=DatasetConfig,
        sequences=sequences,
        articulators=articulators,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        worker_init_fn=set_seeds,
        num_workers=config.get("num_workers", 0)
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
    data_p2cp = torch.zeros(size=(0, n_articulators))
    data_latents = torch.zeros(size=(0, latent_size))
    data_reconstructions = torch.zeros(size=(0, n_articulators, 2, 50))
    data_targets = torch.zeros(size=(0, n_articulators, 2, 50))
    for _, inputs, _, _ in tqdm(dataloader):
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
        ) * DatasetConfig.PIXEL_SPACING * DatasetConfig.RES

        data_reconstructions = torch.cat([data_reconstructions, reconstructions])
        data_latents = torch.cat([data_latents, latents])
        data_targets = torch.cat([data_targets, targets])
        data_p2cp = torch.cat([data_p2cp, p2cp])

    errors_filepath = os.path.join(exp_dir, "reconstruction_errors.npy")
    np.save(errors_filepath, data_p2cp.numpy())

    ################################################################################################
    #
    # Nomogram plots
    #
    ################################################################################################

    idx = 100
    orig_shape = data_targets[idx]
    orig_reconstruction = data_reconstructions[idx]

    for i_PC in range(latent_size):
        plt.figure(figsize=(10, 10))

        PC_range = np.arange(-1, 1.01, 0.1)
        for v in PC_range:
            latents = data_latents[idx].clone()
            latents[i_PC] = v
            latents = latents.unsqueeze(dim=0).to(device)

            reconstruction = torch.concat([
                autoencoder.decoders[articulator](latents[:, autoencoder.indices_dict[articulator]]).unsqueeze(dim=1)
                for articulator in autoencoder.sorted_articulators
            ], dim=1)
            reconstruction = reconstruction.detach().cpu()
            reconstruction = reconstruction.reshape(1, n_articulators, 2, 50)

            for i, articulator in enumerate(articulators):
                denorm_fn = dataset.normalize[articulator].inverse
                reconstruction[:, i, :, :] = denorm_fn(reconstruction[:, i, :, :])
            reconstruction = reconstruction.squeeze(dim=0)

            plt.title(f"{i_PC + 1}{ORDER_STR.get(i_PC + 1, 'th')} component", fontsize=56)

            if v <= 0:
                color = PINK
            else:
                color = BLUE

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
        plt.savefig(os.path.join(saves_dir, f"C{i_PC + 1}.pdf"))
        plt.savefig(os.path.join(saves_dir, f"C{i_PC + 1}.png"))
        plt.close()

    plt.figure(figsize=(5.7, 2.3))
    plt.rcParams.update({"mathtext.fontset": "cm"})

    rec_neg_handler = mlines.Line2D([0, 1], [0, 1], color=PINK, linestyle="-", lw=5, label=r"reconstructions ($z_i \leq 0$)")
    rec_pos_handler = mlines.Line2D([0, 1], [0, 1], color=BLUE, linestyle="-", lw=5, label=r"reconstructions ($z_i > 0$)")
    orig_rec_handler = mlines.Line2D([0, 1], [0, 1], color="limegreen", linestyle="-", lw=5, label="original reconstruction")
    orig_handler = mlines.Line2D([0, 1], [0, 1], color="red", linestyle="--", lw=5, label="original curve")

    plt.legend(
        handles=[
            rec_neg_handler,
            rec_pos_handler,
            orig_rec_handler,
            orig_handler
        ],
        fontsize=22,
        loc="center"
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(saves_dir, "legend.pdf"))
    plt.savefig(os.path.join(saves_dir, "legend.png"))


def main(
    datadir,
    exp_dir,
    batch_size,
    model_params,
    seq_dict,
    save_to,
    clip_tails=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    articulators_indices_dict = model_params["indices_dict"]
    test_sequences = sequences_from_dict(datadir, seq_dict)
    articulators = sorted(articulators_indices_dict.keys())
    test_dataset = PrincipalComponentsMultiArticulatorAutoencoderDataset(
        datadir=datadir,
        dataset_config=DatasetConfig,
        sequences=test_sequences,
        articulators=articulators,
        clip_tails=clip_tails
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=set_seeds
    )

    best_autoencoder = MultiArticulatorAutoencoder(**model_params)
    encoders_filepath = os.path.join(exp_dir, "best_encoders.pt")
    best_encoders_state_dict = torch.load(encoders_filepath, map_location=device)
    best_autoencoder.encoders.load_state_dict(best_encoders_state_dict)

    decoders_filepath = os.path.join(exp_dir, "best_decoders.pt")
    best_decoders_state_dict = torch.load(decoders_filepath, map_location=device)
    best_autoencoder.decoders.load_state_dict(best_decoders_state_dict)
    best_autoencoder.to(device)

    test_outputs_dir = os.path.join(save_to, "test_outputs")
    if not os.path.exists(test_outputs_dir):
        os.makedirs(test_outputs_dir)
    plots_dir = os.path.join(save_to, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    loss_fn = MultiArtRegularizedLatentsMSELoss(
        indices_dict=articulators_indices_dict,
        alpha=0.1,
    )

    info_test = run_multiart_autoencoder_test(
        epoch=0,
        model=best_autoencoder,
        dataloader=test_dataloader,
        criterion=loss_fn,
        # outputs_dir=test_outputs_dir,
        plots_dir=plots_dir,
        device=device
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

    datadir = cfg["datadir"]
    exp_dir = cfg["exp_dir"]
    evaluate_autoencoder(datadir, exp_dir)
