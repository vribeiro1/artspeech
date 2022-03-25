import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import torch
import ujson

from sklearn.metrics import explained_variance_score
from statsmodels.stats.weightstats import  ttest_ind
from torch.utils.data import DataLoader
from tqdm import tqdm

from helpers import sequences_from_dict, set_seeds
from phoneme_to_articulation.principal_components.metrics import MeanP2CPDistance
from phoneme_to_articulation.principal_components.models import Autoencoder
from phoneme_to_articulation.principal_components.dataset import PrincipalComponentsAutoencoderDataset
from settings import DatasetConfig

PINK = np.array([255, 0, 85, 255]) / 255
BLUE = np.array([0, 139, 231, 255]) / 255
ORDER_STR = {
    1: "st",
    2: "nd"
}


def evaluate_autoencoder(cfg):
    datadir = cfg["datadir"]
    exp_dir = cfg["exp_dir"]

    config_filepath = os.path.join(exp_dir, "config.json")
    encoder_filepath = os.path.join(exp_dir, "best_encoder.pt")
    decoder_filepath = os.path.join(exp_dir, "best_decoder.pt")

    saves_dir = os.path.join(exp_dir, "plots")
    if not os.path.exists(saves_dir):
        os.makedirs(saves_dir)

    with open(config_filepath) as f:
        config = ujson.load(f)
    n_components = config["n_components"]
    sequences_dict = config["test_seq_dict"]

    sequences = sequences_from_dict(datadir, sequences_dict)
    dataset = PrincipalComponentsAutoencoderDataset(
        datadir=datadir,
        sequences=sequences,
        articulator=cfg["articulator"],
        sync_shift=DatasetConfig.SYNC_SHIFT,
        framerate=DatasetConfig.FRAMERATE
    )

    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        worker_init_fn=set_seeds
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = Autoencoder(
        in_features=100,
        n_components=n_components
    )
    encoder_state_dict = torch.load(encoder_filepath, map_location=device)
    autoencoder.encoder.load_state_dict(encoder_state_dict)
    decoder_state_dict = torch.load(decoder_filepath, map_location=device)
    autoencoder.decoder.load_state_dict(decoder_state_dict)

    autoencoder.to(device)
    autoencoder.eval()

    p2cp_fn = MeanP2CPDistance(reduction="none")

    data_p2cp = torch.zeros(size=(0,))
    data_latents = torch.zeros(size=(0, n_components))
    data_reconstructions = torch.zeros(size=(0, 2, 50))
    data_targets = torch.zeros(size=(0, 2, 50))
    for _, inputs, _, _ in tqdm(dataloader):
        bs, _ = inputs.shape

        inputs = inputs.to(device)
        reconstructions, latents = autoencoder(inputs)
        reconstructions = reconstructions.reshape(bs, 2, 50)
        reconstructions = dataset.normalize.inverse(reconstructions)
        reconstructions = reconstructions.detach().cpu()
        latents = latents.detach().cpu()
        inputs = inputs.detach().cpu()

        targets = inputs.reshape(bs, 2, 50)
        targets = dataset.normalize.inverse(targets)
        p2cp = p2cp_fn(
            reconstructions.permute(0, 2, 1),
            targets.permute(0, 2, 1)
        ) * DatasetConfig.PIXEL_SPACING * DatasetConfig.RES

        data_reconstructions = torch.cat([data_reconstructions, reconstructions])
        data_latents = torch.cat([data_latents, latents])
        data_targets = torch.cat([data_targets, targets])
        data_p2cp = torch.cat([data_p2cp, p2cp])

    errors_filepath = os.path.join(exp_dir, "reconstruction_errors.npy")
    np.save(errors_filepath, data_p2cp.numpy())

    mean_error = data_p2cp.mean()
    std_error = data_p2cp.std()

    N, _, _ = data_targets.shape
    rho2 = explained_variance_score(
        data_reconstructions.reshape(N, 2 * 50),
        data_targets.reshape(N, 2 * 50),
    )

    mean_error_str = "%.03f" % mean_error.item()
    std_error_str = "%.03f" % std_error.item()
    rho2_str = "%.03f" % rho2

    latents_cov = torch.cov(data_latents.T)

    percent_variance_main_diag = latents_cov.diag().abs().sum() / latents_cov.abs().sum()
    percent_variance_main_diag_str = "%.01f" % (percent_variance_main_diag * 100)

    print(f"""
     P2CP   ---    rho2    ----    Var(diag) / Total_Var
{mean_error_str} +- {std_error_str} |   {rho2_str}    |    {percent_variance_main_diag_str}
""")

    plt.figure(figsize=(10, 10))

    sns.heatmap(
        latents_cov,
        cmap="BuPu",
        linewidths=.5,
        annot=True,
        cbar=False,
        xticklabels=[i + 1 for i in range(n_components)],
        yticklabels=[i + 1 for i in range(n_components)],
        annot_kws={
            "fontsize": 16
        }
    )

    plt.tick_params(axis="both", which="major", labelsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(saves_dir, f"covariance_matrix.pdf"))

    idx = 0
    orig_shape = data_targets[idx]
    orig_reconstruction = data_reconstructions[idx]

    for i_PC in range(n_components):
        plt.figure(figsize=(10, 10))

        PC_range = np.arange(-1, 1.01, 0.1)
        for v in PC_range:
            latents = data_latents[idx].clone()
            latents[i_PC] = v
            latents = latents.unsqueeze(dim=0).to(device)

            reconstruction = autoencoder.decoder(latents).detach().cpu().squeeze(dim=0)
            reconstruction = dataset.normalize.inverse(reconstruction.reshape(2, 50))

            plt.title(f"{i_PC + 1}{ORDER_STR.get(i_PC + 1, 'th')} component", fontsize=56)

            if v <= 0:
                color = PINK
            else:
                color = BLUE
            plt.plot(*reconstruction, color=color, lw=3, alpha=0.2)

        plt.plot(*orig_shape, "--", color="red", lw=5, alpha=1.)
        plt.plot(*orig_reconstruction, color="limegreen", lw=5, alpha=1.)

        plt.xlim([0.25, 0.75])
        plt.ylim([0.7, 0.2])

        plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(saves_dir, f"C{i_PC + 1}.pdf"))
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
    plt.savefig(os.path.join(saves_dir, "legend.pdf"), format="pdf")


def compare_autoencoders(cfg):
    exp_dirs = {
        n_components: os.path.join(cfg["base_results_dir"], str(exp))
        for exp, n_components in cfg["exps"]
    }

    reconstruction_errors = {
        n_components: np.load(os.path.join(exp_dir, "reconstruction_errors.npy"))
        for n_components, exp_dir in exp_dirs.items()
    }

    df_erros = pd.concat([pd.DataFrame({
        "n_comp": [n_components] * len(errors),
        "error": errors
    }) for n_components, errors in reconstruction_errors.items()])

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()

    sns.violinplot(
        x="n_comp",
        y="error",
        data=df_erros,
        palette="colorblind"
    )

    ax.set_xlabel("Number of components", fontsize=22)
    ax.set_ylabel("Reconstruction Error (mm)" + "\n" + "(Mean P2CP Distance)", fontsize=22)

    ax.grid(which="major")
    ax.grid(which="minor", alpha=0.4)
    ax.minorticks_on()

    ax.tick_params(axis="both", which="major", labelsize=18)

    plt.tight_layout()
    plt.savefig(os.path.join(cfg["base_results_dir"], "rec_error_distributions.pdf"))

    pval_data = []
    error_items = sorted(reconstruction_errors.items(), key=lambda t: t[0])
    for i, (n_components, n_errors) in enumerate(error_items[:-1]):
        for m_components, m_errors in error_items[i+1:]:
            _, pvalue, dof = ttest_ind(n_errors, m_errors)

            pval_data.append({
                "components_1": n_components,
                "components_2": m_components,
                "pvalue": pvalue,
                "degrees_of_freedom": dof
            })

    pd.DataFrame(pval_data).to_csv(
        os.path.join(cfg["base_results_dir"], "stats_diff.csv"),
        index=False
    )


if __name__ == "__main__":
    cfg = {
        "datadir": "/home/vribeiro/Documents/loria/datasets/Gottingen_Database",
        "exp_dir": "/home/vribeiro/Documents/loria/workspace/artspeech/phoneme_to_articulation/principal_components/results/autoencoder/4",
        "articulator": "tongue"
    }

    evaluate_autoencoder(cfg)

    cfg = {
        "base_results_dir": "/home/vribeiro/Documents/loria/workspace/artspeech/phoneme_to_articulation/principal_components/results/autoencoder",
        "exps": [(1, 8), (2, 10), (3, 12), (4, 16)]
    }

    # compare_autoencoders(cfg)
