import argparse
import os
import torch
import yaml

from torch.utils.data import DataLoader

from helpers import sequences_from_dict, set_seeds
from phoneme_to_articulation.principal_components.dataset import PrincipalComponentsAutoencoderDataset
from phoneme_to_articulation.principal_components.evaluation import run_autoencoder_test
from phoneme_to_articulation.principal_components.models import Autoencoder
from phoneme_to_articulation.principal_components.losses import RegularizedLatentsMSELoss


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_sequences = sequences_from_dict(cfg["datadir"], cfg["test_seq_dict"])
    test_dataset = PrincipalComponentsAutoencoderDataset(
        datadir=cfg["datadir"],
        sequences=test_sequences,
        articulator=cfg["articulator"],
        sync_shift=0,
        framerate=55,
        clip_tails=cfg["clip_tails"]
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        worker_init_fn=set_seeds
    )

    best_autoencoder = Autoencoder(
        in_features=100,
        n_components=cfg["n_components"]
    )
    best_encoder_state_dict = torch.load(cfg["encoder_state_dict_fpath"], map_location=device)
    best_autoencoder.encoder.load_state_dict(best_encoder_state_dict)
    best_decoder_state_dict = torch.load(cfg["decoder_state_dict_fpath"], map_location=device)
    best_autoencoder.decoder.load_state_dict(best_decoder_state_dict)
    best_autoencoder.to(device)

    test_outputs_dir = os.path.join(cfg["save_to"], "test_outputs")
    if not os.path.exists(test_outputs_dir):
        os.makedirs(test_outputs_dir)

    loss_fn = RegularizedLatentsMSELoss(alpha=1e-2)

    info_test = run_autoencoder_test(
        epoch=0,
        model=best_autoencoder,
        dataloader=test_dataloader,
        criterion=loss_fn,
        outputs_dir=test_outputs_dir,
        device=device
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="cfg_filepath")
    args = parser.parse_args()

    with open(args.cfg_filepath) as f:
        cfg = yaml.safe_load(f.read())

    main(cfg)
