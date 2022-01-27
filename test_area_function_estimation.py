import argparse
import os
import yaml
import torch
import torch.nn as nn

from area_function_estimation.dataset import AreaFunctionDataset2
from area_function_estimation.evaluation import run_test
from articulation_to_melspec.model import ArticulatorsEmbedding
from helpers import set_seeds
from torch.utils.data import DataLoader


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    articulators = cfg["articulators"]
    n_articulators = len(articulators)

    test_dataset = AreaFunctionDataset2(cfg["datadir"], cfg["test_seq_dict"], clip_tails=cfg["clip_tails"])
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        worker_init_fn=set_seeds
    )

    embed = ArticulatorsEmbedding(n_curves=n_articulators, n_samples=50, embed_size=200)
    state_dict = torch.load(cfg["state_dict_fpath"], map_location=device)
    embed.load_state_dict(state_dict)
    embed = embed.to(device)

    test_outputs_dir = os.path.join(cfg["save_to"], "test_outputs")
    if not os.path.exists(test_outputs_dir):
        os.makedirs(test_outputs_dir)

    loss_fn = nn.L1Loss()

    info_test = run_test(
        epoch=0,
        model=embed,
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
