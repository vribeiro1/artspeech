import argparse
import os
import torch
import yaml

from torch.utils.data import DataLoader
from torchaudio.models import Tacotron2

from articul_to_melspec.dataset import ArticulToMelSpecDataset, pad_sequence_collate_fn
from articul_to_melspec.evaluation import run_inference
from articul_to_melspec.model import ArticulatoryTacotron2
from helpers import set_seeds, sequences_from_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="cfg_filepath")
    args = parser.parse_args()

    with open(args.cfg_filepath) as f:
        cfg = yaml.safe_load(f.read())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sequences = sequences_from_dict(cfg["datadir"], cfg["seq_dict"])
    dataset = ArticulToMelSpecDataset(cfg["datadir"], sequences, cfg["articulators"])
    batch_size = cfg["batch_size"] if len(sequences) % cfg["batch_size"] != 1 else cfg["batch_size"] - 1
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=set_seeds,
        collate_fn=pad_sequence_collate_fn
    )

    model = ArticulatoryTacotron2(len(cfg["articulators"]))
    state_dict = torch.load(cfg["state_dict_filepath"], map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    save_to = os.path.join(cfg["save_to"])
    if not os.path.exists(save_to):
        os.makedirs(save_to)

    run_inference(model, dataloader, device, save_to)
