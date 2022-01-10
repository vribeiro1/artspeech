import argparse
import os
import torch
import yaml

from torch.utils.data import DataLoader

from articulation_to_melspec.dataset import ArticulToMelSpecDataset, pad_sequence_collate_fn
from articulation_to_melspec.evaluation import run_glow_tts_inference
from articulation_to_melspec.model import GlowATS
from helpers import set_seeds, sequences_from_dict


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_sequences = sequences_from_dict(cfg["datadir"], cfg["test_seq_dict"])
    test_dataset = ArticulToMelSpecDataset(cfg["datadir"], test_sequences, cfg["articulators"])
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=4,
        worker_init_fn=set_seeds,
        collate_fn=pad_sequence_collate_fn
    )

    glow_tts_hparams = {
        "n_vocab": 148,
        "out_channels": 80,  # n_mel_channels on the Mel Spectrogram
        "hidden_channels": 192,
        "filter_channels": 768,
        "filter_channels_dp": 256,
        "kernel_size": 3,
        "p_dropout": 0.1,
        "n_blocks_dec": 12,
        "n_layers_enc": 6,
        "n_heads": 2,
        "p_dropout_dec": 0.05,
        "dilation_rate": 1,
        "kernel_size_dec": 5,
        "n_block_layers": 4,
        "n_sqz": 2,
        "prenet": True,
        "mean_only": True,
        "hidden_channels_enc": 192,
        "hidden_channels_dec": 192,
        "window_size": 4
    }

    best_model = GlowATS(
        n_articulators=len(cfg["articulators"]),
        n_samples=50,
        pretrained=True,
        **glow_tts_hparams
    )
    if cfg["state_dict_fpath"] is not None:
        state_dict = torch.load(cfg["state_dict_fpath"], map_location=device)
        best_model.load_state_dict(state_dict)
    best_model.to(device)

    test_outputs_dir = os.path.join(cfg["save_to"], "test_outputs")
    if not os.path.exists(test_outputs_dir):
        os.makedirs(test_outputs_dir)

    run_glow_tts_inference(
        model=best_model,
        dataloader=test_dataloader,
        device=device,
        save_to=test_outputs_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="cfg_filepath")
    args = parser.parse_args()

    with open(args.cfg_filepath) as f:
        cfg = yaml.safe_load(f.read())

    main(cfg)
