import argparse
import os
import torch
import ujson
import yaml

from torch.utils.data import DataLoader

from helpers import sequences_from_dict, set_seeds
from phoneme_to_articulation.principal_components.dataset import PrincipalComponentsPhonemeToArticulationDataset, pad_sequence_collate_fn
from phoneme_to_articulation.principal_components.evaluation import run_phoneme_to_PC_test
from phoneme_to_articulation.principal_components.losses import AutoencoderLoss
from phoneme_to_articulation.principal_components.models import PrincipalComponentsArtSpeech


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(cfg["vocab_fpath"]) as f:
        tokens = ujson.load(f)
        vocabulary = {token: i for i, token in enumerate(tokens)}

    test_sequences = sequences_from_dict(cfg["datadir"], cfg["test_seq_dict"])
    test_dataset = PrincipalComponentsPhonemeToArticulationDataset(
        datadir=cfg["datadir"],
        sequences=test_sequences,
        vocabulary=vocabulary,
        articulator=cfg["articulator"],
        sync_shift=0,
        framerate=55,
        clip_tails=cfg["clip_tails"]
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        worker_init_fn=set_seeds,
        collate_fn=pad_sequence_collate_fn
    )

    best_model = PrincipalComponentsArtSpeech(vocab_size=len(vocabulary), n_components=12, gru_dropout=0.2)
    best_model_state_dict = torch.load(cfg["state_dict_fpath"], map_location=device)
    best_model.load_state_dict(best_model_state_dict)
    best_model.to(device)

    test_outputs_dir = os.path.join(cfg["save_to"], "test_outputs")
    if not os.path.exists(test_outputs_dir):
        os.makedirs(test_outputs_dir)

    loss_fn = AutoencoderLoss(
        in_features=100,
        n_components=12,
        encoder_state_dict_fpath=cfg["encoder_state_dict_fpath"],
        decoder_state_dict_fpath=cfg["decoder_state_dict_fpath"],
        device=device
    )

    info_test = run_phoneme_to_PC_test(
        epoch=0,
        model=best_model,
        decoder_state_dict_fpath=cfg["decoder_state_dict_fpath"],
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
