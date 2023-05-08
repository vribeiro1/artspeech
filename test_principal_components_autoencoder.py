import argparse
import os
import torch
import ujson
import yaml

from torch.utils.data import DataLoader

from helpers import sequences_from_dict, set_seeds
from phoneme_to_articulation.principal_components.dataset import PrincipalComponentsAutoencoderDataset
from phoneme_to_articulation.principal_components.evaluation import run_autoencoder_test
from phoneme_to_articulation.principal_components.models.autoencoder import Autoencoder
from phoneme_to_articulation.principal_components.losses import RegularizedLatentsMSELoss


def main(
    database_name,
    datadir,
    batch_size,
    seq_dict,
    articulator,
    model_params,
    encoder_state_dict_fpath,
    decoder_state_dict_fpath,
    save_to,
    clip_tails=True,
    num_workers=0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_sequences = sequences_from_dict(datadir, seq_dict)
    test_dataset = PrincipalComponentsAutoencoderDataset(
        database_name=database_name,
        datadir=datadir,
        sequences=test_sequences,
        articulator=articulator,
        clip_tails=clip_tails
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=set_seeds,
        num_workers=num_workers
    )

    best_autoencoder = Autoencoder(**model_params)
    best_encoder_state_dict = torch.load(encoder_state_dict_fpath, map_location=device)
    best_autoencoder.encoder.load_state_dict(best_encoder_state_dict)
    best_decoder_state_dict = torch.load(decoder_state_dict_fpath, map_location=device)
    best_autoencoder.decoder.load_state_dict(best_decoder_state_dict)
    best_autoencoder.to(device)

    test_outputs_dir = os.path.join(save_to, "test_outputs")
    if not os.path.exists(test_outputs_dir):
        os.makedirs(test_outputs_dir)

    loss_fn = RegularizedLatentsMSELoss(alpha=1e-2)

    info_test = run_autoencoder_test(
        epoch=0,
        model=best_autoencoder,
        dataloader=test_dataloader,
        criterion=loss_fn,
        # outputs_dir=test_outputs_dir,
        plots_dir=save_to,
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
