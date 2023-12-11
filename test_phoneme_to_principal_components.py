####################################################################################################
#
# Test the autoencoder-based phoneme-to-articulation
#
####################################################################################################
import argparse
import os
import torch
import ujson
import yaml

from functools import reduce
from torch.utils.data import DataLoader

from helpers import sequences_from_dict, set_seeds, make_indices_dict
from phoneme_recognition.deepspeech2 import DeepSpeech2
from phoneme_to_articulation import RNNType
from phoneme_to_articulation.principal_components.dataset import (
    PrincipalComponentsPhonemeToArticulationDataset2,
    pad_sequence_collate_fn
)
from phoneme_to_articulation.principal_components.evaluation import run_phoneme_to_principal_components_test
from phoneme_to_articulation.principal_components.losses import AutoencoderLoss2
from phoneme_to_articulation.principal_components.models import PrincipalComponentsArtSpeech
from settings import BLANK, UNKNOWN


def main(
    database_name,
    datadir,
    batch_size,
    seq_dict,
    indices_dict,
    vocab_filepath,
    state_dict_filepath,
    modelkwargs,
    autoencoder_kwargs,
    save_to,
    encoder_state_dict_filepath,
    decoder_state_dict_filepath,
    rnn_type="GRU",
    beta1=1.0,
    beta2=1.0,
    beta3=1.0,
    beta4=0.0,
    recognizer_filepath=None,
    recognizer_params=None,
    voicing_filepath=None,
    num_workers=0,
    TV_to_phoneme_map=None,
    clip_tails=True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    default_tokens = [BLANK, UNKNOWN]
    vocabulary = {token: i for i, token in enumerate(default_tokens)}
    with open(vocab_filepath) as f:
        tokens = ujson.load(f)
        for i, token in enumerate(tokens, start=len(vocabulary)):
            vocabulary[token] = i
    if voicing_filepath is not None:
        with open(voicing_filepath) as f:
            voiced_tokens = ujson.load(f)
    else:
        voiced_tokens = None

    if isinstance(list(indices_dict.values())[0], int):
        indices_dict = make_indices_dict(indices_dict)
    articulators = sorted(indices_dict.keys())

    sequences = sequences_from_dict(datadir, seq_dict)
    test_dataset = PrincipalComponentsPhonemeToArticulationDataset2(
        database_name,
        datadir,
        sequences,
        vocabulary,
        articulators,
        TV_to_phoneme_map,
        clip_tails=clip_tails,
        voiced_tokens=voiced_tokens,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=set_seeds,
        collate_fn=pad_sequence_collate_fn,
    )

    if TV_to_phoneme_map is None:
        TV_to_phoneme_map = {}
    TVs = sorted(TV_to_phoneme_map.keys())

    if recognizer_filepath:
        recognizer = DeepSpeech2(num_classes=len(vocabulary), **recognizer_params)
        recog_state_dict = torch.load(recognizer_filepath, map_location=device)
        recognizer.load_state_dict(recog_state_dict)
        recognizer.to(device)

        for p in recognizer.parameters():
            p.requires_grad = False
    else:
        recognizer = None
    denorm_fn = {
        articulator: normalize.inverse
        for articulator, normalize in test_dataset.normalize.items()
    }
    loss_fn = AutoencoderLoss2(
        indices_dict=indices_dict,
        TVs=TVs,
        device=device,
        encoder_state_dict_filepath=encoder_state_dict_filepath,
        decoder_state_dict_filepath=decoder_state_dict_filepath,
        denormalize_fn=denorm_fn,
        beta1=beta1,
        beta2=beta2,
        beta3=beta3,
        beta4=beta4,
        recognizer=recognizer,
        **autoencoder_kwargs,
    )

    model = PrincipalComponentsArtSpeech(
        vocab_size=len(vocabulary),
        indices_dict=indices_dict,
        rnn=RNNType[rnn_type.upper()],
        **modelkwargs,
    )
    model_state_dict = torch.load(state_dict_filepath, map_location=device)
    model.load_state_dict(model_state_dict)
    model.to(device)

    print(f"""
PrincipalComponentsArtSpeech -- {model.total_parameters} parameters
""")

    test_outputs_dir = os.path.join(save_to, "test_outputs")
    if not os.path.exists(test_outputs_dir):
        os.makedirs(test_outputs_dir)

    info_test = run_phoneme_to_principal_components_test(
        epoch=0,
        model=model,
        dataloader=test_dataloader,
        criterion=loss_fn,
        outputs_dir=test_outputs_dir,
        decode_transform=loss_fn.decode,
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
