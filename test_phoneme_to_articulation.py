import pdb

import argparse
import os
import pandas as pd
import torch
import yaml
import ujson

from torch.utils.data import DataLoader

from helpers import set_seeds, sequences_from_dict
from phoneme_recognition.deepspeech2 import DeepSpeech2
from phoneme_to_articulation.encoder_decoder.dataset import ArtSpeechDataset, pad_sequence_collate_fn
from phoneme_to_articulation.encoder_decoder.evaluation import run_test
from phoneme_to_articulation.encoder_decoder.loss import ArtSpeechLoss
from phoneme_to_articulation.encoder_decoder.models import ArtSpeech
from settings import UNKNOWN, BLANK


def main(
    datadir,
    database_name,
    batch_size,
    test_seq_dict,
    state_dict_fpath,
    vocab_filepath,
    articulators,
    save_to,
    beta1,
    beta2,
    recognizer_filepath=None,
    recognizer_params=None,
    voicing_filepath=None,
    clip_tails=True,
    num_workers=0
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

    test_sequences = sequences_from_dict(datadir, test_seq_dict)
    test_dataset = ArtSpeechDataset(
        datadir,
        database_name,
        test_sequences,
        vocabulary,
        articulators,
        clip_tails=clip_tails,
        voiced_tokens=voiced_tokens,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=set_seeds,
        collate_fn=pad_sequence_collate_fn
    )

    num_articulators = len(articulators)
    best_model = ArtSpeech(len(vocabulary), num_articulators, gru_dropout=0.2)
    state_dict = torch.load(state_dict_fpath, map_location=device)
    best_model.load_state_dict(state_dict)
    best_model.to(device)

    test_outputs_dir = os.path.join(save_to, "test_outputs")
    if not os.path.exists(test_outputs_dir):
        os.makedirs(test_outputs_dir)

    if recognizer_filepath:
        recognizer = DeepSpeech2(num_classes=len(vocabulary), **recognizer_params)
        recog_state_dict = torch.load(recognizer_filepath, map_location=device)
        recognizer.load_state_dict(recog_state_dict)
        recognizer.to(device)

        for p in recognizer.parameters():
            p.requires_grad = False
    else:
        recognizer = None
    loss_fn = ArtSpeechLoss(recognizer)

    test_results = run_test(
        epoch=0,
        model=best_model,
        dataloader=test_dataloader,
        criterion=loss_fn,
        outputs_dir=test_outputs_dir,
        articulators=articulators,
        device=device,
        beta1=beta1,
        beta2=beta2,
        regularize_out=True,
    )

    test_results_filepath = os.path.join(save_to, "test_results.json")
    with open(test_results_filepath, "w") as f:
        ujson.dump(test_results, f)

    results_item = {
        "exp": None,
        "loss": test_results["loss"],
    }

    for articulator in test_dataset.articulators:
        results_item[f"p2cp_{articulator}"] = test_results[articulator]["p2cp"]
        results_item[f"p2cp_mm_{articulator}"] = test_results[articulator]["p2cp_mm"]
        results_item[f"med_{articulator}"] = test_results[articulator]["med"]
        results_item[f"med_mm_{articulator}"] = test_results[articulator]["med_mm"]

    df = pd.DataFrame([results_item])
    df_filepath = os.path.join(save_to, "test_results.csv")
    df.to_csv(df_filepath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="cfg_filepath")
    args = parser.parse_args()

    with open(args.cfg_filepath) as f:
        cfg = yaml.safe_load(f.read())

    main(**cfg)
