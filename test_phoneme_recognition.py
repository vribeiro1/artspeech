import argparse
import logging
import os
import tempfile
import torch
import torch.nn as nn
import yaml
import ujson

from functools import partial
from torch.utils.data import DataLoader
from torchaudio.models.decoder import ctc_decoder

from helpers import set_seeds, sequences_from_dict
from phoneme_recognition import run_test, SIL, BLANK, UNKNOWN
from phoneme_recognition.datasets import Feature, PhonemeRecognitionDataset, collate_fn
from phoneme_recognition.deepspeech2 import DeepSpeech2
from phoneme_recognition.metrics import EditDistance
from settings import DatasetConfig, BASE_DIR

TMPFILES = os.path.join(BASE_DIR, "tmp")
TMP_DIR = tempfile.mkdtemp(dir=TMPFILES)
RESULTS_DIR = os.path.join(TMP_DIR, "results")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


def main(
    datadir,
    database,
    batch_size,
    seq_dict,
    vocab_fpath,
    pretrained,
    feature,
    model_params,
    state_dict_filepath,
    num_workers=0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on '{device.type}'")

    vocabulary = {
        BLANK: 0,
        UNKNOWN: 1
    }
    with open(vocab_fpath) as f:
        tokens = ujson.load(f)
        for i, token in enumerate(tokens, start=2):
            vocabulary[token] = i

    tokens = [k for k, v in sorted(vocabulary.items(), key=lambda t: t[1])]
    decoder = ctc_decoder(
        lexicon=None,
        tokens=tokens,
        sil_token=SIL,
        blank_token=BLANK,
        unk_word=UNKNOWN,
    )

    if pretrained:
        model = DeepSpeech2.load_librispeech_model(
            model_params["num_features"],
            adapter_out_features=model_params.get("adapter_out_features")
        )
        hidden_size = model_params["rnn_hidden_size"]
        model.classifier[-1] = nn.Linear(hidden_size, len(vocabulary))
    else:
        model = DeepSpeech2(num_classes=len(vocabulary), **model_params)

    state_dict = torch.load(state_dict_filepath, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.to(device)

    feature = Feature(feature)

    sequences = sequences_from_dict(datadir, seq_dict)
    dataset = PhonemeRecognitionDataset(
        datadir=datadir,
        database=database,
        sequences=sequences,
        vocabulary=vocabulary,
        framerate=DatasetConfig.FRAMERATE,
        sync_shift=DatasetConfig.SYNC_SHIFT,
        features=[feature],
        tmp_dir=TMP_DIR,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=set_seeds,
        collate_fn=partial(collate_fn, features_names=[feature]),
    )

    metrics = {
        "edit_distance": EditDistance(decoder),
    }

    info_test = run_test(
        model=model,
        dataloader=dataloader,
        fn_metrics=metrics,
        device=device,
        feature=feature
    )
    print(info_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_filepath")
    args = parser.parse_args()

    with open(args.config_filepath) as f:
        cfg = yaml.safe_load(f)

    main(**cfg)
