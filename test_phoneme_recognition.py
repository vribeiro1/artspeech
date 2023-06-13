import argparse
import logging
import os
import shutil
import tempfile
import torch
import torch.nn as nn
import yaml
import ujson

from functools import partial
from torch.utils.data import DataLoader
from torchaudio.models.decoder import ctc_decoder

from helpers import set_seeds, sequences_from_dict
from phoneme_recognition import (
    run_test,
    Criterion,
    Feature,
    Target,
    SIL,
    BLANK,
    UNKNOWN
)
from phoneme_recognition.datasets import PhonemeRecognitionDataset, collate_fn
from phoneme_recognition.decoders import TopKDecoder
from phoneme_recognition.deepspeech2 import DeepSpeech2
from phoneme_recognition.metrics import EditDistance, WordInfoLost, Accuracy, AUROC
from phoneme_recognition.synthetic_shapes import SyntheticPhonemeRecognitionDataset
from settings import BASE_DIR, DATASET_CONFIG

TMPFILES = os.path.join(BASE_DIR, "tmp")
TMP_DIR = tempfile.mkdtemp(dir=TMPFILES)
RESULTS_DIR = os.path.join(TMP_DIR, "results")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


def main(
    database_name,
    datadir,
    batch_size,
    seq_dict,
    vocab_filepath,
    pretrained,
    feature,
    loss,
    model_params,
    target,
    state_dict_filepath,
    plot_target=None,
    voicing_filepath=None,
    num_workers=0,
    save_dir=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on '{device.type}'")

    feature = Feature(feature)
    target = Target(target)
    plot_target = Target(plot_target) if plot_target else target
    criterion = Criterion[loss]

    default_tokens = [BLANK, UNKNOWN] if criterion == Criterion.CTC else [UNKNOWN]
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

    tokens = [k for k, v in sorted(vocabulary.items(), key=lambda t: t[1])]
    decoder_fn = ctc_decoder if criterion == Criterion.CTC else TopKDecoder
    decoder = decoder_fn(
        lexicon=None,
        tokens=tokens,
        sil_token=SIL,
        blank_token=BLANK if criterion == Criterion.CTC else None,
        unk_word=UNKNOWN,
    )

    if pretrained:
        model = DeepSpeech2.load_librispeech_model(
            model_params["num_features"],
            adapter_out_features=model_params.get("adapter_out_features")
        )
        hidden_size = model_params["rnn_hidden_size"]
        model.classifier = nn.Linear(hidden_size, len(vocabulary))
    else:
        model = DeepSpeech2(num_classes=len(vocabulary), **model_params)

    state_dict = torch.load(state_dict_filepath, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.to(device)

    sequences = sequences_from_dict(datadir, seq_dict)
    # dataset = SyntheticPhonemeRecognitionDataset(
    dataset = PhonemeRecognitionDataset(
        datadir=datadir,
        database_name=database_name,
        sequences=sequences,
        vocabulary=vocabulary,
        features=[feature],
        tmp_dir=TMP_DIR,
        voiced_tokens=voiced_tokens,
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
        "word_info_lost": WordInfoLost(decoder),
        # "accuracy": Accuracy(len(vocabulary)),
        # "accuracy_per_class": Accuracy(len(vocabulary), average=None),
        # "auroc": AUROC(len(vocabulary))
    }

    info_test = run_test(
        model=model,
        dataloader=dataloader,
        fn_metrics=metrics,
        decoder=decoder,
        device=device,
        feature=feature,
        target=target,
        plot_target=plot_target,
        use_voicing=(voicing_filepath is not None),
        save_dir=save_dir,
    )

    if save_dir is not None:
        info_test_filepath = os.path.join(save_dir, "info_test.json")
        with open(info_test_filepath, "w") as f:
            ujson.dump(info_test, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_filepath")
    args = parser.parse_args()

    with open(args.config_filepath) as f:
        cfg = yaml.safe_load(f)

    try:
        main(**cfg)
    finally:
        shutil.rmtree(TMP_DIR)
