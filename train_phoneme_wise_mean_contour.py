import json
import os
import pandas as pd

from sacred import Experiment
from sacred.observers import FileStorageObserver
from articul_to_melspec import dataset

from dataset import ArtSpeechDataset
from phoneme_wise_mean_contour import train, test

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ex = Experiment()
fs_observer = FileStorageObserver.create(os.path.join(BASE_DIR, "phoneme_wise_mean_contour", "results"))
ex.observers.append(fs_observer)


@ex.automain
def main(_run, datadir, n_epochs, patience, learning_rate, weight_decay, train_filepath, valid_filepath, test_filepath, vocab_filepath, articulators, p_aug=0., state_dict_fpath=None):
    with open(vocab_filepath) as f:
        tokens = json.load(f)
        vocabulary = {token: i for i, token in enumerate(tokens)}

    train_dataset = ArtSpeechDataset(
        os.path.dirname(datadir),
        train_filepath,
        vocabulary,
        articulators,
        lazy_load=True,
        clip_tails=True
    )

    valid_dataset = ArtSpeechDataset(
        os.path.dirname(datadir),
        valid_filepath,
        vocabulary,
        articulators,
        lazy_load=True,
        clip_tails=True
    )

    if state_dict_fpath is None:
        save_to = os.path.join(fs_observer.dir, "phoneme_wise_articulators.csv")
        df = train(train_dataset, save_to)
    else:
        df = pd.read_csv(state_dict_fpath)
        for articulator in train_dataset.articulators:
            df[articulator] = df[articulator].apply(eval)

    test_dataset = ArtSpeechDataset(
        os.path.dirname(datadir),
        test_filepath,
        vocabulary,
        articulators,
        lazy_load=True,
        clip_tails=True
    )

    test_outputs_dir = os.path.join(fs_observer.dir, "test_outputs")
    if not os.path.exists(test_outputs_dir):
        os.makedirs(test_outputs_dir)

    test_results = test(test_dataset, df, test_outputs_dir)

    test_results_filepath = os.path.join(fs_observer.dir, "test_results.json")
    with open(test_results_filepath, "w") as f:
        json.dump(test_results, f)