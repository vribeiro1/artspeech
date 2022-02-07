import pdb

import funcy
import functools
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F

from tqdm import tqdm

from loss import EuclideanDistanceLoss
from metrics import pearsons_correlation, p2cp_distance, euclidean_distance
from phoneme_to_articulation.encoder_decoder.evaluation import save_outputs, tract_variables


def calculate_tokens_sequences_len(tokens):
    lengths = []
    seq_length = 0
    for curr_token, next_token in zip(tokens[:-1], tokens[1:]):
        if curr_token == next_token:
            seq_length += 1
        else:
            lengths.append((curr_token, seq_length + 1))
            seq_length = 0
    lengths.append((next_token, seq_length + 1))

    return lengths


def calculate_tokens_positions_in_sequence(tokens):
    token_positions = []
    position = 0
    for curr_token, next_token in zip(tokens[:-1], tokens[1:]):
        token_positions.append((curr_token, position))
        if curr_token == next_token:
            position += 1
        else:
            position = 0
    token_positions.append((next_token, position + 1))

    return token_positions


def process_sentence(sentence, articulators):
    _, sentence_targets, sentence_tokens = sentence

    tokens_pos = calculate_tokens_positions_in_sequence(sentence_tokens)
    tokens_seqs_len = calculate_tokens_sequences_len(sentence_tokens)
    tokens_seqs_len = functools.reduce(lambda l1, l2: l1 + l2, [[token_seq_len for _ in range(token_seq_len[1])] for token_seq_len in tokens_seqs_len])
    tokens_rel_pos = [(token, pos, seq_len, pos / seq_len) for (token, pos), (token, seq_len)  in zip(tokens_pos, tokens_seqs_len)]

    data = []
    for (token, token_abs_pos, seq_len, token_rel_pos), token_targets in zip(tokens_rel_pos, sentence_targets):
        item = {
            "token": token,
            "abs_pos": token_abs_pos,
            "seq_len": seq_len,
            "rel_pos": token_rel_pos
        }

        item.update({
            articulator: token_targets[i].numpy().tolist()
            for i, articulator in enumerate(articulators)
        })

        data.append(item)

    return data


def train(dataset, save_to=None):
    data = funcy.lflatten([
        process_sentence(sentence, dataset.articulators)
        for sentence in tqdm(dataset, desc="train")
    ])

    df = pd.DataFrame(data)
    if save_to is not None:
        df.to_csv(save_to, index=False)

    return df


def forward_mean_contour(sentence_tokens, df, articulators, n_samples=50):
    n_articulators = len(articulators)
    tokens_pos = calculate_tokens_positions_in_sequence(sentence_tokens)
    tokens_seqs_len = calculate_tokens_sequences_len(sentence_tokens)
    tokens_seqs_len = functools.reduce(
        lambda l1, l2: l1 + l2,
        [
            [token_seq_len for _ in range(token_seq_len[1])]
            for token_seq_len in tokens_seqs_len
        ]
    )
    tokens_rel_pos = [
        (token, pos, seq_len, pos / seq_len)
        for (token, pos), (token, seq_len) in zip(tokens_pos, tokens_seqs_len)
    ]

    sentence_outputs = torch.zeros(0, n_articulators, 2, n_samples)
    for token, _, _, rel_pos in tokens_rel_pos:
        df_token_wise = df[df.token == token]

        weights = (df_token_wise.rel_pos - rel_pos).abs().to_numpy()
        n_weights, = weights.shape
        softmin_weights = F.softmin(torch.from_numpy(weights), dim=0).view(n_weights, 1, 1, 1)

        token_wise_articulators = [torch.stack([
            torch.tensor((row[articulator])) for articulator in articulators
        ]) for _, row in df_token_wise.iterrows()]

        token_wise_articulators = torch.stack(token_wise_articulators)
        weighted_mean_token_wise_articulators = (softmin_weights * token_wise_articulators).sum(dim=0)
        weighted_mean_token_wise_articulators = weighted_mean_token_wise_articulators.unsqueeze(dim=0)

        sentence_outputs = torch.cat([sentence_outputs, weighted_mean_token_wise_articulators])

    return sentence_outputs


def test(dataset, df, save_to):
    if isinstance(df, str):
        df = pd.read_csv(df)

    n_articulators = len(dataset.articulators)
    n_samples = dataset.n_samples

    criterion = EuclideanDistanceLoss()

    losses = []
    euclidean_per_articulator = [[] for _ in dataset.articulators]
    p2cp_per_articulator = [[] for _ in dataset.articulators]
    x_corrs = [[] for _ in dataset.articulators]
    y_corrs = [[] for _ in dataset.articulators]
    for i_sentence, (_, _, sentence_targets, sentence_tokens) in enumerate(tqdm(dataset, "test")):
        sentence_outputs = forward_mean_contour(sentence_tokens, df, dataset.articulators)
        sentence_outputs = sentence_outputs.unsqueeze(dim=0)
        sentence_targets = sentence_targets.unsqueeze(dim=0)

        loss = criterion(sentence_outputs, sentence_targets)
        p2cp = p2cp_distance(sentence_outputs, sentence_targets).mean(dim=1)  # (bs, n_articulators)
        euclidean = euclidean_distance(sentence_outputs, sentence_targets).mean(dim=1)  # (bs, n_articulators)

        x_corr, y_corr = pearsons_correlation(sentence_outputs, sentence_targets)
        x_corr = x_corr.mean(dim=-1)[0]
        y_corr = y_corr.mean(dim=-1)[0]

        losses.append(loss.item())
        for i_art, _ in enumerate(dataset.articulators):
            x_corrs[i_art].append(x_corr[i_art].item())
            y_corrs[i_art].append(y_corr[i_art].item())

            p2cp_per_articulator[i_art].extend([dist.item() for dist in p2cp[:, i_art]])
            euclidean_per_articulator[i_art].extend([dist.item() for dist in euclidean[:, i_art]])

        saves_i_dir = os.path.join(save_to, str(i_sentence))
        if not os.path.exists(saves_i_dir):
            os.makedirs(saves_i_dir)

        tract_variables(
            sentence_outputs,
            sentence_targets,
            [len(sentence_tokens)],
            [sentence_tokens],
            dataset.articulators,
            saves_i_dir,
            offset=i_sentence
        )

        save_outputs(
            sentence_outputs,
            sentence_targets,
            [len(sentence_tokens)],
            [sentence_tokens],
            dataset.articulators,
            saves_i_dir,
            regularize_out=True,
            offset=i_sentence
        )

    mean_loss = np.mean(losses)
    info = {
        "loss": mean_loss
    }

    info.update({
        art: {
            "x_corr": np.mean(x_corrs[i_art]),
            "y_corr": np.mean(y_corrs[i_art])
        }
        for i_art, art in enumerate(dataset.articulators)
    })

    return info
