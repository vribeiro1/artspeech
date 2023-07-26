import pdb

import funcy
import functools
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F

from functools import partial
from itertools import groupby
from tqdm import tqdm
from vt_tools import UPPER_INCISOR

from phoneme_to_articulation.metrics import EuclideanDistance
from metrics import pearsons_correlation, p2cp_distance, euclidean_distance
from phoneme_to_articulation import save_outputs, tract_variables


def _calculate_tokens_lengths_and_positions(tokens):
    lengths = [(token, sum(1 for _ in group)) for token, group in groupby(tokens)]
    positions = functools.reduce(
        lambda l1, l2: l1 + l2,
        [
            [(token, i) for i, _ in enumerate(group)]
            for token, group in groupby(tokens)
        ]
    )

    return lengths, positions


def process_sentence_with_pos(sentence, articulators):
    _, _, sentence_targets, sentence_tokens, _, _, _, _ = sentence

    tokens_seqs_len, tokens_pos = _calculate_tokens_lengths_and_positions(sentence_tokens)
    tokens_seqs_len = functools.reduce(
        lambda l1, l2: l1 + l2,
        [
            [token_seq_len for _ in range(token_seq_len[1])]
            for token_seq_len in tokens_seqs_len
        ]
    )
    tokens_rel_pos = [
        (token, pos, seq_len, pos / seq_len)
        for (token, pos), (token, seq_len)
        in zip(tokens_pos, tokens_seqs_len)
    ]

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


def process_sentence(sentence, articulators):
    _, _, sentence_targets, sentence_tokens, _, _, _, _ = sentence

    data = []
    for token, token_targets in zip(sentence_tokens, sentence_targets):
        item = {
            "token": token,
        }

        item.update({
            articulator: token_targets[i].numpy().tolist()
            for i, articulator in enumerate(articulators)
        })
        data.append(item)

    return data


def forward_weighted_mean_contour(sentence_tokens, df, articulators, n_samples=50):
    n_articulators = len(articulators)
    tokens_seqs_len, tokens_pos = _calculate_tokens_lengths_and_positions(sentence_tokens)
    tokens_seqs_len = functools.reduce(

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
        df_token_wise = df[df.token == token].sample(frac=0.1, random_state=0)

        weights = (df_token_wise.rel_pos - rel_pos).abs().to_numpy()
        n_weights, = weights.shape
        softmin_weights = F.softmin(torch.from_numpy(weights), dim=0).view(n_weights, 1, 1, 1)

        token_wise_articulators = [torch.stack([
            torch.tensor((row[articulator])) for articulator in articulators
        ]) for _, row in df_token_wise.iterrows()]

        token_wise_articulators = torch.stack(token_wise_articulators)
        weighted_mean_token_wise_articulators = (softmin_weights * token_wise_articulators).sum(dim=0)
        weighted_mean_token_wise_articulators = weighted_mean_token_wise_articulators.unsqueeze(dim=0)

        sentence_outputs = torch.cat([
            sentence_outputs,
            weighted_mean_token_wise_articulators
        ])

    return sentence_outputs


def forward_mean_contour(sentence_tokens, df, articulators, n_samples=50):
    n_articulators = len(articulators)

    sentence_outputs = torch.zeros(0, n_articulators, 2, n_samples)
    for token in sentence_tokens:
        df_token_wise = df[df.token == token].sample(frac=0.1, random_state=0)

        token_wise_articulators = [torch.stack([
            torch.tensor((row[articulator])) for articulator in articulators
        ]) for _, row in df_token_wise.iterrows()]

        token_wise_articulators = torch.stack(token_wise_articulators)
        mean_token_wise_articulators = token_wise_articulators.mean(dim=0)
        mean_token_wise_articulators = mean_token_wise_articulators.unsqueeze(dim=0)

        sentence_outputs = torch.cat([
            sentence_outputs,
            mean_token_wise_articulators
        ])

    return sentence_outputs


def train(dataset, save_to=None, weighted=False):
    process_fn = process_sentence_with_pos if weighted else process_sentence
    data = funcy.flatten(map(
        partial(process_fn, articulators=dataset.articulators),
        tqdm(dataset, desc="train")
    ))

    df = pd.DataFrame(data)
    if save_to is not None:
        df.to_csv(save_to, index=False)

    return df


def test(dataset, df, save_to, weighted=False):
    save_to = os.path.join(save_to, "0")  # Keep compatibility with other methods
    if isinstance(df, str):
        df = pd.read_csv(df)

    criterion = EuclideanDistance()
    losses = []
    euclidean_per_articulator = [[] for _ in dataset.articulators]
    p2cp_per_articulator = [[] for _ in dataset.articulators]
    x_corrs = [[] for _ in dataset.articulators]
    y_corrs = [[] for _ in dataset.articulators]
    forward_fn = forward_weighted_mean_contour if weighted else forward_mean_contour
    for sentence_name, _, sentence_targets, sentence_tokens, reference_arrays, _, frame_ids, _ in tqdm(dataset, "test"):
        sentence_outputs = forward_fn(sentence_tokens, df, dataset.articulators)
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

        # The upper incisor is the reference of the coordinate system and since it has a fixed
        # shape, it is non-sense to include it in the prediction. However, it is important for
        # tract variables and visualization. Therefore, we inject it in the arrays in order to
        # have it available for the next steps.
        if UPPER_INCISOR not in dataset.articulators:
            tv_articulators = sorted(dataset.articulators + [UPPER_INCISOR])
            ref_idx = tv_articulators.index(UPPER_INCISOR)

            sentence_outputs = torch.concat([
                sentence_outputs[:, :, :ref_idx, :, :],
                reference_arrays,
                sentence_outputs[:, :, ref_idx:, :, :],
            ], dim=2)

            sentence_targets = torch.concat([
                sentence_targets[:, :, :ref_idx, :, :],
                reference_arrays,
                sentence_targets[:, :, ref_idx:, :, :],
            ], dim=2)
        else:
            tv_articulators = dataset.articulators

        tract_variables(
            sentences_ids=[sentence_name],
            frame_ids=[frame_ids],
            outputs=sentence_outputs,
            targets=sentence_targets,
            lengths=[len(frame_ids)],
            phonemes=[sentence_tokens],
            articulators=tv_articulators,
            save_to=save_to,
        )

        save_outputs(
            sentences_ids=[sentence_name],
            frame_ids=[frame_ids],
            outputs=sentence_outputs,
            targets=sentence_targets,
            lengths=[len(frame_ids)],
            phonemes=[sentence_tokens],
            articulators=tv_articulators,
            save_to=save_to,
            regularize_out=True
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
