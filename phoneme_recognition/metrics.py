import heapq
import numpy as np
import torch
import torch.nn as nn
import ujson

from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, MulticlassF1Score
from torchmetrics.functional import word_error_rate, word_information_lost
from typing import Union, List


class MetricsMixin:
    @classmethod
    def get_pad_mask(
        cls,
        inputs,
        lengths
    ):
        if len(inputs.shape) == 2:
            bs, time = inputs.shape
        elif len(inputs.shape) == 3:
            bs, time, _ = inputs.shape
        else:
            raise Exception("invalid inputs")

        cumsum = torch.cumsum(torch.ones((bs, time)), dim=1)
        lengths = lengths.repeat(time, 1).T
        pad_mask = (cumsum <= lengths).type(torch.long)
        pad_mask = pad_mask.to(inputs.device)
        return pad_mask

    @classmethod
    def prepare_inputs(
        cls,
        emissions,
        targets,
        emissions_lengths,
        targets_lengths,
        detach_inputs=True
    ):
        if detach_inputs:
            emissions = emissions.detach().cpu()
            targets = targets.detach().cpu()  # (B, T)

        emissions_pad_mask = cls.get_pad_mask(emissions, emissions_lengths)
        emissions_pad_mask = torch.flatten(emissions_pad_mask, start_dim=0, end_dim=1)
        emissions = torch.flatten(emissions, start_dim=0, end_dim=1)
        emissions = emissions[emissions_pad_mask == 1]

        targets_pad_mask = cls.get_pad_mask(targets, targets_lengths)
        targets_pad_mask = torch.flatten(targets_pad_mask, start_dim=0, end_dim=1)
        targets = torch.flatten(targets, start_dim=0, end_dim=1)
        targets = targets[targets_pad_mask == 1]

        return emissions, targets

    @classmethod
    def make_pred_and_target_sentences(
        cls,
        decoder,
        emissions,
        targets,
        emissions_lengths,
        targets_lengths,
        detach_inputs=True
    ):
        if detach_inputs:
            emissions = emissions.detach().cpu()
            targets = targets.detach().cpu()  # (B, T)

        target_sequences = []
        for target, length in zip(targets, targets_lengths):
            target_no_pad = target[:length]
            tokens = [str(token.item()) for token in target_no_pad]
            target_sequences.append(" ".join(tokens))

        results = decoder(emissions, emissions_lengths)
        pred_sequences = []
        for result in results:
            best_hyp = result[0]
            tokens = [str(token.item()) for token in best_hyp.tokens]
            pred_sequences.append(" ".join(tokens))

        return pred_sequences, target_sequences


class CrossEntropyLoss(nn.Module, MetricsMixin):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if class_weights is not None:
            if isinstance(class_weights, str):
                with open(class_weights) as f:
                    class_weights = ujson.load(f)

            # Add a 1.0 weight for the unknown class
            weight = torch.tensor(
                [1.0] +
                [w for _, w in sorted(
                    class_weights.items(),
                    key=lambda t: t[0]
                )]
            ).to(device)
        else:
            weight = None

        self.ce = nn.CrossEntropyLoss(*args, weight=weight, **kwargs)

    def forward(self, emissions, targets, emissions_lengths, targets_lengths):
        emissions = emissions.permute(1, 0, 2)
        emissions, targets = self.prepare_inputs(
            emissions,
            targets,
            emissions_lengths,
            targets_lengths,
            detach_inputs=False
        )
        out = self.ce(emissions, targets)
        return out


class EditDistance(MetricsMixin):
    def __init__(self, decoder):
        self.decoder = decoder

    def __call__(self, emissions, targets, input_lengths, target_lengths):
        pred_sequences, target_sequences = self.make_pred_and_target_sentences(
            self.decoder,
            emissions,
            targets,
            input_lengths,
            target_lengths
        )
        edit_dist = word_error_rate(pred_sequences, target_sequences)
        return edit_dist


class WordInfoLost(MetricsMixin):
    def __init__(self, decoder):
        self.decoder = decoder

    def __call__(self, emissions, targets, input_lengths, target_lengths):
        pred_sequences, target_sequences = self.make_pred_and_target_sentences(
            self.decoder,
            emissions,
            targets,
            input_lengths,
            target_lengths
        )
        wil = word_information_lost(pred_sequences, target_sequences)
        return wil


class F1Score(MetricsMixin):
    def __init__(self, num_classes, average="macro"):
        self.f1_score = MulticlassF1Score(num_classes=num_classes, average=average)

    def __call__(self, emissions, targets, emissions_lengths, targets_lengths):
        emissions, targets = self.prepare_inputs(
            emissions,
            targets,
            emissions_lengths,
            targets_lengths
        )
        f1 = self.f1_score(emissions, targets)
        return f1


class Accuracy(MetricsMixin):
    def __init__(self, num_classes, average="macro"):
        self.accuracy = MulticlassAccuracy(num_classes=num_classes, average=average)

    def __call__(self, emissions, targets, emissions_lengths, targets_lengths):
        emissions, targets = self.prepare_inputs(
            emissions,
            targets,
            emissions_lengths,
            targets_lengths
        )
        acc = self.accuracy(emissions, targets)
        return acc


class AUROC(MetricsMixin):
    def __init__(self, num_classes, average="macro"):
        self.auroc = MulticlassAUROC(num_classes=num_classes, average=average)

    def __call__(self, emissions, targets, emissions_lengths, targets_lengths):
        emissions, targets = self.prepare_inputs(
            emissions,
            targets,
            emissions_lengths,
            targets_lengths
        )
        auc = self.auroc(emissions, targets)
        return auc


def edit_matrix(prediction_tokens, reference_tokens):
    """
    Edit distance matrix from torchmetrics.
    """
    dp = [[0] * (len(reference_tokens) + 1) for _ in range(len(prediction_tokens) + 1)]
    for i in range(len(prediction_tokens) + 1):
        dp[i][0] = i
    for j in range(len(reference_tokens) + 1):
        dp[0][j] = j
    for i in range(1, len(prediction_tokens) + 1):
        for j in range(1, len(reference_tokens) + 1):
            if prediction_tokens[i - 1] == reference_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp


def shortest_path(M):
    """
    Dijkstra's shortest path for a matrix.
    """
    rows = len(M)
    cols = len(M[0])

    orig = (0, 0)
    dest = (rows - 1, cols - 1)
    nodes = set((i, j) for i in range(rows) for j in range(cols))
    shortest_path_table = {
        node: (0 if node == orig else np.inf, node, None) for node in nodes
    }

    visited = set()
    queue = [(0, orig)]
    heapq.heapify(queue)

    while queue:
        dist_from_orig, curr_node = heapq.heappop(queue)

        if curr_node in visited:
            continue

        visited.add(curr_node)
        curr_i, curr_j = curr_node
        neighbors = [
            (curr_i + 1, curr_j),
            (curr_i, curr_j + 1),
            (curr_i + 1, curr_j + 1)
        ]

        for neighbor in neighbors:
            if neighbor not in shortest_path_table:
                continue

            neighbor_i, neighbor_j = neighbor
            weight = M[neighbor_i][neighbor_j]
            neighbor_dist_from_orig = dist_from_orig + weight

            if neighbor_dist_from_orig < shortest_path_table[neighbor][0]:
                shortest_path_table[neighbor] = neighbor_dist_from_orig, neighbor, curr_node

            if neighbor not in visited:
                heapq.heappush(queue, (weight, neighbor))

    path = []
    previous = dest
    while previous:
        path = [previous] + path
        previous = shortest_path_table[previous][-1]

    return path


def _compute_transitions(path):
    """
    Compute the deletions, insertions and substitutions for one path.
    """
    deletions = []
    insertions = []
    substitutions = []

    for curr_node, next_node in zip(path[:-1], path[1:]):
        curr_i, curr_j = curr_node
        next_i, next_j = next_node

        if curr_i == next_i:
            deletions.append(curr_j)
        elif curr_j == next_j:
            insertions.append(curr_i)
        else:
            substitutions.append((curr_j, curr_i))

    return deletions, insertions, substitutions


def compute_transitions(
    preds: Union[str, List[str]],
    target: Union[str, List[str]]
):
    """
    >>> targets = ["a b c", "a b c", "a b c", "a b d e a",]
    >>> preds = ["a b c", "b c", "a b c d", "c b d e",]
    >>> expected = [([], [], [(0, 0), (1, 1), (2, 2)]), ([0], [], [(1, 0), (2, 1)]), ([], [3], [(0, 0), (1, 1), (2, 2)]), ([4], [], [(0, 0), (1, 1), (2, 2), (3, 3)])]
    >>> all_transitions = compute_transitions(preds, targets)
    all_transitions == expected
    """
    if isinstance(preds, str):
        preds = [preds]
    if isinstance(target, str):
        target = [target]

    all_transitions = []
    for pred, tgt in zip(preds, target):
        pred = pred.split()
        tgt = tgt.split()

        dp = edit_matrix(pred, tgt)
        path = shortest_path(dp)
        transitions = _compute_transitions(path)
        all_transitions.append(transitions)

    return all_transitions


def substitution_matrix(
    preds: Union[str, List[str]],
    target: Union[str, List[str]],
    vocab: List[str],
    insertions_and_deletions: str = None,
    normalize: str = None,
):
    """
    Compute the substitution matrix between the predictions and the targets. The substitution matrix
    is equivalente to a confusion matrix for regular classification. The x-axis holds the predictions
    and the y-axis holds the targets. The main diagonal represents correct transcriptions, and the
    other cells represents the token i replaced by token j. An extra row or column can be added to
    represent the number of insertions and deletions of a token.
    """
    if isinstance(preds, str):
        preds = [preds]
    if isinstance(target, str):
        target = [target]

    rows = len(vocab)
    include_insertions = False
    if insertions_and_deletions in ["insertions", "both"]:
        include_insertions = True
        rows += 1

    cols = len(vocab)
    include_deletions = False
    if insertions_and_deletions in ["deletions", "both"]:
        include_deletions = True
        cols += 1

    cm = np.zeros(shape=(len(vocab) + 1, len(vocab) + 1))
    all_transitions = compute_transitions(preds, target)
    for pred, tgt, (deletions, insertions, substitutions) in zip(preds, target, all_transitions):
        pred = pred.split()
        tgt = tgt.split()

        for i, j in substitutions:
            tgt_token = tgt[i]
            pred_token = pred[j]

            tgt_token_index = vocab.index(tgt_token)
            pred_token_index = vocab.index(pred_token)

            cm[tgt_token_index, pred_token_index] += 1

        if include_deletions:
            for i in deletions:
                tgt_token = tgt[i]
                tgt_token_index = vocab.index(tgt_token)
                cm[tgt_token_index, -1] += 1

        if include_insertions:
            for j in insertions:
                pred_token = pred[j]
                pred_token_index = vocab.index(pred_token)
                cm[-1, pred_token_index] += 1


    with np.errstate(all="ignore"):
        if normalize == "true":
            cm = cm / cm.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            cm = cm / cm.sum(axis=0, keepdims=True)
        elif normalize == "all":
            cm = cm / cm.sum()
        cm = np.nan_to_num(cm)

    return cm
