import pdb
from typing import Any
import torch
import torch.nn as nn
import ujson

from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, MulticlassF1Score
from torchmetrics.functional import word_error_rate, word_information_lost


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
