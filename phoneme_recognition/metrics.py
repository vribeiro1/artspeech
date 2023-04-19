import pdb
import torch
import torch.nn as nn
import ujson

from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC
from torchmetrics.functional import word_error_rate


class CrossEntropyLoss(nn.Module):
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

    @staticmethod
    def get_pad_mask(inputs, lengths):
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

    def forward(self, inputs, targets, inputs_lengths, targets_lengths):
        inputs = inputs.permute(1, 0, 2)

        inputs_pad_mask = self.get_pad_mask(inputs, inputs_lengths)
        inputs_pad_mask = torch.flatten(inputs_pad_mask, start_dim=0, end_dim=1)
        inputs = torch.flatten(inputs, start_dim=0, end_dim=1)
        inputs = inputs[inputs_pad_mask == 1]

        targets_pad_mask = self.get_pad_mask(targets, targets_lengths)
        targets_pad_mask = torch.flatten(targets_pad_mask, start_dim=0, end_dim=1)
        targets = torch.flatten(targets, start_dim=0, end_dim=1)
        targets = targets[targets_pad_mask == 1]
        out = self.ce(inputs, targets)

        return out


class EditDistance:
    def __init__(self, decoder):
        self.decoder = decoder

    def __call__(self, emissions, targets, input_lengths, target_lengths):
        emissions = emissions  # (B, T, C)
        emissions = emissions.detach().cpu()

        targets = targets.detach().cpu()  # (B, T)
        target_sequences = []
        for target, length in zip(targets, target_lengths):
            target_no_pad = target[:length]
            tokens = [str(token.item()) for token in target_no_pad]
            target_sequences.append(" ".join(tokens))

        results = self.decoder(emissions, input_lengths)
        pred_sequences = []
        for result in results:
            best_hyp = result[0]
            tokens = [str(token.item()) for token in best_hyp.tokens]
            pred_sequences.append(" ".join(tokens))

        edit_dist = word_error_rate(pred_sequences, target_sequences)
        return edit_dist


class Accuracy:
    def __init__(self, num_classes, average="macro"):
        self.accuracy = MulticlassAccuracy(num_classes=num_classes, average=average)

    @staticmethod
    def get_pad_mask(inputs, lengths):
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

    def __call__(self, emissions, targets, inputs_lengths, targets_lengths):
        emissions = emissions.detach().cpu()
        emissions_pad_mask = self.get_pad_mask(emissions, inputs_lengths)
        emissions_pad_mask = torch.flatten(emissions_pad_mask, start_dim=0, end_dim=1)
        emissions = torch.flatten(emissions, start_dim=0, end_dim=1)
        emissions = emissions[emissions_pad_mask == 1]

        targets = targets.detach().cpu()  # (B, T)
        targets_pad_mask = self.get_pad_mask(targets, targets_lengths)
        targets_pad_mask = torch.flatten(targets_pad_mask, start_dim=0, end_dim=1)
        targets = torch.flatten(targets, start_dim=0, end_dim=1)
        targets = targets[targets_pad_mask == 1]

        acc = self.accuracy(emissions, targets)
        return acc


class AUROC:
    def __init__(self, num_classes, average="macro"):
        self.auroc = MulticlassAUROC(num_classes=num_classes, average=average)

    @staticmethod
    def get_pad_mask(inputs, lengths):
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

    def __call__(self, emissions, targets, inputs_lengths, targets_lengths):
        emissions = emissions  # (B, T, C)

        emissions = emissions.detach().cpu()
        emissions_pad_mask = self.get_pad_mask(emissions, inputs_lengths)
        emissions_pad_mask = torch.flatten(emissions_pad_mask, start_dim=0, end_dim=1)
        emissions = torch.flatten(emissions, start_dim=0, end_dim=1)
        emissions = emissions[emissions_pad_mask == 1]

        targets = targets.detach().cpu()  # (B, T)
        targets_pad_mask = self.get_pad_mask(targets, targets_lengths)
        targets_pad_mask = torch.flatten(targets_pad_mask, start_dim=0, end_dim=1)
        targets = torch.flatten(targets, start_dim=0, end_dim=1)
        targets = targets[targets_pad_mask == 1]

        auc = self.auroc(emissions, targets)
        return auc
