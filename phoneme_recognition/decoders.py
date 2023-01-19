import funcy
import torch

from collections import namedtuple

Hypothesis = namedtuple("Hypothesis", ["tokens"])


class TopKDecoder:
    def __init__(
        self,
        tokens,
        sil_token=None,
        blank_token=None,
        unk_word=None,
        **kwargs,
    ):
        self.num_tokens = len(tokens)
        self.sil_token = sil_token
        self.blank_token = blank_token
        self.unk_word = unk_word

    def filter_blank(self, indices):
        if self.blank_token is None:
            return indices

        non_blank_indices = funcy.lfilter(lambda i: i != self.blank_token, indices)
        return non_blank_indices

    def __call__(self, emissions, lengths):
        """
        Args:
            emissions (torch.tensor): Tensor of shape (batch, time, classes) after softmax
            lengths (torch.tensor): Tensor of shape (batch,)
        """
        top_list = torch.topk(emissions, k=1, dim=-1).indices
        top_list = top_list.squeeze(dim=-1)
        results = [
            [Hypothesis(tokens=self.filter_blank(top))]
            for top in top_list
        ]
        return results
