import funcy
import numpy as np
import os
import random
import torch


def set_seeds(worker_id):
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed + 1)
    random.seed(seed + 2)


def assert_expression(expression, exception=AssertionError, message=""):
    """
    Asserts if a given expression is True. If it is False, raises exception with a message.

    Args:
    expression (Any): Expression to be evaluated.
    exception (type): Exception class to raise.
    message (str): Exception message.
    """
    if not expression:
        raise exception(message)


def npy_to_xarticul(array, filepath=None):
    """
    Converts a numpy array of (x, y) coordinates into a file readable by Xarticul.

    Args:
    array (np.ndarray): (N, 2) shaped numpy array with x and y coordinates.
    filepath (str): Target path to save the file.
    """
    pt_list = [f"{x} {y}" for x, y in array]

    # Add (-1, -1) in the end to tag the eof to xarticul
    pt_list.append("-1 -1")

    pt_string = "\n".join(pt_list)
    if filepath is not None:
        with open(filepath, "w") as f:
            f.write(pt_string)

    return pt_list


def xarticul_to_npy(filepath):
    """
    Converts a file readable by Xarticul to a numpy array of (x, y) coordinates.

    Args:
    filepath (str): Path of the Xarticul file.
    """
    with open(filepath, "r") as f:
        # The last line is "-1 -1" and indicates the EOF
        lines = funcy.lmap(str.strip, f.readlines())[:-1]

    data = funcy.lmap(lambda x: funcy.lmap(float, x), map(str.split, lines))
    return np.array(data)


def sequences_from_dict(datadir, sequences_dict):
    sequences = []
    for subj, seqs in sequences_dict.items():
        use_seqs = seqs
        if len(seqs) == 0:
            # Use all sequences
            use_seqs = filter(
                lambda s: os.path.isdir(os.path.join(datadir, subj, s)),
                os.listdir(os.path.join(datadir, subj))
            )

        sequences.extend([(subj, seq) for seq in use_seqs])

    return sequences


def make_padding_mask(lengths):
    """
    Make a padding mask from a tensor lengths.

    Args:
        lengths (torch.tensor): tensor of shape (B,)
    """
    bs = len(lengths)
    max_length = lengths.max()
    mask = torch.ones(size=(bs, max_length))
    mask = torch.cumsum(mask, dim=1)
    mask = mask <= lengths.unsqueeze(dim=1)
    return mask


def make_indices_dict(num_components):
    """
    Converts a dictionary of number of components per articulator into a dictionary of indices per
    articulator.

    Args:
        num_components (Dict[str, int]):
    Returns:
        indices_dict (Dict[str, list])

    >>> num_components = {'a': 3, 'b': 3, 'c': 2}
    >>> make_indices_dict(num_components)
    {'a': [0, 1, 2], 'b': [3, 4, 5], 'c': [6, 7]}
    """
    indices_dict = {}
    start = 0
    for key, val in num_components.items():
        indices_dict[key] = list(range(start, start + val))
        start = start + val

    return indices_dict
