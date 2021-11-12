import funcy
import numpy as np
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
