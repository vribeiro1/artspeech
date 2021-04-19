import numpy as np
import random
import torch


def set_seeds(worker_id):
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed + 1)
    random.seed(seed + 2)


def assert_expression(expression, exception=Exception, message=""):
    if not expression:
        raise exception(message)
