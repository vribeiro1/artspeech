import pdb

import numpy as np
import torch
import time

from numba import jit


@jit(nopython=True)
def euclidean(u, v):
    x_u, y_u = u
    x_v, y_v = v
    return np.sqrt((x_u - x_v) ** 2 + (y_u - y_v) ** 2)


@jit(nopython=True)
def p2cp(i, u, v):
    ui = u[i]
    ui2cp = min(euclidean(ui, vj) for vj in v)
    return ui2cp


@jit(nopython=True)
def distance_matrix(u, v):
    n = len(u)
    m = len(v)

    dist_mtx = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            dist_mtx[i][j] = euclidean(u[i], v[j])

    return dist_mtx


def p2cp_mean(u_, v_):
    # u_: [N_samples, 2]
    # v_: [N_samples, 2]
    n = len(u_)
    m = len(v_)

    dist_mtx = distance_matrix(u_, v_)

    u2cv = dist_mtx.min(axis=1)
    v2cu = dist_mtx.min(axis=0)
    mean_p2cp = (np.sum(u2cv) + np.sum(v2cu)) / (n + m)

    return mean_p2cp


def p2cp_distance(outputs, targets):
    # outputs: torch.Size([bs, seq_len, N_art, 2, N_samples])
    # targets: torch.Size([bs, seq_len, N_art, 2, N_samples])
    bs, seq_len, N_art, _, N_samples = outputs.shape

    results = torch.zeros(0, seq_len, N_art)
    for batch_out, batch_target in zip(outputs, targets):
        batch_results = torch.zeros(0, N_art)
        for seq_out, seq_target in zip(batch_out, batch_target):
            seq_results = []
            for output, target in zip(seq_out, seq_target):
                output_transpose = output.transpose(1, 0)
                target_transpose = target.transpose(1, 0)

                p2cp = p2cp_mean(output_transpose.numpy(), target_transpose.numpy())
                seq_results.append(p2cp)

            batch_results = torch.cat([batch_results, torch.tensor(seq_results).unsqueeze(dim=0)])
        results = torch.cat([results, batch_results.unsqueeze(dim=0)])

    return results
