import numpy as np

from scipy.spatial.distance import euclidean


def p2cp(i, u, v):
    ui = u[i]
    ui2cp = min(euclidean(ui, vj) for vj in v)
    return ui2cp


def p2cp_mean(u_, v_):
    n = len(u_)
    m = len(v_)

    dist_mtx = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            dist_mtx[i][j] = euclidean(u_[i], v_[j])

    u2cv = dist_mtx.min(axis=1)
    v2cu = dist_mtx.min(axis=0)
    mean_p2cp = (sum(u2cv) + sum(v2cu)) / (n + m)

    return mean_p2cp
