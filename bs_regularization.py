import numpy as np
import math


def bspline (tl, bs, a, b, n, degree):
    """
    For a real number tl in the interval [a, b], calculate the index i such that tl is in
    [u[i], u[i + 1]] and calculate the base function values of the non-null splines of tl that the
    return is the index i.
    """
    # step calculation
    h = (b - a) / n
    r = (tl - a) / h
    i = math.floor(r)

    if tl == b:
        i = n - 1

    # compute relative coordinates in [u[i], u[i+1]]
    p = r - i
    q = 1 - p

    # determination of the non zero spline functions at tl
    if degree == 1:
        bs[1] = q
        bs[0] = p
    elif degree == 2:
        bs[2] = q * q / 2
        bs[1] = (2 * p * q + 1) /2
        bs[0] = p * p / 2
    elif degree == 3:
        bs[3] = q * q * q / 6
        bs[2] = (3*q*(p*q+1)+1) / 6
        bs[1] = (3*p*(p*q+1)+1) / 6
        bs[0] = p * p * p / 6

    return i


def solve_system_cholesky(M, B):
    L = np.linalg.cholesky(M)
    m, _ = M.shape
    X = np.zeros(shape=m)

    for j in range(0, m):
        s = 0.
        for k in range(0, j):
           s += L[j][k] * X[k]
        X[j] = (B[j] - s) / L[j][j]

    for j in range (m - 1, -1, -1):
        s = 0.
        for k in range(j + 1, m):
            s += L[k][j] * X[k]

        X[j] = (X[j] - s) / L[j][j]

    return X


def bsreg(T, Z, degree, a, b, n, tau, nres, resampling):
    """
    Args:
    T (np.ndarray): Array of the data in the abscissa
    Z (np.ndarray): Array of the data in the ordinates
    degree (int): Degree of the interpolation polynom
    a (float): Start of the interval
    b (float): End of the interval
    n (int): Number of sub-intervals in [a, b]
    tau (float): Regularizing parameter
    nres (int): Number of points for spline computation
    resampling (bool): True if resampling the curve, false otherwise
    """
    if (len(T) != len(Z)):
        raise Exception("Dimensions of T and Z must match.")
    nd = len(T)  # number of input data points

    degree = 2
    c = np.array([0, 0, 0, 0], dtype=float)
    if degree == 1:
        c = np.array([1, -1, 0, 0], dtype=float)
    elif degree == 2:
        c = np.array([1, -2, 1, 0], dtype=float)
    elif degree == 3:
        c = np.array([1, -3, 3, -1], dtype=float)
    else:
        raise Exception(f"Polynom degree must be 1, 2, or 3. {degree} was given.")

    # Formation de la matrice M et du second membre beta partie moindre carres
    bs = np.array([0, 0, 0, 0], dtype=float)
    beta = np.array([0], dtype=float)
    beta = np.resize(beta, n + degree)

    M = np.zeros(shape=(n + degree, n + degree), dtype=float)
    for l in range(nd):
        i = bspline (T[l], bs, a, b, n, degree)
        for j0 in range(degree, -1, -1):
            j1 = i + degree - j0
            beta[j1] += Z[l] * bs[j0]
            for k0 in range(degree, j0 - 1, -1):
                k1 = i - k0 + degree
                M[j1][k1] += bs[j0] * bs[k0]

    # Partie regularisation
    for i in range(n):
        for j0 in range(degree + 1):
            for k0 in range(j0 + 1):
                M[i  + j0][i + k0] += tau * c[j0] * c[k0]

    # alfa: coefficients on the spline base
    # resolution du systeme lineaire
    alfa = solve_system_cholesky(M, beta)

    # Evaluation de la spline
    Tspline = np.array([0], dtype=float)
    Zspline = np.array([0], dtype=float)
    if resampling:
        hs = (b - a) / nres  # -1
        Tspline = np.resize(Tspline, nres)
        Zspline = np.resize(Zspline, nres)
        for l in range(0, nres):
            tl = a + l * hs
            i = bspline(tl, bs, a, b, n, degree)
            som = 0.
            for j0 in range(degree + 1):
                som += alfa[i + j0] * bs[degree - j0]
            Tspline[l] = tl
            Zspline[l] = som
    else:
        Tspline.resize(nd)
        Zspline.resize(nd)
        for l in range (0, nd):
            tl = T[l]
            i = bspline(tl, bs, a, b, n, degree)
            som = 0.
            for j0 in range(degree + 1):
                som += alfa[i + j0] * bs[degree - j0]
            Tspline[l] = tl
            Zspline[l] = som

    return Tspline, Zspline


def regularize_Bsplines(curve, degree, n_samples=50):
    T = np.arange(len(curve), dtype=float)
    X = curve[:, 0].astype(float)
    Y = curve[:, 1].astype(float)

    _, resX = bsreg(T, X, degree, 0., len(curve), 20, 0.1, n_samples, True)
    _, resY = bsreg(T, Y, degree, 0., len(curve), 20, 0.1, n_samples, True)
    return resX, resY
