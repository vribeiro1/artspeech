import numpy as np
import math as m
import matplotlib.pyplot as plt
import random as r

# Pour un reel tl de [a,b], calcule
# l'indice i tel que tl in [u[i],u[i+1] et calcule les valeurs des fonctions
# de base des splines non nulles en tl
# la valeur retournee est l'indice i

def bspline (tl, bs, a, b, n, degree):
    # step calculation
    h = (b-a) / n
    r = (tl-a) / h
    i = m.floor(r)

    if tl == b:
        i = n-1

    # compute relative coordinates in [u[i],u[i+1]]
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
        bs[3] = q*q*q/6
        bs[2] = (3*q*(p*q+1)+1) / 6
        bs[1] = (3*p*(p*q+1)+1) / 6
        bs[0] = p*p*p/6
#    print("a b ", a, b, "tl ", tl, "i =", i, "bs[0] =", bs[0])
    return i


def solveSystemCholesky(M, B):
    L = np.linalg.cholesky(M)
    X = [0 for x in range(len(M))]
    for j in range(0, len(M)):
        s = 0.
        for k in range(0, j):
           s += L[j][k] * X[k]
        X[j] = (B[j]-s) / L[j][j]

    for j in range (len(M) - 1, -1, -1):
        s = 0.
        for k in range (j + 1, len(M)):
            s += L[k][j] * X[k]
        X[j] = (X[j]-s) / L[j][j]
    return X


def bsreg(
    T, # array of the data abscissa
    Z,  # array of the data ordinates
    degree,  # degreee of the interpolation polynom
    a,     # interval
    b,
    n, # number of sub intervals in [a, b]
    tau, # regularizing parameter
    nres, # number of points where the spline function is computed
    resampling): # True if resampling of the curve , false otherwise

    # X, Y --> coordinates of the discretized resulting spline

    if (len(T) != len(Z)):
        return [],[]

    nd = len(T) # number of input data
    for j in range(0,5):
        pass

    degree = 2
    c = np.array([0,0,0,0], dtype=float)
    if degree == 1:
         c[0] = 1
         c[1] = -1
    elif degree == 2:
        c[0] = 1
        c[1] = -2
        c[2] = 1
    elif degree == 3:
        c[0] = 1
        c[1] = -3
        c[2] = 3
        c[3] = - 1

    # Formation de la matrice M et du second membre beta
    # partie moindre carres */
    bs = np.array([0,0,0,0], dtype = float)
    beta = np.array([0], dtype = float)
    beta = np.resize(beta, n + degree)
    M = np.array([[0, 0], [0, 0]], dtype = float)
    M = np.resize(M, (n + degree, n + degree))
    for l in range(nd):
        if T[l] < a or T[l] > b: status = 1
        i = bspline (T[l], bs, a, b, n, degree)
        for j0 in range(degree, -1, -1):
            j1 = i - j0 + degree
            beta[j1] += Z[l] * bs[j0]
            for k0 in range(degree, j0-1, -1):
                k1 = i - k0 + degree
                M[j1][k1] += bs[j0] * bs[k0]

    for i in range(0, n + degree):
        if m.fabs(M[i][j]) < 1.e-10: status = 1

    # partie regularisation
    for i in range(0, n):
        for j0 in range(0, degree + 1):
            for k0 in range (0, j0 + 1):
                M[i+j0][i+k0] += tau * c[j0] * c[k0]

    # alfa: coefficients on the spline base
    # resolution du systeme lineaire
    alfa = solveSystemCholesky(M, beta)

    # evaluation de la spline
    Tspline = np.array([0], dtype = float)
    Zspline = np.array([0], dtype = float)
    if resampling:
        hs = (b - a) / (nres) # - 1
        Tspline = np.resize(Tspline, nres)
        Zspline = np.resize(Zspline, nres)
        for l in range(0, nres):
            tl = a + l * hs
            i = bspline(tl, bs, a, b, n, degree)
            som = 0.
            for j0 in range(0, degree + 1):
                som += alfa[i + j0] * bs[degree - j0]
            Tspline[l] = tl
            Zspline[l] = som
    else:
        Xspline.resize(nd)
        Yspline.resize(nd)
        for l in range (0, nd):
            tl = T[l]
            i = bspline(tl, bs, a, b, n, degree)
            som = 0.
            for j0 in range(0, degree + 1):
                som += alfa[i+j0] * bs[degree-j0]
            Tspline[l] = tl
            Zspline[l] = som
    return Tspline, Zspline


def generateArcPointCloud(R, N, noise):
    points = []
    for j in range(N + 1):
        i = j #r.randint(0, N) #to check the effect of the index
        p = (R * m.cos(i / N * m.pi + m.pi/2) + noise * r.random(),
             R * m.sin(i / N * m.pi + m.pi/2) + noise * r.random())
        points.append(p)
    return points


def regularize_Bsplines(curve, degree) :
    T = np.zeros((len(curve),), dtype = float)
    X = np.zeros((len(curve),), dtype = float)
    Y = np.zeros((len(curve),), dtype = float)
    for i in range(0, len(curve)):
        T[i] = i
        X[i] = curve[i][0]
        Y[i] = curve[i][1]
    resT, resX = bsreg(T, X, degree, 0., len(curve), 20, 0.1, 50, True)
    resT, resY = bsreg(T, Y, degree, 0., len(curve), 20, 0.1, 50, True)
    return resX, resY
