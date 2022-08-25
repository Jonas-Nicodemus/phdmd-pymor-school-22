import numpy as np

from pymor.algorithms.to_matrix import to_matrix

def implicit_midpoint(lti, U, T, x0):
    delta = T[1] - T[0]

    M = lti.E - delta / 2 * lti.A
    A = lti.E + delta / 2 * lti.A

    X = np.zeros((lti.order, len(T)))
    X[:, 0] = x0

    for i in range(len(T) - 1):
        U_midpoint = 1 / 2 * (U[:, i] + U[:, i + 1])
        X[:, i + 1] = np.linalg.solve(to_matrix(M), to_matrix(A) @ X[:, i] + delta * to_matrix(lti.B) @ U_midpoint)

    Y = to_matrix(lti.C) @ X + to_matrix(lti.D) @ U

    return U, X, Y

def sim(lti, U, T=None, x0=None):
    if T is None:
        assert isinstance(U, np.ndarray)
        T = np.linspace(0, len(U))
    if not isinstance(U, np.ndarray):
        U = U(T)
        if U.ndim < 2:
            U = U[np.newaxis, :]

    if x0 is None:
        x0 = np.zeros(lti.order)

    return implicit_midpoint(lti, U, T, x0)