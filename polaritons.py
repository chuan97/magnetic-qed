import numpy as np


def normal(w1, w2, lam):
    A = w1**2 + w2**2
    B = np.sqrt((w1**2 - w2**2) ** 2 + 16 * lam**2 * w1 * w2)

    return np.sqrt(0.5 * (A - B)), np.sqrt(0.5 * (A + B))


def super_dicke(wz, wc, lam):
    mu = wz * wc / (4 * lam**2)

    A = (wz / mu) ** 2 + wc**2
    B = np.sqrt(((wz / mu) ** 2 - wc**2) ** 2 + 4 * wc**2 * wz**2)

    return np.sqrt(0.5 * (A - B)), np.sqrt(0.5 * (A + B))


def dicke(wz, wc, lam):
    if 4 * lam**2 < wz * wc:
        # paramagnetic phase
        return normal(wz, wc, lam)
    else:
        # ferromagnetic phase
        return super_dicke(wz, wc, lam)
