"""PCA factor model of the curve (mirror of include/cetk/factors.hpp)."""
import numpy as np


def fit_panel(X, months, K=3):
    """X: (days, tenors) prices on a ladder -> dict with cov, evals, loadings (sign-fixed), explained."""
    X = np.asarray(X, float); D = np.diff(X, axis=0); D = D[~np.isnan(D).any(axis=1)]
    cov = np.cov(D, rowvar=False, ddof=1); evals, V = np.linalg.eigh(cov); order = np.argsort(evals)[::-1]; evals = evals[order]; V = V[:, order]
    n = len(months); L = []
    for f in range(K):
        v = V[:, f].copy()
        s = v.sum() if f == 0 else (v[-1] - v[0] if f == 1 else v[n // 2] - 0.5 * (v[0] + v[-1]))
        if s < 0:
            v = -v
        L.append(v)
    tot = evals.sum()
    return {"months": list(months), "cov": cov, "evals": evals, "loadings": np.array(L), "explained": [float(evals[:k].sum() / tot) for k in range(1, K + 1)], "n_days": len(D)}


def covariance_at(fm, tenors):
    months = np.array(fm["months"], float); S = np.zeros((len(tenors), len(tenors)))
    for f in range(len(fm["loadings"])):
        l = np.interp(tenors, months, fm["loadings"][f]); S += fm["evals"][f] * np.outer(l, l)
    return S
