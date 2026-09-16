"""Bootstrap and permutation helpers, ridge and walk-forward fits."""
import numpy as np


def bootstrap_ci(x, stat=np.mean, B=2000, alpha=0.05, seed=7):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    if len(x) < 2:
        return float(stat(x)) if len(x) else float("nan"), float("nan"), float("nan")
    s = np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(B)])
    return float(stat(x)), float(np.quantile(s, alpha / 2)), float(np.quantile(s, 1 - alpha / 2))


def paired_ratio_ci(a, b, B=2000, seed=7):
    """bootstrap CI of std(a)/std(b) over paired observations"""
    a = np.asarray(a, float); b = np.asarray(b, float); rng = np.random.default_rng(seed); n = len(a)
    r = np.array([np.std(a[i]) / max(np.std(b[i]), 1e-12) for i in (rng.integers(0, n, n) for _ in range(B))])
    return float(np.std(a) / max(np.std(b), 1e-12)), float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))


def corr_ci(x, y, B=2000, seed=7):
    x = np.asarray(x, float); y = np.asarray(y, float); rng = np.random.default_rng(seed); n = len(x)
    if n < 4:
        return float("nan"), float("nan"), float("nan")
    c = float(np.corrcoef(x, y)[0, 1])
    s = np.array([np.corrcoef(x[i], y[i])[0, 1] for i in (rng.integers(0, n, n) for _ in range(B))])
    return c, float(np.nanquantile(s, 0.025)), float(np.nanquantile(s, 0.975))


def permutation_p(x, y, n=2000, seed=7):
    """two-sided permutation p-value of the correlation"""
    x = np.asarray(x, float); y = np.asarray(y, float); rng = np.random.default_rng(seed)
    if len(x) < 4:
        return float("nan")
    c = abs(np.corrcoef(x, y)[0, 1]); cnt = 0
    for _ in range(n):
        if abs(np.corrcoef(x, rng.permutation(y))[0, 1]) >= c:
            cnt += 1
    return (cnt + 1) / (n + 1)


def ridge(X, y, lam=1e-3):
    X = np.asarray(X, float); y = np.asarray(y, float)
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def walk_forward(X, y, window, lam=1e-3):
    """out-of-sample predictions with a trailing window: prediction for t uses rows t-window..t-1"""
    X = np.asarray(X, float); y = np.asarray(y, float); n = len(y); pred = np.full(n, np.nan)
    for t in range(window, n):
        b = ridge(X[t - window:t], y[t - window:t], lam); pred[t] = X[t] @ b
    return pred
