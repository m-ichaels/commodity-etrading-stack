"""numpy port of include/cetk/curve.hpp: the three futures-curve models, leave-one-out, key-rate sensitivities and
the calendar-month-average swap.  Same weights, grids and jitter as the C++, so the two agree to numerical precision
(checked by `python -m cetk compare` and in CI)."""
import datetime as dt
import math

import numpy as np

from . import calendar as cal

NSS_TAU1 = (0.25, 0.5, 1.0, 2.0)
NSS_TAU2 = (3.0, 5.0, 10.0)


def ridge_solve(G, y, w, P):
    G = np.asarray(G, float); y = np.asarray(y, float); w = np.asarray(w, float)
    M = (G * w[:, None]).T @ G
    if P is not None:
        M = M + P
    M = M + np.diag(1e-12 * (1 + np.abs(np.diag(M))))
    v = (G * w[:, None]).T @ y
    return np.linalg.solve(M, v)


def nss_basis(T, phi, tau1, tau2, harm=2):
    T = np.maximum(np.asarray(T, float), 1e-6); phi = np.asarray(phi, float)
    x1, x2 = T / tau1, T / tau2
    e1, e2 = np.exp(-x1), np.exp(-x2)
    l1, l2 = (1 - e1) / x1, (1 - e2) / x2
    cols = [np.ones_like(T), l1, l1 - e1, l2 - e2]
    for k in range(1, harm + 1):
        cols.append(np.cos(2 * np.pi * k * phi)); cols.append(np.sin(2 * np.pi * k * phi))
    return np.column_stack(cols)


def bspline_basis(knots, x, k=3):
    """Cox-de Boor cubic B-splines on the (already extended) knot vector, evaluated at x (array)."""
    x = np.asarray(x, float); nb = len(knots) - 4
    B = np.zeros((len(x), len(knots) - 1))
    for i in range(len(knots) - 1):
        B[:, i] = ((x >= knots[i]) & (x < knots[i + 1])).astype(float)
    for d in range(1, k + 1):
        Bn = np.zeros((len(x), len(knots) - 1 - d))
        for i in range(len(knots) - 1 - d):
            a = (knots[i + d] - knots[i]); b = (knots[i + d + 1] - knots[i + 1])
            ta = ((x - knots[i]) / a * B[:, i]) if a > 0 else 0
            tb = ((knots[i + d + 1] - x) / b * B[:, i + 1]) if b > 0 else 0
            Bn[:, i] = ta + tb
        B = Bn
    return B[:, :nb]


class CurveModel:
    def __init__(self, method="kernel_ridge", lam=1e-3, theta=1.0, ell=0.5, knot_spacing=1.0 / 12, harmonics=2, liquidity_weights=True):
        self.method, self.lam, self.theta, self.ell, self.knot_spacing, self.harmonics, self.lw = method, lam, theta, ell, knot_spacing, harmonics, liquidity_weights

    # kernels
    @staticmethod
    def k_sob(s, t):
        a = np.minimum(s, t); b = np.maximum(s, t)
        return a * a * (3 * b - a) / 6.0

    def k_per(self, p, q):
        sn = np.sin(np.pi * (p - q))
        return np.exp(-2 * sn * sn / (self.ell * self.ell))

    def kern(self, T, phi, T2, phi2):
        return self.k_sob(T[:, None], T2[None, :]) + self.theta * self.k_per(phi[:, None], phi2[None, :])

    def fit(self, T, phi, price):
        T = np.asarray(T, float); phi = np.asarray(phi, float); y = np.asarray(price, float); n = len(T)
        w = 1.0 / (1.0 + T) if self.lw else np.ones(n)
        if self.method == "seasonal_nss":
            best = None
            for t1 in NSS_TAU1:
                for t2 in NSS_TAU2:
                    G = nss_basis(T, phi, t1, t2, self.harmonics); P = np.eye(G.shape[1]) * 1e-6
                    c = ridge_solve(G, y, w, P); sse = float(np.sum(w * (y - G @ c) ** 2))
                    if best is None or sse < best[0]:
                        best = (sse, c, t1, t2)
            self.coef, self.tau1, self.tau2 = best[1], best[2], best[3]
        elif self.method == "pspline":
            tmax = T.max(); t_hi = tmax + self.knot_spacing; h = self.knot_spacing; nk = int(math.ceil((t_hi - 0) / h)) + 1
            self.knots = np.array([i * h for i in range(-3, nk + 3)]); self.t_lo = 0.0; self.t_hi = self.knots[-4]
            G = self.pspline_basis(T); m = G.shape[1]; P = np.zeros((m, m)); d = np.array([1.0, -2.0, 1.0])
            for i in range(m - 2):
                P[i:i + 3, i:i + 3] += self.lam * np.outer(d, d)
            self.coef = ridge_solve(G, y, w, P)
        else:
            self.Tj, self.Pj = T.copy(), phi.copy(); K = self.kern(T, phi, T, phi)
            G = np.column_stack([K, np.ones(n), T]); P = np.zeros((n + 2, n + 2)); P[:n, :n] = self.lam * K
            u = ridge_solve(G, y, w, P); self.w, self.a0, self.b0 = u[:n], u[n], u[n + 1]
        pred = self.predict(T, phi); self.in_sample_rmse = float(np.sqrt(np.mean((pred - y) ** 2))); self.max_abs_err = float(np.max(np.abs(pred - y)))
        return self

    def pspline_basis(self, T):
        x = np.clip(np.asarray(T, float), self.t_lo, self.t_hi - 1e-9)
        return bspline_basis(self.knots, x)

    def predict(self, T, phi):
        T = np.atleast_1d(np.asarray(T, float)); phi = np.atleast_1d(np.asarray(phi, float))
        if self.method == "seasonal_nss":
            return nss_basis(T, phi, self.tau1, self.tau2, self.harmonics) @ self.coef
        if self.method == "pspline":
            return self.pspline_basis(T) @ self.coef
        return self.a0 + self.b0 * T + self.kern(T, phi, self.Tj, self.Pj) @ self.w


def leave_one_out(T, phi, price, months=None, **spec):
    T = np.asarray(T, float); phi = np.asarray(phi, float); price = np.asarray(price, float); n = len(T)
    err = np.zeros(n)
    for k in range(n):
        idx = np.arange(n) != k
        m = CurveModel(**spec).fit(T[idx], phi[idx], price[idx]); err[k] = float(m.predict(T[k], phi[k])[0] - price[k])
    out = {"err": err, "mean_abs": float(np.mean(np.abs(err))), "median_abs": float(np.median(np.abs(err))), "mean_abs_pct": float(np.mean(np.abs(100 * err / price)))}
    if months is not None:
        months = np.asarray(months); inter = (months > months.min()) & (months < months.max())
        out["mean_abs_interior"] = float(np.mean(np.abs(err[inter]))) if inter.any() else float("nan")
    return out


def key_rate_sensitivities(T, phi, price, Tq, phiq, **spec):
    """Bump each input by 1 and refit (NSS shape frozen at the base fit): the hat-matrix row of the prediction."""
    T = np.asarray(T, float); phi = np.asarray(phi, float); price = np.asarray(price, float)
    base = CurveModel(**spec).fit(T, phi, price); p0 = base.predict(Tq, phiq)[0]; s = np.zeros(len(T))
    for i in range(len(T)):
        up = price.copy(); up[i] += 1.0
        if spec.get("method") == "seasonal_nss":
            w = 1.0 / (1.0 + T) if base.lw else np.ones(len(T)); G = nss_basis(T, phi, base.tau1, base.tau2, base.harmonics)
            c = ridge_solve(G, up, w, np.eye(G.shape[1]) * 1e-6); pi = float(nss_basis(np.array([Tq]), np.array([phiq]), base.tau1, base.tau2, base.harmonics) @ c)
        else:
            pi = float(CurveModel(**spec).fit(T, phi, up).predict(Tq, phiq)[0])
        s[i] = pi - p0
    return s


def calendar_swap(root, date, points, model, y, m):
    """points: dict delivery 'YYYY-MM' -> price observed on `date`; model predicts missing deliveries."""
    d0 = dt.date(y, m, 1); ny, nm = (y, m + 1) if m < 12 else (y + 1, 1); d1 = dt.date(ny, nm, 1)
    counts = {}; nbd = 0; d = d0
    while d < d1:
        if cal.is_bd(d):
            py, pm = cal.nth_delivery(root, d, 1); counts[f"{py}-{pm:02d}"] = counts.get(f"{py}-{pm:02d}", 0) + 1; nbd += 1
        d += dt.timedelta(days=1)
    price = 0.0; legs = []
    for dely, c in sorted(counts.items()):
        wgt = c / nbd
        if dely in points:
            p = points[dely]; obs = True
        else:
            dy, dm = map(int, dely.split("-")); p = float(model.predict(cal.years_to_expiry(root, date, dy, dm), cal.season_phase(dm))[0]); obs = False
        price += wgt * p; legs.append((dely, wgt, p, obs))
    return price, legs, nbd
