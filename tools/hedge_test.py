#!/usr/bin/env python3
"""Hedge test on the EIA front contracts (WTI 1983-2024, Henry Hub 1994-2024): the contract-3 position hedged with
contracts 1, 2 and 4 using (a) the desk default (100% in the front), (b) 50/50 neighbours, (c) the pricer's own
key-rate sensitivities (seasonal kernel ridge fitted to the three observed contracts, i.e. what the curve implies for
an unobserved month), (d) ratios learned from co-movements (ridge on the trailing 250-day window, walk-forward).
Score: standard deviation of the realised next-day P&L of the hedged package in price units, by decade and overall,
with a paired bootstrap band on the learned / curve ratio.     python tools/hedge_test.py [--roots CL,NG] [--window 250]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "python")
from cetk import calendar as cal, curve as cv, data, stats  # noqa: E402

ARGS = sys.argv[1:]
ROOTS = [r for r in (next((a.split("=", 1)[1] for a in ARGS if a.startswith("--roots=")), "CL,NG")).split(",") if r]
WINDOW = int(next((a.split("=", 1)[1] for a in ARGS if a.startswith("--window=")), 250))
os.makedirs("results", exist_ok=True)


def run(root):
    import datetime as dt
    dates, X = data.load_front(root)
    n = len(dates); dX = np.diff(X, axis=0)                       # daily changes of contracts 1..4
    y = dX[:, 2]; Z = dX[:, [0, 1, 3]]                             # target contract 3, instruments 1, 2, 4
    # (c) curve sensitivities: per day, fit the seasonal kernel ridge to contracts 1, 2, 4 and read the hat-matrix row at contract 3
    curve_w = np.zeros((n - 1, 3)); ell = 0.2; lam = 1e-4
    step = max(1, (n - 1) // 4000)                                 # sensitivities move slowly; refit every `step` days and hold
    last = None
    for i in range(n - 1):
        if i % step == 0 or last is None:
            d = dt.date.fromisoformat(dates[i]); dels = [cal.nth_delivery(root, d, k) for k in (1, 2, 3, 4)]
            T = np.array([cal.years_to_expiry(root, d, *dels[k]) for k in (0, 1, 3)]); ph = np.array([cal.season_phase(dels[k][1]) for k in (0, 1, 3)])
            Tq = cal.years_to_expiry(root, d, *dels[2]); phq = cal.season_phase(dels[2][1])
            last = cv.key_rate_sensitivities(T, ph, X[i, [0, 1, 3]], Tq, phq, method="kernel_ridge", lam=lam, ell=ell)
        curve_w[i] = last
    hedged = {"unhedged": y, "front_stack": y - Z[:, 0], "neighbours_50_50": y - 0.5 * (Z[:, 1] + Z[:, 2]), "curve_sensitivities": y - np.sum(curve_w * Z, axis=1)}
    pred = stats.walk_forward(Z, y, WINDOW, lam=1e-3); hedged["learned_ridge"] = y - pred
    valid = ~np.isnan(pred)
    out = {"root": root, "n_days": int(valid.sum()), "from": dates[WINDOW], "to": dates[-1], "window": WINDOW, "curve_spec": {"method": "kernel_ridge", "lam": lam, "ell": ell}, "methods": {}}
    for k, v in hedged.items():
        vv = v[valid]; out["methods"][k] = {"std": float(np.std(vv)), "reduction_vs_unhedged": float(1 - np.std(vv) / np.std(y[valid]))}
    r, lo, hi = stats.paired_ratio_ci(hedged["learned_ridge"][valid], hedged["curve_sensitivities"][valid]); out["learned_over_curve_std_ratio"] = {"point": r, "ci95": [lo, hi]}
    r2, lo2, hi2 = stats.paired_ratio_ci(hedged["curve_sensitivities"][valid], hedged["front_stack"][valid]); out["curve_over_front_std_ratio"] = {"point": r2, "ci95": [lo2, hi2]}
    by_decade = {}
    yrs = np.array([int(d[:4]) for d in dates[1:]])
    for dec in sorted(set(yrs[valid] // 10 * 10)):
        m = valid & (yrs // 10 * 10 == dec); by_decade[str(dec)] = {k: float(np.std(v[m])) for k, v in hedged.items()}
    out["by_decade"] = by_decade; out["mean_curve_weights"] = curve_w[valid].mean(axis=0).tolist()
    json.dump(out, open(f"results/hedge_test_{root.lower()}.json", "w"), indent=1)
    print(root, out["n_days"], "days:", {k: round(v["std"], 4) for k, v in out["methods"].items()}, "learned/curve", round(r, 3), [round(lo, 3), round(hi, 3)], "curve/front", round(r2, 3))


if __name__ == "__main__":
    for root in ROOTS:
        run(root)
