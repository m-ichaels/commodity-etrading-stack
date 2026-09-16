"""python -m cetk compare --root CL --date D    numpy rebuild of a day's curves vs the C++ results/curve_<root>_<date>.json
   python -m cetk curve --root CL --date D      fit the three models in numpy and print LOO
   python -m cetk summary                       headline numbers from results/"""
import argparse
import json
import os
import sys

import numpy as np

from . import curve as cv
from . import data, results


def cmd_curve(a):
    days = data.load_curves(a.root, since=a.date[:4] + "-01-01" if a.date else None)
    d = a.date or max(days); day = days[d]
    out = {"date": d, "n": len(day["price"]), "gap": int(day["gap"])}
    for method, spec in (("seasonal_nss", {"method": "seasonal_nss"}), ("pspline", {"method": "pspline", "lam": 1e-2}), ("kernel_ridge", {"method": "kernel_ridge", "lam": 1e-3})):
        m = cv.CurveModel(**spec).fit(day["T"], day["phase"], day["price"]); loo = cv.leave_one_out(day["T"], day["phase"], day["price"], day["months"], **spec)
        out[method] = {"in_sample_rmse": m.in_sample_rmse, "loo_mean_abs": loo["mean_abs"], "loo_mean_abs_interior": loo["mean_abs_interior"]}
        print(f"{method}: rmse {m.in_sample_rmse:.5f}  LOO {loo['mean_abs']:.5f}  interior {loo['mean_abs_interior']:.5f}")
    return out


def cmd_compare(a):
    p = a.file or f"results/curve_{a.root.lower()}_{a.date}.json"
    c = json.load(open(p, encoding="utf-8")); day = data.load_curves(a.root, since=c["date"][:4] + "-01-01")[c["date"]]
    worst = 0.0
    for method in ("seasonal_nss", "pspline", "kernel_ridge"):
        j = c["methods"][method]; spec = {"method": method, "lam": j["lambda"], "knot_spacing": j.get("knot_spacing", 1.0 / 12), "ell": j.get("ell", 0.5), "theta": j.get("theta", 1.0), "harmonics": int(j.get("harmonics", 2))}
        m = cv.CurveModel(**spec).fit(day["T"], day["phase"], day["price"]); loo = cv.leave_one_out(day["T"], day["phase"], day["price"], day["months"], **spec)
        d1 = abs(m.in_sample_rmse - j["in_sample_rmse"]); d2 = abs(loo["mean_abs"] - j["loo_mean_abs"]); worst = max(worst, d1, d2)
        print(f"{method}: rmse C++ {j['in_sample_rmse']:.6f} py {m.in_sample_rmse:.6f} | LOO C++ {j['loo_mean_abs']:.6f} py {loo['mean_abs']:.6f}")
        if method == "seasonal_nss":
            print(f"   tau C++ ({j['tau1']}, {j['tau2']}) py ({m.tau1}, {m.tau2})")
    print("max |difference|", worst)
    tol = 1e-4 * max(1.0, float(np.mean(day["price"])))
    if worst > tol:
        print("MISMATCH beyond tolerance", tol); sys.exit(1)
    print("python rebuild agrees with the C++ within", tol)


def cmd_summary(a):
    for root in ("CL", "NG"):
        h = results.curvehist_headline(root)
        if h:
            print(f"{root} curves {h['from']}..{h['to']} ({h['n_days']} days): best {h['best_method']} ({h['best_label']}) test LOO {h['test_loo']:.4f} ({h['test_loo_pct']:.2f}%), {h['vs_nss_pct']:+.1f}% vs seasonal NSS {h['vs_nss_ci']}")
        i = results.impact_headline(root)
        if i:
            print(f"{root} impact: delta {i['delta']:.3f} {i['delta_ci']} R2 self {i['r2_self']:.3f} -> +cross {i['r2_cross']:.3f} over {i['n_windows']} windows")
        g = results.hedge_headline(root)
        if g:
            print(f"{root} hedge: at the front stack's risk the impact-aware hedge costs {g['equal_risk_saving_pct']:.1f}% less per $1M {g['equal_risk_ci']}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cetk"); sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("curve", "compare"):
        s = sub.add_parser(name); s.add_argument("--root", default="CL"); s.add_argument("--date", default=None); s.add_argument("--file", default=None)
    sub.add_parser("summary")
    a = ap.parse_args(argv)
    if a.cmd == "curve":
        cmd_curve(a)
    elif a.cmd == "compare":
        cmd_compare(a)
    else:
        cmd_summary(a)


if __name__ == "__main__":
    main()
