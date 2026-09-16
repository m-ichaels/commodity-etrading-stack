"""Tidy access to results/*.json and the headline numbers the README quotes."""
import json
import os

R = "results"


def load(name, root=None):
    p = os.path.join(R, name if root is None else name.replace("<root>", root.lower()))
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def curvehist_headline(root):
    d = load("curvehist_<root>.json", root)
    if not d:
        return None
    sel = d["selected"]; best = min(sel.items(), key=lambda kv: kv[1]["loo_test_mean_abs"])
    return {"root": root, "n_days": d["n_days"], "from": d["from"], "to": d["to"], "best_method": best[0], "best_label": best[1]["label"], "test_loo": best[1]["loo_test_mean_abs"], "test_loo_pct": best[1]["loo_test_mean_abs_pct"], "vs_nss_pct": best[1]["vs_nss_pct"], "vs_nss_ci": best[1]["vs_nss_ci95"], "selected": sel, "hedge_test": d["hedge_test"]}


def impact_headline(root):
    d = load("impact_<root>.json", root)
    if not d:
        return None
    w = d["windows"][0]
    return {"root": root, "delta": w["delta"], "delta_ci": w["delta_ci95"], "Y": w["Y"], "r2_linear": w["r2_linear"], "r2_concave": w["r2_concave"], "r2_self": w["r2_self_only"], "r2_cross": w["r2_self_plus_cross"], "n_windows": w["n_windows"], "n_days": w["n_days"], "perm_frac": d["simulator_model"]["perm_frac"], "windows": d["windows"]}


def hedge_headline(root):
    d = load("hedge_<root>.json", root)
    if not d:
        return None
    cells = {c["policy"]: c for c in d["cells"] if c["gamma"] == d["gamma"]}
    eq = cells.get("ImpactAware@FrontRisk"); fs = cells.get("FrontStack")
    return {"root": root, "cells": cells, "equal_risk_saving_pct": -eq["vs_front_pct"] if eq else None, "equal_risk_ci": eq["vs_front_ci95"] if eq else None, "front_cost_per_mn": fs["cost_per_mn"] if fs else None, "equal_risk_cost_per_mn": eq["cost_per_mn"] if eq else None}


def sim_headline(root):
    d = load("sim_<root>.json", root)
    if not d:
        return None
    out = {}
    for c in d["cells"]:
        out[(c["hedger"], c["quoter"], c["gamma"])] = {"pnl_per_day": c["pnl"] / d["days"], "pnl_ci": [x / d["days"] for x in c["pnl_ci95"]], "pnl_per_var": c["pnl_per_var"], "std5": c["pnl_5min_std"], "internalisation": c["internalisation_ratio"], "hedge_cost_per_mn": c["hedge_cost_per_mn_notional"], "decomposition": c["decomposition"], "adverse_per_day": c["adverse"] / d["days"], "hits": c["hit_ratio_by_tier"]}
    return {"root": root, "days": d["days"], "seeds": d["seeds"], "cells": out}
