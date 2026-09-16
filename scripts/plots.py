#!/usr/bin/env python3
"""Figures from results/*.json.   python scripts/plots.py [results] [results/figures]"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = sys.argv[1] if len(sys.argv) > 1 else "results"
F = sys.argv[2] if len(sys.argv) > 2 else os.path.join(R, "figures")
os.makedirs(F, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7, "figure.dpi": 130})
ROOTS = ["CL", "NG"]
NAMES = {"CL": "WTI (CL)", "NG": "Henry Hub (NG)"}
UNIT = {"CL": "$/bbl", "NG": "$/MMBtu"}
METHODS = [("seasonal_nss", "seasonal NSS", "C3"), ("pspline", "P-spline", "C0"), ("kernel_ridge", "seasonal kernel ridge", "C2")]
QCOL = {"Symmetric": "C7", "BEGV": "C0", "BEGV_adv": "C1", "BEGV_adv_cal": "C2", "Oracle": "C3"}
HCOL = {"NoHedge": "C7", "FrontStack": "C3", "Strip": "C1", "FactorMin": "C4", "ImpactAware": "C2"}


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def latest_curve(root):
    fs = sorted(glob.glob(os.path.join(R, f"curve_{root.lower()}_*.json")))
    return json.load(open(fs[-1], encoding="utf-8")) if fs else None


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(F, name)); plt.close(fig); print("wrote", os.path.join(F, name))


# ---- 1. curve construction on one day ---------------------------------------------------------------------------------
def fig_curve():
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    for r, root in enumerate(ROOTS):
        c = latest_curve(root)
        if not c:
            continue
        pts = c["points"]; T = np.array([p["T"] for p in pts]); P = np.array([p["price"] for p in pts]); src = [p["source"] for p in pts]
        ax = axes[r, 0]
        for m, lab, col in METHODS:
            g = c["methods"][m]["grid"]; ax.plot([x["T"] for x in g], [x["price"] for x in g], color=col, lw=1, label=lab)
        for s, mk in (("yahoo", "o"), ("yahoo_front", "s"), ("snapshot", "^"), ("eia", "d")):
            i = [k for k in range(len(pts)) if src[k] == s]
            if i:
                ax.plot(T[i], P[i], mk, ms=3, color="k", mfc="none", label=f"observed ({s})")
        ax.set_title(f"{NAMES[root]} curve, {c['date']}: {c['n_points']} contracts, {c['gap_months']} missing in first 24"); ax.set_xlabel("years to expiry"); ax.set_ylabel(UNIT[root]); ax.legend()
        ax = axes[r, 1]; w = 0.27
        for k, (m, lab, col) in enumerate(METHODS):
            j = c["methods"][m]; ax.bar(np.array(j["loo_months"]) + (k - 1) * w, j["loo_err"], width=w, color=col, label=f"{lab}: mean |err| {j['loo_mean_abs']:.3f}")
        ax.axhline(0, color="k", lw=0.5); ax.set_title("leave-one-out repricing error by contract"); ax.set_xlabel("months ahead"); ax.set_ylabel(UNIT[root]); ax.legend()
        ax = axes[r, 2]
        for m, lab, col in METHODS:
            j = c["methods"][m]; ax.plot([p["months"] for p in pts], j["sens_month3_to_contracts"], "-o", ms=2.5, color=col, lw=1, label=f"{lab} (sum {sum(j['sens_month3_to_contracts']):.2f})")
        ax.axhline(0, color="k", lw=0.5); ax.set_xlim(0, 14); ax.set_title("key-rate sensitivities of the month-3 swap"); ax.set_xlabel("months ahead"); ax.set_ylabel("d swap / d contract price"); ax.legend()
    save(fig, "curve.png")


# ---- 2. daily builds ----------------------------------------------------------------------------------------------------
def fig_curvehist():
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4))
    for r, root in enumerate(ROOTS):
        d = load(f"curvehist_{root.lower()}.json")
        if not d:
            continue
        cands = d["candidates"]; sel = d["selected"]
        ax = axes[r, 0]
        for m, lab, col in METHODS:
            cs = [c for c in cands if c["method"] == m]; ax.plot(range(len(cs)), [c["loo_test"] for c in cs], "o-", ms=3, color=col, lw=1, label=lab)
            k = [i for i, c in enumerate(cs) if c["label"] == sel[m]["label"]]
            if k:
                ax.plot(k[0], cs[k[0]]["loo_test"], "*", ms=11, color=col)
        ax.set_title(f"{NAMES[root]}: test LOO error by candidate (* = selected)"); ax.set_xlabel("candidate index within method"); ax.set_ylabel(f"mean |LOO err| ({UNIT[root]})"); ax.legend()
        ax = axes[r, 1]; gaps = ["0", "1-3", "4-12", ">12"]; w = 0.27
        for k, (m, lab, col) in enumerate(METHODS):
            c = next(c for c in cands if c["method"] == m and c["label"] == sel[m]["label"]); ax.bar(np.arange(4) + (k - 1) * w, [c["by_gap"][g]["loo_mean_abs"] for g in gaps], width=w, color=col, label=lab)
        ax.set_xticks(range(4)); ax.set_xticklabels([f"{g}\n(n={next(c for c in cands if c['method'] == 'pspline')['by_gap'][g]['n']})" for g in gaps]); ax.set_xlabel("missing months within the first 24"); ax.set_title("LOO error by curve completeness"); ax.set_ylabel(UNIT[root]); ax.legend()
        ax = axes[r, 2]; dates = d["dates"]; x = np.arange(len(dates))
        for m, lab, col in METHODS:
            c = next(c for c in cands if c["method"] == m and c["label"] == sel[m]["label"]); y = np.array(c["loo_daily"]); k = 20; roll = np.convolve(y, np.ones(k) / k, mode="valid"); ax.plot(x[k - 1:], roll, color=col, lw=1, label=lab)
        ticks = [i for i in range(0, len(dates), max(1, len(dates) // 6))]; ax.set_xticks(ticks); ax.set_xticklabels([dates[i][:7] for i in ticks], rotation=30); ax.axvline(d["n_train"], color="k", ls="--", lw=0.7); ax.text(d["n_train"], ax.get_ylim()[1] * 0.95, " selection | test", fontsize=7, va="top")
        ax.set_title("daily LOO error, 20-day rolling mean"); ax.set_ylabel(UNIT[root]); ax.legend()
        ax = axes[r, 3]; ht = d["hedge_test"]; names = [("front_stack", "front stack", "C3"), ("neighbours_50_50", "50/50 neighbours", "C8"), ("seasonal_nss", "NSS key rates", "C3"), ("pspline", "P-spline key rates", "C0"), ("kernel_ridge", "kernel-ridge key rates", "C2")]
        vals = [ht["unhedged_std"]] + [ht["methods"][n]["hedged_std"] for n, _, _ in names]; cols = ["C7"] + [c for _, _, c in names]
        ax.bar(range(len(vals)), vals, color=cols); ax.set_xticks(range(len(vals))); ax.set_xticklabels(["unhedged"] + [l for _, l, _ in names], rotation=30, ha="right"); ax.set_yscale("log")
        ax.set_title(f"hedge test: month {ht['target_month']} dropped, {ht['n_days']} days"); ax.set_ylabel(f"next-day hedged P&L std ({UNIT[root]})")
    save(fig, "curvehist.png")


# ---- 3. factors ---------------------------------------------------------------------------------------------------------
def fig_factors():
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for r, root in enumerate(ROOTS):
        d = load(f"factors_{root.lower()}.json")
        if not d:
            continue
        ax = axes[0, r]; rec = d["recent"]
        for f, lab in enumerate(["level", "slope", "curvature"]):
            ax.plot(rec["months"], rec["loadings"][f], "o-", ms=3, label=f"{lab} ({100 * rec['explained'][f] - (100 * rec['explained'][f - 1] if f else 0):.1f}%, sd {np.sqrt(rec['evals_usd2_per_day'][f]):.3f}/day)")
        ax.axhline(0, color="k", lw=0.5); ax.set_title(f"{NAMES[root]}: PCA loadings, {rec['from']}..{rec['to']} ({rec['n_days']} days)"); ax.set_xlabel("months ahead"); ax.legend()
        ax = axes[1, r]; hist = d["front_history"]; keys = [k for k in hist if hist[k]["n_days"] > 0]; x = np.arange(len(keys)); w = 0.27
        for f, lab in enumerate(["level", "slope", "curvature"]):
            ax.bar(x + (f - 1) * w, [100 * (hist[k]["explained"][f] - (hist[k]["explained"][f - 1] if f else 0)) for k in keys], width=w, label=lab)
        ax.set_xticks(x); ax.set_xticklabels([f"{k[:4]}-{k[11:15]}\n(sd {np.sqrt(hist[k]['evals_usd2_per_day'][0]):.2f})" for k in keys], fontsize=7); ax.set_ylabel("% of daily variance"); ax.set_title("EIA contracts 1-4 by decade: variance explained"); ax.legend()
    save(fig, "factors.png")


# ---- 4. impact ----------------------------------------------------------------------------------------------------------
def fig_impact():
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    for r, root in enumerate(ROOTS):
        d = load(f"impact_{root.lower()}.json")
        if not d:
            continue
        ws = d["windows"]; w1 = ws[0]
        ax = axes[r, 0]; x = np.array(w1["bin_abs_q"]); y = np.array(w1["bin_response"]); ok = (x > 0) & (y > 0)
        ax.loglog(x[ok], y[ok], "o", ms=4, color="C0", label=f"binned response, {w1['window_bars']}h windows (n={w1['n_windows']})")
        xx = np.geomspace(x[ok].min(), x[ok].max(), 50); ax.loglog(xx, w1["Y"] * xx ** w1["delta"], "-", color="C3", label=f"power law: delta {w1['delta']:.2f} [{w1['delta_ci95'][0]:.2f}, {w1['delta_ci95'][1]:.2f}]  (R² bins {w1.get('r2_concave_bins', float('nan')):.2f})")
        ax.loglog(xx, w1["Y_linear"] * xx, "--", color="C7", label=f"linear through the origin (R² bins {w1.get('r2_linear_bins', float('nan')):.2f})")
        ax.set_title(f"{NAMES[root]}: self-impact vs order-flow imbalance"); ax.set_xlabel("|Q| = |signed volume| / daily volume"); ax.set_ylabel("sign(Q) · return / sigma"); ax.legend()
        ax = axes[r, 1]
        for w, col in zip(ws, ["C0", "C1", "C2"]):
            ax.plot(range(1, len(w["decay"]) + 1), w["decay"], "o-", ms=3, color=col, lw=1, label=f"{w['window_bars']}h windows")
        ax.axhline(d["simulator_model"]["perm_frac"], color="k", ls="--", lw=0.7, label=f"permanent fraction used: {d['simulator_model']['perm_frac']:.2f}"); ax.axhline(1, color="k", lw=0.4)
        ax.set_title("cumulative response after the window, G(tau)/G(0)"); ax.set_xlabel("windows later"); ax.set_ylabel("G(tau) / G(0)"); ax.legend()
        ax = axes[r, 2]; x = np.arange(len(ws)); w = 0.2
        ax.bar(x - 1.5 * w, [w_["r2_self_only"] for w_ in ws], width=w, color="C0", label="self-impact only")
        ax.bar(x - 0.5 * w, [w_["r2_self_plus_cross"] for w_ in ws], width=w, color="C2", label="self + cross-impact")
        ax.bar(x + 0.5 * w, [w_["r2_concave"] for w_ in ws], width=w, color="C1", label="pooled concave (raw windows)")
        ax.bar(x + 1.5 * w, [max(w_["r2_linear"], -0.3) for w_ in ws], width=w, color="C7", label="pooled linear (raw windows)")
        ax.set_xticks(x); ax.set_xticklabels([f"{w_['window_bars']}h\n({w_['n_contracts']} contracts)" for w_ in ws]); ax.axhline(0, color="k", lw=0.5); ax.set_ylabel("R²"); ax.set_title("R² on the raw windows"); ax.legend()
    save(fig, "impact.png")


# ---- 5. hedging ---------------------------------------------------------------------------------------------------------
def fig_hedge():
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4))
    for r, root in enumerate(ROOTS):
        d = load(f"hedge_{root.lower()}.json")
        if d:
            ax = axes[0, r]; cells = [c for c in d["cells"] if c["policy"] != "NoHedge"]
            x = np.arange(len(cells)); y = [c["cost_per_mn"] for c in cells]; lo = [c["cost_per_mn"] - c["cost_per_mn_ci95"][0] for c in cells]; hi = [c["cost_per_mn_ci95"][1] - c["cost_per_mn"] for c in cells]
            ax.bar(x, y, color=[HCOL.get(c["policy"].split("_")[0].split("@")[0], "C5") for c in cells], yerr=[lo, hi], capsize=3)
            for i, c in enumerate(cells):
                ax.text(i, c["cost_per_mn"] + hi[i] + 20, f"risk ${c['risk_usd'] / 1e3:.0f}k\n{c['vs_front_pct']:+.0f}% vs front", ha="center", fontsize=6.5)
            ax.set_xticks(x); ax.set_xticklabels([c["policy"].replace("@FrontRisk", "\n@ front-stack risk") for c in cells], rotation=25, ha="right", fontsize=7); ax.set_ylabel("expected execution cost per \$1M notional (USD)")
            ax.set_title(f"{NAMES[root]}: {d['books']} random swap books, cost per USD 1M (risk above bars)"); ax.set_ylim(0, max(y[i] + hi[i] for i in range(len(y))) * 1.35)
        ht = load(f"hedge_test_{root.lower()}.json")
        if ht:
            ax = axes[1, r]; decs = sorted(ht["by_decade"]); names = [("unhedged", "C7"), ("front_stack", "C3"), ("neighbours_50_50", "C8"), ("curve_sensitivities", "C0"), ("learned_ridge", "C2")]; x = np.arange(len(decs)); w = 0.16
            for k, (n, col) in enumerate(names):
                ax.bar(x + (k - 2) * w, [ht["by_decade"][dc][n] for dc in decs], width=w, color=col, label=n.replace("_", " "))
            ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{dc}s" for dc in decs]); ax.set_ylabel(f"hedged next-day P&L std ({UNIT[root]})"); ax.legend(ncol=2)
            r1 = ht["learned_over_curve_std_ratio"]; r2 = ht["curve_over_front_std_ratio"]
            ax.set_title(f"EIA panel {ht['from'][:4]}-{ht['to'][:4]}, contract 3 hedged with 1,2,4: learned/curve {r1['point']:.2f} [{r1['ci95'][0]:.2f}, {r1['ci95'][1]:.2f}], curve/front {r2['point']:.2f}", fontsize=7.5)
    save(fig, "hedge.png")


# ---- 6. simulator -------------------------------------------------------------------------------------------------------
def fig_sim():
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.8))
    for r, root in enumerate(ROOTS):
        d = load(f"sim_{root.lower()}.json")
        if not d:
            continue
        n = d["days"]; cells = d["cells"]; gammas = sorted(set(c["gamma"] for c in cells)); hedgers = [h for h in ["FrontStack", "Strip", "ImpactAware"] if any(c["hedger"] == h for c in cells)]
        main_h = "ImpactAware" if "ImpactAware" in hedgers else hedgers[-1]
        ax = axes[r, 0]
        for q in QCOL:
            cs = [c for c in cells if c["quoter"] == q and c["hedger"] == main_h]; cs.sort(key=lambda c: c["gamma"])
            if cs:
                ax.plot([c["pnl_5min_std"] / 1e3 for c in cs], [c["pnl"] / n / 1e3 for c in cs], "o-", ms=3.5, lw=1, color=QCOL[q], label=q)
                for c in cs:
                    ax.annotate(f"{c['gamma']:.0e}", (c["pnl_5min_std"] / 1e3, c["pnl"] / n / 1e3), fontsize=5.5, xytext=(2, 2), textcoords="offset points")
        ax.set_xlabel("5-minute P&L std ($k)"); ax.set_ylabel("P&L per day ($k)"); ax.set_title(f"{NAMES[root]}: frontier over gamma, hedger {main_h}, {n} days x {d['seeds']} seeds"); ax.legend()
        g0 = gammas[len(gammas) // 2]; ax = axes[r, 1]; comps = ["spread", "inventory", "adverse", "hedge_exec", "hedge_impact"]; ccol = ["C2", "C0", "C3", "C1", "C4"]
        rows = []
        for h in hedgers:
            for q in QCOL:
                cs = [c for c in cells if c["quoter"] == q and c["hedger"] == h and abs(c["gamma"] - g0) < 1e-15]
                if cs:
                    rows.append((f"{h}\n{q}", cs[0]))
        x = np.arange(len(rows)); bottom_pos = np.zeros(len(rows)); bottom_neg = np.zeros(len(rows))
        for comp, col in zip(comps, ccol):
            v = np.array([c["decomposition"][comp] / n / 1e3 for _, c in rows]); b = np.where(v >= 0, bottom_pos, bottom_neg); ax.bar(x, v, bottom=b, color=col, label=comp, width=0.7); bottom_pos += np.maximum(v, 0); bottom_neg += np.minimum(v, 0)
        ax.plot(x, [c["pnl"] / n / 1e3 for _, c in rows], "k_", ms=12, label="total"); ax.set_xticks(x); ax.set_xticklabels([l for l, _ in rows], fontsize=5.5, rotation=90); ax.set_ylabel("$k per day"); ax.set_title(f"P&L decomposition at gamma {g0:.0e}"); ax.legend(fontsize=6, ncol=2)
        ax = axes[r, 2]; w = 0.8 / max(1, len(hedgers))
        for k, h in enumerate(hedgers):
            cs = [next((c for c in cells if c["quoter"] == q and c["hedger"] == h and abs(c["gamma"] - g0) < 1e-15), None) for q in QCOL]
            ax.bar(np.arange(len(QCOL)) + (k - (len(hedgers) - 1) / 2) * w, [1e3 * c["pnl_per_var"] if c else 0 for c in cs], width=w, color=HCOL[h], label=f"{h} (internalisation {np.mean([c['internalisation_ratio'] for c in cs if c]):.2f})")
        ax.set_xticks(range(len(QCOL))); ax.set_xticklabels(list(QCOL), rotation=20); ax.set_ylabel("daily P&L / 5-min P&L variance (x1e-3)"); ax.set_title(f"daily P&L / 5-min variance, gamma {g0:.0e}"); ax.legend(fontsize=6)
    save(fig, "sim.png")


def fig_policy():
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4))
    for r, root in enumerate(ROOTS):
        d = load(f"sim_{root.lower()}.json")
        if not d:
            continue
        cells = d["cells"]; gammas = sorted(set(c["gamma"] for c in cells)); g0 = gammas[len(gammas) // 2]; hedgers = [h for h in ["ImpactAware", "Strip", "FrontStack"] if any(c["hedger"] == h for c in cells)]; h0 = hedgers[0]
        ax = axes[r, 0]; tiers = ["consumer", "producer", "fund"]; x = np.arange(len(tiers) * 2); w = 0.8 / len(QCOL)
        for k, q in enumerate(QCOL):
            c = next((c for c in cells if c["quoter"] == q and c["hedger"] == h0 and abs(c["gamma"] - g0) < 1e-15), None)
            if not c:
                continue
            vals = []
            for t in tiers:
                for pre in (False, True):
                    m = next((m for m in c["markouts"] if m["tier"] == t and m["pre_event"] == pre), None); vals.append(m["markout_30m"] if m and m["n"] > 0 else 0)
            ax.bar(x + (k - 2) * w, vals, width=w, color=QCOL[q], label=q)
        ax.axhline(0, color="k", lw=0.5); ax.set_xticks(x); ax.set_xticklabels([f"{t}\n{'pre-release' if pre else 'normal'}" for t in tiers for pre in (False, True)], fontsize=7); ax.set_ylabel(f"30-minute mark-out vs the dealer ({UNIT[root]})"); ax.set_title(f"{NAMES[root]}: 30-min mark-outs by tier and window (gamma {g0:.0e}, {h0})"); ax.legend(fontsize=6)
        ax = axes[r, 1]; x = np.arange(len(tiers) * 2)
        for k, q in enumerate(QCOL):
            c = next((c for c in cells if c["quoter"] == q and c["hedger"] == h0 and abs(c["gamma"] - g0) < 1e-15), None)
            if not c:
                continue
            vals = []
            for t in tiers:
                for adding in (False, True):
                    qs = [s for s in c["quote_structure"] if s["tier"] == t and s["risk_adding"] == adding and not s["pre_event"]]; nn = sum(s["n"] for s in qs); vals.append(sum(s["n"] * s["mean_delta"] for s in qs) / nn if nn else 0)
            ax.bar(x + (k - 2) * w, vals, width=w, color=QCOL[q], label=q)
        ax.set_xticks(x); ax.set_xticklabels([f"{t}\n{'risk-adding' if a else 'risk-reducing'}" for t in tiers for a in (False, True)], fontsize=7); ax.set_ylabel(f"mean half-spread ({UNIT[root]})"); ax.set_title("half-spread: risk-adding vs risk-reducing RFQs"); ax.legend(fontsize=6)
    save(fig, "policy.png")


# ---- 7. real-day replay -------------------------------------------------------------------------------------------------
def fig_events():
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4))
    for r, root in enumerate(ROOTS):
        d = load(f"events_{root.lower()}.json")
        if not d:
            continue
        n = d["n_days"]; cells = d["cells"]; x = np.arange(len(cells))
        ax = axes[0, r]; y = [c["pnl"] / n / 1e3 for c in cells]; lo = [(c["pnl"] - c["pnl_ci95"][0]) / n / 1e3 for c in cells]; hi = [(c["pnl_ci95"][1] - c["pnl"]) / n / 1e3 for c in cells]
        ax.bar(x, y, yerr=[lo, hi], capsize=3, color=[QCOL.get(c["quoter"], "C5") for c in cells])
        for i, c in enumerate(cells):
            ax.text(i, 0.5, f"P&L/var {1e3 * c['pnl_per_var']:.2f}\nadverse {c['decomposition']['adverse'] / n / 1e3:+.1f}k\ninternal. {c['internalisation_ratio']:.2f}", ha="center", va="bottom", fontsize=6, color="w")
        ax.set_xticks(x); ax.set_xticklabels([c["quoter"] for c in cells], rotation=15); ax.set_ylabel("P&L per day ($k), 95% CI over seeds"); ax.set_title(f"{NAMES[root]}: {n} real days {d['from']}..{d['to']}, {d['n_events']} releases, {cells[0]['hedger']}", fontsize=8)
        ax = axes[1, r]; tiers = ["consumer", "producer", "fund"]; xx = np.arange(len(tiers) * 2); w = 0.8 / len(cells)
        for k, c in enumerate(cells):
            vals = []
            for t in tiers:
                for pre in (False, True):
                    m = next((m for m in c["markouts"] if m["tier"] == t and m["pre_event"] == pre), None); vals.append(m["markout_30m"] if m and m["n"] > 0 else 0)
            ax.bar(xx + (k - (len(cells) - 1) / 2) * w, vals, width=w, color=QCOL.get(c["quoter"], "C5"), label=c["quoter"])
        ax.axhline(0, color="k", lw=0.5); ax.set_xticks(xx); ax.set_xticklabels([f"{t}\n{'pre-release' if pre else 'normal'}" for t in tiers for pre in (False, True)], fontsize=7); ax.set_ylabel(f"30-minute mark-out vs the dealer ({UNIT[root]})"); ax.set_title("30-min mark-outs: normal vs pre-release window"); ax.legend(fontsize=6)
    save(fig, "events.png")


# ---- 8. release-day study -----------------------------------------------------------------------------------------------
def fig_events_study():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    ds = {root: load(f"events_study_{root.lower()}.json") for root in ROOTS}
    ax = axes[0]; x = np.arange(2); w = 0.35
    for k, root in enumerate(ROOTS):
        d = ds[root]
        if not d:
            continue
        v = [d["release_abs_move"], d["other_abs_move"]]; err = [[d["release_abs_move"] - d["release_abs_move_ci95"][0], 0], [d["release_abs_move_ci95"][1] - d["release_abs_move"], 0]]
        ax.bar(x + (k - 0.5) * w, np.array(v) / d["other_abs_move"], width=w, yerr=np.array(err) / d["other_abs_move"], capsize=3, label=f"{root}: variance ratio {d['release_variance_ratio']:.2f}, jump share kappa {d['kappa_jump_share']:.2f}")
    ax.set_xticks(x); ax.set_xticklabels(["release days", "other days"]); ax.set_ylabel("mean |front-month move| / other-day mean"); ax.set_title("EIA release days vs other days (front contract)"); ax.legend(fontsize=6.5)
    for j, (key, title) in enumerate([("surprise_nowcast", "storage / stocks surprise -> release-day move"), ("hedging_pressure", "CFTC hedging pressure -> next-week move")]):
        ax = axes[j + 1]
        for k, root in enumerate(ROOTS):
            d = ds[root]
            if not d:
                continue
            s = d[key]; ax.errorbar([k - 0.15], [s["corr_train"]], fmt="s", color="C7", label="in-sample (train)" if k == 0 else None)
            ax.errorbar([k + 0.15], [s["corr_oos"]], yerr=[[s["corr_oos"] - s["corr_oos_ci95"][0]], [s["corr_oos_ci95"][1] - s["corr_oos"]]], fmt="o", capsize=4, color="C0", label="out of sample, 95% CI" if k == 0 else None)
            ax.text(k + 0.15, s["corr_oos_ci95"][1] + 0.02, f"perm. p {s['permutation_p']:.2f}\nn={s['test_n']}", ha="center", fontsize=6.5)
        ax.axhline(0, color="k", lw=0.5); ax.set_xticks(range(len(ROOTS))); ax.set_xticklabels(ROOTS); ax.set_ylabel("correlation"); ax.set_title(title); ax.legend(fontsize=6.5); ax.set_ylim(-0.4, 0.4)
    save(fig, "events_study.png")


if __name__ == "__main__":
    for f in (fig_curve, fig_curvehist, fig_factors, fig_impact, fig_hedge, fig_sim, fig_policy, fig_events, fig_events_study):
        try:
            f()
        except Exception as e:  # noqa: BLE001
            print(f"{f.__name__}: skipped ({e})")
