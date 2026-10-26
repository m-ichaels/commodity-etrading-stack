#!/usr/bin/env python3
"""README.md and docs/method.md from the results files, so every number in them is the run's.   python scripts/readme.py [--check]
--check: regenerate in memory and fail if either file on disk differs (CI)."""
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
OUT = ("README.md", "docs/method.md")
NAME = {"seasonal_nss": "seasonal NSS", "pspline": "P-spline", "kernel_ridge": "seasonal kernel ridge"}
QUOTERS = ["Symmetric", "BEGV", "BEGV_adv", "BEGV_adv_cal", "Oracle"]
QLAB = {"Symmetric": "Symmetric", "BEGV": "BEGV (skew)", "BEGV_adv": "BEGV_adv (+ tier widening)", "BEGV_adv_cal": "BEGV_adv_cal (+ pre-release widening)", "Oracle": "Oracle (true client model)"}
POLICY = {"FrontStack": "front stack", "Strip": "strip (one for one)", "FactorMin": "factor-minimum", "ImpactAware": "impact-aware, unconstrained",
          "ImpactAware_noCross@FrontRisk": "impact-aware, no cross terms @ front-stack risk", "ImpactAware_linear@FrontRisk": "impact-aware, linear impact @ front-stack risk",
          "ImpactAware@FrontRisk": "impact-aware @ front-stack risk"}


def load(name):
    return json.load(open(os.path.join(R, name), encoding="utf-8"))


def paired(a, b, B=2000, seed=7):
    """mean of a - b with a bootstrap 95% CI over the paired seeds (the same draw as scripts/summarize.py)"""
    a = np.asarray(a, float); b = np.asarray(b, float); d = a - b; rng = np.random.default_rng(seed)
    if len(d) < 2:
        return float(d.mean()), float("nan"), float("nan")
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def cell(d, quoter, hedger="ImpactAware", gamma=None):
    for c in d["cells"]:
        if c["quoter"] == quoter and c["hedger"] == hedger and (gamma is None or abs(c["gamma"] - gamma) < 1e-15):
            return c
    raise KeyError((quoter, hedger, gamma))


def vs(c, base, n):
    """paired P&L difference per day in $k with its 95% CI, and the change relative to the base"""
    m, lo, hi = paired(c["pnl_by_seed"], base["pnl_by_seed"])
    return m / n / 1e3, lo / n / 1e3, hi / n / 1e3, 100 * m / base["pnl"]


def vs_txt(v, rel=True):
    return f"{v[0]:+.1f} [{v[1]:+.1f}, {v[2]:+.1f}]" + (f" ({v[3]:+.0f}%)" if rel else "")


def hedge_cost(c, n):
    return -(c["decomposition"]["hedge_exec"] + c["decomposition"]["hedge_impact"]) / n / 1e3


def markout(c, pre, tier="fund"):
    return next(m for m in c["markouts"] if m["tier"] == tier and m["pre_event"] == pre)


def risk(x):
    return "0" if x < 500 else f"${x / 1e3:.0f}k"


def pc(x):
    """signed whole percent, with no '-0'"""
    t = f"{x:+.0f}"
    return "0" if t in ("+0", "-0") else t


def reduction(s):
    """a selected curve's change against the seasonal NSS as a reduction with its CI: '26% [18, 33]'"""
    return f"{-s['vs_nss_pct']:.0f}% [{-s['vs_nss_ci95'][1]:.0f}, {-s['vs_nss_ci95'][0]:.0f}]"


def tests():
    t = open(os.path.join(R, "tests.txt"), encoding="utf-8").read()
    return int(re.search(r"\d+/(\d+) checks passed", t).group(1)), int(re.search(r"^(\d+) passed", t, flags=re.M).group(1))


def build():
    """(README.md, docs/method.md)"""
    rs = ("cl", "ng")
    ch = {r: load(f"curvehist_{r}.json") for r in rs}
    imp = {r: load(f"impact_{r}.json") for r in rs}
    hed = {r: load(f"hedge_{r}.json") for r in rs}
    sim = {r: load(f"sim_{r}.json") for r in rs}
    rep = {r: load(f"events_{r}.json") for r in rs}
    stu = {r: load(f"events_study_{r}.json") for r in rs}
    dck = {r: load(f"datacheck_{r}.json") for r in rs}
    fac = {r: load(f"factors_{r}.json") for r in rs}
    eia = {r: load(f"hedge_test_{r}.json") for r in rs}
    tox = {r: load(f"sim_tox_{r}.json") for r in rs}
    ant = {r: load(f"sim_anticipate_{r}.json") for r in rs}
    scale = {"half": load("sim_impact_half_cl.json"), "base": sim["cl"], "double": load("sim_impact_double_cl.json")}
    n_cpp, n_py = tests()

    best = {}
    for r in rs:
        sel = ch[r]["selected"]; m = min(sel, key=lambda k: sel[k]["loo_test_mean_abs"]); best[r] = (m, sel[m], sel["seasonal_nss"])
    (bc, sc, nc), (bn, sn, nn) = best["cl"], best["ng"]
    ht = {r: ch[r]["hedge_test"] for r in rs}
    w1 = {r: imp[r]["windows"][0] for r in rs}
    pol = {r: {c["policy"]: c for c in hed[r]["cells"]} for r in rs}
    eq = {r: pol[r]["ImpactAware@FrontRisk"] for r in rs}
    fs = {r: pol[r]["FrontStack"] for r in rs}
    g0 = 1e-6
    band = {h: cell(sim["cl"], "BEGV_adv_cal", h, g0)["hedge_cost_per_mn_notional"] for h in ("FrontStack", "ImpactAware")}
    nrep = rep["cl"]["n_days"]
    rv = {r: vs(cell(rep[r], "BEGV"), cell(rep[r], "Symmetric"), rep[r]["n_days"]) for r in rs}
    std_ng = 100 * (cell(rep["ng"], "BEGV")["pnl_5min_std"] / cell(rep["ng"], "Symmetric")["pnl_5min_std"] - 1)
    corrs = [stu[r][k]["corr_oos"] for r in rs for k in ("surprise_nowcast", "hedging_pressure")]
    complete = {r: dck[r]["gap_histogram_since_2024_09"] for r in rs}

    # ---------------------------------------------------------------- README.md
    rd = ["# commodity-etrading-stack", ""]
    rd.append("A pricing, hedging and quoting stack for calendar-month WTI and Henry Hub swaps, in header-only C++20 behind one CLI (`cetk`) with a numpy reference port (`python/cetk`): "
              "seasonal futures curves, a factor risk model, price impact fitted from hourly bars, an impact-aware hedger, and an RFQ engine with closed-form quotes and a FIX 4.4 gateway. "
              "All data is free (EIA, CFTC, Yahoo) and checked in.")
    rd.append("")
    rd.append("The question: do recent curve, impact and quoting models beat a swap desk's classical answers out of sample?")
    rd.append("")
    rd.append("## Results")
    rd.append("")
    rd.append(f"- **Curves.** Over {ch['cl']['n_days']} / {ch['ng']['n_days']} daily builds ({ch['cl']['from']} to {ch['cl']['to']}; settings chosen on the first third, scored on the rest), "
              f"the {NAME[bc]} cuts WTI's leave-one-out repricing error {reduction(sc)} against a seasonal Nelson–Siegel–Svensson ({sc['loo_test_mean_abs']:.3f} vs {nc['loo_test_mean_abs']:.3f} $/bbl). "
              f"The {NAME[bn]} cuts Henry Hub's {reduction(sn)} ({sn['loo_test_mean_abs']:.3f} vs {nn['loo_test_mean_abs']:.3f} $/MMBtu).")
    rd.append(f"- **Key-rate hedges.** A month-3 contract dropped from the curve and hedged with its key-rate sensitivities leaves a next-day P&L std of "
              f"{ht['cl']['methods'][bc]['hedged_std']:.2g} $/bbl and {ht['ng']['methods'][bn]['hedged_std']:.2g} $/MMBtu, against "
              f"{ht['cl']['methods']['front_stack']['hedged_std']:.2g} and {ht['ng']['methods']['front_stack']['hedged_std']:.2g} for a front-month stack.")
    rd.append(f"- **Impact.** Hourly order-flow impact is concave: δ = {w1['cl']['delta']:.2f} [{w1['cl']['delta_ci95'][0]:.2f}, {w1['cl']['delta_ci95'][1]:.2f}] for WTI and "
              f"{w1['ng']['delta']:.2f} [{w1['ng']['delta_ci95'][0]:.2f}, {w1['ng']['delta_ci95'][1]:.2f}] for gas. Cross-impact lifts the explained variance of hourly returns from "
              f"{w1['cl']['r2_self_only']:.2f} to {w1['cl']['r2_self_plus_cross']:.2f} (WTI) and from {w1['ng']['r2_self_only']:.2f} to {w1['ng']['r2_self_plus_cross']:.2f} (gas).")
    rd.append(f"- **Hedging.** In the fitted impact model, an impact-aware hedge at the front stack's residual risk costs {-eq['cl']['vs_front_pct']:.0f}% "
              f"[{-eq['cl']['vs_front_ci95'][1]:.0f}, {-eq['cl']['vs_front_ci95'][0]:.0f}] less per $1M on {hed['cl']['books']} random WTI swap books, and "
              f"{-eq['ng']['vs_front_pct']:.0f}% [{-eq['ng']['vs_front_ci95'][1]:.0f}, {-eq['ng']['vs_front_ci95'][0]:.0f}] less on gas. "
              f"Hedging back to a risk band every 30 minutes in the RFQ simulator, the two cost the same ({band['FrontStack']:.0f} and {band['ImpactAware']:.0f} $ per $1M on WTI): the saving is a large-book effect.")
    rd.append(f"- **Quoting.** On a replay of {nrep} real days, skewing quotes by inventory with the closed form adds {rv['cl'][0]:.1f} [{rv['cl'][1]:.1f}, {rv['cl'][2]:.1f}] $k/day "
              f"({rv['cl'][3]:.0f}%) over symmetric quotes on WTI. On gas it leaves P&L unchanged ({vs_txt(rv['ng'], rel=False)}) and cuts the 5-minute P&L std {-std_ng:.0f}%. "
              f"EIA inventory surprises and CFTC hedging pressure predict nothing out of sample (|correlation| ≤ {max(abs(c) for c in corrs):.2f}).")
    rd.append("")
    rd.append("| | WTI | Henry Hub |")
    rd.append("|---|---|---|")
    rd.append(f"| repricing error, seasonal NSS → best curve | {nc['loo_test_mean_abs']:.3f} → {sc['loo_test_mean_abs']:.3f} $/bbl | {nn['loo_test_mean_abs']:.3f} → {sn['loo_test_mean_abs']:.3f} $/MMBtu |")
    rd.append("| impact exponent δ, 1 h [95% CI] | " + " | ".join(f"{w1[r]['delta']:.2f} [{w1[r]['delta_ci95'][0]:.2f}, {w1[r]['delta_ci95'][1]:.2f}]" for r in rs) + " |")
    rd.append("| hedge cost per $1M at equal risk: front stack → impact-aware | " + " | ".join(f"{fs[r]['cost_per_mn']:,.0f} → {eq[r]['cost_per_mn']:,.0f}" for r in rs) + " |")
    rd.append("| skewed minus symmetric quotes, real-day replay ($k/day) | " + " | ".join(vs_txt(rv[r], rel=False) for r in rs) + " |")
    rd.append("")
    rd.append("![curves](results/figures/curvehist.png)")
    rd.append("![hedging](results/figures/hedge.png)")
    rd.append("")
    rd.append("All numbers are in `results/summary.md`; the detail is in [docs/method.md](docs/method.md).")
    rd.append("")
    rd.append("## How it works")
    rd.append("")
    rd.append(r"$$F(T,\phi) = a + bT + \sum_j w_j\,k_{\mathrm{Sob}}(T,T_j)\,k_{\mathrm{per}}(\phi,\phi_j)$$")
    rd.append("")
    rd.append(r"$$h^\ast = \arg\min_h\ \mathrm{cost}(h) + \gamma\,\tau\,(q+h)^\top\Sigma\,(q+h)$$")
    rd.append("")
    rd.append("- **Calendar.** NYMEX expiry rules for both contracts. A calendar-month swap settles on the business-day average of the prompt contract, so its legs span two contracts around expiry.")
    rd.append(r"- **Curves.** A seasonal NSS, a cubic P-spline and a seasonal kernel ridge (a Sobolev kernel in time to expiry $T$ times a periodic kernel in the delivery month's phase $\phi$, after Filipović–Pelger–Ye). "
              "All three are linear in prices, so leave-one-out errors and key-rate sensitivities are exact.")
    rd.append(r"- **Impact and hedging.** Power-law self-impact and ridge cross-impact from tick-rule signed volume on hourly bars, bootstrapped over days. The hedge $h$ for a swap book $q$ "
              r"minimises spread and impact cost plus risk under the factor covariance $\Sigma$, dialled to the front stack's residual risk before costs are compared.")
    rd.append("- **RFQ engine.** Client tiers with logistic hit probabilities; quotes from the Bergault–Evangelista–Guéant–Vieira closed form, widened for informed flow (Barzykin–Bergault–Guéant) and before EIA releases; "
              "a 5-minute market with release jumps, an exact P&L decomposition, mark-outs, limits and a kill switch. FIX 4.4 fills are asserted identical to the in-process engine.")
    rd.append("")
    rd.append("## Limitations")
    rd.append("")
    rd.append(f"- There is no free history of full futures curves: Yahoo lists only live contracts, so the curve is complete on only {complete['cl']['0']} (WTI) and {complete['ng']['0']} (gas) days since September 2024. "
              "Every curve result is reported by completeness.")
    rd.append("- Order flow is a tick-rule proxy on hourly bars, not order-book data.")
    rd.append("- RFQ flow is simulated, since no public dealer RFQ data exists; the replay takes only the market path and release times from real days.")
    rd.append("- The hedging saving is relative to the impact model fitted here.")
    rd.append("")
    rd.append("## Run")
    rd.append("")
    rd.append("```")
    rd.append(f"cmake -S . -B build -G Ninja && cmake --build build && ./build/cetk_tests    # {n_cpp} known-answer checks")
    rd.append(f"pip install -e .[test] && python -m pytest                                 # {n_py} tests")
    rd.append("scripts/run_all.sh --skip-download                                         # the pipeline on the checked-in data")
    rd.append("```")
    rd.append("")
    rd.append("The README and `docs/method.md` are written by `scripts/readme.py` from `results/`. CI builds and tests the C++, runs a short pipeline, and checks both files against the results.")

    # ---------------------------------------------------------------- docs/method.md
    md = ["# commodity-etrading-stack: method and full results", ""]
    md.append(QUESTION)
    md.append("")
    md.append("Literature cut-off October 2025. Header-only C++20 behind one CLI (`cetk`), a FIX 4.4 RFQ gateway (`cetk_fixgw`) whose fills are asserted identical to the in-process engine, "
              "and a numpy port of the curve engine, calendar and factor model (`python/cetk`) as the independent reference. "
              f"{n_cpp} C++ known-answer checks, the FIX self-test and {n_py} Python tests; CI builds with g++, runs all of it and a short pipeline on the committed data, "
              "checks that the Python rebuild of a day's curves agrees with the C++ it just built, and checks the README and this page against `results/` (`python scripts/readme.py --check`).")
    md += ["", "---", "", LAYOUT, "", "---", "", DATA, ""]
    md.append(f"Curve completeness since September 2024 (WTI / gas): {complete['cl']['0']} / {complete['ng']['0']} complete days, {complete['cl']['1-3']} / {complete['ng']['1-3']} with 1–3 missing months, "
              f"{complete['cl']['4-12']} / {complete['ng']['4-12']} with 4–12, {complete['cl']['>12']} / {complete['ng']['>12']} with more than 12; the first complete day is "
              f"{dck['cl']['first_complete_day']} (WTI) / {dck['ng']['first_complete_day']} (gas). Every curve result is reported by completeness class. "
              "The hedge test uses the consecutive days on which every contract of the first day is still quoted on the next.")
    md += ["", "---", "", METHOD, "", "---", "", "## Results", "", "Figures from `scripts/plots.py`; tables in `results/summary.md`; the run in `report.pdf`.", ""]

    # curves
    md.append(f"### Curve: three models on {ch['cl']['n_days']} / {ch['ng']['n_days']} daily builds")
    md += ["", "![curve](../results/figures/curve.png)", "", "![curvehist](../results/figures/curvehist.png)", ""]
    md.append("| | WTI (CL) | Henry Hub (NG) |")
    md.append("|---|---|---|")
    md.append(f"| test-period LOO repricing error, seasonal NSS | {nc['loo_test_mean_abs']:.3f} $/bbl ({nc['loo_test_mean_abs_pct']:.2f}%) | {nn['loo_test_mean_abs']:.3f} $/MMBtu ({nn['loo_test_mean_abs_pct']:.2f}%) |")
    for m in ("pspline", "kernel_ridge"):
        row = []
        for r in rs:
            s = ch[r]["selected"][m]; t = f"{s['loo_test_mean_abs']:.3f} ({s['loo_test_mean_abs_pct']:.2f}%), {pc(s['vs_nss_pct'])}% [{pc(s['vs_nss_ci95'][0])}, {pc(s['vs_nss_ci95'][1])}]"
            row.append(f"**{t}**" if best[r][0] == m else t)
        md.append(f"| {NAME[m]} (selected: {ch['cl']['selected'][m]['label']} / {ch['ng']['selected'][m]['label']}) | " + " | ".join(row) + " |")
    md.append("| test-period extrapolation error, last three contracts (NSS / P-spline / KR) | "
              + " | ".join(" / ".join(f"{ch[r]['selected'][m]['extrapolation_test_mean_abs']:.3f}" for m in ("seasonal_nss", "pspline", "kernel_ridge")) for r in rs) + " |")
    md.append(f"| hedge test, month-3 contract dropped, next-day P&L std ({ht['cl']['n_days']} / {ht['ng']['n_days']} days; unhedged / front stack / 50-50 neighbours / NSS / P-spline / KR key rates) | "
              + " | ".join(" / ".join([f"{ht[r]['unhedged_std']:.3g}"] + [f"{ht[r]['methods'][k]['hedged_std']:.3g}" for k in ("front_stack", "neighbours_50_50", "seasonal_nss", "pspline", "kernel_ridge")]) for r in rs) + " |")
    md.append("")
    em = {r: eia[r]["methods"] for r in rs}
    md.append("The two markets want different machines. WTI's curve is smooth and nearly seasonless, so a P-spline with a light penalty wins and the seasonal kernel's extra flexibility costs it at the front. "
              "Henry Hub's curve is a strong annual wave, so the periodic kernel is the right prior: the seasonal NSS with two or three harmonics cannot follow the winter peaks and the P-spline has to bend through them month by month. "
              f"On the {min(eia[r]['from'][:4] for r in rs)}–{eia['cl']['to'][:4]} EIA panel (contracts 1–4 only) the curve's key rates are no better than a 50/50 neighbour hedge "
              f"(WTI {em['cl']['curve_sensitivities']['std']:.3f} vs {em['cl']['neighbours_50_50']['std']:.3f} $/bbl; gas {em['ng']['curve_sensitivities']['std']:.3f} vs {em['ng']['neighbours_50_50']['std']:.3f}) "
              f"and a ridge learned on a trailing window sits between ({em['cl']['learned_ridge']['std']:.3f} / {em['ng']['learned_ridge']['std']:.3f}; learned-over-curve ratio "
              + ", ".join(f"{eia[r]['learned_over_curve_std_ratio']['point']:.2f} [{eia[r]['learned_over_curve_std_ratio']['ci95'][0]:.2f}, {eia[r]['learned_over_curve_std_ratio']['ci95'][1]:.2f}] {lab}" for r, lab in (("cl", "WTI"), ("ng", "gas")))
              + "): with four points a curve has nothing to add, and the modern hedge test above is where the curve earns its keep.")
    md.append("")

    # factors
    rec = {r: fac[r]["recent"] for r in rs}
    decades = {r: [v["explained"][0] for k, v in fac[r]["front_history"].items() if v["n_days"] and int(k[11:15]) - int(k[:4]) < 10] for r in rs}
    yrs = sorted({int(k[:4]) for r in rs for k, v in fac[r]["front_history"].items() if v["n_days"]})
    md += ["### Factor risk", "", "![factors](../results/figures/factors.png)", ""]
    md.append(f"The first factor explains {100 * rec['cl']['explained'][0]:.0f}% of the daily ladder variance for WTI ({rec['cl']['n_days']} days) and is a level shift; "
              f"for Henry Hub ({rec['ng']['n_days']} days) it explains {100 * rec['ng']['explained'][0]:.0f}% and is not parallel — the 6-month contract loads with the opposite sign because the winter and summer contracts move apart — "
              f"and the second factor carries another {100 * (rec['ng']['explained'][1] - rec['ng']['explained'][0]):.0f}%. "
              f"Three factors explain over {int(100 * min(rec[r]['explained'][2] for r in rs))}% of both. "
              f"On the {yrs[0]}–{eia['cl']['to'][:4]} EIA front contracts, four adjacent months, the level share is {100 * min(decades['cl']):.0f}–{100 * max(decades['cl']):.0f}% for crude "
              f"and {100 * min(decades['ng']):.0f}–{100 * max(decades['ng']):.0f}% for gas in every decade.")
    md.append("")

    # impact
    md += ["### Impact from hourly bars", "", "![impact](../results/figures/impact.png)", ""]
    md.append("| window | WTI δ [95% CI] | gas δ [95% CI] | R² self-only → self + cross (WTI / gas) |")
    md.append("|---|---|---|---|")
    for i, w in enumerate(imp["cl"]["windows"]):
        g = imp["ng"]["windows"][i]
        lab = f"{w['window_bars']} h" + (f" ({w['n_windows']:,} / {g['n_windows']:,} windows, {w['n_days']} days, {w['n_contracts']} contracts)" if i == 0 else "")
        md.append(f"| {lab} | {w['delta']:.3f} [{w['delta_ci95'][0]:.3f}, {w['delta_ci95'][1]:.3f}] | {g['delta']:.3f} [{g['delta_ci95'][0]:.3f}, {g['delta_ci95'][1]:.3f}] | "
                  f"{w['r2_self_only']:.3f} → {w['r2_self_plus_cross']:.3f} / {g['r2_self_only']:.3f} → {g['r2_self_plus_cross']:.3f} |")
    md.append("")
    md.append(f"The concave power law fits the binned one-hour response with R² {w1['cl']['r2_concave_bins']:.3f} (WTI) and {w1['ng']['r2_concave_bins']:.3f} (gas); "
              f"a line through the origin has R² {w1['cl']['r2_linear_bins']:.2f} and {w1['ng']['r2_linear_bins']:.2f} on the same bins. "
              "The exponent is low at one hour (the flow-imbalance proxy is noisy at that resolution) and rises toward the 0.4–0.5 the literature reports at longer horizons. "
              + ("After the window the WTI response does not decay (the estimate is treated as permanent); " if imp["cl"]["simulator_model"]["perm_frac"] >= 0.995 else
                 f"After the window the WTI response decays to {imp['cl']['simulator_model']['perm_frac']:.2f} of the concurrent level; ")
              + f"the gas response decays to {imp['ng']['simulator_model']['perm_frac']:.2f} of the concurrent level.")
    md.append("")

    # hedging
    hc = hed["cl"]
    md += ["### Hedging swap books at equal risk", "", "![hedge](../results/figures/hedge.png)", ""]
    md.append(f"| policy (WTI, {hc['books']} books, γ = {hc['gamma']:.0e}) | cost per $1M notional [95% CI] | residual risk | vs front stack [95% CI] |")
    md.append("|---|---|---|---|")
    for k, lab in POLICY.items():
        c = pol["cl"][k]; bold = k == "ImpactAware@FrontRisk"
        cells_ = [lab, f"{c['cost_per_mn']:,.0f} [{c['cost_per_mn_ci95'][0]:,.0f}, {c['cost_per_mn_ci95'][1]:,.0f}]", risk(c["risk_usd"]),
                  "—" if k == "FrontStack" else f"{pc(c['vs_front_pct'])}% [{pc(c['vs_front_ci95'][0])}, {pc(c['vs_front_ci95'][1])}]"]
        md.append("| " + " | ".join(f"**{x}**" if bold else x for x in cells_) + " |")
    md.append("")
    abl = [eq[r]["vs_front_pct"] - pol[r][k]["vs_front_pct"] for r in rs for k in ("ImpactAware_noCross@FrontRisk", "ImpactAware_linear@FrontRisk")]
    md.append(f"Henry Hub: front stack {fs['ng']['cost_per_mn']:,.0f} per $1M at {risk(fs['ng']['risk_usd'])} residual risk; impact-aware at the same risk {eq['ng']['cost_per_mn']:,.0f} "
              f"[{eq['ng']['cost_per_mn_ci95'][0]:,.0f}, {eq['ng']['cost_per_mn_ci95'][1]:,.0f}], {eq['ng']['vs_front_pct']:+.0f}% [{eq['ng']['vs_front_ci95'][0]:+.0f}, {eq['ng']['vs_front_ci95'][1]:+.0f}]; "
              f"ablations {pol['ng']['ImpactAware_noCross@FrontRisk']['vs_front_pct']:+.0f}% / {pol['ng']['ImpactAware_linear@FrontRisk']['vs_front_pct']:+.0f}%. "
              "The saving comes from splitting the hedge across contracts where the book's risk actually sits instead of dumping it into the front month (the concave cost makes many small clips cheaper than one large one); "
              f"the cross terms and the concavity are each worth {-max(abl):.1f} to {-min(abl):.1f} points on top.")
    md.append("")

    # quoting: simulator
    n = sim["cl"]["days"]
    md += ["### Quoting: the closed forms in the simulator and on the real days", "", "![sim](../results/figures/sim.png)", "", "![policy](../results/figures/policy.png)", ""]
    md.append(f"Synthetic days: {n} days × {sim['cl']['seeds']} seeds, common random numbers, so the differences between cells are paired; the levels carry ±$10–20k/day of inventory noise, the paired differences do not. "
              "Risk aversion γ = 1e-06, hedger impact-aware to a $100k band.")
    md.append("")
    for r, lab in (("cl", "WTI"), ("ng", "Henry Hub")):
        d = sim[r]; base = cell(d, "Symmetric", gamma=g0)
        md.append(f"| quoter, {lab} | P&L/day $k | vs Symmetric, paired [95% CI] | 5-min std $k | daily P&L / 5-min var ×10⁻³ | adverse $k/day | fund hit ratio |")
        md.append("|---|---|---|---|---|---|---|")
        for q in QUOTERS:
            c = cell(d, q, gamma=g0)
            md.append(f"| {QLAB[q] if r == 'cl' else q} | {c['pnl'] / n / 1e3:.1f} | {'—' if q == 'Symmetric' else vs_txt(vs(c, base, n))} | {c['pnl_5min_std'] / 1e3:.2f} | "
                      f"{1e3 * c['pnl_per_var']:.2f} | {c['decomposition']['adverse'] / n / 1e3:.1f} | {c['hit_ratio_by_tier'][2]:.2f} |")
        md.append("")
    hi_g = max(c["gamma"] for c in sim["cl"]["cells"])
    cal_lo, cal_hi = cell(sim["cl"], "BEGV_adv_cal", gamma=g0), cell(sim["cl"], "BEGV_adv_cal", gamma=hi_g)
    sym_lo, sym_hi, sym_cl = cell(sim["cl"], "Symmetric", gamma=g0), cell(sim["cl"], "Symmetric", gamma=hi_g), cell(sim["cl"], "Symmetric", gamma=g0)
    hedgers = {}
    for r in rs:
        fr = cell(sim[r], "BEGV_adv_cal", "FrontStack", g0)
        hedgers[r] = {h: (vs(cell(sim[r], "BEGV_adv_cal", h, g0), fr, sim[r]["days"]), hedge_cost(cell(sim[r], "BEGV_adv_cal", h, g0), sim[r]["days"])) for h in ("Strip", "ImpactAware")}
        hedgers[r]["FrontStack"] = (None, hedge_cost(fr, sim[r]["days"]))
    hcpm = {k: {h: cell(v, "BEGV_adv_cal", h, g0)["hedge_cost_per_mn_notional"] for h in ("FrontStack", "ImpactAware")} for k, v in scale.items()}
    variants = {}
    for r in rs:
        d = sim[r]; nd = d["days"]
        a = ant[r]; ac = cell(a, "BEGV_adv_cal", gamma=g0); base_c = cell(d, "BEGV_adv_cal", gamma=g0)
        variants[("anticipate", r)] = (vs(ac, base_c, nd), 100 * (ac["pnl_5min_std"] / base_c["pnl_5min_std"] - 1))
        t = tox[r]
        for q in ("Symmetric", "BEGV_adv_cal"):
            tc = cell(t, q, gamma=g0); variants[("tox", r, q)] = (vs(tc, cell(d, q, gamma=g0), nd), tc["n_externalised"])
    ext_ng = max(variants[("tox", "ng", q)][1] for q in ("Symmetric", "BEGV_adv_cal"))
    md.append(f"Over γ the skewed quoters move up and to the left together — at γ = {hi_g:.0e} the WTI `BEGV_adv_cal` book earns {cal_hi['pnl'] / n / 1e3:.1f} $k/day at a {cal_hi['pnl_5min_std'] / 1e3:.1f} $k 5-minute std "
              f"against {cal_lo['pnl'] / n / 1e3:.1f} / {cal_lo['pnl_5min_std'] / 1e3:.1f} at {g0:.0e} — while the symmetric quoter stays at {sym_hi['pnl'] / n / 1e3:.1f} / {sym_hi['pnl_5min_std'] / 1e3:.1f} "
              f"(against {sym_lo['pnl'] / n / 1e3:.1f} / {sym_lo['pnl_5min_std'] / 1e3:.1f}): with quotes that do not depend on the book, γ only changes the hedge. "
              f"The widening features do what they are meant to (fund hit ratio {sym_cl['hit_ratio_by_tier'][2]:.2f} → {cal_lo['hit_ratio_by_tier'][2]:.2f}, adverse "
              f"{sym_cl['decomposition']['adverse'] / n / 1e3:.1f} → {cal_lo['decomposition']['adverse'] / n / 1e3:.1f} $k/day for `BEGV_adv_cal`) but the informed drift the data support is small next to the spread, so their P&L is a few $k/day. "
              f"Hedgers (`BEGV_adv_cal`, paired vs the front stack): WTI strip {vs_txt(hedgers['cl']['Strip'][0], rel=False)} $k/day at {hedgers['cl']['Strip'][1] / hedgers['cl']['FrontStack'][1]:.1f}× the hedge cost "
              f"({hedgers['cl']['Strip'][1]:.1f} vs {hedgers['cl']['FrontStack'][1]:.1f} $k/day), impact-aware {vs_txt(hedgers['cl']['ImpactAware'][0], rel=False)} at {hedgers['cl']['ImpactAware'][1]:.1f}; "
              f"gas strip {vs_txt(hedgers['ng']['Strip'][0], rel=False)}, impact-aware {vs_txt(hedgers['ng']['ImpactAware'][0], rel=False)}. "
              f"Halving or doubling the impact model scales the band-hedging cost (front stack {hcpm['half']['FrontStack']:.0f} / {hcpm['base']['FrontStack']:.0f} / {hcpm['double']['FrontStack']:.0f} $ per $1M on WTI, "
              f"impact-aware {hcpm['half']['ImpactAware']:.0f} / {hcpm['base']['ImpactAware']:.0f} / {hcpm['double']['ImpactAware']:.0f}) and the two stay level. "
              f"Variants at γ = {g0:.0e} for `BEGV_adv_cal`: anticipatory hedging (hedge the EWMA of recent signed flow before it arrives) {vs_txt(variants[('anticipate', 'cl')][0], rel=False)} $k/day and "
              f"{variants[('anticipate', 'cl')][1]:+.0f}% std on WTI, {vs_txt(variants[('anticipate', 'ng')][0], rel=False)} on gas; toxicity routing {vs_txt(variants[('tox', 'cl', 'Symmetric')][0], rel=False)} "
              f"for the symmetric quoter and {vs_txt(variants[('tox', 'cl', 'BEGV_adv_cal')][0], rel=False)} for the skewed one on WTI ({variants[('tox', 'cl', 'BEGV_adv_cal')][1]} RFQs externalised), "
              + ("no RFQ externalised on gas." if ext_ng == 0 else f"{ext_ng} RFQs externalised on gas."))
    md.append("")

    # quoting: replay
    md += ["![events](../results/figures/events.png)", ""]
    md.append(f"Real days: {nrep} days from {rep['cl']['from']} to {rep['cl']['to']} with {rep['cl']['n_events']} (WTI) / {rep['ng']['n_events']} (gas) EIA release days, "
              f"{rep['cl']['cells'][0]['seeds']} seeds of client flow, impact-aware hedger.")
    md.append("")
    md.append("| quoter | WTI P&L/day $k | vs Symmetric, paired | 5-min std | P&L / var ×10⁻³ | gas P&L/day $k | vs Symmetric | 5-min std | P&L / var ×10⁻³ |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for q in QUOTERS:
        row = [q]
        for r in rs:
            d = rep[r]; c = cell(d, q); nd = d["n_days"]
            row += [f"{c['pnl'] / nd / 1e3:.1f}", "—" if q == "Symmetric" else vs_txt(vs(c, cell(d, "Symmetric"), nd)), f"{c['pnl_5min_std'] / 1e3:.2f}", f"{1e3 * c['pnl_per_var']:.2f}"]
        md.append("| " + " | ".join(row) + " |")
    md.append("")
    ppv = 100 * (cell(rep["cl"], "BEGV")["pnl_per_var"] / cell(rep["cl"], "Symmetric")["pnl_per_var"] - 1)
    std_skew = [100 * (cell(rep["ng"], q)["pnl_5min_std"] / cell(rep["ng"], "Symmetric")["pnl_5min_std"] - 1) for q in QUOTERS[1:4]]
    mo = {r: (markout(cell(rep[r], "Symmetric"), False), markout(cell(rep[r], "Symmetric"), True), markout(cell(rep[r], "BEGV_adv_cal"), True)) for r in rs}
    md.append(f"On the real path the skew is the whole story: {rv['cl'][3]:+.0f}% P&L and {ppv:+.0f}% P&L per unit variance on WTI (`BEGV` against `Symmetric`); "
              f"on gas the skewed quoters leave P&L unchanged and cut the 5-minute std {-max(std_skew):.0f}–{-min(std_skew):.0f}%. "
              f"The fund tier's 30-minute mark-out is {mo['cl'][0]['markout_30m']:+.3f} $/bbl on a normal WTI RFQ and {mo['cl'][1]['markout_30m']:+.3f} in the 30 minutes before a release "
              f"(n = {mo['cl'][1]['n']} under symmetric quotes, against a {100 * mo['cl'][1]['mean_delta']:.0f}-cent half-spread) — the informed pre-release flow is there, but it is a small share of the flow: "
              f"the widening cuts those fills from {mo['cl'][1]['n']} to {mo['cl'][2]['n']} and its P&L effect is inside the noise. "
              f"The gas replay says the same ({mo['ng'][1]['markout_30m']:+.3f} → {mo['ng'][2]['markout_30m']:+.3f} $/MMBtu on {mo['ng'][1]['n']} → {mo['ng'][2]['n']} fills).")
    md.append("")

    # releases
    md += ["### Releases: what the EIA history says", "", "![events_study](../results/figures/events_study.png)", ""]
    md.append("| | WTI (crude stocks, Wednesday) | Henry Hub (storage, Thursday) |")
    md.append("|---|---|---|")
    md.append("| release days / other days | " + " | ".join(f"{stu[r]['n_release_days']:,} / {stu[r]['n_other_days']:,}" for r in rs) + " |")
    md.append("| release-day variance ratio, jump share κ | " + " | ".join(f"{stu[r]['release_variance_ratio']:.2f}, {stu[r]['kappa_jump_share']:.2f}" for r in rs) + " |")
    for k, lab in (("surprise_nowcast", "surprise → release-day move, out of sample (2020–24)"), ("hedging_pressure", "hedging pressure → next-week move, out of sample")):
        md.append(f"| {lab} | " + " | ".join(f"corr {stu[r][k]['corr_oos']:+.2f} [{stu[r][k]['corr_oos_ci95'][0]:+.2f}, {stu[r][k]['corr_oos_ci95'][1]:+.2f}], p = {stu[r][k]['permutation_p']:.2f}" for r in rs) + " |")
    md.append("")
    md.append(f"A crude inventory release is not an event for the daily variance of WTI (variance ratio {stu['cl']['release_variance_ratio']:.2f}); a storage release is for gas "
              f"({stu['ng']['release_variance_ratio']:.2f}), and {100 * stu['ng']['kappa_jump_share']:.0f}% of that day's variance sits in the jump. "
              "Neither the inventory surprise measured against a seasonal-mean nowcast (the free substitute for the analyst consensus) nor the CFTC positioning predicts anything out of sample; "
              "both are reported and neither is used by a quoter.")
    md += ["", "---", "", VALIDATION, f"- {n_cpp} C++ checks, {n_py} Python tests, CI on every push.", "", TRAPS, "", REFERENCES]
    return "\n".join(rd) + "\n", "\n".join(md) + "\n"


def main(argv):
    texts = dict(zip(OUT, build()))
    if "--check" in argv:
        stale = []
        for rel, text in texts.items():
            path = os.path.join(ROOT, rel)
            cur = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
            if cur != text:
                stale.append(rel)
        if stale:
            print(" and ".join(stale), "differ from the regenerated text: run python scripts/readme.py")
            sys.exit(1)
        print("README.md and docs/method.md match the results")
        return
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    for rel, text in texts.items():
        open(os.path.join(ROOT, rel), "w", encoding="utf-8", newline="\n").write(text)
        print(rel, "written", len(text), "chars")


# ------------------------------------------------------------------------------------------------ fixed text of docs/method.md
QUESTION = r"""**Question.** A commodity swap desk quoting calendar-month WTI and Henry Hub swaps by RFQ needs a futures curve to price off (including the months no contract trades), hedge ratios for a book of swaps, an execution policy for the hedge that respects price impact, and a quoting rule that survives informed flow around the weekly EIA releases. Does the recent research — seasonal kernel-ridge and P-spline curves (Filipović–Pelger–Ye kernels), nonparametric cross-impact (Hey–Neuman–Tuschmann), impact-aware hedging, and the Barzykin–Bergault–Guéant closed-form multi-asset quotes with adverse-selection and calendar widening — improve on the desk's classical answers (seasonal Nelson–Siegel–Svensson, a front-month stack or strip, symmetric spreads) when each is tested out of sample on free public data and, for quoting, in an RFQ simulator with an exact P&L decomposition and a replay of the real 2024–2026 days?"""

LAYOUT = r"""## Layout

| path | what |
|---|---|
| `tools/download.py`, `tools/build_curves.py` | free data (EIA, Yahoo chart API, CFTC) → `data/raw/`; daily curves and hourly bar panels → `data/derived/` (committed, so everything below runs without downloads) |
| `include/cetk/calendar.hpp` | NYMEX expiry rules (WTI: 3 business days before the 25th of the prior month; Henry Hub: 3 business days before the delivery month), n-th delivery, years to expiry, seasonal phase |
| `include/cetk/curve.hpp` | seasonal NSS (τ grid, two annual harmonics), cubic P-spline (second-difference penalty), seasonal kernel ridge (Sobolev cubic × periodic kernel); leave-one-out, extrapolation, hat-matrix key-rate sensitivities; calendar-month swaps and strips |
| `include/cetk/factors.hpp`, `linalg.hpp` | PCA (Jacobi) of the ladder, covariance at any tenors; ridge solves, symmetric powers |
| `include/cetk/impact.hpp` | Hey–Neuman–Tuschmann-style estimation from bars: pooled power-law self-impact, ridge cross-impact, block bootstrap over days, cumulative decay; the `ImpactModel` the hedger and simulator use (`configs/impact_*.json`) |
| `include/cetk/hedge.hpp` | policies FrontStack, Strip, FactorMin, ImpactAware (coordinate descent on the exact cost with the kink at zero), ablations without cross terms / linear; `hedge_to_risk` (log-bisection on γ) for equal-risk comparisons; the random-book experiment |
| `include/cetk/clients.hpp`, `quoter.hpp` | client tiers (consumer / producer / fund) with logistic hit probabilities and intensity seasonality; Symmetric, BEGV closed form, `_adv` (tier widening), `_adv_cal` (pre-release widening), Oracle |
| `include/cetk/engine.hpp`, `risk.hpp`, `toxicity.hpp` | the RFQ engine: 5-minute market on the factor model with a hidden informed direction, release-day jumps, transient and permanent own impact, mark-outs, exact P&L decomposition, band hedging, internalisation, toxicity classifier (EKF logistic) with routing, anticipatory hedging, position / risk / drawdown limits with a kill switch, common random numbers; the real-day replay |
| `include/cetk/fix.hpp`, `src/fix_gateway.cpp` | FIX 4.4 session layer and RFQ application messages (QuoteRequest / Quote / NewOrderSingle / ExecutionReport); `--serve` and the identical-fills `--selftest` |
| `src/main.cpp` → `cetk` | `curve`, `curvehist`, `factors`, `impact`, `hedge`, `sim`, `events`, `price`, `datacheck` |
| `tools/hedge_test.py`, `tools/event_features.py` | learned vs curve hedge ratios on the 1986–2024 EIA panel; release calendar, storage / stocks surprises, hedging pressure → `data/derived/events_*.csv`, `results/events_study_*.json` |
| `tools/api.py`, `tools/fix_client.py`, `tools/record_snapshot.py` | REST front end over `cetk price`; a Python FIX client for the gateway; the daily full-curve recorder |
| `python/cetk/` | `calendar.py`, `curve.py`, `factors.py`: numpy port agreeing with the C++ to a few 10⁻⁶ on a day's fits; `data.py`, `results.py`, `stats.py`; `python -m cetk curve|compare|summary` |
| `python/tests/`, `tests/tests.cpp` | pytest: expiry rules, models on synthetic curves, sensitivities sum to one, swap weights, PCA signs, bootstrap and ridge, every model number of the committed C++ curve build re-derived; C++: calendar, models, swaps, factors, impact on synthetic bars, hedging at equal risk, quoter identities, engine decomposition / CRN / mark-outs / routing / limits, FIX round trip and loopback |
| `scripts/run_all.sh`, `plots.py`, `summarize.py`, `report.py`, `readme.py` | pipeline, figures, `results/summary.md`, `report.pdf`, the README and `docs/method.md`; `notebooks/results.ipynb` |

Build: `./build.ps1` (MSVC + Ninja) or `cmake -S . -B build -G Ninja && cmake --build build`; `python -m pytest` (`PYTHONPATH=python` or `pip install -e .[test]`). Python: `numpy matplotlib fpdf2 xlrd` (xlrd only for the EIA spreadsheets)."""

DATA = r"""## Data

There is no free source of historical *full* futures curves, and the plan said so before any code was written. What exists, and what the stack does with it:

| layer | source | coverage | use |
|---|---|---|---|
| front contracts 1–4, daily settlements | EIA (NYMEX via EIA, spreadsheets) | WTI 1983-03 – 2024-04-05, Henry Hub 1993-12 – 2024-04-05 | long-history factor model by decade, learned-vs-curve hedge test, release-day study |
| every listed contract, daily closes | Yahoo chart API (`CLX26.NYM`, …), 2y range | 2024-09 – 2026-09, contracts listed today only | daily curve builds; a contract that has since expired is missing on past dates, so the curve is complete only from the last expiry (2026-07-21 WTI, 2026-07-29 gas) and every day records `gap_months`, the missing delivery months within the first 24 |
| continuous front `CL=F` / `NG=F` | Yahoo, daily | 2024-09 – 2026-09 | month 1 on every day (median difference to the EIA front on the overlap 0.000) |
| hourly bars per listed contract | Yahoo, 730d range | 2024-09 – 2026-09, the last 90 days used | impact estimation (the twelve nearest deliveries, 76 trading days, ~1,430 hourly windows) |
| weekly storage / crude stocks; release calendar | EIA | 1994– / 1982– | release days, surprises (vs a seasonal-mean nowcast), the jump share of release-day variance |
| Commitments of Traders, disaggregated | CFTC | 2010–2026 | hedging pressure (Kang–Rouwenhorst–Tang) |
| daily full-curve snapshots | `tools/record_snapshot.py` | from the day it is first run | grows the complete-curve history one day at a time |"""

METHOD = r"""## Method

**Calendar.** WTI's delivery-month contract expires three business days before the 25th of the prior month (the 25th moved back to a business day); Henry Hub's three business days before the delivery month. A calendar-month swap settles on the business-day average of the prompt contract, so a December WTI swap is 65 % January and 35 % February; a Henry Hub calendar month is the single next-month contract except in months whose contract expires early. Legs whose contract is not observed are priced off the model and flagged.

**Curve models** (all linear in prices, so leave-one-out, extrapolation and sensitivities are exact hat-matrix identities). With $T$ years to expiry and $\phi$ the seasonal phase of the delivery month:
- seasonal NSS: $F(T,\phi) = \beta_0 + \beta_1 e^{-T/\tau_1} + \beta_2 \tfrac{T}{\tau_1}e^{-T/\tau_1} + \beta_3 \tfrac{T}{\tau_2}e^{-T/\tau_2} + \sum_{h=1}^{H}\big(a_h\cos 2\pi h\phi + b_h \sin 2\pi h\phi\big)$, $(\tau_1,\tau_2)$ on a grid, $H \in \{2,3\}$;
- P-spline: cubic B-splines with knots every 1 or 2 months and a second-difference penalty $\lambda\sum(\Delta^2 c)^2$;
- seasonal kernel ridge: $F(T,\phi) = a + bT + \sum_j w_j\,k\big((T,\phi),(T_j,\phi_j)\big)$ with $k = k_{\text{Sob}}(T,T')\,k_{\text{per}}(\phi,\phi')$, $k_{\text{Sob}}(s,t) = \min(s,t)^2(3\max(s,t)-\min(s,t))/6$ (Filipović–Pelger–Ye) and $k_{\text{per}} = \exp\!\big(-2\sin^2(\pi(\phi-\phi'))/\ell^2\big)$, ridge $\lambda$.

Candidates (NSS $H$; P-spline knots × $\lambda \in [10^{-4}, 1]$; kernel ridge $\ell \in \{0.2, 0.35, 0.5\}$ × $\lambda \in [10^{-5}, 10^{-1}]$) are scored by leave-one-out repricing on the first third of the daily builds; the best per model is fixed and scored on the remaining two thirds. Key-rate sensitivities $\partial F(T^\ast)/\partial P_i$ are the hat-matrix row (NSS with the shape frozen); they sum to one. Hedge test: the month-3 contract is dropped, repriced off the curve built from the rest and hedged with the other contracts using that curve's sensitivities; the realised next-day P&L of the package is the score.

**Factors.** PCA of daily changes of the fitted curve on the $\{1,2,3,6,12\}$-month ladder (recent window, days with at most two missing months) and of the EIA front panel by decade; $\Sigma = \sum_k \lambda_k L_k L_k^\top$ at the ladder in \$/contract-unit per day.

**Impact.** For each hourly window and contract $i$: $Q_i$ = tick-rule signed volume / average daily volume, $R_i$ = log return / its std. Pooled self-impact $\text{sign}(Q)\,R = Y|Q|^\delta$ fitted in log–log on 16 volume bins; $\delta$'s 95 % CI by block bootstrap over days; cross-impact by ridge of $R_j$ on $\text{sign}(Q_i)|Q_i|^\delta$ for all $i$ (self-only vs full R² on the raw windows); the response after the window $G(\tau)/G(0)$ gives the transient decay rate and the permanent fraction. The simulator's `ImpactModel` (σ, daily volume, $Y$, $\Lambda$, decay, permanent fraction, half-spreads) is restricted to the ladder.

**Hedging.** For a swap book $q$ (contracts on the ladder) the hedge $h$ minimises $\text{cost}(h) + \gamma\,\tau\,(q+h)^\top\Sigma(q+h)$ where the cost is spread + transient impact $\sum_j \sigma_j Y_j |h_j/V_j|^{\delta} |h_j|$ + permanent cross-impact $\tfrac12 h^\top \Lambda h$-type terms, all in dollars; coordinate descent with a golden-section line search and the kink at zero. FrontStack puts the whole delta in month 1, Strip hedges each month one for one, FactorMin minimises the factor risk only. For the comparison every policy's cost is evaluated under the full model, and the impact-aware hedge and its ablations (no cross terms; linear impact) are dialled by log-bisection on $\gamma$ to the front stack's residual risk, so the saving is at equal risk; 400 random books, bootstrap CIs.

**Quoting** (half-spread $\delta$ in \$/unit; tier hit probability $f_i(\delta) = 1/(1+e^{\alpha_i+\beta_i\delta})$):
- Symmetric: $f_i(\delta_i) = h^\ast$ per tier.
- BEGV closed form: exponential intensities fitted at $\arg\max_\delta \delta f(\delta)$, $D_j = A_j \bar z k_j / e$, $\Gamma = D^{-1/2}\big(D^{1/2}\tfrac{\gamma}{2}\Sigma D^{1/2}\big)^{1/2} D^{-1/2}$ (so $\Gamma D \Gamma = \tfrac{\gamma}{2}\Sigma$, tested), $\delta_j^{\pm} = \tfrac{1}{k} + \tfrac12 \Gamma_{jj} z \mp (\Gamma q)_j$.
- `BEGV_adv`: plus each tier's expected adverse move $\mu_i$ over the holding period (the Barzykin–Bergault–Guéant treatment of informed flow); `BEGV_adv_cal`: plus a fixed widening in the 30 minutes before a scheduled release; Oracle: the true $\mu_i$ and $p_i\,E|\text{jump}|$ — the closed form given the client model.

**Simulator.** Poisson RFQs per (tier, tenor, side) with seasonal intensity, sizes 10 / 25 / 50 / 100 contracts; a fill moves inventory and earns $z\,\delta$ per unit; the market moves on a 5-minute grid from the factor model with a hidden informed direction that switches at rate 0.5/day and drifts the curve (0.6 \$/bbl/day WTI, 0.03 \$/MMBtu/day gas, scaled by the level loadings), which the fund tier trades with; release days add a jump at the release time (half at once, half over 30 minutes); the dealer's own hedges move the market with the fitted transient and permanent impact; the book is hedged every 30 minutes to a risk band by the chosen policy. Exact decomposition: spread + inventory + adverse + hedge execution + hedge impact = total, residual $10^{-7}$. Mark-outs at 5 / 30 / 60 minutes by tier and window, a toxicity classifier (EKF logistic on mark-outs) with optional routing of toxic RFQs, anticipatory hedging (Nutz–Webster–Zhao), position / risk / drawdown limits with a kill switch. Common random numbers: every quoter sees the same market and the same RFQ script. The replay uses the real 2024-09 – 2026-09 days: the front contract follows the realised close-to-close change (Brownian bridge intraday, the release jump at the release time), the rest of the ladder moves with the factor model."""

VALIDATION = r"""## Validation

- Calendar: `CLX26` expires 2026-10-20, `NGX26` 2026-10-28, `CLV26` 2026-09-22 (tests, and the same rules in Python); the December WTI swap is 15 January + 8 February business days.
- Every curve model reprices its own inputs within the ridge; key-rate sensitivities sum to one; the P-spline penalty orders smoothness; LOO and extrapolation are computed by refitting, and the candidate selection was fixed on the first third of the days before the test numbers were seen.
- Two implementations of the curve: the C++ that produced the numbers and an independent numpy port compared number by number on the committed day (`python -m cetk compare`: a maximum difference of a few 10⁻⁶; it fails beyond 10⁻⁴ of the mean price) and in CI on the day it just built.
- Impact: on synthetic bars generated with exponent 0.5 and known cross terms the estimator recovers a concave exponent, its bootstrap band contains the point estimate, and the cross terms add explanatory power only when they are there; the decay is a cumulative response, not a lag correlation.
- Hedging: the equal-risk cells sit within 3 % of the front stack's residual risk; the ablations use the same books and seeds; bootstrap CIs over books.
- Quoting: $\Gamma D\Gamma = \tfrac{\gamma}{2}\Sigma$ to $10^{-6}$; quotes skew the right way; the decomposition sums to the total to $10^{-7}$; identical quotes on the same seed give identical P&L; the fund tier's 30-minute mark-outs are more favourable to the client than the consumers'; the adverse widening is wider for the fund tier and wider again before a release; the kill switch and the position limit fire when they should; the FIX loopback produces the same fills and P&L as the in-process run (`cetk_fixgw --selftest`)."""

TRAPS = r"""## Traps stated

- *No free curve history.* The daily builds cover two years and are complete only from July 2026; the completeness class is on every table, and the hedge test uses only consecutive days on which every contract is still quoted. `tools/record_snapshot.py` is the fix, one day at a time.
- *Client flow is simulated.* The quoting results measure what the closed forms do in a stated client model, not desk P&L; the replay borrows only the real market path and release times.
- *Impact from bars, not ticks.* Tick-rule signed volume on hourly bars is a proxy for order-flow imbalance; the low one-hour exponent is partly that noise. The hedge-cost saving is model-relative: the dollar numbers move with the model, the ranking survives halving or doubling it (`sim_impact_half`, `sim_impact_double`).
- *P&L levels are noisy; differences are not.* Forty days × eight seeds leaves ±$10–20k/day on a level; all quoter and hedger comparisons are paired on the same seeds. Spreads come from the assumed client hit model (half-spreads of roughly 40 c/bbl and 4 c/MMBtu on the closed forms), not from a real desk's flow, so the P&L levels are stylised.
- *Yahoo is a convenience feed*: unofficial, rate-limited, and the intraday history is ephemeral (730 days), which is why the derived bars are committed."""

REFERENCES = r"""## References

- Filipović, Pelger, Ye (2022) *Stripping the discount curve — a robust machine learning approach*, [arXiv:2206.05002](https://arxiv.org/abs/2206.05002).
- Hey, Neuman, Tuschmann (2025) *Nonparametric estimation of cross-impact* — the estimation design followed here, [arXiv:2510.06879](https://arxiv.org/abs/2510.06879); Tomas, Mastromatteo, Benzaquen (2021) *Cross-impact in multi-asset markets*, [arXiv:2102.02834](https://arxiv.org/abs/2102.02834).
- Bergault, Evangelista, Guéant, Vieira (2021) *Closed-form approximations in multi-asset market making*, [arXiv:1810.04383](https://arxiv.org/abs/1810.04383).
- Barzykin, Bergault, Guéant (2023, 2024) *Dealing with multi-currency inventory risk in FX cash markets* and *Market making in spot precious metals*, [arXiv:2404.15478](https://arxiv.org/abs/2404.15478); Barzykin, Bergault, Guéant, Lu (2025) *Market making with informed flow*, [arXiv:2508.20225](https://arxiv.org/abs/2508.20225).
- Nutz, Webster, Zhao (2025) *Anticipatory hedging under price impact*; Cartea, Jaimungal, Sánchez-Betancourt on toxicity and mark-outs — implemented as described in the project plan.
- Kang, Rouwenhorst, Tang (2020) *A tale of two premiums: the role of hedgers and speculators in commodity futures markets*, Journal of Finance 75(1).
- Seasonal curve studies: the TTF curve comparison in *Energies* (2023); [arXiv:2501.15596](https://arxiv.org/abs/2501.15596), [arXiv:2503.18199](https://arxiv.org/abs/2503.18199).
- Data: EIA petroleum and natural-gas spreadsheets; CFTC disaggregated Commitments of Traders; Yahoo Finance chart API."""

if __name__ == "__main__":
    main(sys.argv[1:])
