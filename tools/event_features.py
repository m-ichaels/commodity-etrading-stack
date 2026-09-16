#!/usr/bin/env python3
"""Scheduled releases and positioning as features, on the long EIA history, and the feature table for the replay.
   python tools/event_features.py [--roots CL,NG] [--train-end 2018-12-31]

  Release days   EIA Weekly Petroleum Status Report (Wednesday, crude stocks) / Weekly Natural Gas Storage Report (Thursday):
                 release date = week-ending Friday + 5 / 6 days (holiday shifts not modelled).  Test: |front-month move| on
                 release days vs other days; the share of release-day variance attributable to the release, kappa, which
                 the replay uses to size the 10:30 jump.
  Storage surprise   actual weekly change minus the seasonal expectation (mean change for the same week of the year over the
                 previous five years, known before the release), standardised on the training window.  Nowcast of the
                 release-day move: slope fitted on the training window, evaluated out of sample with a bootstrap band and a
                 permutation placebo.  Consensus surveys are not free; the seasonal expectation is the stated proxy.
  Hedging pressure   CFTC disaggregated positions: HP = (producer/merchant short - long) / open interest, Tuesday positions
                 published Friday; the following week's front-month return regressed on the weekly change in HP
                 (Kang, Rouwenhorst & Tang 2020).  Out of sample with placebo, as above.
Writes results/events_study_<root>.json and data/derived/events_<root>.csv (date, front_change, release, release_jump,
surprise_z, hp_change) for `cetk events`.
"""
import datetime as dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, "python")
from cetk import data, stats  # noqa: E402

ARGS = sys.argv[1:]
ROOTS = [r for r in (next((a.split("=", 1)[1] for a in ARGS if a.startswith("--roots=")), "CL,NG")).split(",") if r]
TRAIN_END = next((a.split("=", 1)[1] for a in ARGS if a.startswith("--train-end=")), "2018-12-31")
os.makedirs("results", exist_ok=True); os.makedirs("data/derived", exist_ok=True)


def front_changes(root):
    """daily change of the month-1 contract, contract-by-contract (same delivery on both days), by date"""
    days = data.load_curves(root)
    dates = sorted(days); out = {}
    for i in range(1, len(dates)):
        a, b = days[dates[i - 1]], days[dates[i]]
        dely = b["delivery"][0]
        if dely in a["delivery"]:
            out[dates[i]] = float(b["price"][0] - a["price"][a["delivery"].index(dely)])
    return out


def run(root):
    ch = front_changes(root); dates = sorted(ch)
    storage = data.load_storage(root); by_week = {w: v for w, v in storage}
    # weekly change and the seasonal expectation
    weeks = sorted(by_week); change = {}; expect = {}
    for i in range(1, len(weeks)):
        change[weeks[i]] = by_week[weeks[i]] - by_week[weeks[i - 1]]
    for w in weeks[1:]:
        d = dt.date.fromisoformat(w); iso = d.isocalendar()[1]; hist = []
        for k in range(1, 6):
            yy = d.year - k
            for w2 in weeks[1:]:
                d2 = dt.date.fromisoformat(w2)
                if d2.year == yy and abs(d2.isocalendar()[1] - iso) <= 1:
                    hist.append(change[w2])
        if len(hist) >= 3:
            expect[w] = float(np.mean(hist))
    release = {}   # release date -> surprise
    for w in weeks[1:]:
        if w in expect:
            release[data.release_date(root, w).isoformat()] = change[w] - expect[w]
    rel_dates = [d for d in dates if d in release]; other = [d for d in dates if d not in release]
    r_rel = np.array([ch[d] for d in rel_dates]); r_oth = np.array([ch[d] for d in other])
    v_rel, v_oth = float(np.var(r_rel)), float(np.var(r_oth)); kappa = float(np.sqrt(max(0.0, 1 - v_oth / v_rel))) if v_rel > 0 else 0.0
    ab_rel, ab_lo, ab_hi = stats.bootstrap_ci(np.abs(r_rel)); ab_oth = float(np.mean(np.abs(r_oth)))
    out = {"root": root, "n_release_days": len(rel_dates), "n_other_days": len(other), "release_abs_move": ab_rel, "release_abs_move_ci95": [ab_lo, ab_hi], "other_abs_move": ab_oth, "release_variance_ratio": v_rel / v_oth if v_oth > 0 else None, "kappa_jump_share": kappa}
    # surprise nowcast: standardise on the training window, slope on training, OOS on the rest
    tr = [d for d in rel_dates if d <= TRAIN_END]; te = [d for d in rel_dates if d > TRAIN_END]
    s_tr = np.array([release[d] for d in tr]); sd = float(np.std(s_tr)) if len(s_tr) > 5 else 1.0
    z = {d: release[d] / sd for d in rel_dates}
    if len(tr) > 20 and len(te) > 10:
        x_tr = np.array([z[d] for d in tr]); y_tr = np.array([ch[d] for d in tr]); b = float(np.sum(x_tr * y_tr) / max(np.sum(x_tr * x_tr), 1e-12))
        x_te = np.array([z[d] for d in te]); y_te = np.array([ch[d] for d in te]); c, lo, hi = stats.corr_ci(b * x_te, y_te); p = stats.permutation_p(x_te, y_te)
        out["surprise_nowcast"] = {"train_n": len(tr), "test_n": len(te), "slope_train": b, "corr_train": float(np.corrcoef(x_tr, y_tr)[0, 1]), "corr_oos": c, "corr_oos_ci95": [lo, hi], "permutation_p": p, "sign_hit_oos": float(np.mean(np.sign(b * x_te) == np.sign(y_te)))}
    # hedging pressure
    cot = data.load_cot(root); hp = {}
    for d, v in cot.items():
        if v["Open_Interest_All"] > 0:
            hp[d] = (v["Prod_Merc_Positions_Short_All"] - v["Prod_Merc_Positions_Long_All"]) / v["Open_Interest_All"]
    hp_dates = sorted(hp); rows = []
    for i in range(1, len(hp_dates)):
        d_t = dt.date.fromisoformat(hp_dates[i]); dhp = hp[hp_dates[i]] - hp[hp_dates[i - 1]]
        # published on Friday: the next week's return runs from the following Monday for five trading days
        mon = d_t + dt.timedelta(days=(7 - d_t.weekday()) % 7 or 7); span = [d for d in dates if mon.isoformat() <= d <= (mon + dt.timedelta(days=6)).isoformat()]
        if span:
            rows.append((hp_dates[i], dhp, float(sum(ch[d] for d in span))))
    if len(rows) > 60:
        tr_r = [r for r in rows if r[0] <= TRAIN_END]; te_r = [r for r in rows if r[0] > TRAIN_END]
        if len(tr_r) > 30 and len(te_r) > 10:
            x_tr = np.array([r[1] for r in tr_r]); y_tr = np.array([r[2] for r in tr_r]); b = float(np.sum(x_tr * y_tr) / max(np.sum(x_tr * x_tr), 1e-12))
            x_te = np.array([r[1] for r in te_r]); y_te = np.array([r[2] for r in te_r]); c, lo, hi = stats.corr_ci(b * x_te, y_te); p = stats.permutation_p(x_te, y_te)
            out["hedging_pressure"] = {"train_n": len(tr_r), "test_n": len(te_r), "slope_train": b, "corr_train": float(np.corrcoef(x_tr, y_tr)[0, 1]), "corr_oos": c, "corr_oos_ci95": [lo, hi], "permutation_p": p}
    json.dump(out, open(f"results/events_study_{root.lower()}.json", "w"), indent=1)
    # feature table for the replay (2024-09 onward, the Yahoo window)
    hp_by_date = {}
    for i in range(1, len(hp_dates)):
        hp_by_date[hp_dates[i]] = hp[hp_dates[i]] - hp[hp_dates[i - 1]]
    last_hp = 0.0
    with open(f"data/derived/events_{root.lower()}.csv", "w") as f:
        f.write("date,front_change,release,release_jump,surprise_z,hp_change\n")
        for d in dates:
            if d < "2024-09-01":
                continue
            if d in hp_by_date:
                last_hp = hp_by_date[d]
            rel = 1 if d in release else 0; jump = kappa * ch[d] if rel else 0.0; sz = z.get(d, 0.0) if rel else 0.0
            f.write(f"{d},{ch[d]:.5f},{rel},{jump:.5f},{sz:.4f},{last_hp:.5f}\n")
    print(root, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items() if not isinstance(v, dict)}), "nowcast", out.get("surprise_nowcast", {}).get("corr_oos"), "hp", out.get("hedging_pressure", {}).get("corr_oos"))


if __name__ == "__main__":
    for root in ROOTS:
        run(root)
