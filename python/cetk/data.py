"""Loaders for the derived data and the raw EIA / CFTC files."""
import csv
import datetime as dt
import os

import numpy as np

from . import calendar as cal


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_curves(root, path=None, since=None):
    """{date: {'delivery': [...], 'months': array, 'T': array, 'phase': array, 'price': array, 'gap': int}}"""
    path = path or f"data/derived/curves_{root.lower()}.csv"
    days = {}
    for r in read_csv(path):
        d = r["date"]
        if since and d < since:
            continue
        days.setdefault(d, []).append(r)
    out = {}
    for d, rows in days.items():
        rows.sort(key=lambda r: int(r["months"]))
        seen = set(); keep = []
        for r in rows:
            if int(r["months"]) in seen:
                continue
            seen.add(int(r["months"])); keep.append(r)
        if len(keep) < 4:
            continue
        months = np.array([int(r["months"]) for r in keep]); T = np.array([float(r["T"]) for r in keep]); price = np.array([float(r["price"]) for r in keep])
        phase = np.array([cal.season_phase(int(r["delivery"][5:7])) for r in keep]); dely = [r["delivery"] for r in keep]
        gap = 0
        for i in range(1, len(months)):
            if months[i - 1] < 24:
                gap += min(24, months[i]) - months[i - 1] - 1
        out[d] = {"delivery": dely, "months": months, "T": T, "phase": phase, "price": price, "gap": gap, "source": [r["source"] for r in keep]}
    return dict(sorted(out.items()))


def load_front(root, path=None):
    """EIA contracts 1-4: dates list and (n, 4) array (rows with all four contracts)."""
    path = path or f"data/raw/eia_{root.lower()}_front.csv"
    dates, X = [], []
    for r in read_csv(path):
        try:
            vals = [float(r[f"c{k}"]) for k in range(1, 5)]
        except ValueError:
            continue
        dates.append(r["date"]); X.append(vals)
    return dates, np.array(X)


def load_storage(root):
    path = "data/raw/eia_ng_storage.csv" if root == "NG" else "data/raw/eia_crude_stocks.csv"
    rows = read_csv(path)
    return [(r["week_ending"], float(r["value"])) for r in rows if r["value"]]


def release_date(root, week_ending):
    """EIA weekly petroleum status: Wednesday 10:30 ET after the Friday week-ending; natural-gas storage: Thursday.
    Holiday shifts are not modelled (stated)."""
    d = dt.date.fromisoformat(week_ending)
    return d + dt.timedelta(days=6 if root == "NG" else 5)


def load_cot(root):
    """Weekly disaggregated COT: date -> dict of positions (producer/merchant, swap dealers, managed money, open interest)."""
    path = f"data/raw/cot_{root.lower()}.csv"
    if not os.path.exists(path):
        return {}
    out = {}
    for r in read_csv(path):
        d = r["Report_Date_as_MM_DD_YYYY"].strip()
        try:
            out[d] = {k: float(r[k]) for k in ("Open_Interest_All", "Prod_Merc_Positions_Long_All", "Prod_Merc_Positions_Short_All", "Swap_Positions_Long_All", "Swap__Positions_Short_All", "M_Money_Positions_Long_All", "M_Money_Positions_Short_All")}
        except (ValueError, KeyError):
            continue
    return dict(sorted(out.items()))
