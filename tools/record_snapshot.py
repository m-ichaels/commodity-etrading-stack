#!/usr/bin/env python3
"""Record today's full futures curve (every listed contract's last price and volume) so the complete-curve history grows
one day at a time.   python tools/record_snapshot.py [CL NG] [--date YYYY-MM-DD]

Writes data/raw/snapshots/<ROOT>_<date>.csv with columns date, delivery, price, volume, symbol; tools/build_curves.py
merges these as source "snapshot".  Free historical full curves do not exist (see README), so a desk that wants the
history simply runs this once a day after the settlement (a cron entry or a scheduled task)."""
import csv
import datetime as dt
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download import MONTH_CODES, yahoo_chart  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
SNAP = os.path.join(RAW, "snapshots")


def listed_symbols(root):
    """Listed contracts: the cached enumeration from download.py when present, else the next 36 months."""
    p = os.path.join(RAW, f"yahoo_listed_{root.lower()}.json")
    if os.path.exists(p):
        return [(c["symbol"], c["delivery"]) for c in json.load(open(p))]
    today = dt.date.today(); out = []
    for k in range(1, 37):
        y, m = today.year + (today.month + k - 1) // 12, (today.month + k - 1) % 12 + 1
        out.append((f"{root}{MONTH_CODES[m - 1]}{y % 100:02d}.NYM", f"{y}-{m:02d}"))
    return out


def record(root, date):
    os.makedirs(SNAP, exist_ok=True)
    rows = []
    for sym, delivery in listed_symbols(root):
        if delivery <= date[:7]:
            continue
        bars = yahoo_chart(sym, "5d", "1d")
        if not bars:
            continue
        # the last bar on or before the requested date
        best = None
        for t, o, h, lo, c, v in bars:
            d = dt.datetime.fromtimestamp(t, dt.timezone.utc).date().isoformat()
            if d <= date:
                best = (d, c, v)
        if best and best[0] == date:
            rows.append([date, delivery, f"{best[1]:.4f}", f"{best[2]:.0f}", sym])
        time.sleep(0.1)
    out = os.path.join(SNAP, f"{root}_{date}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["date", "delivery", "price", "volume", "symbol"]); w.writerows(rows)
    print(f"{root} {date}: {len(rows)} contracts -> {out}")
    return len(rows)


def main():
    roots = [a for a in sys.argv[1:] if not a.startswith("--")] or ["CL", "NG"]
    date = dt.date.today().isoformat()
    if "--date" in sys.argv:
        date = sys.argv[sys.argv.index("--date") + 1]
    for r in roots:
        record(r.upper(), date)


if __name__ == "__main__":
    main()
