#!/usr/bin/env python3
"""Assemble daily futures curves and bar panels from the raw downloads.   python tools/build_curves.py [--roots CL,NG]

Curves (data/derived/curves_<root>.csv: date, delivery, months, T, price, volume, source):
  * EIA contracts 1-4 (1983/1993 to April 2024, when EIA stopped publishing NYMEX settlements), mapped to delivery
    months with the NYMEX expiry rule in include/cetk/calendar.hpp (weekday business days; the mapping is checked
    against Yahoo where both exist and the agreement rate is written to results/data_parity.json);
  * Yahoo daily closes of every currently listed contract (2 years); the continuous front contract (CL=F / NG=F) fills
    month 1 on every day.  Contracts that have expired since are missing on past dates, so the gap between month 1 and
    the first listed contract shrinks toward the present and is zero from the last expiry onward.  Each day records
    `gap_months` so every test can be reported by completeness.
  * The snapshot recorder (tools/record_snapshot.py) appends today's full curve so the complete-curve window grows.
Bars (data/derived/bars_<root>_<interval>.csv: ts, delivery, months, open, high, low, close, volume).
"""
import csv
import datetime as dt
import glob
import json
import os
import sys

RAW, DER = "data/raw", "data/derived"
os.makedirs(DER, exist_ok=True)
os.makedirs("results", exist_ok=True)
ARGS = sys.argv[1:]
ROOTS = [r for r in (next((a.split("=", 1)[1] for a in ARGS if a.startswith("--roots=")), "CL,NG")).split(",") if r]
MONTH_CODES = "FGHJKMNQUVXZ"


# ---- calendar (mirrors calendar.hpp; weekday business days) --------------------------------------------------------
def is_bd(d):
    return d.weekday() < 5


def add_bd(d, n):
    step = 1 if n > 0 else -1
    left = abs(n)
    while left:
        d += dt.timedelta(days=step)
        if is_bd(d):
            left -= 1
    return d


def prev_bd_or_self(d):
    while not is_bd(d):
        d -= dt.timedelta(days=1)
    return d


def expiry(root, y, m):
    if root == "NG":
        return add_bd(dt.date(y, m, 1), -3)
    py, pm = (y, m - 1) if m > 1 else (y - 1, 12)
    return add_bd(prev_bd_or_self(dt.date(py, pm, 25)), -3)


def nth_delivery(root, d, n):
    y, m, count = d.year, d.month, 0
    for _ in range(400):
        if expiry(root, y, m) > d:
            count += 1
            if count == n:
                return y, m
        m += 1
        if m == 13:
            m, y = 1, y + 1
    raise RuntimeError("nth_delivery")


def months_between(y0, m0, y1, m1):
    return (y1 - y0) * 12 + (m1 - m0)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_root(root):
    rows = {}   # (date, delivery) -> (price, volume, source)

    def put(date, dely, dm, price, vol, source, override=False):
        key = (date, f"{dely}-{dm:02d}")
        if key not in rows or override:
            rows[key] = (price, vol, source)

    # EIA contracts 1-4
    eia_path = os.path.join(RAW, f"eia_{root.lower()}_front.csv")
    n_eia = 0
    if os.path.exists(eia_path):
        for r in read_csv(eia_path):
            d = dt.date.fromisoformat(r["date"])
            for k in range(1, 5):
                v = r.get(f"c{k}", "")
                if not v:
                    continue
                y, m = nth_delivery(root, d, k)
                put(r["date"], y, m, float(v), 0.0, "eia")
                n_eia += 1
    # Yahoo continuous front contract = contract 1
    front_path = os.path.join(RAW, "yahoo", f"{root}=F_1d.csv")
    n_front = 0
    if os.path.exists(front_path):
        for r in read_csv(front_path):
            d = dt.date.fromisoformat(r["date"])
            y, m = nth_delivery(root, d, 1)
            put(r["date"], y, m, float(r["close"]), float(r["volume"] or 0), "yahoo_front")
            n_front += 1
    # Yahoo listed contracts
    listed = json.load(open(os.path.join(RAW, f"yahoo_listed_{root.lower()}.json")))
    parity = []
    n_yahoo = 0
    for c in listed:
        y, m = map(int, c["delivery"].split("-"))
        for r in read_csv(os.path.join(RAW, "yahoo", f"{c['symbol']}_1d.csv")):
            d = dt.date.fromisoformat(r["date"])
            if expiry(root, y, m) <= d:
                continue
            price = float(r["close"])
            key = (r["date"], c["delivery"])
            if key in rows and rows[key][2] != "yahoo":
                parity.append((rows[key][2], rows[key][0], price))
                if rows[key][2] == "yahoo_front":
                    rows[key] = (price, float(r["volume"] or 0), "yahoo")   # the contract's own series beats the continuous proxy
                continue
            put(r["date"], y, m, price, float(r["volume"] or 0), "yahoo")
            n_yahoo += 1
    # snapshots recorded by tools/record_snapshot.py
    n_snap = 0
    for p in sorted(glob.glob(os.path.join(RAW, "snapshots", f"{root}_*.csv"))):
        for r in read_csv(p):
            y, m = map(int, r["delivery"].split("-"))
            put(r["date"], y, m, float(r["price"]), float(r.get("volume") or 0), "snapshot")
            n_snap += 1
    # write, with months index and years to expiry
    out = os.path.join(DER, f"curves_{root.lower()}.csv")
    by_date = {}
    for (date, dely), (price, vol, src) in rows.items():
        by_date.setdefault(date, []).append((dely, price, vol, src))
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "delivery", "months", "T", "price", "volume", "source"])
        for date in sorted(by_date):
            d = dt.date.fromisoformat(date)
            fy, fm = nth_delivery(root, d, 1)
            for dely, price, vol, src in sorted(by_date[date]):
                y, m = map(int, dely.split("-"))
                months = months_between(fy, fm, y, m) + 1
                T = (expiry(root, y, m) - d).days / 365.0
                if T <= 0 or months < 1:
                    continue
                w.writerow([date, dely, months, f"{T:.6f}", f"{price:.4f}", f"{vol:.0f}", src])
    # parity of the EIA / continuous-front mapping against the contracts' own prices
    par = {"n": len(parity)}
    if parity:
        for src in ("eia", "yahoo_front"):
            diffs = [abs(a - b) for s, a, b in parity if s == src]
            if diffs:
                par[src] = {"n": len(diffs), "median_abs_diff": sorted(diffs)[len(diffs) // 2], "share_within_1pct_of_price": sum(1 for s, a, b in parity if s == src and abs(a - b) <= 0.01 * b) / len(diffs)}
    pj = os.path.join("results", "data_parity.json")
    allp = json.load(open(pj)) if os.path.exists(pj) else {}
    allp[root] = par
    json.dump(allp, open(pj, "w"), indent=1)
    print(f"  {root}: {len(by_date)} days, EIA {n_eia} points, front {n_front}, listed {n_yahoo}, snapshots {n_snap}; parity {par}")


def build_bars(root, interval):
    listed = json.load(open(os.path.join(RAW, f"yahoo_listed_{root.lower()}.json")))
    out = os.path.join(DER, f"bars_{root.lower()}_{interval}.csv")
    n = 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "delivery", "months", "open", "high", "low", "close", "volume"])
        for c in listed:
            p = os.path.join(RAW, "yahoo", f"{c['symbol']}_{interval}.csv")
            if not os.path.exists(p):
                continue
            y, m = map(int, c["delivery"].split("-"))
            for r in read_csv(p):
                ts = int(r["ts"])
                d = dt.datetime.fromtimestamp(ts, dt.timezone.utc).date()
                fy, fm = nth_delivery(root, d, 1)
                months = months_between(fy, fm, y, m) + 1
                if months < 1:
                    continue
                w.writerow([ts, c["delivery"], months, f"{float(r['open']):.4f}", f"{float(r['high']):.4f}", f"{float(r['low']):.4f}", f"{float(r['close']):.4f}", f"{float(r['volume'] or 0):.0f}"])
                n += 1
    print(f"  {root} {interval}: {n} bars -> {out}")


if __name__ == "__main__":
    for root in ROOTS:
        build_root(root)
        for interval in ("1h",):
            build_bars(root, interval)
    print("done")
