#!/usr/bin/env python3
"""Free data for the commodity e-trading stack.   python tools/download.py [--skip-yahoo] [--skip-cftc] [--roots CL,NG]

  EIA     daily NYMEX settlements for futures contracts 1-4 (WTI since 1983, Henry Hub since 1994), weekly natural-gas
          storage and weekly crude stocks (the scheduled-release series)               -> data/raw/eia_*.csv
  Yahoo   every listed CL / NG contract: 2y of daily closes and 730d of hourly bars (chart API, no key); CL=F / NG=F continuous front
                                                                                        -> data/raw/yahoo/<sym>_<interval>.csv
  CFTC    disaggregated futures-only Commitments of Traders, 2010-2026 (hedging pressure) -> data/raw/cot_<root>.csv
Then tools/build_curves.py assembles the daily curves and bar panels in data/derived/.
Nothing here needs an API key.  Yahoo's "max" range is weekly-sampled; the 1y/2y ranges are daily, which is why they are used.
"""
import csv
import datetime as dt
import io
import json
import os
import sys
import time
import urllib.request
import zipfile

RAW = "data/raw"
YH = os.path.join(RAW, "yahoo")
os.makedirs(YH, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/128 Safari/537.36"}
MONTH_CODES = "FGHJKMNQUVXZ"
ARGS = sys.argv[1:]
ROOTS = [r for r in (next((a.split("=", 1)[1] for a in ARGS if a.startswith("--roots=")), "CL,NG")).split(",") if r]


def get(url, retries=3, timeout=60):
    for k in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if k == retries - 1:
                print("  failed", url, e)
                return None
            time.sleep(1.5 * (k + 1))


# ---- EIA ---------------------------------------------------------------------------------------------------------
def eia_xls(url):
    """EIA dnav workbook -> [(date, value)] from sheet 'Data 1' (needs xlrd for the .xls format)."""
    import xlrd
    raw = get(url)
    if raw is None:
        return []
    wb = xlrd.open_workbook(file_contents=raw)
    sh = wb.sheet_by_name("Data 1")
    out = []
    for r in range(sh.nrows):
        c0 = sh.cell(r, 0)
        if c0.ctype != xlrd.XL_CELL_DATE:
            continue
        d = xlrd.xldate_as_datetime(c0.value, wb.datemode).date()
        v = sh.cell(r, 1).value
        if isinstance(v, (int, float)) and v != "":
            out.append((d.isoformat(), float(v)))
    return out


def eia_front_contracts():
    for root, codes in (("CL", ["RCLC1d", "RCLC2d", "RCLC3d", "RCLC4d"]), ("NG", ["RNGC1d", "RNGC2d", "RNGC3d", "RNGC4d"])):
        if root not in ROOTS:
            continue
        folder = "pet" if root == "CL" else "ng"
        series = {}
        for k, code in enumerate(codes, 1):
            rows = eia_xls(f"https://www.eia.gov/dnav/{folder}/hist_xls/{code}.xls")
            print(f"  EIA {code}: {len(rows)} rows")
            for d, v in rows:
                series.setdefault(d, {})[f"c{k}"] = v
        path = os.path.join(RAW, f"eia_{root.lower()}_front.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "c1", "c2", "c3", "c4"])
            for d in sorted(series):
                r = series[d]
                w.writerow([d] + [r.get(f"c{k}", "") for k in range(1, 5)])
        print("  wrote", path)


def eia_weekly():
    for name, url in (("eia_ng_storage", "https://www.eia.gov/dnav/ng/hist_xls/NW2_EPG0_SWO_R48_BCFw.xls"),
                      ("eia_crude_stocks", "https://www.eia.gov/dnav/pet/hist_xls/WCESTUS1w.xls")):
        rows = eia_xls(url)
        path = os.path.join(RAW, name + ".csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["week_ending", "value"])
            w.writerows(rows)
        print(f"  {name}: {len(rows)} weeks -> {path}")


# ---- Yahoo -------------------------------------------------------------------------------------------------------
def yahoo_chart(sym, rng, interval):
    raw = get(f"https://query2.finance.yahoo.com/v8/finance/chart/{sym}?range={rng}&interval={interval}", retries=2, timeout=30)
    if raw is None:
        return None
    try:
        d = json.loads(raw)["chart"]
    except Exception:  # noqa: BLE001
        return None
    res = d.get("result")
    if not res:
        return None
    r = res[0]
    ts = r.get("timestamp") or []
    q = r["indicators"]["quote"][0]
    rows = []
    for t, o, h, lo, c, v in zip(ts, q.get("open", []), q.get("high", []), q.get("low", []), q.get("close", []), q.get("volume", [])):
        if c is None:
            continue
        rows.append((t, o if o is not None else c, h if h is not None else c, lo if lo is not None else c, c, v or 0))
    return rows


def yahoo_contracts(root):
    """Enumerate listed contracts for the next years; a symbol is listed when the daily chart returns rows."""
    listed = []
    today = dt.date.today()
    for year in range(today.year, today.year + 6):
        for mi, code in enumerate(MONTH_CODES, 1):
            if dt.date(year, mi, 1) < dt.date(today.year, today.month, 1):
                continue
            sym = f"{root}{code}{year % 100:02d}.NYM"
            rows = yahoo_chart(sym, "2y", "1d")
            if not rows:
                continue
            listed.append({"symbol": sym, "root": root, "delivery": f"{year}-{mi:02d}", "n_daily": len(rows)})
            with open(os.path.join(YH, f"{sym}_1d.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["ts", "date", "open", "high", "low", "close", "volume"])
                for t, o, h, lo, c, v in rows:
                    w.writerow([t, dt.datetime.fromtimestamp(t, dt.timezone.utc).date().isoformat(), o, h, lo, c, v])
            time.sleep(0.2)
    with open(os.path.join(RAW, f"yahoo_listed_{root.lower()}.json"), "w") as f:
        json.dump(listed, f, indent=1)
    print(f"  {root}: {len(listed)} listed contracts with daily history")
    return listed


def yahoo_bars(listed, interval, rng, max_contracts):
    """Intraday bars for the nearest `max_contracts` contracts (the liquid part of the curve)."""
    for c in listed[:max_contracts]:
        sym = c["symbol"]
        rows = yahoo_chart(sym, rng, interval)
        if not rows:
            print("  no", interval, "bars for", sym)
            continue
        with open(os.path.join(YH, f"{sym}_{interval}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts", "open", "high", "low", "close", "volume"])
            w.writerows(rows)
        print(f"  {sym} {interval}: {len(rows)} bars")
        time.sleep(0.25)


# ---- CFTC --------------------------------------------------------------------------------------------------------
COT_CODES = {"CL": "067651", "NG": "023651"}   # CFTC contract market codes: WTI light sweet crude (NYMEX), Henry Hub natural gas (NYMEX)


def cftc():
    keep = {r: [] for r in ROOTS if r in COT_CODES}
    header = None
    for year in range(2010, dt.date.today().year + 1):
        raw = get(f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip")
        if raw is None:
            continue
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            name = [n for n in z.namelist() if n.lower().endswith(".txt")][0]
            text = z.read(name).decode("latin-1")
        rd = csv.reader(io.StringIO(text))
        hdr = next(rd)
        if header is None:
            header = hdr
        ci = {h.strip(): i for i, h in enumerate(hdr)}
        code_col = ci.get("CFTC_Contract_Market_Code")
        for row in rd:
            if len(row) <= code_col:
                continue
            code = row[code_col].strip()
            for r, c in COT_CODES.items():
                if r in keep and code == c:
                    keep[r].append(row)
        print(f"  CFTC {year}: " + ", ".join(f"{r} {len(v)}" for r, v in keep.items()))
    for r, rows in keep.items():
        path = os.path.join(RAW, f"cot_{r.lower()}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([h.strip() for h in header])
            w.writerows(rows)
        print("  wrote", path, len(rows), "weeks")


def main():
    print("EIA")
    eia_front_contracts()
    eia_weekly()
    if "--skip-yahoo" not in ARGS:
        print("Yahoo")
        for root in ROOTS:
            listed = yahoo_contracts(root)
            yahoo_bars(listed, "1h", "730d", 24)
    if "--skip-cftc" not in ARGS:
        print("CFTC")
        cftc()
    print("done")


if __name__ == "__main__":
    main()
