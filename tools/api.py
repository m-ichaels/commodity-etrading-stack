#!/usr/bin/env python3
"""Tiny REST front end over the C++ pricer (stdlib only).   python tools/api.py [--port 8080] [--bin build/cetk]

  GET /health
  GET /quote?root=CL&tier=consumer&tenor=3&side=1&qty=25&mid=70&inventory=0,0,50,0,0&quoter=BEGV_adv_cal[&pre_event=1]
        -> the JSON `cetk price` prints: two-way price, half-spread, risk before the trade
  GET /curve?root=CL[&date=YYYY-MM-DD]      -> results/curve_<root>_<date>.json (the latest build when no date is given)
  GET /swaps?root=CL[&date=...]             -> the calendar-month swaps and strips from that build

Each quote is one short-lived process of the CLI, so the endpoint is a demonstration of the interface rather than a
production server: the engine itself is the FIX gateway (cetk_fixgw) and the in-process RfqEngine."""
import argparse
import glob
import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_bin(given):
    for c in ([given] if given else []) + [os.path.join(ROOT, "build", "cetk.exe"), os.path.join(ROOT, "build", "cetk")]:
        if c and os.path.exists(c):
            return c
    raise SystemExit("cetk binary not found: build it first (build.ps1 or cmake)")


def price(binary, q):
    args = [binary, "price"]
    for k in ("root", "tier", "tenor", "side", "qty", "mid", "inventory", "quoter", "gamma"):
        if k in q:
            args += [f"--{k}", q[k][0]]
    if q.get("pre_event", ["0"])[0] not in ("0", "", "false"):
        args.append("--pre-event")
    out = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or "pricer failed")
    return json.loads(out.stdout)


def curve_file(root, date):
    pat = os.path.join(ROOT, "results", f"curve_{root.lower()}_{date or '*'}.json")
    fs = sorted(glob.glob(pat))
    if not fs:
        raise FileNotFoundError(f"no curve build matches {pat}")
    return json.load(open(fs[-1], encoding="utf-8"))


class Handler(BaseHTTPRequestHandler):
    binary = None

    def _send(self, code, obj):
        body = json.dumps(obj, indent=1).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path); q = parse_qs(u.query)
        try:
            if u.path == "/health":
                self._send(200, {"ok": True, "binary": self.binary})
            elif u.path == "/quote":
                self._send(200, price(self.binary, q))
            elif u.path == "/curve":
                self._send(200, curve_file(q.get("root", ["CL"])[0], q.get("date", [None])[0]))
            elif u.path == "/swaps":
                c = curve_file(q.get("root", ["CL"])[0], q.get("date", [None])[0]); sel = q.get("model", ["kernel_ridge" if c["root"] == "NG" else "pspline"])[0]
                m = c["methods"][sel]; self._send(200, {"date": c["date"], "root": c["root"], "model": sel, "calendar_swaps": m["calendar_swaps"], "strip_3m": m["strip_3m"], "strip_12m": m["strip_12m"]})
            else:
                self._send(404, {"error": "unknown path", "paths": ["/health", "/quote", "/curve", "/swaps"]})
        except Exception as e:  # noqa: BLE001
            self._send(400, {"error": str(e)})

    def log_message(self, fmt, *args):
        sys.stderr.write("[api] " + fmt % args + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8080); ap.add_argument("--bin", default=None); a = ap.parse_args()
    Handler.binary = find_bin(a.bin)
    print(f"[api] serving on http://127.0.0.1:{a.port}  (binary {Handler.binary})"); HTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
