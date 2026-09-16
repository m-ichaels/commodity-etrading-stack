#!/usr/bin/env python3
"""A FIX 4.4 client for the gateway, in plain Python: logs on, starts a day, sends RFQs, hits the quotes, ends the day.
   1) ./build/cetk_fixgw --serve --root CL --port 9879 --days 1
   2) python tools/fix_client.py [--port 9879] [--root CL] [--n 20] [--seed 1]
Prints every quote and fill (a QuickFIX initiator sees the same messages: standard header, BodyLength, CheckSum, MsgSeqNum)."""
import argparse
import datetime as dt
import random
import socket

SOH = "\x01"
TIERS = ["consumer", "producer", "fund"]
MONTHS = [1, 2, 3, 6, 12]


class Session:
    def __init__(self, sock, sender="CLIENT", target="CETK"):
        self.s, self.sender, self.target, self.seq, self.buf = sock, sender, target, 0, b""

    def send(self, fields):
        self.seq += 1
        body = f"35={fields['35']}{SOH}49={self.sender}{SOH}56={self.target}{SOH}34={self.seq}{SOH}52={dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d-%H:%M:%S.%f')[:-3]}{SOH}"
        body += "".join(f"{k}={v}{SOH}" for k, v in fields.items() if k != "35")
        msg = f"8=FIX.4.4{SOH}9={len(body)}{SOH}{body}"
        msg += f"10={sum(msg.encode()) % 256:03d}{SOH}"
        self.s.sendall(msg.encode())

    def recv(self):
        while True:
            p = self.buf.find(b"\x0110=")
            if p >= 0:
                e = self.buf.find(b"\x01", p + 1)
                if e >= 0:
                    raw, self.buf = self.buf[:e + 1], self.buf[e + 1:]
                    fields = {}
                    for kv in raw.decode().split(SOH):
                        if "=" in kv:
                            k, v = kv.split("=", 1); fields[k] = v
                    body = raw[:raw.rfind(b"10=")]
                    if int(fields["10"]) != sum(body) % 256:
                        raise RuntimeError("bad checksum")
                    return fields
            chunk = self.s.recv(65536)
            if not chunk:
                raise ConnectionError("disconnected")
            self.buf += chunk

    def wait(self, msg_type, **match):
        while True:
            m = self.recv()
            if m.get("35") == msg_type and all(m.get(k) == v for k, v in match.items()):
                return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=9879); ap.add_argument("--root", default="CL"); ap.add_argument("--n", type=int, default=20); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--day", type=int, default=0)
    a = ap.parse_args(); rng = random.Random(a.seed)
    with socket.create_connection(("127.0.0.1", a.port)) as sock:
        s = Session(sock)
        s.send({"35": "A", "98": "0", "108": "30"}); s.wait("A"); print("logged on")
        s.send({"35": "0", "112": f"SOD:{a.day}"}); sod = s.wait("0", **{"112": f"SOD:{a.day}"}); print(f"day {a.day} started (hidden flow direction {sod.get('20120')})")
        fills = 0; t = 60.0
        for i in range(a.n):
            t += rng.expovariate(1 / 600.0)
            tier, tenor, side, qty = rng.randrange(3), rng.randrange(len(MONTHS)), rng.choice([1, 2]), rng.choice([10, 25, 50, 100])
            s.send({"35": "R", "131": str(i + 1), "55": a.root, "54": str(side), "38": str(qty), "20101": str(tenor), "20102": str(tier), "20103": f"{t:.3f}"})
            q = s.wait("S", **{"131": str(i + 1)})
            if q.get("20111") != "1":
                print(f"  RFQ {i + 1}: {TIERS[tier]:8s} {MONTHS[tenor]:2d}m {'buy ' if side == 1 else 'sell'} {qty:3d} -> no quote"); continue
            px = q["133"] if side == 1 else q["132"]
            print(f"  RFQ {i + 1}: {TIERS[tier]:8s} {MONTHS[tenor]:2d}m {'buy ' if side == 1 else 'sell'} {qty:3d} -> bid {float(q['132']):.4f} ask {float(q['133']):.4f} (mid {float(q['20112']):.4f}, half-spread {float(q['20110']):.4f})", end="")
            if rng.random() < 0.5:
                s.send({"35": "D", "11": f"O{i + 1}", "117": q["117"], "55": a.root, "54": str(side), "38": str(qty), "44": px, "40": "2"})
                er = s.wait("8", **{"11": f"O{i + 1}"})
                if er.get("150") == "F":
                    fills += 1; print(f"  FILLED {er['32']} @ {float(er['31']):.4f}")
                else:
                    print(f"  rejected: {er.get('58')}")
            else:
                print("  passed")
        s.send({"35": "0", "112": "EOD"}); s.wait("0", **{"112": "EOD"})
        s.send({"35": "5"}); print(f"day ended: {fills} fills out of {a.n} RFQs; logged out")


if __name__ == "__main__":
    main()
