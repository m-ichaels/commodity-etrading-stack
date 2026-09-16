"""Contract calendar (mirror of include/cetk/calendar.hpp): NYMEX expiry rules with weekday business days."""
import datetime as dt

MONTH_CODES = "FGHJKMNQUVXZ"


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
    """Last trading day of the contract for delivery month (y, m)."""
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


def years_to_expiry(root, d, y, m):
    return (expiry(root, y, m) - d).days / 365.0


def season_phase(m):
    return (m - 1) / 12.0


def months_between(y0, m0, y1, m1):
    return (y1 - y0) * 12 + (m1 - m0)
