"""pytest for the numpy reference: calendar rules, curve models (against the committed C++ results), sensitivities, swaps,
factors and statistics.   python -m pytest"""
import datetime as dt
import glob
import json
import os

import numpy as np
import pytest

from cetk import calendar as cal
from cetk import curve as cv
from cetk import data, factors, stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def chdir_root():
    os.chdir(ROOT)


# ---- calendar ------------------------------------------------------------------------------------------------------
def test_wti_expiry_is_three_business_days_before_the_25th_of_the_prior_month():
    assert cal.expiry("CL", 2026, 11) == dt.date(2026, 10, 20)   # 25 Oct 2026 is a Sunday -> 23 Oct -> back 3 bd
    assert cal.expiry("CL", 2026, 10) == dt.date(2026, 9, 22)


def test_henry_hub_expiry_is_three_business_days_before_the_delivery_month():
    assert cal.expiry("NG", 2026, 11) == dt.date(2026, 10, 28)
    assert cal.expiry("NG", 2027, 1) == dt.date(2026, 12, 29)


def test_front_month_rolls_after_expiry():
    assert cal.nth_delivery("CL", dt.date(2026, 9, 22), 1) == (2026, 11)   # on the expiry day the next contract is front
    assert cal.nth_delivery("CL", dt.date(2026, 9, 21), 1) == (2026, 10)
    assert cal.nth_delivery("NG", dt.date(2026, 9, 16), 2) == (2026, 11)


def test_season_phase_and_months_between():
    assert cal.season_phase(1) == 0.0 and abs(cal.season_phase(7) - 0.5) < 1e-12
    assert cal.months_between(2026, 9, 2027, 3) == 6


# ---- curve models on synthetic data -----------------------------------------------------------------------------------
def synthetic_curve(n=24, seed=3):
    rng = np.random.default_rng(seed)
    months = np.arange(1, n + 1); T = months / 12.0; phase = ((9 + months - 1) % 12) / 12.0
    price = 70 - 4 * np.exp(-T / 0.8) + 0.6 * np.cos(2 * np.pi * phase) + rng.normal(0, 0.01, n)
    return months, T, phase, price


@pytest.mark.parametrize("spec", [{"method": "seasonal_nss"}, {"method": "pspline", "lam": 1e-2}, {"method": "kernel_ridge", "lam": 1e-4}])
def test_models_interpolate_a_smooth_seasonal_curve(spec):
    months, T, phase, price = synthetic_curve()
    m = cv.CurveModel(**spec).fit(T, phase, price)
    assert m.in_sample_rmse < 0.05
    loo = cv.leave_one_out(T, phase, price, months, **spec)
    assert loo["mean_abs_interior"] < 0.08


def test_pspline_penalty_orders_smoothness():
    months, T, phase, price = synthetic_curve()
    rough = cv.CurveModel(method="pspline", lam=1e-6).fit(T, phase, price); smooth = cv.CurveModel(method="pspline", lam=1e2).fit(T, phase, price)
    assert rough.in_sample_rmse <= smooth.in_sample_rmse + 1e-9


def test_key_rate_sensitivities_sum_to_one_for_linear_smoothers():
    months, T, phase, price = synthetic_curve()
    for spec in ({"method": "pspline", "lam": 1e-2}, {"method": "kernel_ridge", "lam": 1e-4}):
        s = cv.key_rate_sensitivities(T, phase, price, 0.25, phase[2], **spec)
        assert abs(s.sum() - 1.0) < 2e-2, spec      # a parallel shift of the inputs moves the curve one for one
        assert s[np.argmin(np.abs(T - 0.25))] > 0.3   # and the nearest contract dominates


def test_calendar_swap_weights_are_business_day_fractions():
    months, T, phase, price = synthetic_curve()
    m = cv.CurveModel(method="pspline", lam=1e-2).fit(T, phase, price)
    date = dt.date(2026, 9, 16); points = {"2026-12": 68.0, "2027-01": 68.5}
    p, legs, nbd = cv.calendar_swap("CL", date, points, m, 2026, 12)   # December 2026 averages Jan (to 18 Dec) then Feb
    assert nbd == 23 and abs(sum(w for _, w, _, _ in legs) - 1.0) < 1e-12
    assert [d for d, _, _, _ in legs] == ["2027-01", "2027-02"]
    assert legs[0][3] is True and legs[1][3] is False
    assert abs(p - sum(w * x for _, w, x, _ in legs)) < 1e-12


# ---- against the C++ results checked into results/ ------------------------------------------------------------------
@pytest.mark.parametrize("root", ["CL", "NG"])
def test_numpy_reference_matches_cpp_curve_build(root):
    chdir_root()
    fs = sorted(glob.glob(f"results/curve_{root.lower()}_*.json"))
    if not fs or not os.path.exists(f"data/derived/curves_{root.lower()}.csv"):
        pytest.skip("no C++ curve build or derived data present")
    c = json.load(open(fs[-1], encoding="utf-8")); day = data.load_curves(root, since=c["date"][:4] + "-01-01")[c["date"]]
    for method in ("seasonal_nss", "pspline", "kernel_ridge"):
        j = c["methods"][method]
        spec = {"method": method, "lam": j["lambda"], "knot_spacing": j.get("knot_spacing", 1.0 / 12), "ell": j.get("ell", 0.5), "theta": j.get("theta", 1.0), "harmonics": int(j.get("harmonics", 2))}
        m = cv.CurveModel(**spec).fit(day["T"], day["phase"], day["price"]); loo = cv.leave_one_out(day["T"], day["phase"], day["price"], day["months"], **spec)
        tol = 1e-4 * max(1.0, float(np.mean(day["price"])))
        assert abs(m.in_sample_rmse - j["in_sample_rmse"]) < tol, method
        assert abs(loo["mean_abs"] - j["loo_mean_abs"]) < tol, method


# ---- factors ---------------------------------------------------------------------------------------------------------
def test_pca_recovers_a_level_factor():
    rng = np.random.default_rng(1); months = [1, 2, 3, 6, 12]
    level = rng.normal(0, 1.0, 400); slope = rng.normal(0, 0.3, 400)
    X = np.cumsum(np.outer(level, np.ones(5)) + np.outer(slope, np.linspace(-1, 1, 5)) + rng.normal(0, 0.02, (400, 5)), axis=0)
    fm = factors.fit_panel(X, months, K=3)
    assert fm["explained"][0] > 0.85 and fm["explained"][1] > 0.98
    assert (fm["loadings"][0] > 0).all()                       # level: same sign everywhere
    assert fm["loadings"][1][-1] > fm["loadings"][1][0]        # slope: back above front
    S = factors.covariance_at(fm, [1, 3, 12]); assert np.allclose(S, S.T) and np.all(np.linalg.eigvalsh(S) > -1e-12)


# ---- statistics ------------------------------------------------------------------------------------------------------
def test_bootstrap_ci_brackets_the_mean_and_ratio_ci_is_paired():
    rng = np.random.default_rng(2); x = rng.normal(1.0, 0.5, 200)
    lo, hi = stats.bootstrap_ci(x)[1:]
    assert lo < x.mean() < hi and hi - lo < 0.3
    a = rng.normal(2.0, 0.1, 100); b = a * 0.5 + rng.normal(0, 0.01, 100)
    r = stats.paired_ratio_ci(b, a)
    assert 0.45 < r[0] < 0.55 and r[1] < r[0] < r[2]


def test_ridge_and_walk_forward_recover_a_linear_rule():
    rng = np.random.default_rng(4); X = rng.normal(size=(300, 3)); beta = np.array([1.0, -0.5, 0.25]); y = X @ beta + rng.normal(0, 0.01, 300)
    assert np.allclose(stats.ridge(X, y, 1e-6), beta, atol=0.01)
    pred = stats.walk_forward(X, y, window=100, lam=1e-6)
    m = ~np.isnan(pred); assert m.sum() == 200 and np.corrcoef(pred[m], y[m])[0, 1] > 0.99
