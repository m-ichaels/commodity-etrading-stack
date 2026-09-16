#pragma once
// Futures term-structure models fitted cross-sectionally each day, all linear in the observed prices given
// their shape parameters, so a fitted curve is a hat matrix and every off-grid price has exact key-rate
// sensitivities to the observed contracts.
//
//   SeasonalNSS   Nelson-Siegel-Svensson in maturity plus two annual harmonics in the delivery month:
//                 f(T, phi) = b0 + b1 (1-e^{-x1})/x1 + b2 [(1-e^{-x1})/x1 - e^{-x1}] + b3 [(1-e^{-x2})/x2 - e^{-x2}]
//                           + sum_{k=1,2} a_k cos(2 pi k phi) + c_k sin(2 pi k phi),   x_i = T / tau_i,
//                 tau_1, tau_2 by a small grid search (the dynamic NSS of the oil-curve literature, seasonal terms for gas).
//   PSpline       cubic B-splines on equally spaced knots with a second-difference penalty (Eilers-Marx); no seasonal
//                 term: the knots are dense enough to follow a monthly curve, which is what the TTF study found best in-line.
//   KernelRidge   f(T, phi) = a + b T + sum_j w_j k((T,phi),(T_j,phi_j)),  k = k_sob(T,T') + theta k_per(phi,phi'),
//                 k_sob(s,t) = min(s,t)^2 (3 max(s,t) - min(s,t)) / 6 (the curvature-penalty kernel of Filipovic, Pelger
//                 & Ye's discount-curve stripping) and k_per = exp(-2 sin^2(pi (phi-phi')) / l^2) (a periodic kernel in the
//                 delivery month); minimise sum_i w_i (F_i - f_i)^2 + lambda w'Kw; the affine part is the unpenalised null space.
// Prices are $/bbl or $/MMBtu; maturities in years; phi = (delivery month - 1)/12.
#include "data.hpp"
#include "linalg.hpp"
#include <memory>
#include <functional>

namespace cetk {

enum class CurveMethod { SeasonalNSS, PSpline, KernelRidge };
inline const char* method_name(CurveMethod m) { return m == CurveMethod::SeasonalNSS ? "seasonal_nss" : m == CurveMethod::PSpline ? "pspline" : "kernel_ridge"; }
inline CurveMethod parse_method(const std::string& s) { if (s == "seasonal_nss" || s == "nss") return CurveMethod::SeasonalNSS; if (s == "pspline") return CurveMethod::PSpline; if (s == "kernel_ridge" || s == "kr") return CurveMethod::KernelRidge; throw Error("unknown curve method " + s); }

struct CurveSpec {
    CurveMethod method = CurveMethod::KernelRidge;
    double lambda = 1e-3;        // penalty (PSpline second differences; KernelRidge w'Kw)
    double theta = 1.0;          // KernelRidge: weight of the periodic kernel
    double ell = 0.5;            // KernelRidge: periodic-kernel length scale
    double knot_spacing = 1.0 / 12;  // PSpline: years between knots (one month)
    int harmonics = 2;           // SeasonalNSS: annual harmonics
    bool liquidity_weights = true;   // weight observations by 1/(1+T): the front is the liquid, reliable part of the curve
};

struct CurveModel {
    CurveSpec spec; CurveMethod method = CurveMethod::KernelRidge;
    // fitted state
    Vec coef;                    // NSS / PSpline coefficients
    double tau1 = 0.5, tau2 = 5; // NSS shape
    Vec knots; double t_lo = 0, t_hi = 1;   // PSpline
    Vec Tj, Pj, w; double a0 = 0, b0 = 0;   // KernelRidge
    double in_sample_rmse = 0, max_abs_err = 0; size_t n = 0;

    // ---- basis functions ------------------------------------------------------------------------------------
    static Vec nss_basis(double T, double phi, double tau1, double tau2, int harm) {
        double x1 = std::max(T, 1e-6) / tau1, x2 = std::max(T, 1e-6) / tau2; double e1 = std::exp(-x1), e2 = std::exp(-x2);
        double l1 = (1 - e1) / x1, l2 = (1 - e2) / x2; Vec b = {1.0, l1, l1 - e1, l2 - e2};
        for (int k = 1; k <= harm; ++k) { b.push_back(std::cos(2 * 3.141592653589793 * k * phi)); b.push_back(std::sin(2 * 3.141592653589793 * k * phi)); }
        return b;
    }
    // cubic B-spline basis on uniform knots (Cox-de Boor); knots vector already extended
    static double bspline(const Vec& t, size_t i, int k, double x) {
        if (k == 0) return (x >= t[i] && x < t[i + 1]) ? 1.0 : 0.0;
        double a = (t[i + k] - t[i]) > 0 ? (x - t[i]) / (t[i + k] - t[i]) * bspline(t, i, k - 1, x) : 0;
        double b = (t[i + k + 1] - t[i + 1]) > 0 ? (t[i + k + 1] - x) / (t[i + k + 1] - t[i + 1]) * bspline(t, i + 1, k - 1, x) : 0;
        return a + b;
    }
    Vec pspline_basis(double T) const { size_t nb = knots.size() - 4; Vec b(nb, 0.0); double x = std::min(std::max(T, t_lo), t_hi - 1e-9); for (size_t i = 0; i < nb; ++i) b[i] = bspline(knots, i, 3, x); return b; }
    static double k_sob(double s, double t) { double a = std::min(s, t), b = std::max(s, t); return a * a * (3 * b - a) / 6.0; }
    double k_per(double p, double q) const { double sn = std::sin(3.141592653589793 * (p - q)); return std::exp(-2 * sn * sn / (spec.ell * spec.ell)); }
    double kern(double T, double phi, double T2, double phi2) const { return k_sob(T, T2) + spec.theta * k_per(phi, phi2); }

    // ---- fitting ---------------------------------------------------------------------------------------------
    void fit(const std::vector<CurvePoint>& pts, const CurveSpec& s) {
        spec = s; method = s.method; n = pts.size(); if (n < 3) throw Error("curve fit needs >= 3 points");
        Vec y(n), wt(n); for (size_t i = 0; i < n; ++i) { y[i] = pts[i].price; wt[i] = s.liquidity_weights ? 1.0 / (1.0 + pts[i].T) : 1.0; }
        if (method == CurveMethod::SeasonalNSS) {
            double best = 1e300; const double T1[4] = {0.25, 0.5, 1.0, 2.0}, T2[3] = {3.0, 5.0, 10.0};
            for (double t1 : T1) for (double t2 : T2) {
                Mat G(n); for (size_t i = 0; i < n; ++i) G[i] = nss_basis(pts[i].T, pts[i].phase, t1, t2, s.harmonics);
                size_t m = G[0].size(); Mat P = zeros(m, m); for (size_t a = 0; a < m; ++a) P[a][a] = 1e-6;   // tiny ridge: the NSS loadings are nearly collinear for tau1 close to tau2
                Vec c = ridge_solve(G, y, wt, P); double sse = 0; for (size_t i = 0; i < n; ++i) { double r = y[i] - dot(G[i], c); sse += wt[i] * r * r; }
                if (sse < best) { best = sse; coef = c; tau1 = t1; tau2 = t2; }
            }
        } else if (method == CurveMethod::PSpline) {
            double tmin = 1e9, tmax = 0; for (auto& p : pts) { tmin = std::min(tmin, p.T); tmax = std::max(tmax, p.T); }
            t_lo = 0; t_hi = tmax + s.knot_spacing; double h = s.knot_spacing; int nk = (int)std::ceil((t_hi - t_lo) / h) + 1;
            knots.clear(); for (int i = -3; i < nk + 3; ++i) knots.push_back(t_lo + i * h); t_hi = knots[knots.size() - 4];
            Mat G(n); for (size_t i = 0; i < n; ++i) G[i] = pspline_basis(pts[i].T); size_t m = G[0].size();
            Mat P = zeros(m, m); for (size_t i = 0; i + 2 < m; ++i) { double d[3] = {1, -2, 1}; for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) P[i + a][i + b] += s.lambda * d[a] * d[b]; }
            coef = ridge_solve(G, y, wt, P);
        } else {
            Tj.resize(n); Pj.resize(n); for (size_t i = 0; i < n; ++i) { Tj[i] = pts[i].T; Pj[i] = pts[i].phase; }
            Mat K = zeros(n, n); for (size_t a = 0; a < n; ++a) for (size_t b = 0; b < n; ++b) K[a][b] = kern(Tj[a], Pj[a], Tj[b], Pj[b]);
            size_t m = n + 2; Mat G(n, Vec(m, 0.0)); for (size_t i = 0; i < n; ++i) { for (size_t j = 0; j < n; ++j) G[i][j] = K[i][j]; G[i][n] = 1.0; G[i][n + 1] = Tj[i]; }
            Mat P = zeros(m, m); for (size_t a = 0; a < n; ++a) for (size_t b = 0; b < n; ++b) P[a][b] = s.lambda * K[a][b];
            Vec u = ridge_solve(G, y, wt, P); w.assign(u.begin(), u.begin() + n); a0 = u[n]; b0 = u[n + 1];
        }
        double sse = 0; max_abs_err = 0; for (auto& p : pts) { double e = predict(p.T, p.phase) - p.price; sse += e * e; max_abs_err = std::max(max_abs_err, std::fabs(e)); } in_sample_rmse = std::sqrt(sse / n);
    }
    double predict(double T, double phi) const {
        if (method == CurveMethod::SeasonalNSS) return dot(nss_basis(T, phi, tau1, tau2, spec.harmonics), coef);
        if (method == CurveMethod::PSpline) return dot(pspline_basis(T), coef);
        double f = a0 + b0 * T; for (size_t j = 0; j < Tj.size(); ++j) f += w[j] * kern(T, phi, Tj[j], Pj[j]); return f;
    }
    // slope of the curve in price per year (for roll-yield / calendar-spread analytics)
    double slope(double T, double phi, double h = 1.0 / 365) const { return (predict(T + h, phi) - predict(std::max(T - h, 1e-6), phi)) / (T + h - std::max(T - h, 1e-6)); }
};

// Leave-one-out repricing: drop each point, refit on the rest (same spec), predict it.  Errors in price units.
struct LooResult { Vec err; Vec err_pct; std::vector<int> months; double mean_abs = 0, median_abs = 0, mean_abs_interior = 0, mean_abs_pct = 0; size_t n_interior = 0; };
inline LooResult leave_one_out(const std::vector<CurvePoint>& pts, const CurveSpec& spec) {
    LooResult R; size_t n = pts.size(); if (n < 5) return R; int mmin = pts.front().months, mmax = pts.back().months;
    Vec abs_all, abs_int, pct;
    for (size_t k = 0; k < n; ++k) {
        std::vector<CurvePoint> sub; sub.reserve(n - 1); for (size_t j = 0; j < n; ++j) if (j != k) sub.push_back(pts[j]);
        CurveModel m; m.fit(sub, spec); double e = m.predict(pts[k].T, pts[k].phase) - pts[k].price;
        R.err.push_back(e); R.err_pct.push_back(100 * e / pts[k].price); R.months.push_back(pts[k].months); abs_all.push_back(std::fabs(e)); pct.push_back(std::fabs(100 * e / pts[k].price));
        if (pts[k].months > mmin && pts[k].months < mmax) abs_int.push_back(std::fabs(e));
    }
    R.mean_abs = mean(abs_all); R.median_abs = median(abs_all); R.mean_abs_interior = mean(abs_int); R.n_interior = abs_int.size(); R.mean_abs_pct = mean(pct);
    return R;
}

// Extrapolation beyond the last k liquid contracts: fit on months <= M-k, predict the rest.
inline Vec extrapolation_errors(const std::vector<CurvePoint>& pts, const CurveSpec& spec, int k) {
    Vec out; if ((int)pts.size() <= k + 3) return out; std::vector<CurvePoint> sub(pts.begin(), pts.end() - k); CurveModel m; m.fit(sub, spec);
    for (size_t j = pts.size() - k; j < pts.size(); ++j) out.push_back(m.predict(pts[j].T, pts[j].phase) - pts[j].price); return out;
}

// Key-rate sensitivities of an off-grid price to the observed contracts: bump each input by 1 and refit (all three
// models are linear in the prices given their shape parameters, so this is the exact hat-matrix row; for the NSS the
// shape parameters are frozen at the base fit).
inline Vec key_rate_sensitivities(const std::vector<CurvePoint>& pts, const CurveSpec& spec, double T, double phi, const CurveModel* base = nullptr) {
    CurveModel b; if (base) b = *base; else b.fit(pts, spec); double p0 = b.predict(T, phi); Vec s(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) { std::vector<CurvePoint> up = pts; up[i].price += 1.0; CurveModel m; if (spec.method == CurveMethod::SeasonalNSS) { m = b; Mat G(up.size()); Vec y(up.size()), wt(up.size()); for (size_t r = 0; r < up.size(); ++r) { G[r] = CurveModel::nss_basis(up[r].T, up[r].phase, b.tau1, b.tau2, spec.harmonics); y[r] = up[r].price; wt[r] = spec.liquidity_weights ? 1.0 / (1.0 + up[r].T) : 1.0; } size_t mm = G[0].size(); Mat P = zeros(mm, mm); for (size_t a = 0; a < mm; ++a) P[a][a] = 1e-6; m.coef = ridge_solve(G, y, wt, P); } else m.fit(up, spec); s[i] = m.predict(T, phi) - p0; }
    return s;
}

// ---- swaps ----------------------------------------------------------------------------------------------------
// Calendar-month average swap on delivery month (y, m): the floating leg is the average, over the business days of
// that calendar month, of the settlement of whichever contract is prompt that day.  Priced off today's curve, the fair
// fixed price is the business-day-weighted average of today's forwards for the contracts that will be prompt.
// Observed contracts are used where they exist, the fitted curve where they do not.
struct SwapLeg { std::string delivery; double weight; double price; bool observed; };
struct SwapPrice { double price = 0; std::vector<SwapLeg> legs; int business_days = 0; };
inline SwapPrice calendar_swap(const std::string& root, const CurveDay& day, const CurveModel* model, int y, int m) {
    SwapPrice sp; long long d0 = days_from_civil(y, (unsigned)m, 1); int ny = y, nm = m + 1; if (nm == 13) { nm = 1; ++ny; } long long d1 = days_from_civil(ny, (unsigned)nm, 1);
    std::map<std::string, int> counts; for (long long d = d0; d < d1; ++d) if (is_business_day(d)) { Ymd x = civil_from_days(d); auto pr = nth_delivery(root, x, 1); counts[fmt_month(pr.first, pr.second)]++; sp.business_days++; }
    for (auto& kv : counts) {
        SwapLeg leg; leg.delivery = kv.first; leg.weight = (double)kv.second / sp.business_days; int dy, dm; std::sscanf(kv.first.c_str(), "%d-%d", &dy, &dm);
        const CurvePoint* obs = nullptr; for (auto& p : day.pts) if (p.delivery == kv.first) obs = &p;
        if (obs) { leg.price = obs->price; leg.observed = true; } else { if (!model) throw Error("calendar_swap: delivery " + kv.first + " not observed and no model"); leg.price = model->predict(years_to_expiry(root, day.ymd, dy, dm), season_phase(dm)); leg.observed = false; }
        sp.price += leg.weight * leg.price; sp.legs.push_back(leg);
    }
    return sp;
}
// Average of calendar swaps over consecutive months (a quarterly or annual strip)
inline double strip_price(const std::string& root, const CurveDay& day, const CurveModel* model, int y, int m, int months) { double s = 0; for (int k = 0; k < months; ++k) { s += calendar_swap(root, day, model, y, m).price; if (++m == 13) { m = 1; ++y; } } return s / months; }

} // namespace cetk
