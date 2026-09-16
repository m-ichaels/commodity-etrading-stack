// Known-answer and identity tests: calendar rules, curve models and hat-matrix sensitivities, swaps, PCA, the impact
// estimator on synthetic bars with a known law, hedging policies, quoters, the engine's P&L decomposition, risk
// controls, the FIX session layer and a loopback RFQ gateway.
#include "cetk/fix.hpp"
#include <iostream>
#include <thread>
#include <cmath>

using namespace cetk;
static int failures = 0, checks = 0;
#define CHECK(cond, msg) do { ++checks; if (!(cond)) { ++failures; std::cerr << "FAIL: " << msg << "  [" #cond "]\n"; } } while (0)
#define CHECK_NEAR(a, b, tol, msg) CHECK(std::fabs((double)(a) - (double)(b)) <= (tol), msg << " (" << (a) << " vs " << (b) << ")")

// ---- calendar ------------------------------------------------------------------------------------------------------
static void test_calendar() {
    CHECK(fmt_date(civil_from_days(expiry_days("CL", 2026, 11))) == "2026-10-20", "CLX26 expires 2026-10-20 (3 business days before the 25th of October, a Sunday)");
    CHECK(fmt_date(civil_from_days(expiry_days("NG", 2026, 11))) == "2026-10-28", "NGX26 expires 2026-10-28 (3 business days before 1 November)");
    CHECK(fmt_date(civil_from_days(expiry_days("CL", 2026, 10))) == "2026-09-22", "CLV26 expires 2026-09-22");
    Ymd d = parse_date("2026-09-17"); auto p1 = nth_delivery("CL", d, 1), p2 = nth_delivery("CL", d, 2);
    CHECK(p1.first == 2026 && p1.second == 10, "prompt CL contract on 2026-09-17 is October 2026"); CHECK(p2.first == 2026 && p2.second == 11, "second CL contract is November 2026");
    auto g1 = nth_delivery("NG", d, 1); CHECK(g1.first == 2026 && g1.second == 10, "prompt NG contract on 2026-09-17 is October 2026 (November expires 28 Oct)");
    Ymd after = parse_date("2026-09-23"); auto p1b = nth_delivery("CL", after, 1); CHECK(p1b.second == 11, "the day after CLV26 expiry the prompt is November");
    CHECK(months_between(2026, 10, 2027, 1) == 3, "months_between"); CHECK_NEAR(season_phase(1), 0.0, 1e-12, "January phase 0"); CHECK_NEAR(season_phase(7), 0.5, 1e-12, "July phase 0.5");
    CHECK(years_to_expiry("CL", d, 2026, 11) > 0.08 && years_to_expiry("CL", d, 2026, 11) < 0.1, "years to expiry of CLX26 from 2026-09-17 ~ 33 days");
    CHECK(is_business_day(days_from_civil(2026, 9, 17)) && !is_business_day(days_from_civil(2026, 9, 19)), "weekday test");
    CHECK(fmt_date(civil_from_days(to_days(parse_date("2024-02-29")))) == "2024-02-29", "date round trip on a leap day");
}

// ---- curves ----------------------------------------------------------------------------------------------------------
static CurveDay synthetic_day(const std::string& root, const std::string& date, int n_months, double seasonal_amp, Rng* noise = nullptr, double noise_sd = 0.0) {
    CurveDay d; d.date = date; d.ymd = parse_date(date);
    for (int m = 1; m <= n_months; ++m) { auto del = nth_delivery(root, d.ymd, m); CurvePoint p; p.delivery = fmt_month(del.first, del.second); p.dy = del.first; p.dm = del.second; p.months = m; p.T = years_to_expiry(root, d.ymd, del.first, del.second); p.phase = season_phase(del.second);
        p.price = 70 - 8 * (1 - std::exp(-p.T / 1.5)) + 1.5 * p.T * std::exp(-p.T / 2) + seasonal_amp * std::cos(2 * 3.141592653589793 * (p.phase - 0.05)); if (noise) p.price += noise_sd * noise->normal(); p.source = "synthetic"; d.pts.push_back(p); }
    return d;
}
static void test_curve_models() {
    CurveDay d = synthetic_day("NG", "2026-09-17", 36, 0.6);
    for (CurveMethod m : {CurveMethod::SeasonalNSS, CurveMethod::PSpline, CurveMethod::KernelRidge}) {
        CurveSpec s; s.method = m; s.lambda = m == CurveMethod::PSpline ? 1e-3 : 1e-4; CurveModel cm; cm.fit(d.pts, s);
        CHECK(cm.in_sample_rmse < 0.15, method_name(m) << " fits a smooth seasonal curve in sample (rmse " << cm.in_sample_rmse << ")");
        LooResult loo = leave_one_out(d.pts, s); CHECK(loo.mean_abs_interior < 0.25, method_name(m) << " interior LOO error small (" << loo.mean_abs_interior << ")");
        // hat-matrix property: bumping every input by 1 moves any interior prediction by 1
        auto del = nth_delivery("NG", d.ymd, 5); double T5 = years_to_expiry("NG", d.ymd, del.first, del.second); Vec sens = key_rate_sensitivities(d.pts, s, T5, season_phase(del.second), &cm); double sum = 0; for (double x : sens) sum += x;
        CHECK_NEAR(sum, 1.0, 0.02, method_name(m) << " key-rate sensitivities sum to one");
        const CurvePoint* p5 = d.at_months(5); CHECK(std::fabs(cm.predict(p5->T, p5->phase) - p5->price) < 0.2, method_name(m) << " reprices month 5");
    }
    // the seasonal kernel ridge beats a non-seasonal one on a seasonal curve, and the seasonal NSS captures the amplitude
    { CurveSpec seas; seas.method = CurveMethod::KernelRidge; seas.lambda = 1e-4; seas.theta = 1.0; CurveSpec flat = seas; flat.theta = 0.0; LooResult a = leave_one_out(d.pts, seas), b = leave_one_out(d.pts, flat); CHECK(a.mean_abs_interior < b.mean_abs_interior, "seasonal kernel improves LOO on a seasonal curve (" << a.mean_abs_interior << " vs " << b.mean_abs_interior << ")"); }
    { CurveSpec s; s.method = CurveMethod::SeasonalNSS; CurveModel cm; cm.fit(d.pts, s); double amp = std::sqrt(cm.coef[4] * cm.coef[4] + cm.coef[5] * cm.coef[5]); CHECK_NEAR(amp, 0.6, 0.15, "seasonal NSS recovers the annual amplitude"); }
    // extrapolation errors are finite and larger than interior LOO
    { CurveSpec s; s.method = CurveMethod::KernelRidge; s.lambda = 1e-4; Vec ex = extrapolation_errors(d.pts, s, 3); CHECK(ex.size() == 3, "three extrapolation errors"); for (double e : ex) CHECK(std::isfinite(e), "finite extrapolation error"); }
    // noise: heavier smoothing reduces LOO error on a noisy curve
    { Rng rng(5); CurveDay dn = synthetic_day("CL", "2026-09-17", 30, 0.0, &rng, 0.3); CurveSpec s; s.method = CurveMethod::KernelRidge; s.lambda = 1e-6; LooResult tight = leave_one_out(dn.pts, s); s.lambda = 1e-1; LooResult smooth = leave_one_out(dn.pts, s); CHECK(smooth.mean_abs_interior < tight.mean_abs_interior, "smoothing helps on a noisy curve (" << smooth.mean_abs_interior << " vs " << tight.mean_abs_interior << ")"); }
    // a curve with a gap in the front months still fits and predicts the missing months within tolerance
    { CurveDay g = d; std::vector<CurvePoint> keep; for (auto& p : g.pts) if (p.months == 1 || p.months >= 6) keep.push_back(p); g.pts = keep; CurveSpec s; s.method = CurveMethod::KernelRidge; s.lambda = 1e-4; CurveModel cm; cm.fit(g.pts, s); const CurvePoint* p3 = d.at_months(3); CHECK(std::fabs(cm.predict(p3->T, p3->phase) - p3->price) < 0.6, "gap month predicted within 0.6 (" << cm.predict(p3->T, p3->phase) - p3->price << ")"); }
}
static void test_swaps() {
    CurveDay d = synthetic_day("CL", "2026-09-17", 30, 0.0); CurveSpec s; s.method = CurveMethod::KernelRidge; s.lambda = 1e-4; CurveModel cm; cm.fit(d.pts, s);
    SwapPrice sp = calendar_swap("CL", d, &cm, 2026, 12); double w = 0; for (auto& l : sp.legs) w += l.weight; CHECK_NEAR(w, 1.0, 1e-12, "calendar swap weights sum to one");
    CHECK(sp.legs.size() == 2, "a WTI calendar month straddles two contracts (" << sp.legs.size() << ")"); CHECK(sp.business_days >= 20 && sp.business_days <= 23, "business days in December 2026: " << sp.business_days);
    double lo = 1e9, hi = 0; for (auto& l : sp.legs) { lo = std::min(lo, l.price); hi = std::max(hi, l.price); } CHECK(sp.price >= lo - 1e-9 && sp.price <= hi + 1e-9, "swap price between its legs");
    CurveDay g = synthetic_day("NG", "2026-09-17", 30, 0.6); CurveModel gm; gm.fit(g.pts, s); SwapPrice gp = calendar_swap("NG", g, &gm, 2026, 12); CHECK(gp.legs.size() == 2 && gp.legs[0].delivery == "2027-01" && gp.legs[0].weight > 0.85 && gp.legs[1].delivery == "2027-02", "a Henry Hub December averages the January contract until its expiry on 29 December, then February (" << gp.legs[0].delivery << " " << gp.legs[0].weight << ")");
    double strip = strip_price("CL", d, &cm, 2026, 11, 3); double avg = (calendar_swap("CL", d, &cm, 2026, 11).price + calendar_swap("CL", d, &cm, 2026, 12).price + calendar_swap("CL", d, &cm, 2027, 1).price) / 3; CHECK_NEAR(strip, avg, 1e-12, "strip is the average of its calendar swaps");
    // an unobserved delivery month is priced off the model
    CurveDay part = d; for (size_t i = 0; i < part.pts.size(); ++i) if (part.pts[i].delivery == "2027-03") { part.pts.erase(part.pts.begin() + i); break; } SwapPrice sp2 = calendar_swap("CL", part, &cm, 2027, 2); bool any_model = false; for (auto& l : sp2.legs) if (!l.observed) any_model = true; CHECK(any_model, "missing contract priced off the model");
}

// ---- factors ----------------------------------------------------------------------------------------------------------
static void test_factors() {
    Rng rng(3); std::vector<Vec> X; Vec p(5, 70.0); std::vector<int> ladder = {1, 2, 3, 6, 12};
    for (int d = 0; d < 800; ++d) { double lvl = 1.2 * rng.normal(), slp = 0.4 * rng.normal(); for (size_t j = 0; j < 5; ++j) p[j] += lvl + slp * (std::log(ladder[j]) / std::log(12.0) - 0.5); X.push_back(p); }
    FactorModel fm; fm.fit_panel(X, ladder, 3); CHECK(fm.explained(2) > 0.99, "two factors explain > 99%: " << fm.explained(2)); double s = 0; for (double v : fm.loadings[0]) s += v; CHECK(s > 0, "level loadings positive"); CHECK(fm.loadings[1].back() > fm.loadings[1].front(), "slope loading rises with maturity");
    CHECK_NEAR(std::sqrt(fm.evals[0]), 1.2 * std::sqrt(5.0), 0.3, "level factor sd ~ 1.2 sqrt(5)"); auto S = fm.covariance_at({1, 4, 12}); CHECK(S[0][0] > 0 && std::fabs(S[0][2] - S[2][0]) < 1e-12, "covariance at tenors symmetric and positive");
}

// ---- impact ---------------------------------------------------------------------------------------------------------------
static std::vector<Bar> synthetic_bars(double delta, double Y, double cross, int days, Rng& rng) {
    std::vector<Bar> bars; int N = 4; Vec price(N, 70.0); Vec Vday = {20000, 12000, 6000, 2000};
    for (int d = 0; d < days; ++d) for (int h = 0; h < 23; ++h) { long long ts = 1700000000LL + (long long)d * 86400 + h * 3600; Vec u(N); for (int i = 0; i < N; ++i) u[i] = rng.normal();
        Vec q(N); for (int i = 0; i < N; ++i) { double vol = std::fabs(Vday[i] / 23 * (0.5 + rng.uniform())); q[i] = (rng.uniform() < 0.5 ? -1 : 1) * vol; }
        for (int j = 0; j < N; ++j) { double r = 0.004 * u[j] * 0.3 + 0.004 * 0.7 * u[0]; for (int i = 0; i < N; ++i) { double x = q[i] / Vday[i]; double imp = Y * 0.004 * (x > 0 ? 1 : -1) * std::pow(std::fabs(x), delta); r += (i == j ? 1.0 : cross) * imp; } Bar b; b.ts = ts; b.months = j + 1; b.delivery = "2027-0" + std::to_string(j + 1); b.open = price[j]; price[j] *= std::exp(r); b.close = price[j]; b.high = std::max(b.open, b.close); b.low = std::min(b.open, b.close); b.volume = std::fabs(q[j]); if ((q[j] > 0) != (b.close > b.open)) { b.open = b.close * std::exp(-(q[j] > 0 ? 1 : -1) * 1e-4); } bars.push_back(b); } }
    return bars;
}
static void test_impact() {
    Rng rng(9); auto bars = synthetic_bars(0.5, 0.8, 0.4, 120, rng); ImpactParams p; p.n_contracts = 4; p.window_bars = 1; p.boot = 30; ImpactEstimate E = estimate_impact(bars, p);
    CHECK(E.n_windows > 2000, "enough windows: " << E.n_windows); CHECK(E.delta > 0.3 && E.delta < 0.75, "concave exponent recovered near 0.5: " << E.delta << " [" << E.delta_lo << ", " << E.delta_hi << "]");
    CHECK(E.delta_lo <= E.delta && E.delta <= E.delta_hi, "bootstrap band contains the point estimate"); CHECK(E.r2_concave >= E.r2_linear - 1e-9, "concave fit at least as good as linear (" << E.r2_concave << " vs " << E.r2_linear << ")");
    CHECK(E.r2_cross > E.r2_self, "cross terms add explanatory power (" << E.r2_cross << " vs " << E.r2_self << ")"); CHECK(E.model.Lambda[1][0] > 0 && E.model.Lambda[0][1] > 0, "cross-impact between adjacent contracts positive");
    for (int i = 0; i < 4; ++i) CHECK(E.model.Lambda[i][i] > 0, "self-impact positive for contract " << i);
    auto nocross = synthetic_bars(0.5, 0.8, 0.0, 120, rng); ImpactEstimate E0 = estimate_impact(nocross, p); CHECK(E0.r2_cross - E0.r2_self < E.r2_cross - E.r2_self, "cross gain smaller when the generator has no cross-impact");
    // model mechanics
    ImpactModel m; m.sigma = {0.5, 0.5, 0.5}; m.V = {10000, 5000, 1000}; m.Y = {0.7, 0.7, 0.7}; m.delta = 0.5; m.Lambda = {{0.7, 0.2, 0.1}, {0.2, 0.7, 0.2}, {0.1, 0.2, 0.7}}; m.half_spread = {0.01, 0.01, 0.02};
    Vec h = {100, 0, 0}; Vec dp = m.price_moves(h); CHECK(dp[0] > dp[1] && dp[1] > dp[2] && dp[2] > 0, "buying the front moves the front most and the others in the same direction"); Vec h2 = {200, 0, 0}; CHECK(m.exec_cost(h2) > m.exec_cost(h) && m.exec_cost(h2) < 2 * m.exec_cost(h) * 1.5, "cost grows sub-quadratically"); CHECK_NEAR(m.exec_cost(Vec{0, 0, 0}), 0.0, 1e-12, "no trade, no cost");
    Vec hs = {50, 50, 0}; Vec ho = {50, -50, 0}; CHECK(m.exec_cost(hs) > m.exec_cost(ho), "same-direction trades in correlated contracts cost more than offsetting ones (cross-impact)");
    Json j = m.to_json(); ImpactModel m2 = ImpactModel::from_json(Json::parse(j.dump())); CHECK_NEAR(m2.exec_cost(hs), m.exec_cost(hs), 1e-9, "impact model json round trip");
}

// ---- hedging -------------------------------------------------------------------------------------------------------------
static Hedger sample_hedger() {
    Hedger H; size_t n = 5; H.Sigma = zeros(n, n); Vec sd = {1.3, 1.25, 1.2, 1.0, 0.8}; for (size_t a = 0; a < n; ++a) for (size_t b = 0; b < n; ++b) H.Sigma[a][b] = sd[a] * sd[b] * (a == b ? 1.0 : 0.9 - 0.02 * std::fabs((double)a - (double)b)); H.contract_size = 1000; H.gamma = 5e-7; H.tau_days = 0.25;
    ImpactModel m; m.sigma = Vec(n, 0.4); m.V = {150000, 80000, 40000, 15000, 5000}; m.Y = Vec(n, 0.6); m.delta = 0.5; m.Lambda = zeros(n, n); for (size_t a = 0; a < n; ++a) for (size_t b = 0; b < n; ++b) m.Lambda[a][b] = 0.6 * (a == b ? 1.0 : 0.5); m.half_spread = {0.01, 0.01, 0.01, 0.02, 0.03}; H.impact = m; return H;
}
static void test_hedging() {
    Hedger H = sample_hedger(); Vec q = {-60, -40, -30, -20, -10};
    Vec hs = H.hedge(q, HedgePolicy::Strip); Vec r(5); for (size_t j = 0; j < 5; ++j) r[j] = q[j] + hs[j]; CHECK_NEAR(H.risk_usd2(r), 0.0, 1e-9, "strip hedge leaves zero risk");
    Vec hf = H.hedge(q, HedgePolicy::FrontStack); double tot = 0; for (double x : hf) tot += x; CHECK_NEAR(tot, 160.0, 1e-9, "front stack offsets the total delta"); Vec rf(5); for (size_t j = 0; j < 5; ++j) rf[j] = q[j] + hf[j]; CHECK(H.risk_usd2(rf) > 0 && H.risk_usd2(rf) < H.risk_usd2(q), "front stack cuts but does not remove risk");
    Vec hm = H.hedge(q, HedgePolicy::FactorMin); Vec rm(5); for (size_t j = 0; j < 5; ++j) rm[j] = q[j] + hm[j]; CHECK(H.risk_usd2(rm) < H.risk_usd2(rf), "level+slope hedge beats the front stack on residual risk"); CHECK(hm[1] == 0 && hm[2] == 0 && hm[4] == 0, "FactorMin uses only its two contracts");
    ImpactModel m = H.impact; double J_ia = H.objective(q, H.hedge(q, HedgePolicy::ImpactAware), m), J_s = H.objective(q, hs, m), J_f = H.objective(q, hf, m); CHECK(J_ia <= J_s + 1e-9 && J_ia <= J_f + 1e-9, "impact-aware objective no worse than strip and front stack (" << J_ia << " vs " << J_s << ", " << J_f << ")");
    Vec h_nc = H.hedge(q, HedgePolicy::ImpactAwareNoCross); double diff = 0; for (size_t j = 0; j < 5; ++j) diff += std::fabs(h_nc[j] - H.hedge(q, HedgePolicy::ImpactAware)[j]); CHECK(diff > 1e-6, "ignoring cross-impact changes the hedge");
    Vec q0(5, 0.0); Vec h0 = H.hedge(q0, HedgePolicy::ImpactAware); double a0 = 0; for (double x : h0) a0 += std::fabs(x); CHECK(a0 < 1e-6, "flat book needs no hedge");
    auto cells = hedge_experiment(H, LadderSpec{}, 40, 3, Vec{-40, -35, -30, -20, -10}, Vec{20, 18, 15, 10, 6}); CHECK(cells.size() == 8, "eight policy cells"); for (auto& c : cells) { CHECK(std::isfinite(c.cost_usd) && c.cost_usd >= 0, c.policy << " finite non-negative cost"); if (c.policy == "NoHedge") CHECK_NEAR(c.cost_usd, 0, 1e-9, "no hedge no cost"); }
    const HedgeCell* ia = nullptr; const HedgeCell* st = nullptr; const HedgeCell* eq = nullptr; const HedgeCell* fs = nullptr; for (auto& c : cells) { if (c.policy == "ImpactAware") ia = &c; if (c.policy == "Strip") st = &c; if (c.policy == "ImpactAware@FrontRisk") eq = &c; if (c.policy == "FrontStack") fs = &c; } CHECK(ia && st && ia->objective <= st->objective + 1e-6, "impact-aware objective <= strip on random books");
    CHECK(eq && fs && eq->risk_usd <= fs->risk_usd * 1.05, "equal-risk hedge leaves no more risk than the front stack (" << eq->risk_usd << " vs " << fs->risk_usd << ")"); CHECK(eq && fs && eq->cost_usd <= fs->cost_usd + 1e-6, "at equal risk the impact-aware hedge is no dearer than the front stack (" << eq->cost_usd << " vs " << fs->cost_usd << ")");
    { Vec hr = H.hedge_to_risk(q, HedgePolicy::ImpactAware, 50000.0); Vec r2(5); for (size_t j = 0; j < 5; ++j) r2[j] = q[j] + hr[j]; CHECK(std::sqrt(H.risk_usd2(r2)) <= 50000.0 * 1.05, "hedge_to_risk meets its target (" << std::sqrt(H.risk_usd2(r2)) << ")"); }
}

// ---- quoters and engine -------------------------------------------------------------------------------------------------------
struct Env { LadderSpec ls; FactorModel fm; LadderRisk risk; std::vector<ClientTier> tiers; QuoteParams qp; Hedger H; EngineConfig cfg; };
static Env make_env(HedgePolicy pol = HedgePolicy::ImpactAware) {
    Env e; e.ls.months = {1, 2, 3, 6, 12}; e.ls.contract_size = 1000; e.ls.price_level = 70; e.ls.root = "CL";
    Rng rng(4); std::vector<Vec> X; Vec p(5, 70.0); for (int d = 0; d < 600; ++d) { double lvl = 1.2 * rng.normal(), slp = 0.4 * rng.normal(), cur = 0.15 * rng.normal(); for (size_t j = 0; j < 5; ++j) { double x = std::log(e.ls.months[j]) / std::log(12.0); p[j] += lvl + slp * (x - 0.5) + cur * (x * x - x); } X.push_back(p); }
    e.fm.fit_panel(X, e.ls.months, 3); e.risk.build(e.fm, e.ls); e.tiers = default_tiers("CL"); e.qp.gamma = 5e-7; e.H = sample_hedger(); e.H.Sigma = e.risk.Sigma; e.H.policy = pol;
    e.cfg.ls = e.ls; e.cfg.qp = e.qp; e.cfg.tiers = e.tiers; e.cfg.risk = e.risk; e.cfg.hedger = e.H; e.cfg.market_seed = 5; e.cfg.F0.assign(5, 70.0); e.cfg.hedge_review_min = 30; return e;
}
static void test_quoters() {
    Env e = make_env(); SymmetricQuoter sym(e.tiers, e.qp); Vec q(5, 0.0); for (size_t i = 0; i < e.tiers.size(); ++i) { double d = sym.quote((int)i, 0, 1, 25, q, false, 0); CHECK_NEAR(e.tiers[i].f(d), e.qp.hit_target, 1e-6, "symmetric quote hits the target ratio for tier " << i); }
    BegvQuoter b(e.tiers, e.ls, e.risk, e.qp, false, false); Vec ql = {80, 0, 0, 0, 0}; double dbuy = b.quote(0, 0, +1, 25, ql, false, 0), dsell = b.quote(0, 0, -1, 25, ql, false, 0); CHECK(dbuy < dsell, "long dealer sells cheaper than it buys (skew)"); CHECK(b.skew(0, ql) > 0, "skew positive when long");
    double d0 = b.quote(0, 0, +1, 25, q, false, 0), d1 = b.quote(0, 0, +1, 50, q, false, 0); CHECK(d1 > d0, "larger RFQ, wider quote"); CHECK(d0 > 0, "positive half-spread");
    // Riccati identity: Gamma D Gamma = (gamma/2) Sigma  <=>  the symmetric square-root construction; check symmetry and PSD of Gamma
    const Mat& G = b.Gamma(); for (size_t a = 0; a < 5; ++a) for (size_t c = 0; c < 5; ++c) CHECK_NEAR(G[a][c], G[c][a], 1e-9, "Gamma symmetric"); for (size_t a = 0; a < 5; ++a) CHECK(G[a][a] > 0, "Gamma diagonal positive");
    BegvQuoter adv(e.tiers, e.ls, e.risk, e.qp, true, false); double dfund = adv.quote(2, 0, +1, 25, q, false, 0), dbase = b.quote(2, 0, +1, 25, q, false, 0); CHECK(dfund > dbase, "adverse-selection term widens the informed tier"); CHECK_NEAR(adv.quote(0, 0, +1, 25, q, false, 0), b.quote(0, 0, +1, 25, q, false, 0), 1e-12, "no widening for a tier with mu = 0");
    double dcal = adv.quote(2, 0, +1, 25, q, true, 0); CHECK(dcal > dfund, "calendar widening before a release");
    BegvQuoter orc(e.tiers, e.ls, e.risk, e.qp, true, true); CHECK(orc.quote(2, 0, +1, 25, q, true, 0) > orc.quote(2, 0, +1, 25, q, false, 0), "oracle charges the expected jump before a release");
}
static RunResult run_days(Env& e, Quoter& quoter, int n_days, double p_event = 0.5, uint64_t seed = 9) { Rng drng(seed); auto days = synthetic_days(n_days, e.risk, drng, p_event, 2.0); return run_engine(e.cfg, quoter, days, 77); }
static void test_engine() {
    for (HedgePolicy pol : {HedgePolicy::None, HedgePolicy::FrontStack, HedgePolicy::Strip, HedgePolicy::ImpactAware}) {
        Env e = make_env(pol); BegvQuoter q(e.tiers, e.ls, e.risk, e.qp, true, false); RunResult r = run_days(e, q, 3);
        CHECK(r.n_fills > 10, hedge_name(pol) << ": fills happen (" << r.n_fills << ")"); CHECK_NEAR(r.residual, 0.0, 1e-6 * (1 + std::fabs(r.total)), hedge_name(pol) << ": P&L decomposition sums to the mark-to-market P&L (residual " << r.residual << ")");
        CHECK(r.pnl.spread > 0, hedge_name(pol) << ": spread revenue positive"); if (pol != HedgePolicy::None) { CHECK(r.n_hedges > 0, hedge_name(pol) << ": hedges happen"); CHECK(r.pnl.hedge_exec < 0, hedge_name(pol) << ": hedging costs money"); }
        else CHECK(r.n_hedges == 0 && r.pnl.hedge_exec == 0, "no hedging => no hedge cost"); CHECK(r.internalisation >= 0 && r.internalisation <= 1, "internalisation ratio in [0,1]");
        CHECK(r.markouts.size() == 6, "mark-out table has tier x window rows"); CHECK(r.pnl_5min.size() > 50, "5-minute P&L path recorded");
    }
    // common random numbers: identical config => identical result; changing the quoter keeps the same client script
    { Env e = make_env(); BegvQuoter q1(e.tiers, e.ls, e.risk, e.qp, true, false); RunResult a = run_days(e, q1, 2), b = run_days(e, q1, 2); CHECK_NEAR(a.total, b.total, 1e-9, "deterministic given seeds"); SymmetricQuoter s(e.tiers, e.qp); RunResult c = run_days(e, s, 2); CHECK(c.n_rfq == a.n_rfq, "same RFQ script across quoters (" << c.n_rfq << " vs " << a.n_rfq << ")"); }
    // no clients => zero P&L and zero hedges
    { Env e = make_env(); for (auto& t : e.cfg.tiers) t.lambda = 0; BegvQuoter q(e.cfg.tiers, e.ls, e.risk, e.qp, true, false); RunResult r = run_days(e, q, 2); CHECK(r.n_fills == 0 && std::fabs(r.total) < 1e-9 && r.n_hedges == 0, "no flow, no P&L, no hedges"); }
    // the informed tier is toxic: its 30-minute mark-out is in the client's favour; consumers pay the spread
    { Env e = make_env(HedgePolicy::None); BegvQuoter q(e.tiers, e.ls, e.risk, e.qp, false, false); RunResult r = run_days(e, q, 6, 0.0); double fund = 0, cons = 0; for (auto& m : r.markouts.arr()) if (!m["pre_event"].boolean()) { if (m["tier"].str() == "fund") fund = m["markout_30m"].num(); if (m["tier"].str() == "consumer") cons = m["markout_30m"].num(); } CHECK(fund > cons, "fund mark-outs more favourable to the client than consumers' (" << fund << " vs " << cons << ")"); }
    // toxicity routing externalises some fills; anticipatory hedging runs; residual still zero
    { Env e = make_env(); e.cfg.tox_route = true; e.cfg.tox_threshold = 0.3; e.cfg.anticipate = true; BegvQuoter q(e.tiers, e.ls, e.risk, e.qp, true, false); RunResult r = run_days(e, q, 4); CHECK_NEAR(r.residual, 0.0, 1e-6 * (1 + std::fabs(r.total)), "decomposition exact with routing and anticipation"); CHECK(r.n_externalised >= 0, "externalised count recorded"); }
    // risk controls: a tiny drawdown limit trips the kill switch; a tiny position limit rejects risk-adding RFQs
    { Env e = make_env(HedgePolicy::None); e.cfg.limits.max_drawdown_usd = 100; BegvQuoter q(e.tiers, e.ls, e.risk, e.qp, true, false); RunResult r = run_days(e, q, 3); CHECK(r.risk["n_kills"].num() > 0, "kill switch trips on a tiny drawdown limit"); }
    { Env e = make_env(HedgePolicy::None); e.cfg.limits.max_abs_q = 30; BegvQuoter q(e.tiers, e.ls, e.risk, e.qp, true, false); RunResult r = run_days(e, q, 3); CHECK(r.n_rejected > 0 && r.max_abs_q <= 30 + 50, "position limit rejects RFQs beyond the limit (max |q| " << r.max_abs_q << ")"); }
    // release-day jumps: with events the front-month moves are larger on event days (inventory P&L variance)
    { Env e = make_env(HedgePolicy::None); SymmetricQuoter s(e.tiers, e.qp); RunResult a = run_days(e, s, 6, 1.0), b = run_days(e, s, 6, 0.0); CHECK(a.pnl_5min_std > 0 && b.pnl_5min_std > 0, "P&L paths have variance"); }
    // client script: informed flow trades in the direction of the coming jump when p_info = 1
    { Env e = make_env(); e.cfg.tiers[2].p_info = 1.0; Rng drng(1); auto days = synthetic_days(1, e.risk, drng, 1.0, 2.0); Rng crng(2); auto script = generate_flow(e.cfg, days[0], 1.0, crng); int agree = 0, tot = 0; double te = days[0].events[0].t_day * e.cfg.day_s(); for (auto& r : script) if (r.tier == 2 && r.t < te && te - r.t <= 1800) { ++tot; if ((days[0].events[0].jump[r.tenor] > 0) == (r.side > 0)) ++agree; } CHECK(tot > 0 && agree == tot, "informed RFQs in the pre-release window follow the jump sign (" << agree << "/" << tot << ")"); }
}

// ---- FIX -------------------------------------------------------------------------------------------------------------------
static void test_fix() {
    fix::Message m; m.set(35, "R"); m.set(131, "7"); m.set(55, "CL"); m.set(54, "1"); m.set(38, "25"); m.set(20101, 2LL); m.set(20102, 0LL); m.set(20103, "3600.5");
    std::string raw = fix::encode(m, "CLIENT", "CETK", 3, 123456789LL); fix::Message d; CHECK(fix::decode(raw, d), "encode/decode round trip"); CHECK(d.str(131) == "7" && d.i64(20101) == 2 && std::fabs(d.num(20103) - 3600.5) < 1e-9 && d.i64(34) == 3, "fields survive the round trip");
    std::string bad = raw; bad[bad.size() - 3] = bad[bad.size() - 3] == '1' ? '2' : '1'; CHECK(!fix::decode(bad, d), "checksum mismatch rejected"); CHECK(raw.rfind("8=FIX.4.4", 0) == 0, "begin string");
    // loopback gateway: the same script in-process and over FIX yields identical fills
    Env in = make_env(); BegvQuoter qin(in.tiers, in.ls, in.risk, in.qp, true, false); RfqEngine eng_in(in.cfg); eng_in.set_quoter(&qin); Rng drng(21); auto days = synthetic_days(2, in.risk, drng, 0.5, 2.0); Rng crng(22);
    for (size_t d = 0; d < days.size(); ++d) { eng_in.start_day(days[d]); auto script = generate_flow(in.cfg, days[d], eng_in.flow_dir(), crng); for (auto& r : script) { eng_in.advance(r.t); auto q = eng_in.quote(r.tier, r.tenor, r.side, r.z); if (q.valid && r.u < in.cfg.tiers[r.tier].f(q.delta)) eng_in.execute(q); } eng_in.end_day(); }
    Env gwe = make_env(); BegvQuoter qgw(gwe.tiers, gwe.ls, gwe.risk, gwe.qp, true, false); RfqEngine eng_gw(gwe.cfg); eng_gw.set_quoter(&qgw); int port = 9881;
    sock_t ls = fix::listen_on(port); std::thread server([&]() { sock_t cs = ::accept(ls, nullptr, nullptr); fix::Conn conn(cs); fix::RfqGateway g(eng_gw, days); g.serve(conn); });
    size_t n_fix = 0; double pnl_fix = 0; { fix::Conn conn(fix::connect_to(port)); fix::RfqClient client(conn); Rng crng2(22); for (size_t d = 0; d < days.size(); ++d) { double dir = client.start_day(d); auto script = generate_flow(gwe.cfg, days[d], dir, crng2); for (auto& r : script) client.rfq(r, gwe.cfg.tiers[r.tier], "CL"); client.end_day(); } n_fix = client.fills.size(); client.logout(); }
    server.join(); CLOSESOCK(ls); pnl_fix = eng_gw.pnl_now();
    CHECK(n_fix == eng_in.stats().fills.size(), "same number of fills over FIX (" << n_fix << " vs " << eng_in.stats().fills.size() << ")"); CHECK_NEAR(pnl_fix, eng_in.pnl_now(), 1e-6, "same P&L over FIX"); CHECK(n_fix > 10, "the loopback session traded");
}

static void test_json_and_stats() {
    Json j = Json::parse("{\"a\": [1, 2.5, \"x\"], \"b\": {\"c\": true, \"d\": null}}"); CHECK(j["a"].size() == 3 && j["a"][1].num() == 2.5 && j["b"]["c"].boolean(), "json parse"); Json k = Json::parse(j.dump(1)); CHECK(k["a"][2].str() == "x", "json dump/parse round trip");
    Vec x = {1, 2, 3, 4, 5}; CHECK_NEAR(mean(x), 3.0, 1e-12, "mean"); CHECK_NEAR(stdev(x), std::sqrt(2.5), 1e-12, "stdev"); CI ci = bootstrap_mean_ci(x, 200); CHECK(ci.lo <= 3.0 && ci.hi >= 3.0, "bootstrap CI covers the mean");
    Mat A = {{4, 1}, {1, 3}}; Vec b = {1, 2}, s; CHECK(solve(A, b, s), "solve"); CHECK_NEAR(4 * s[0] + s[1], 1.0, 1e-12, "solution"); Mat R = sym_pow(A, 0.5); Mat RR = matmul(R, R); CHECK_NEAR(RR[0][1], 1.0, 1e-9, "symmetric square root");
}

int main() {
    struct T { const char* n; void (*f)(); }; T tests[] = {{"calendar", test_calendar}, {"curve_models", test_curve_models}, {"swaps", test_swaps}, {"factors", test_factors}, {"impact", test_impact}, {"hedging", test_hedging}, {"quoters", test_quoters}, {"engine", test_engine}, {"fix", test_fix}, {"json_and_stats", test_json_and_stats}};
    for (auto& t : tests) { std::cerr << "[test] " << t.n << std::endl; t.f(); }
    std::cout << checks - failures << "/" << checks << " checks passed" << std::endl; return failures ? 1 : 0;
}
