#pragma once
// Self- and cross-impact along the futures curve, estimated from public bars.
//
// Data: bars of the first N contracts.  Signed volume per bar is the tick-rule proxy volume * sign(close - open)
// (sign(close - previous close) on doji bars).  Over windows of h bars, Q_i = signed volume / daily volume V_i of
// contract i and R_i = log return.  The public-data metaorder proxy follows arXiv:2503.18199 in spirit (order-flow
// imbalance stands in for the metaorder); the estimator follows Hey, Neuman & Tuschmann (arXiv:2510.06879):
//   self-impact   E[ sign(Q) R_i / sigma_i | Q ] = Y |Q_i|^delta          (concave when delta < 1; delta = 1 is linear)
//   cross-impact  R_j / sigma_j = sum_i Lambda_ji sign(Q_i) |Q_i|^delta  (multi-asset propagator at lag 0)
//   decay         G(tau) = E[ sign(Q_t) R_{t+tau} ] / E[ sign(Q_t) R_t ]  for tau = 1..L windows
// delta and Y from a log-log fit on volume-imbalance bins; block-bootstrap (by day) confidence bands; R^2 of the linear
// and concave self models and of the self-only versus self-plus-cross regressions.
//
// The fitted ImpactModel is what the hedging optimiser and the simulator use: executing h_i contracts (h in contracts,
// daily volume V_i in contracts, sigma_i the window volatility in price units) moves contract j by
//   dP_j = sigma_j Lambda_ji sign(h_i) (|h_i| / V_i)^delta,   and the trade in i pays on average the self-impact
//   integral Y sigma_i (|h_i|/V_i)^delta / (1 + delta) per contract plus the half spread.
#include "data.hpp"
#include "stats.hpp"
#include "linalg.hpp"
#include <map>
#include <set>

namespace cetk {

struct ImpactParams { int n_contracts = 6; int window_bars = 1; int n_bins = 16; int max_lag = 12; int boot = 200; double bars_per_day = 23; };

struct ImpactModel {
    Vec sigma;      // per-window volatility in price units, per contract
    Vec V;          // daily volume, contracts
    Vec Y;          // self-impact prefactor per contract (pooled by default)
    double delta = 0.5;
    Mat Lambda;     // [j][i] normalised cross-impact (diagonal = self, in units of sigma_j)
    double decay_per_window = 0.5;   // exponential decay of the transient part per window
    double perm_frac = 0.3;          // fraction of the initial impact that does not decay
    Vec half_spread;                 // price units, per contract
    size_t n() const { return sigma.size(); }
    // average impact paid per contract executing h_i contracts (self only), price units
    double self_cost_per_contract(size_t i, double h) const { return Y[i] * sigma[i] * std::pow(std::fabs(h) / std::max(V[i], 1.0), delta) / (1 + delta); }
    // price move of contract j caused by executing the vector h (contracts), price units
    Vec price_moves(const Vec& h) const { size_t m = n(); Vec dp(m, 0.0); for (size_t i = 0; i < m; ++i) { if (h[i] == 0) continue; double u = (h[i] > 0 ? 1 : -1) * std::pow(std::fabs(h[i]) / std::max(V[i], 1.0), delta); for (size_t j = 0; j < m; ++j) dp[j] += sigma[j] * Lambda[j][i] * u; } return dp; }
    // execution cost of the trade vector: spread + self-impact integral + half of the cross-impact moved against our own trades
    double exec_cost(const Vec& h) const { size_t m = n(); double c = 0; Vec dp = price_moves(h); for (size_t i = 0; i < m; ++i) { c += std::fabs(h[i]) * (half_spread[i] + self_cost_per_contract(i, h[i])); double self_move = sigma[i] * Lambda[i][i] * (h[i] > 0 ? 1 : h[i] < 0 ? -1 : 0) * std::pow(std::fabs(h[i]) / std::max(V[i], 1.0), delta); c += 0.5 * h[i] * (dp[i] - self_move); } return c; }
    Json to_json() const { Json j = Json::object(); j["delta"] = delta; j["sigma"] = Json(sigma); j["daily_volume"] = Json(V); j["Y"] = Json(Y); Json L = Json::array(); for (auto& r : Lambda) L.push(Json(r)); j["Lambda"] = L; j["decay_per_window"] = decay_per_window; j["perm_frac"] = perm_frac; j["half_spread"] = Json(half_spread); return j; }
    static ImpactModel from_json(const Json& j) { ImpactModel m; m.delta = j["delta"].num(); for (auto& v : j["sigma"].arr()) m.sigma.push_back(v.num()); for (auto& v : j["daily_volume"].arr()) m.V.push_back(v.num()); for (auto& v : j["Y"].arr()) m.Y.push_back(v.num()); for (auto& r : j["Lambda"].arr()) { Vec row; for (auto& v : r.arr()) row.push_back(v.num()); m.Lambda.push_back(row); } m.decay_per_window = j["decay_per_window"].num(); m.perm_frac = j["perm_frac"].num(); for (auto& v : j["half_spread"].arr()) m.half_spread.push_back(v.num()); return m; }
};

struct ImpactEstimate {
    ImpactModel model; int window_bars = 1; size_t n_windows = 0, n_days = 0; int n_contracts = 0; std::vector<std::string> deliveries; double Y_linear = 0;
    double delta = 0, Y = 0, delta_lo = 0, delta_hi = 0, r2_linear = 0, r2_concave = 0, r2_self = 0, r2_cross = 0, r2_linear_bins = 0, r2_concave_bins = 0;
    Vec bin_x, bin_y;                       // binned self-impact curve (|Q|, mean signed normalised return)
    Vec decay;                              // G(tau)/G(0), tau = 1..L
    Vec delta_by_contract, Y_by_contract;
    Json to_json() const { Json j = Json::object(); j["window_bars"] = window_bars; j["n_windows"] = (long long)n_windows; j["n_days"] = (long long)n_days; j["n_contracts"] = n_contracts; Json dl = Json::array(); for (auto& d : deliveries) dl.push(d); j["deliveries"] = dl; j["Y_linear"] = Y_linear; j["delta"] = delta; j["delta_ci95"] = Json(Vec{delta_lo, delta_hi}); j["Y"] = Y; j["r2_linear"] = r2_linear; j["r2_concave"] = r2_concave; j["r2_linear_bins"] = r2_linear_bins; j["r2_concave_bins"] = r2_concave_bins; j["r2_self_only"] = r2_self; j["r2_self_plus_cross"] = r2_cross; j["bin_abs_q"] = Json(bin_x); j["bin_response"] = Json(bin_y); j["decay"] = Json(decay); j["delta_by_contract"] = Json(delta_by_contract); j["Y_by_contract"] = Json(Y_by_contract); j["model"] = model.to_json(); return j; }
};

namespace detail {
struct Win { long long day; Vec R, Q, sigmaR; };   // per window: day id, returns and normalised signed volume per contract
// log-log fit of the binned response: returns {delta, Y}
inline std::pair<double, double> fit_power(const std::vector<std::pair<double, double>>& xy, int n_bins, Vec* bx = nullptr, Vec* by = nullptr) {
    std::vector<std::pair<double, double>> s = xy; std::sort(s.begin(), s.end()); if (s.size() < (size_t)n_bins * 4) n_bins = std::max(4, (int)s.size() / 4);
    Vec lx, ly; for (int b = 0; b < n_bins; ++b) { size_t a = s.size() * b / n_bins, e = s.size() * (b + 1) / n_bins; if (e <= a) continue; double mx = 0, my = 0; for (size_t k = a; k < e; ++k) { mx += s[k].first; my += s[k].second; } mx /= (e - a); my /= (e - a); if (bx) bx->push_back(mx); if (by) by->push_back(my); if (mx > 0 && my > 0) { lx.push_back(std::log(mx)); ly.push_back(std::log(my)); } }
    if (lx.size() < 3) return {1.0, 0.0}; Ols o = ols(lx, ly); return {std::max(0.05, std::min(1.5, o.b)), std::exp(o.a)};
}
}

inline ImpactEstimate estimate_impact(const std::vector<Bar>& bars, const ImpactParams& p, const std::set<long long>* only_days = nullptr) {
    using namespace detail;
    ImpactEstimate E; E.window_bars = p.window_bars; int N = p.n_contracts;
    // contracts are identified by delivery month over the estimation window (the "months ahead" index changes at every
    // expiry); the N nearest deliveries with bars inside the window form the panel, index 0 = nearest
    std::map<std::string, size_t> nbars; for (auto& b : bars) if (!only_days || only_days->count(b.ts / 86400)) nbars[b.delivery]++;
    std::vector<std::string> deliveries; for (auto& kv : nbars) if (kv.second >= 20) deliveries.push_back(kv.first); std::sort(deliveries.begin(), deliveries.end()); if ((int)deliveries.size() > N) deliveries.resize(N); N = (int)deliveries.size(); E.deliveries = deliveries; if (N < 2) return E;
    std::map<std::string, int> idx; for (int i = 0; i < N; ++i) idx[deliveries[i]] = i;
    std::map<long long, std::map<int, const Bar*>> by_ts; for (auto& b : bars) { auto it = idx.find(b.delivery); if (it != idx.end()) by_ts[b.ts][it->second] = &b; }
    std::map<std::string, double> prev_close; std::map<int, double> vol_sum; std::set<long long> days;
    std::vector<Vec> r_rows, s_rows; std::vector<long long> ts_rows;
    for (auto& kv : by_ts) {
        Vec r(N, NAN), s(N, 0.0); bool any = false; long long day = kv.first / 86400;
        if (only_days && !only_days->count(day)) continue;
        for (auto& mb : kv.second) { const Bar& b = *mb.second; int i = mb.first; auto it = prev_close.find(b.delivery); double pc = it == prev_close.end() ? b.open : it->second; prev_close[b.delivery] = b.close; if (pc > 0 && b.close > 0) { r[i] = std::log(b.close / pc); any = true; } double sg = b.close > b.open ? 1 : b.close < b.open ? -1 : (b.close > pc ? 1 : b.close < pc ? -1 : 0); s[i] = sg * b.volume; vol_sum[i] += b.volume; }
        if (!any) continue; days.insert(day); r_rows.push_back(r); s_rows.push_back(s); ts_rows.push_back(kv.first);
    }
    E.n_days = days.size(); if (r_rows.size() < 50) return E;
    Vec V(N); for (int i = 0; i < N; ++i) V[i] = vol_sum[i] / std::max<size_t>(1, days.size());   // daily volume in contracts
    // windows of h bars (non-overlapping)
    std::vector<Win> W; for (size_t a = 0; a + p.window_bars <= r_rows.size(); a += p.window_bars) { Win w; w.day = ts_rows[a] / 86400; w.R.assign(N, 0.0); w.Q.assign(N, 0.0); bool ok = true; for (int k = 0; k < p.window_bars; ++k) for (int i = 0; i < N; ++i) { if (std::isnan(r_rows[a + k][i])) { if (i == 0) ok = false; continue; } w.R[i] += r_rows[a + k][i]; w.Q[i] += s_rows[a + k][i] / std::max(V[i], 1.0); } if (ok) W.push_back(w); }
    E.n_contracts = N;
    E.n_windows = W.size(); if (W.size() < 50) return E;
    Vec sig(N, 0.0); for (int i = 0; i < N; ++i) { Vec x; for (auto& w : W) x.push_back(w.R[i]); sig[i] = std::max(stdev(x), 1e-8); }
    // pooled self-impact: (|Q|, sign(Q) R/sigma) over all contracts with volume
    auto pooled = [&](const std::vector<Win>& ws) { std::vector<std::pair<double, double>> xy; for (auto& w : ws) for (int i = 0; i < N; ++i) if (w.Q[i] != 0 && V[i] > 0) xy.push_back({std::fabs(w.Q[i]), (w.Q[i] > 0 ? 1 : -1) * w.R[i] / sig[i]}); return xy; };
    auto xy = pooled(W); auto dy = fit_power(xy, p.n_bins, &E.bin_x, &E.bin_y); E.delta = dy.first; E.Y = dy.second;
    // R^2 of linear vs concave self models on the raw windows
    auto r2_of = [&](double d, double Y) { double ss = 0, st = 0, my = 0; for (auto& q : xy) my += q.second; my /= xy.size(); for (auto& q : xy) { double pred = Y * std::pow(q.first, d); ss += (q.second - pred) * (q.second - pred); st += (q.second - my) * (q.second - my); } return st > 0 ? 1 - ss / st : 0; };
    { double sxy = 0, sxx = 0; for (auto& q : xy) { sxy += q.first * q.second; sxx += q.first * q.first; } double Ylin = sxx > 0 ? sxy / sxx : 0; E.Y_linear = Ylin; E.r2_linear = r2_of(1.0, std::max(Ylin, 1e-12)); E.r2_concave = r2_of(E.delta, E.Y);
      // and on the volume bins (the usual presentation): linear through the origin fitted on the bins vs the power law
      double bxy = 0, bxx = 0, bm = 0; for (size_t k = 0; k < E.bin_x.size(); ++k) { bxy += E.bin_x[k] * E.bin_y[k]; bxx += E.bin_x[k] * E.bin_x[k]; bm += E.bin_y[k]; } bm /= std::max<size_t>(1, E.bin_y.size()); double Ylb = bxx > 0 ? bxy / bxx : 0, ssl = 0, ssc = 0, sst = 0;
      for (size_t k = 0; k < E.bin_x.size(); ++k) { double pl = Ylb * E.bin_x[k], pc = E.Y * std::pow(E.bin_x[k], E.delta); ssl += (E.bin_y[k] - pl) * (E.bin_y[k] - pl); ssc += (E.bin_y[k] - pc) * (E.bin_y[k] - pc); sst += (E.bin_y[k] - bm) * (E.bin_y[k] - bm); }
      E.r2_linear_bins = sst > 0 ? 1 - ssl / sst : 0; E.r2_concave_bins = sst > 0 ? 1 - ssc / sst : 0; }
    // per contract
    for (int i = 0; i < N; ++i) { std::vector<std::pair<double, double>> q; for (auto& w : W) if (w.Q[i] != 0) q.push_back({std::fabs(w.Q[i]), (w.Q[i] > 0 ? 1 : -1) * w.R[i] / sig[i]}); auto f = fit_power(q, p.n_bins); E.delta_by_contract.push_back(f.first); E.Y_by_contract.push_back(f.second); }
    // block bootstrap by day for delta
    { std::vector<long long> dl(days.begin(), days.end()); std::map<long long, std::vector<size_t>> idx; for (size_t k = 0; k < W.size(); ++k) idx[W[k].day].push_back(k); Rng rng(11); Vec ds;
      for (int b = 0; b < p.boot; ++b) { std::vector<Win> ws; for (size_t d = 0; d < dl.size(); ++d) for (size_t k : idx[dl[rng.index(dl.size())]]) ws.push_back(W[k]); ds.push_back(fit_power(pooled(ws), p.n_bins).first); }
      E.delta_lo = quantile(ds, 0.025); E.delta_hi = quantile(ds, 0.975); }
    // cross-impact: R_j/sigma_j on x_i = sign(Q_i)|Q_i|^delta, self-only vs full
    Mat Lam = zeros(N, N); double ss_self = 0, ss_full = 0, st = 0;
    for (int j = 0; j < N; ++j) {
        Mat G; Vec y, wt; for (auto& w : W) { Vec x(N); bool any = false; for (int i = 0; i < N; ++i) { x[i] = w.Q[i] == 0 ? 0 : (w.Q[i] > 0 ? 1 : -1) * std::pow(std::fabs(w.Q[i]), E.delta); if (x[i] != 0) any = true; } if (!any) continue; G.push_back(x); y.push_back(w.R[j] / sig[j]); wt.push_back(1.0); }
        if (G.size() < 20) continue; Mat P = zeros(N, N); for (int i = 0; i < N; ++i) P[i][i] = 1e-6; Vec beta = ridge_solve(G, y, wt, P); for (int i = 0; i < N; ++i) Lam[j][i] = beta[i];
        double my = mean(y); Mat Gs(G.size(), Vec(1)); for (size_t k = 0; k < G.size(); ++k) Gs[k][0] = G[k][j]; Vec bs = ridge_solve(Gs, y, wt, Mat{{1e-6}});
        for (size_t k = 0; k < G.size(); ++k) { double pf = dot(G[k], beta), ps = G[k][j] * bs[0]; ss_full += (y[k] - pf) * (y[k] - pf); ss_self += (y[k] - ps) * (y[k] - ps); st += (y[k] - my) * (y[k] - my); }
    }
    E.r2_self = st > 0 ? 1 - ss_self / st : 0; E.r2_cross = st > 0 ? 1 - ss_full / st : 0;
    // decay of the front contract's response
    { Vec g; for (int tau = 0; tau <= p.max_lag; ++tau) { double s = 0; size_t c = 0; for (size_t k = 0; k + tau < W.size(); ++k) if (W[k].Q[0] != 0) { s += (W[k].Q[0] > 0 ? 1 : -1) * W[k + tau].R[0] / sig[0]; ++c; } g.push_back(c ? s / c : 0); }
      // cumulative response G(tau) = sum_{k<=tau} g_k relative to the concurrent response g_0: the level of the impact tau windows later
      double g0 = g[0] != 0 ? g[0] : 1; double cum = g[0]; for (int tau = 1; tau <= p.max_lag; ++tau) { cum += g[tau]; E.decay.push_back(cum / g0); }
      // permanent fraction = the late-lag level; the transient excess decays exponentially at the fitted rate
      double perm = 0; int tail = std::max(1, p.max_lag / 3); for (int tau = p.max_lag - tail; tau < p.max_lag; ++tau) perm += E.decay[tau] / tail; perm = std::max(0.0, std::min(1.0, perm));
      Vec lx, ly; for (int tau = 1; tau <= p.max_lag; ++tau) { double ex = E.decay[tau - 1] - perm; if (ex > 1e-6) { lx.push_back(tau); ly.push_back(std::log(ex)); } }
      double rate = lx.size() >= 2 ? std::max(0.05, std::min(0.99, 1 - std::exp(ols(lx, ly).b))) : 0.5;   // fraction of the transient part lost per window
      E.model.decay_per_window = rate; E.model.perm_frac = perm; }
    // model for the simulator: sigma in price units needs the price level; the caller rescales (sigma here is in log-return units)
    E.model.sigma = sig; E.model.V = V; E.model.Y.assign(N, E.Y); E.model.delta = E.delta; E.model.Lambda = Lam; E.model.half_spread.assign(N, 0.0);
    return E;
}

} // namespace cetk
