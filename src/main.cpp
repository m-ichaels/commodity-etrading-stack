// cetk - commodity swap e-trading stack: curve, factors, impact, hedge, RFQ simulator, events, pricer.
//   cetk curve     --root CL --date D                       three curve models on one day: fit, leave-one-out, extrapolation, swaps, sensitivities
//   cetk curvehist --root CL [--from D0 --to D1]            daily builds: LOO by method and completeness, selection rule, hedge test
//   cetk factors   --root CL                                PCA on the fitted ladder (recent window) and on the EIA front contracts (long history)
//   cetk impact    --root CL --bars F                       self- and cross-impact from bars; writes the impact model for the simulator
//   cetk hedge     --root CL                                hedging policies on random swap books: cost per $1M notional, residual risk, CIs
//   cetk sim       --root CL [--days 40 --seeds 8 ...]      RFQ market making: quoters x hedgers, P&L decomposition, mark-outs, risk controls
//   cetk events    --root CL --features F --from --to       replay real days (front-month bridge) with release-day jumps and the calendar feature
//   cetk price     --tier consumer --tenor 2 --side 1 --qty 25   one two-way price (the REST endpoint calls this)
//   cetk datacheck --root CL                                data coverage and parity summary
#include "cetk/engine.hpp"
#include <iostream>
#include <map>
#include <memory>
#include <set>

using namespace cetk;

struct Args { std::map<std::string, std::string> kv; std::vector<std::string> pos;
    std::string get(const std::string& k, const std::string& d = "") const { auto it = kv.find(k); return it == kv.end() ? d : it->second; }
    double num(const std::string& k, double d) const { auto it = kv.find(k); return it == kv.end() ? d : std::atof(it->second.c_str()); }
    bool has(const std::string& k) const { return kv.count(k) > 0; }
    std::vector<std::string> list(const std::string& k, const std::vector<std::string>& d) const { if (!has(k)) return d; std::vector<std::string> v; std::string cur; for (char c : get(k)) { if (c == ',') { if (!cur.empty()) v.push_back(cur); cur.clear(); } else cur += c; } if (!cur.empty()) v.push_back(cur); return v; }
    std::vector<double> nums(const std::string& k, const std::vector<double>& d) const { if (!has(k)) return d; std::vector<double> v; for (auto& s : list(k, {})) v.push_back(std::atof(s.c_str())); return v; }
    std::vector<int> ints(const std::string& k, const std::vector<int>& d) const { if (!has(k)) return d; std::vector<int> v; for (auto& s : list(k, {})) v.push_back(std::atoi(s.c_str())); return v; } };
static Args parse_args(int argc, char** argv) { Args a; for (int i = 2; i < argc; ++i) { std::string s = argv[i]; if (s.rfind("--", 0) == 0) { std::string k = s.substr(2); if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) a.kv[k] = argv[++i]; else a.kv[k] = "1"; } else a.pos.push_back(s); } return a; }
static void logmsg(const std::string& s) { std::cerr << "[cetk] " << s << std::endl; }
static std::string lower(std::string s) { for (auto& c : s) c = (char)std::tolower(c); return s; }
static std::string root_of(const Args& a) { return a.get("root", "CL"); }
static std::string curves_path(const Args& a) { return a.get("curves", "data/derived/curves_" + lower(root_of(a)) + ".csv"); }
static double contract_size_of(const std::string& root) { return root == "NG" ? 10000.0 : 1000.0; }

static CurveSpec spec_of(CurveMethod m, double lambda, double theta = 1.0) { CurveSpec s; s.method = m; s.lambda = lambda; s.theta = theta; return s; }
static const std::vector<CurveMethod> METHODS = {CurveMethod::SeasonalNSS, CurveMethod::PSpline, CurveMethod::KernelRidge};
static double default_lambda(CurveMethod m) { return m == CurveMethod::PSpline ? 1e-2 : m == CurveMethod::KernelRidge ? 1e-3 : 0; }

// ---- factor model json round trip --------------------------------------------------------------------------------
static FactorModel factors_from_json(const Json& j) { FactorModel f; for (auto& v : j["months"].arr()) f.months.push_back((int)v.num()); for (auto& v : j["evals_usd2_per_day"].arr()) f.evals.push_back(v.num()); for (auto& r : j["loadings"].arr()) { Vec row; for (auto& v : r.arr()) row.push_back(v.num()); f.loadings.push_back(row); } f.K = (int)f.loadings.size(); f.n_days = (size_t)j.get("n_days", 0LL); for (auto& r : j["cov"].arr()) { Vec row; for (auto& v : r.arr()) row.push_back(v.num()); f.cov.push_back(row); } return f; }
static Json load_json(const std::string& p) { return Json::parse(read_file(p)); }

// ---- curve ---------------------------------------------------------------------------------------------------------
static Json curve_report(const std::string& root, const CurveDay& d, const std::map<CurveMethod, CurveSpec>& specs) {
    Json out = Json::object(); out["date"] = d.date; out["root"] = root; out["n_points"] = (long long)d.pts.size(); out["gap_months"] = d.gap_months;
    Json pts = Json::array(); for (auto& p : d.pts) { Json e = Json::object(); e["delivery"] = p.delivery; e["months"] = p.months; e["T"] = p.T; e["price"] = p.price; e["source"] = p.source; e["volume"] = p.volume; pts.push(e); } out["points"] = pts;
    Json methods = Json::object();
    for (CurveMethod m : METHODS) {
        CurveSpec s = specs.at(m); CurveModel cm; cm.fit(d.pts, s); LooResult loo = leave_one_out(d.pts, s); Vec ex = extrapolation_errors(d.pts, s, 3);
        Json j = Json::object(); j["lambda"] = s.lambda; j["knot_spacing"] = s.knot_spacing; j["ell"] = s.ell; j["theta"] = s.theta; j["harmonics"] = s.harmonics; j["in_sample_rmse"] = cm.in_sample_rmse; j["max_abs_err"] = cm.max_abs_err; j["loo_mean_abs"] = loo.mean_abs; j["loo_mean_abs_interior"] = loo.mean_abs_interior; j["loo_mean_abs_pct"] = loo.mean_abs_pct; j["loo_err"] = Json(loo.err); j["loo_months"] = Json(loo.months); j["extrapolation_err_last3"] = Json(ex);
        if (m == CurveMethod::SeasonalNSS) { j["tau1"] = cm.tau1; j["tau2"] = cm.tau2; }
        Json grid = Json::array(); for (double T = 0.02; T <= d.pts.back().T + 0.5; T += 0.02) { double phase = 0; { long long dd = to_days(d.ymd) + (long long)(T * 365); Ymd x = civil_from_days(dd); phase = season_phase(x.m); } Json g = Json::object(); g["T"] = T; g["price"] = cm.predict(T, phase); grid.push(g); } j["grid"] = grid;
        // calendar swaps for the next 12 delivery months and the two front quarterly strips
        Json sw = Json::array(); auto first = nth_delivery(root, d.ymd, 1); int y = first.first, mo = first.second; for (int k = 0; k < 12; ++k) { SwapPrice sp = calendar_swap(root, d, &cm, y, mo); Json e = Json::object(); e["month"] = fmt_month(y, mo); e["price"] = sp.price; e["business_days"] = sp.business_days; Json legs = Json::array(); for (auto& l : sp.legs) { Json le = Json::object(); le["delivery"] = l.delivery; le["weight"] = l.weight; le["price"] = l.price; le["observed"] = l.observed; legs.push(le); } e["legs"] = legs; sw.push(e); if (++mo == 13) { mo = 1; ++y; } } j["calendar_swaps"] = sw;
        { auto q = nth_delivery(root, d.ymd, 2); j["strip_3m"] = strip_price(root, d, &cm, q.first, q.second, 3); j["strip_12m"] = strip_price(root, d, &cm, q.first, q.second, 12); }
        // key-rate sensitivities of the month-3 calendar swap to the observed contracts
        if (d.pts.size() > 4) { auto del = nth_delivery(root, d.ymd, 3); double T = years_to_expiry(root, d.ymd, del.first, del.second); Vec sens = key_rate_sensitivities(d.pts, s, T, season_phase(del.second), &cm); j["sens_month3_to_contracts"] = Json(sens); }
        methods[method_name(m)] = j;
    }
    out["methods"] = methods; return out;
}
static int cmd_curve(const Args& a) {
    std::string root = root_of(a); auto days = read_curves(curves_path(a), root); if (days.empty()) { std::cerr << "no curves\n"; return 1; }
    std::string want = a.get("date", days.back().date); const CurveDay* d = nullptr; for (auto& x : days) if (x.date <= want) d = &x; if (!d) { std::cerr << "no such date\n"; return 1; }
    // specs: the candidates selected by curvehist when its results exist (or --specs FILE), else the defaults; --lambda_<method> overrides
    std::map<CurveMethod, CurveSpec> specs; for (CurveMethod m : METHODS) specs[m] = spec_of(m, default_lambda(m));
    std::string sp = a.get("specs", "results/curvehist_" + lower(root) + ".json");
    if (file_exists(sp)) { Json ch = load_json(sp); for (CurveMethod m : METHODS) { std::string lab = ch["selected"][method_name(m)]["label"].str(); for (auto& c : ch["candidates"].arr()) if (c["method"].str() == method_name(m) && c["label"].str() == lab) { CurveSpec& cs = specs[m]; cs.lambda = c["lambda"].num(); cs.ell = c["ell"].num(); cs.knot_spacing = c["knots_years"].num(); cs.harmonics = (int)c["harmonics"].num(); } } std::cout << "using the candidates selected in " << sp << "\n"; }
    for (CurveMethod m : METHODS) if (a.has(std::string("lambda_") + method_name(m))) specs[m].lambda = a.num(std::string("lambda_") + method_name(m), specs[m].lambda);
    Json out = curve_report(root, *d, specs); write_file(a.get("out", "results/curve_" + lower(root) + "_" + d->date + ".json"), out.dump(1));
    std::cout << root << " " << d->date << ": " << d->pts.size() << " contracts, gap " << d->gap_months << " months\n";
    for (CurveMethod m : METHODS) { auto& j = out["methods"][method_name(m)]; std::cout << "  " << method_name(m) << ": in-sample rmse " << j["in_sample_rmse"].num() << ", LOO mean |err| " << j["loo_mean_abs"].num() << " (" << j["loo_mean_abs_pct"].num() << "%), interior " << j["loo_mean_abs_interior"].num() << ", strip 3m " << j["strip_3m"].num() << "\n"; }
    return 0;
}

// ---- curvehist ------------------------------------------------------------------------------------------------------
static int cmd_curvehist(const Args& a) {
    std::string root = root_of(a); auto days = read_curves(curves_path(a), root); std::string from = a.get("from", "2024-09-01"), to = a.get("to", "2099-12-31"); int max_gap = (int)a.num("max-gap", 999), min_pts = (int)a.num("min-points", 8);
    std::vector<const CurveDay*> use; for (auto& d : days) if (d.date >= from && d.date <= to && d.gap_months <= max_gap && (int)d.pts.size() >= min_pts) use.push_back(&d);
    if (use.size() < 5) { std::cerr << "too few days (" << use.size() << ")\n"; return 1; }
    logmsg("curvehist " + root + ": " + std::to_string(use.size()) + " days");
    struct Cand { CurveMethod m; double lambda; double ell; double knots; int harm; std::string label() const { char b[96]; if (m == CurveMethod::SeasonalNSS) std::snprintf(b, sizeof b, "harmonics=%d", harm); else if (m == CurveMethod::PSpline) std::snprintf(b, sizeof b, "lambda=%g knots=%gm", lambda, knots * 12); else std::snprintf(b, sizeof b, "lambda=%g ell=%g", lambda, ell); return b; } };
    std::vector<Cand> cands; for (int h : a.ints("nss-harmonics", {2, 3})) cands.push_back({CurveMethod::SeasonalNSS, 0, 0.5, 1.0 / 12, h});
    for (double k : a.nums("pspline-knots", {1.0 / 12, 1.0 / 6})) for (double l : a.nums("pspline-lambdas", {1e-4, 1e-3, 1e-2, 1e-1, 1.0})) cands.push_back({CurveMethod::PSpline, l, 0.5, k, 2});
    for (double e : a.nums("kr-ells", {0.2, 0.35, 0.5})) for (double l : a.nums("kr-lambdas", {1e-5, 1e-4, 1e-3, 1e-2, 1e-1})) cands.push_back({CurveMethod::KernelRidge, l, e, 1.0 / 12, 2});
    auto spec_c = [](const Cand& c) { CurveSpec s = spec_of(c.m, c.lambda); s.ell = c.ell; s.knot_spacing = c.knots; s.harmonics = c.harm; return s; };
    size_t n_train = std::max<size_t>(5, use.size() / 3);   // selection rule fixed in advance: lambda by LOO on the first third of days, frozen
    Json out = Json::object(); out["root"] = root; out["n_days"] = (long long)use.size(); out["from"] = use.front()->date; out["to"] = use.back()->date; out["n_train"] = (long long)n_train;
    Json cj = Json::array(); std::map<std::string, std::vector<double>> loo_test_by_key;   // for the selection
    std::vector<std::vector<double>> loo_all(cands.size()), pct_all(cands.size()), rmse_all(cands.size()), ext_all(cands.size()); std::vector<std::map<std::string, std::vector<double>>> by_gap(cands.size()), by_year(cands.size());
    for (size_t c = 0; c < cands.size(); ++c) {
        CurveSpec s = spec_c(cands[c]);
        for (size_t i = 0; i < use.size(); ++i) { const CurveDay& d = *use[i]; LooResult loo = leave_one_out(d.pts, s); CurveModel cm; cm.fit(d.pts, s); Vec ex = extrapolation_errors(d.pts, s, 3); double exm = 0; for (double e : ex) exm += std::fabs(e) / std::max<size_t>(1, ex.size());
            loo_all[c].push_back(loo.mean_abs); pct_all[c].push_back(loo.mean_abs_pct); rmse_all[c].push_back(cm.in_sample_rmse); ext_all[c].push_back(exm);
            std::string gk = d.gap_months == 0 ? "0" : d.gap_months <= 3 ? "1-3" : d.gap_months <= 12 ? "4-12" : ">12"; by_gap[c][gk].push_back(loo.mean_abs); by_year[c][d.date.substr(0, 4)].push_back(loo.mean_abs); }
        Json j = Json::object(); j["method"] = method_name(cands[c].m); j["lambda"] = cands[c].lambda; j["ell"] = cands[c].ell; j["knots_years"] = cands[c].knots; j["harmonics"] = cands[c].harm; j["label"] = cands[c].label(); j["loo_mean_abs"] = mean(loo_all[c]); j["loo_mean_abs_pct"] = mean(pct_all[c]); j["loo_median_abs"] = median(loo_all[c]); j["in_sample_rmse"] = mean(rmse_all[c]); j["extrapolation_mean_abs"] = mean(ext_all[c]);
        Vec tr(loo_all[c].begin(), loo_all[c].begin() + n_train), te(loo_all[c].begin() + n_train, loo_all[c].end()); j["loo_train"] = mean(tr); j["loo_test"] = mean(te);
        Json g = Json::object(); for (auto& kv : by_gap[c]) { Json e = Json::object(); e["n"] = (long long)kv.second.size(); e["loo_mean_abs"] = mean(kv.second); g[kv.first] = e; } j["by_gap"] = g;
        Json yj = Json::object(); for (auto& kv : by_year[c]) yj[kv.first] = mean(kv.second); j["by_year"] = yj; j["loo_daily"] = Json(loo_all[c]); cj.push(j);
    }
    out["candidates"] = cj;
    // selection per method on the training window; test-window comparison against the seasonal NSS
    Json sel = Json::object(); std::map<CurveMethod, size_t> chosen;
    for (CurveMethod m : METHODS) { double best = 1e300; size_t bi = 0; for (size_t c = 0; c < cands.size(); ++c) if (cands[c].m == m) { double tr = mean(Vec(loo_all[c].begin(), loo_all[c].begin() + n_train)); if (tr < best) { best = tr; bi = c; } } chosen[m] = bi; }
    size_t nss_ref = chosen[CurveMethod::SeasonalNSS];
    Vec dates; Json dj = Json::array(); for (auto* d : use) dj.push(d->date); out["dates"] = dj; Json gaps = Json::array(); for (auto* d : use) gaps.push(d->gap_months); out["gaps"] = gaps;
    for (auto& kv : chosen) { size_t c = kv.second; Json e = Json::object(); e["lambda"] = cands[c].lambda; e["label"] = cands[c].label(); Vec te(loo_all[c].begin() + n_train, loo_all[c].end()), te0(loo_all[nss_ref].begin() + n_train, loo_all[nss_ref].end()); e["loo_test_mean_abs"] = mean(te); e["loo_test_mean_abs_pct"] = mean(Vec(pct_all[c].begin() + n_train, pct_all[c].end()));
        Vec diff; for (size_t i = 0; i < te.size(); ++i) diff.push_back(100 * (te[i] - te0[i]) / std::max(te0[i], 1e-9)); CI ci = bootstrap_mean_ci(diff); e["vs_nss_pct"] = ci.mean; e["vs_nss_ci95"] = Json(Vec{ci.lo, ci.hi}); e["extrapolation_test_mean_abs"] = mean(Vec(ext_all[c].begin() + n_train, ext_all[c].end())); sel[method_name(kv.first)] = e; }
    out["selected"] = sel;
    // hedge test: the month-3 contract priced off the curve without it and hedged with that curve's key-rate sensitivities; realised next-day P&L
    { Json ht = Json::object(); int target_m = (int)a.num("hedge-month", 3); std::map<std::string, Vec> pnl; Vec unhedged;
      for (size_t i = 0; i + 1 < use.size(); ++i) { const CurveDay& d = *use[i]; const CurveDay& d1 = *use[i + 1]; const CurvePoint* tgt = d.at_months(target_m); if (!tgt) continue; const CurvePoint* tgt1 = nullptr; for (auto& p : d1.pts) if (p.delivery == tgt->delivery) tgt1 = &p; if (!tgt1) continue;
        std::vector<CurvePoint> sub; for (auto& p : d.pts) if (p.months != target_m) sub.push_back(p); if (sub.size() < 5) continue;
        std::vector<double> dP; bool ok = true; for (auto& p : sub) { const CurvePoint* p1 = nullptr; for (auto& x : d1.pts) if (x.delivery == p.delivery) p1 = &x; if (!p1) { ok = false; break; } dP.push_back(p1->price - p.price); } if (!ok) continue;
        double dT = tgt1->price - tgt->price; unhedged.push_back(dT);
        for (auto& kv : chosen) { CurveSpec s = spec_c(cands[kv.second]); Vec sens = key_rate_sensitivities(sub, s, tgt->T, tgt->phase); double h = 0; for (size_t k = 0; k < sub.size(); ++k) h += sens[k] * dP[k]; pnl[method_name(kv.first)].push_back(dT - h); }
        { double h = 0; for (size_t k = 0; k < sub.size(); ++k) if (sub[k].months == 1) h = dP[k]; pnl["front_stack"].push_back(dT - h); }
        { double h = 0; int cnt = 0; for (size_t k = 0; k < sub.size(); ++k) if (sub[k].months == target_m - 1 || sub[k].months == target_m + 1) { h += dP[k]; ++cnt; } if (cnt == 2) pnl["neighbours_50_50"].push_back(dT - 0.5 * h); } }
      ht["target_month"] = target_m; ht["n_days"] = (long long)unhedged.size(); ht["unhedged_std"] = stdev(unhedged); Json m = Json::object(); for (auto& kv : pnl) { Json e = Json::object(); e["n"] = (long long)kv.second.size(); e["hedged_std"] = stdev(kv.second); e["reduction_vs_unhedged"] = unhedged.empty() ? 0 : 1 - stdev(kv.second) / std::max(stdev(unhedged), 1e-12); m[kv.first] = e; } ht["methods"] = m; out["hedge_test"] = ht; }
    write_file(a.get("out", "results/curvehist_" + lower(root) + ".json"), out.dump(1));
    for (auto& kv : chosen) std::cout << method_name(kv.first) << " (" << cands[kv.second].label() << "): test LOO " << out["selected"][method_name(kv.first)]["loo_test_mean_abs"].num() << " (" << out["selected"][method_name(kv.first)]["vs_nss_pct"].num() << "% vs NSS)\n";
    return 0;
}

// ---- factors ------------------------------------------------------------------------------------------------------------
static int cmd_factors(const Args& a) {
    std::string root = root_of(a); auto days = read_curves(curves_path(a), root); std::vector<int> ladder = a.ints("ladder", {1, 2, 3, 6, 12}); int max_gap = (int)a.num("max-gap", 2);
    CurveSpec s = spec_of(CurveMethod::KernelRidge, a.num("lambda", 1e-3)); std::vector<std::string> dates; std::vector<CurveDay> recent; for (auto& d : days) if (d.date >= a.get("from", "2024-09-01")) recent.push_back(d);
    auto X = ladder_panel(recent, s, ladder, root, max_gap, &dates); Json out = Json::object(); out["root"] = root; out["ladder"] = Json(ladder);
    if (X.size() >= 10) { FactorModel fm; fm.fit_panel(X, ladder, 3); fm.from = dates.front(); fm.to = dates.back(); out["recent"] = fm.to_json(); out["recent"]["max_gap"] = max_gap; std::cout << "recent window " << fm.from << ".." << fm.to << " (" << fm.n_days << " days): explained " << fm.explained(1) << "/" << fm.explained(2) << "/" << fm.explained(3) << ", level sd " << std::sqrt(fm.evals[0]) << " $/unit/day\n"; }
    else std::cerr << "recent window too short for the factor model (" << X.size() << " days)\n";
    // long history on the EIA front contracts, by window
    std::string fp = a.get("front", "data/raw/eia_" + lower(root) + "_front.csv"); Json hist = Json::object();
    try { auto rows = read_front(fp); std::vector<std::pair<std::string, std::string>> windows = {{"1990-01-01", "1999-12-31"}, {"2000-01-01", "2009-12-31"}, {"2010-01-01", "2019-12-31"}, {"2020-01-01", "2024-12-31"}, {"2010-01-01", "2024-12-31"}};
        for (auto& w : windows) { auto P = front_panel(rows, w.first, w.second); if (P.size() < 100) continue; FactorModel fm; fm.fit_panel(P, {1, 2, 3, 4}, 3); fm.from = w.first; fm.to = w.second; hist[w.first + ":" + w.second] = fm.to_json(); }
    } catch (const std::exception& e) { logmsg(std::string("front panel skipped: ") + e.what()); }
    out["front_history"] = hist; write_file(a.get("out", "results/factors_" + lower(root) + ".json"), out.dump(1)); return 0;
}

// ---- impact ------------------------------------------------------------------------------------------------------------
static int cmd_impact(const Args& a) {
    std::string root = root_of(a); std::string bp = a.get("bars", "data/derived/bars_" + lower(root) + "_1h.csv"); auto bars = read_bars(bp); if (bars.empty()) { std::cerr << "no bars\n"; return 1; }
    ImpactParams p; p.n_contracts = (int)a.num("n", 12); p.boot = (int)a.num("boot", 200); p.bars_per_day = a.num("bars-per-day", 23);
    // restrict to the last `days` days, when the listed contracts were the liquid front of the curve
    std::set<long long> only; { long long last = bars.back().ts / 86400; int nd = (int)a.num("days", 90); for (long long d = last - nd; d <= last; ++d) only.insert(d); }
    double price = 0; { size_t c = 0; for (auto it = bars.rbegin(); it != bars.rend() && c < 200; ++it) if (it->months == 1) { price += it->close; ++c; } price /= std::max<size_t>(1, c); }
    Json out = Json::object(); out["root"] = root; out["bars"] = bp; out["price_level"] = price; Json ws = Json::array(); ImpactEstimate main;
    for (int w : a.ints("windows", {1, 3, 6})) { p.window_bars = w; ImpactEstimate E = estimate_impact(bars, p, &only); ws.push(E.to_json()); std::cout << "window " << w << " bars: " << E.n_windows << " windows over " << E.n_days << " days; delta " << E.delta << " [" << E.delta_lo << ", " << E.delta_hi << "], Y " << E.Y << "; R2 linear " << E.r2_linear << " concave " << E.r2_concave << "; R2 self " << E.r2_self << " self+cross " << E.r2_cross << "\n"; if (w == (int)a.num("model-window", 1)) main = E; }
    out["windows"] = ws;
    // simulator model at the ladder tenors: sigma in price units, half spread one tick, Lambda restricted to the ladder
    std::vector<int> ladder = a.ints("ladder", {1, 2, 3, 6, 12}); ImpactModel m; m.delta = main.model.delta; m.decay_per_window = main.model.decay_per_window; m.perm_frac = main.model.perm_frac; double tick = root == "NG" ? 0.001 : 0.01;
    for (int mo : ladder) { size_t i = (size_t)std::min(mo, p.n_contracts) - 1; m.sigma.push_back(main.model.sigma.empty() ? 0.01 * price : main.model.sigma[i] * price); m.V.push_back(main.model.V.empty() ? 1000 : std::max(main.model.V[i], 50.0)); m.Y.push_back(main.model.Y.empty() ? 0.5 : main.model.Y[i]); m.half_spread.push_back(tick * (mo <= 3 ? 1 : mo <= 6 ? 2 : 3)); }
    m.Lambda = zeros(ladder.size(), ladder.size()); for (size_t ja = 0; ja < ladder.size(); ++ja) for (size_t ia = 0; ia < ladder.size(); ++ia) { size_t j = (size_t)std::min(ladder[ja], p.n_contracts) - 1, i = (size_t)std::min(ladder[ia], p.n_contracts) - 1; m.Lambda[ja][ia] = main.model.Lambda.empty() ? (ja == ia ? m.Y[ja] : 0.5 * m.Y[ja]) : std::max(0.0, main.model.Lambda[j][i]); }
    out["simulator_model"] = m.to_json(); out["ladder"] = Json(ladder);
    write_file(a.get("out", "results/impact_" + lower(root) + ".json"), out.dump(1)); write_file(a.get("model-out", "configs/impact_" + lower(root) + ".json"), m.to_json().dump(1)); return 0;
}

// ---- shared setup for hedge / sim / events / price ------------------------------------------------------------------------
struct Setup { LadderSpec ls; LadderRisk risk; FactorModel fm; ImpactModel impact; std::vector<ClientTier> tiers; QuoteParams qp; Hedger hedger; };
static Setup make_setup(const Args& a) {
    Setup S; std::string root = root_of(a); S.ls.root = root; S.ls.contract_size = contract_size_of(root); S.ls.months = a.ints("ladder", {1, 2, 3, 6, 12});
    Json fj = load_json(a.get("factors", "results/factors_" + lower(root) + ".json")); if (!fj.has("recent")) throw Error("factors file has no recent window"); S.fm = factors_from_json(fj["recent"]);
    S.impact = ImpactModel::from_json(load_json(a.get("impact", "configs/impact_" + lower(root) + ".json"))); if (S.impact.n() != S.ls.months.size()) throw Error("impact model ladder size mismatch");
    if (double isc = a.num("impact-scale", 1.0); isc != 1.0) { for (auto& y : S.impact.Y) y *= isc; for (auto& row : S.impact.Lambda) for (auto& v : row) v *= isc; }   // sensitivity: the estimated impact scaled up or down
    S.ls.price_level = a.num("price", root == "NG" ? 3.0 : 70.0); S.ls.tenor_years.clear(); for (int m : S.ls.months) S.ls.tenor_years.push_back(m / 12.0);
    S.risk.build(S.fm, S.ls); S.tiers = default_tiers(root); if (root == "NG") { for (auto& t : S.tiers) { t.beta *= 20; t.mu /= 20; } }   // gas prices are ~20x smaller per unit; spreads and drifts scale with them
    S.qp.gamma = a.num("gamma", root == "NG" ? 5e-7 : 5e-7); S.qp.hit_target = a.num("hit-target", 0.4); S.qp.event_widen = a.num("event-widen", root == "NG" ? 0.005 : 0.05);
    S.hedger.impact = S.impact; S.hedger.Sigma = S.risk.Sigma; S.hedger.contract_size = S.ls.contract_size; S.hedger.gamma = S.qp.gamma; S.hedger.tau_days = a.num("tau-days", 0.25); S.hedger.belly = 3;
    return S;
}
static std::unique_ptr<Quoter> make_quoter(const std::string& name, const Setup& S, QuoteParams qp) {
    if (name == "Symmetric") return std::make_unique<SymmetricQuoter>(S.tiers, qp);
    if (name == "BEGV") return std::make_unique<BegvQuoter>(S.tiers, S.ls, S.risk, qp, false, false);
    if (name == "BEGV_adv") { qp.calendar = false; return std::make_unique<BegvQuoter>(S.tiers, S.ls, S.risk, qp, true, false); }
    if (name == "BEGV_adv_cal") { qp.calendar = true; return std::make_unique<BegvQuoter>(S.tiers, S.ls, S.risk, qp, true, false); }
    if (name == "Oracle") return std::make_unique<BegvQuoter>(S.tiers, S.ls, S.risk, qp, true, true);
    throw Error("unknown quoter " + name);
}

static int cmd_hedge(const Args& a) {
    Setup S = make_setup(a); int books = (int)a.num("books", 400); size_t n = S.ls.months.size();
    // book distribution: consumers leave the desk short the front two years, front-loaded; stated as the calibration target
    Vec mean_q = a.nums("mean-q", {-40, -35, -30, -20, -10}), sd_q = a.nums("sd-q", {40, 35, 30, 20, 12}); mean_q.resize(n, mean_q.back()); sd_q.resize(n, sd_q.back());
    Json out = Json::object(); out["root"] = root_of(a); out["books"] = books; out["gamma"] = S.hedger.gamma; out["tau_days"] = S.hedger.tau_days; out["mean_q"] = Json(mean_q); out["sd_q"] = Json(sd_q); out["impact"] = S.impact.to_json();
    Json cells = Json::array(); for (double g : a.nums("gammas", {S.hedger.gamma})) { Hedger H = S.hedger; H.gamma = g; auto res = hedge_experiment(H, S.ls, books, (uint64_t)a.num("seed", 7), mean_q, sd_q);
        for (auto& c : res) { Json e = Json::object(); e["gamma"] = g; e["policy"] = c.policy; e["equal_risk"] = c.equal_risk; e["cost_usd"] = c.cost_usd; e["cost_ci95"] = Json(Vec{c.cost_lo, c.cost_hi}); e["cost_per_mn"] = c.cost_per_mn; e["cost_per_mn_ci95"] = Json(Vec{c.cost_per_mn_lo, c.cost_per_mn_hi}); e["risk_usd"] = c.risk_usd; e["objective"] = c.objective; e["vs_front_pct"] = c.vs_front_pct; e["vs_front_ci95"] = Json(Vec{c.vs_front_lo, c.vs_front_hi}); e["mean_h"] = Json(c.mean_h); e["mean_abs_h"] = c.mean_abs_h; cells.push(e);
            if (g == S.hedger.gamma) std::cout << c.policy << ": cost $" << c.cost_usd << " (" << c.cost_per_mn << " per $1M, " << c.vs_front_pct << "% vs front stack [" << c.vs_front_lo << ", " << c.vs_front_hi << "]), residual risk $" << c.risk_usd << "/day, objective $" << c.objective << "\n"; } }
    out["cells"] = cells; write_file(a.get("out", "results/hedge_" + lower(root_of(a)) + ".json"), out.dump(1)); return 0;
}

static Json cell_json(const std::vector<RunResult>& runs, double gamma, const std::string& hedger_name) {
    Json c = runs.front().to_json(); c["gamma"] = gamma; c["seeds"] = (long long)runs.size(); Vec pnl, pv, std5, hc, internal, adv;
    for (auto& r : runs) { pnl.push_back(r.total); pv.push_back(r.pnl_per_var()); std5.push_back(r.pnl_5min_std); hc.push_back(r.hedge_cost_per_mn); internal.push_back(r.internalisation); adv.push_back(r.pnl.adverse); }
    CI ci = bootstrap_mean_ci(pnl); c["pnl"] = ci.mean; c["pnl_ci95"] = Json(Vec{ci.lo, ci.hi}); c["pnl_per_var"] = mean(pv); c["pnl_5min_std"] = mean(std5); c["hedge_cost_per_mn_notional"] = mean(hc); c["internalisation_ratio"] = mean(internal); c["adverse"] = mean(adv);
    double sp = 0, inv = 0, ad = 0, he = 0, hi = 0, res = 0; for (auto& r : runs) { sp += r.pnl.spread; inv += r.pnl.inventory; ad += r.pnl.adverse; he += r.pnl.hedge_exec; hi += r.pnl.hedge_impact; res = std::max(res, std::fabs(r.residual)); } size_t ns = runs.size();
    Json d = Json::object(); d["spread"] = sp / ns; d["inventory"] = inv / ns; d["adverse"] = ad / ns; d["hedge_exec"] = he / ns; d["hedge_impact"] = hi / ns; d["max_abs_residual"] = res; c["decomposition"] = d; c["hedger"] = hedger_name; c["pnl_by_seed"] = Json(pnl); return c;
}

static int cmd_sim(const Args& a) {
    Setup S = make_setup(a); int n_days = (int)a.num("days", 40), n_seeds = (int)a.num("seeds", 8);
    auto quoters = a.list("quoters", {"Symmetric", "BEGV", "BEGV_adv", "BEGV_adv_cal", "Oracle"}); auto hedgers = a.list("hedgers", {"FrontStack", "Strip", "ImpactAware"}); auto gammas = a.nums("gammas", {S.qp.gamma});
    Json out = Json::object(); out["root"] = root_of(a); out["days"] = n_days; out["seeds"] = n_seeds; out["ladder"] = Json(S.ls.months); Json cells = Json::array();
    for (double g : gammas) for (auto& hn : hedgers) for (auto& qn : quoters) {
        std::vector<RunResult> runs;
        for (int s = 0; s < n_seeds; ++s) {
            EngineConfig cfg; cfg.ls = S.ls; cfg.qp = S.qp; cfg.qp.gamma = g; cfg.tiers = S.tiers; cfg.risk = S.risk; cfg.hedger = S.hedger; cfg.hedger.gamma = g; cfg.hedger.policy = parse_hedge(hn);
            cfg.hedge_review_min = a.num("review-min", 30); cfg.band_usd = a.num("band-usd", 1e5); cfg.tox_route = a.has("tox-route"); cfg.anticipate = a.has("anticipate"); cfg.market_seed = 1000 + s; cfg.F0.assign(S.ls.months.size(), S.ls.price_level); cfg.info_drift_per_day = a.num("info-drift", root_of(a) == "NG" ? 0.03 : 0.6);
            cfg.limits.max_abs_q = a.num("max-abs-q", 1000); cfg.limits.max_risk_usd = a.num("max-risk-usd", 2e6); cfg.limits.max_drawdown_usd = a.num("max-drawdown-usd", 1e6);
            Rng drng(500 + s); auto days = synthetic_days(n_days, S.risk, drng, a.num("p-event", 0.4), a.num("jump-sigma", 2.0));
            auto quoter = make_quoter(qn, S, cfg.qp); runs.push_back(run_engine(cfg, *quoter, days, 900 + s));
        }
        Json c = cell_json(runs, g, hn); cells.push(c);
        auto& dd = c["decomposition"]; std::cout << hn << " / " << qn << " gamma " << g << ": P&L $" << c["pnl"].num() / n_days << "/day [" << c["pnl_ci95"][0].num() / n_days << ", " << c["pnl_ci95"][1].num() / n_days << "] = spread " << dd["spread"].num() / n_days << " + inventory " << dd["inventory"].num() / n_days << " + adverse " << dd["adverse"].num() / n_days << " + hedge_exec " << dd["hedge_exec"].num() / n_days << " + hedge_impact " << dd["hedge_impact"].num() / n_days << "; 5-min std $" << c["pnl_5min_std"].num() << ", P&L/var " << c["pnl_per_var"].num() << ", internalisation " << c["internalisation_ratio"].num() << ", hedge cost " << c["hedge_cost_per_mn_notional"].num() << "/$1M, residual " << dd["max_abs_residual"].num() << "\n";
    }
    out["cells"] = cells; write_file(a.get("out", "results/sim_" + lower(root_of(a)) + ".json"), out.dump(1)); return 0;
}

// ---- events: real days with the front-month bridge ----------------------------------------------------------------------
static int cmd_events(const Args& a) {
    Setup S = make_setup(a); Features F = read_features(a.get("features", "data/derived/events_" + lower(root_of(a)) + ".csv"));
    std::string from = a.get("from", "2024-09-01"), to = a.get("to", "2099-12-31"); std::vector<DaySpec> days; size_t n = S.ls.months.size();
    for (auto& kv : F.by_date) { if (kv.first < from || kv.first > to) continue; DaySpec d; d.date = kv.first; Ymd y = parse_date(kv.first); d.t_year = (to_days(y) - days_from_civil(y.y, 1, 1)) / 365.0; d.close_change_front = F.get(kv.first, "front_change", NAN);
        if (F.get(kv.first, "release", 0) > 0) { EventSpec e; e.t_day = a.num("t-event", 1.5 / 8); double jump1 = F.get(kv.first, "release_jump", 0); e.jump.assign(n, 0.0); double l1 = S.risk.L[0][0] != 0 ? S.risk.L[0][0] : 1; for (size_t j = 0; j < n; ++j) e.jump[j] = jump1 * S.risk.L[0][j] / l1; d.events.push_back(e); }
        days.push_back(d); }
    if (days.empty()) { std::cerr << "no days\n"; return 1; } logmsg("events: " + std::to_string(days.size()) + " days from " + days.front().date + " to " + days.back().date);
    int n_seeds = (int)a.num("seeds", 3); auto quoters = a.list("quoters", {"Symmetric", "BEGV", "BEGV_adv", "BEGV_adv_cal", "Oracle"}); std::string hn = a.get("hedger", "ImpactAware");
    Json out = Json::object(); out["root"] = root_of(a); out["n_days"] = (long long)days.size(); out["from"] = days.front().date; out["to"] = days.back().date; size_t n_ev = 0; for (auto& d : days) n_ev += d.events.size(); out["n_events"] = (long long)n_ev; Json cells = Json::array();
    for (auto& qn : quoters) { std::vector<RunResult> runs; for (int s = 0; s < n_seeds; ++s) { EngineConfig cfg; cfg.ls = S.ls; cfg.qp = S.qp; cfg.tiers = S.tiers; cfg.risk = S.risk; cfg.hedger = S.hedger; cfg.hedger.policy = parse_hedge(hn); cfg.market_seed = 2000 + s; cfg.F0.assign(n, S.ls.price_level); cfg.hedge_review_min = a.num("review-min", 30); cfg.band_usd = a.num("band-usd", 1e5); cfg.info_drift_per_day = a.num("info-drift", root_of(a) == "NG" ? 0.03 : 0.6); auto quoter = make_quoter(qn, S, cfg.qp); runs.push_back(run_engine(cfg, *quoter, days, 1900 + s)); }
        Json c = cell_json(runs, S.qp.gamma, hn); cells.push(c); std::cout << qn << ": P&L $" << c["pnl"].num() / days.size() << "/day, P&L/var " << c["pnl_per_var"].num() << ", adverse $" << c["adverse"].num() / days.size() << "/day\n"; }
    out["cells"] = cells; write_file(a.get("out", "results/events_" + lower(root_of(a)) + ".json"), out.dump(1)); return 0;
}

// ---- price: one two-way quote (the REST endpoint calls this) -----------------------------------------------------------------
static int cmd_price(const Args& a) {
    Setup S = make_setup(a); QuoteParams qp = S.qp; auto quoter = make_quoter(a.get("quoter", "BEGV_adv_cal"), S, qp);
    int tier = 0; for (size_t i = 0; i < S.tiers.size(); ++i) if (S.tiers[i].name == a.get("tier", "consumer")) tier = (int)i;
    int tenor = 0; for (size_t j = 0; j < S.ls.months.size(); ++j) if (S.ls.months[j] == (int)a.num("tenor", 1)) tenor = (int)j;
    Vec q = a.nums("inventory", Vec(S.ls.months.size(), 0.0)); q.resize(S.ls.months.size(), 0.0); double mid = a.num("mid", S.ls.price_level); int side = (int)a.num("side", 1); double z = a.num("qty", 25);
    double delta = quoter->quote(tier, tenor, side, z, q, a.has("pre-event"), 0.5);
    Json o = Json::object(); o["tier"] = S.tiers[tier].name; o["tenor_months"] = S.ls.months[tenor]; o["side"] = side; o["qty_contracts"] = z; o["mid"] = mid; o["half_spread"] = delta; o["bid"] = mid - delta; o["ask"] = mid + delta; o["quoter"] = quoter->name(); o["risk_usd_before"] = std::sqrt(std::max(0.0, S.hedger.risk_usd2(q)));   // $/day, one sigma
    std::cout << o.dump(1) << std::endl; return 0;
}

static int cmd_datacheck(const Args& a) {
    std::string root = root_of(a); auto days = read_curves(curves_path(a), root); Json out = Json::object(); out["root"] = root; out["n_days"] = (long long)days.size(); if (days.empty()) return 1; out["first"] = days.front().date; out["last"] = days.back().date;
    std::map<std::string, long long> gap_hist; long long complete = 0; std::string first_complete; for (auto& d : days) if (d.date >= "2024-09-01") { std::string k = d.gap_months == 0 ? "0" : d.gap_months <= 3 ? "1-3" : d.gap_months <= 12 ? "4-12" : ">12"; gap_hist[k]++; if (d.gap_months == 0 && d.pts.size() >= 12) { ++complete; if (first_complete.empty()) first_complete = d.date; } }
    Json gh = Json::object(); for (auto& kv : gap_hist) gh[kv.first] = kv.second; out["gap_histogram_since_2024_09"] = gh; out["complete_days"] = complete; out["first_complete_day"] = first_complete;
    long long eia = 0; for (auto& d : days) if (d.date < "2024-09-01") ++eia; out["eia_days"] = eia; write_file(a.get("out", "results/datacheck_" + lower(root) + ".json"), out.dump(1)); std::cout << out.dump(1) << std::endl; return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: cetk <curve|curvehist|factors|impact|hedge|sim|events|price|datacheck> [--options]\n"; return 1; }
    std::string cmd = argv[1]; Args a = parse_args(argc, argv);
    try {
        if (cmd == "curve") return cmd_curve(a); if (cmd == "curvehist") return cmd_curvehist(a); if (cmd == "factors") return cmd_factors(a); if (cmd == "impact") return cmd_impact(a);
        if (cmd == "hedge") return cmd_hedge(a); if (cmd == "sim") return cmd_sim(a); if (cmd == "events") return cmd_events(a); if (cmd == "price") return cmd_price(a); if (cmd == "datacheck") return cmd_datacheck(a);
        std::cerr << "unknown command " << cmd << "\n"; return 1;
    } catch (const std::exception& e) { std::cerr << "error: " << e.what() << std::endl; return 1; }
}
