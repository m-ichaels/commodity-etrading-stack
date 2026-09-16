#pragma once
// Hedging a swap book in listed futures along the curve.  Inventory q (contracts) sits on the ladder tenors; the
// hedge h (contracts) goes into the same contracts, each with its own liquidity, self-impact and cross-impact.
//
//   FrontStack    everything into the front contract (the desk default: stack-and-roll)
//   Strip         each tenor hedged in its own contract (key-rate strip)
//   FactorMin     minimum residual variance using only the two liquid contracts {1, belly} (level + slope hedge)
//   ImpactAware   minimise  execution cost (spread + concave self-impact + cross-impact) + gamma tau (q+h)' Sigma (q+h)
//                 over all contracts by coordinate descent; ablations: no cross terms, linear impact (delta = 1)
// Costs in $ (contract_size units per contract); risk in $^2 per day.
#include "impact.hpp"
#include "quoter.hpp"

namespace cetk {

enum class HedgePolicy { None, FrontStack, Strip, FactorMin, ImpactAware, ImpactAwareNoCross, ImpactAwareLinear };
inline const char* hedge_name(HedgePolicy p) { switch (p) { case HedgePolicy::None: return "NoHedge"; case HedgePolicy::FrontStack: return "FrontStack"; case HedgePolicy::Strip: return "Strip"; case HedgePolicy::FactorMin: return "FactorMin"; case HedgePolicy::ImpactAware: return "ImpactAware"; case HedgePolicy::ImpactAwareNoCross: return "ImpactAware_noCross"; case HedgePolicy::ImpactAwareLinear: return "ImpactAware_linear"; } return "?"; }
inline HedgePolicy parse_hedge(const std::string& s) { for (HedgePolicy p : {HedgePolicy::None, HedgePolicy::FrontStack, HedgePolicy::Strip, HedgePolicy::FactorMin, HedgePolicy::ImpactAware, HedgePolicy::ImpactAwareNoCross, HedgePolicy::ImpactAwareLinear}) if (s == hedge_name(p)) return p; throw Error("unknown hedge policy " + s); }

struct Hedger {
    HedgePolicy policy = HedgePolicy::ImpactAware;
    ImpactModel impact;            // price units per unit, contracts as quantities
    Mat Sigma;                     // ($/unit)^2/day on the ladder
    double contract_size = 1000, gamma = 5e-7, tau_days = 1.0;   // risk horizon until the next review
    size_t belly = 3;              // FactorMin's second contract index

    double risk_usd2(const Vec& q) const { Vec u(q.size()); for (size_t j = 0; j < q.size(); ++j) u[j] = q[j] * contract_size; return quad(u, Sigma); }   // $^2/day
    double cost_usd(const Vec& h, const ImpactModel& m) const { return contract_size * m.exec_cost(h); }
    double objective(const Vec& q, const Vec& h, const ImpactModel& m) const { Vec r(q.size()); for (size_t j = 0; j < q.size(); ++j) r[j] = q[j] + h[j]; return cost_usd(h, m) + gamma * tau_days * risk_usd2(r); }

    ImpactModel variant(HedgePolicy p) const { ImpactModel m = impact; if (p == HedgePolicy::ImpactAwareNoCross) { for (size_t a = 0; a < m.Lambda.size(); ++a) for (size_t b = 0; b < m.Lambda.size(); ++b) if (a != b) m.Lambda[a][b] = 0; } if (p == HedgePolicy::ImpactAwareLinear) { m.delta = 1.0; } return m; }

    Vec hedge(const Vec& q) const { return hedge(q, policy); }
    Vec hedge(const Vec& q, HedgePolicy p) const { return hedge(q, p, gamma); }
    // impact-aware hedge whose residual risk matches a target ($/day): log-bisection on the risk aversion
    Vec hedge_to_risk(const Vec& q, HedgePolicy p, double target_risk_usd) const {
        double lo = 1e-10, hi = 1e-2; Vec best = hedge(q, p, hi);
        for (int it = 0; it < 30; ++it) { double g = std::sqrt(lo * hi); Vec h = hedge(q, p, g); Vec r(q.size()); for (size_t j = 0; j < q.size(); ++j) r[j] = q[j] + h[j]; double risk = std::sqrt(risk_usd2(r)); if (risk <= target_risk_usd) { best = h; hi = g; } else lo = g; if (hi / lo < 1.05) break; }
        return best;
    }
    Vec hedge(const Vec& q, HedgePolicy p, double gam) const {
        size_t n = q.size(); Vec h(n, 0.0);
        switch (p) {
            case HedgePolicy::None: return h;
            case HedgePolicy::FrontStack: { double s = 0; for (double x : q) s += x; h[0] = -s; return h; }
            case HedgePolicy::Strip: { for (size_t j = 0; j < n; ++j) h[j] = -q[j]; return h; }
            case HedgePolicy::FactorMin: {   // least squares on contracts {0, belly}: minimise (q + a e0 + b eb)' S (q + a e0 + b eb)
                size_t b = std::min(belly, n - 1); Mat A = {{Sigma[0][0], Sigma[0][b]}, {Sigma[b][0], Sigma[b][b]}}; Vec rhs(2, 0.0);
                for (size_t j = 0; j < n; ++j) { rhs[0] -= Sigma[0][j] * q[j]; rhs[1] -= Sigma[b][j] * q[j]; }
                Vec x; if (solve(A, rhs, x)) { h[0] = x[0]; h[b] = x[1]; } return h; }
            default: {
                ImpactModel m = variant(p); double bound = 0; for (double x : q) bound += std::fabs(x); bound = std::max(bound * 2, 1.0);
                Hedger self = *this; self.gamma = gam; auto objective = [&](const Vec& qq, const Vec& hh, const ImpactModel& mm) { return self.Hedger::objective(qq, hh, mm); };
                // start from the strip (a feasible, cheap-to-improve point), then coordinate descent with golden-section line search
                for (size_t j = 0; j < n; ++j) h[j] = -q[j];
                double J = objective(q, h, m);
                for (int it = 0; it < 60; ++it) {
                    double J0 = J;
                    for (size_t j = 0; j < n; ++j) {
                        double lo = -bound, hi = bound; const double gr = 0.6180339887;
                        auto f = [&](double x) { Vec hh = h; hh[j] = x; return objective(q, hh, m); };
                        double c = hi - gr * (hi - lo), d = lo + gr * (hi - lo), fc = f(c), fd = f(d);
                        for (int k = 0; k < 60; ++k) { if (fc < fd) { hi = d; d = c; fd = fc; c = hi - gr * (hi - lo); fc = f(c); } else { lo = c; c = d; fc = fd; d = lo + gr * (hi - lo); fd = f(d); } }
                        double x = 0.5 * (lo + hi), fx = f(x); double f0 = f(0.0);   // zero is a kink of the concave cost: test it explicitly
                        if (f0 <= fx) { x = 0; fx = f0; }
                        if (fx < J) { h[j] = x; J = fx; }
                    }
                    if (J0 - J < 1e-6 * (1 + std::fabs(J0))) break;
                }
                return h;
            }
        }
    }
};

// Random-book experiment: books drawn from a stated distribution (net short from consumer demand, front-loaded),
// every policy applied to every book.  Costs are always evaluated under the full estimated impact model; the ablation
// variants (no cross-impact, linear impact) only change the decision.  Two comparisons:
//   at the desk's risk aversion  : each policy's cost, residual risk and objective cost + gamma tau risk^2
//   at equal risk                : the impact-aware hedge (and its ablations) tuned by bisection to leave exactly the
//                                  front stack's residual risk, so cost differences are like for like
struct HedgeCell { std::string policy; bool equal_risk = false; double cost_usd = 0, cost_lo = 0, cost_hi = 0, cost_per_mn = 0, cost_per_mn_lo = 0, cost_per_mn_hi = 0, risk_usd = 0, objective = 0, vs_front_pct = 0, vs_front_lo = 0, vs_front_hi = 0, mean_abs_h = 0; Vec mean_h; };
inline std::vector<HedgeCell> hedge_experiment(const Hedger& H, const LadderSpec& ls, int n_books, uint64_t seed, const Vec& mean_q, const Vec& sd_q) {
    Rng rng(seed); size_t n = ls.months.size();
    struct Pol { HedgePolicy p; bool equal_risk; std::string name; };
    std::vector<Pol> pols = {{HedgePolicy::None, false, "NoHedge"}, {HedgePolicy::FrontStack, false, "FrontStack"}, {HedgePolicy::Strip, false, "Strip"}, {HedgePolicy::FactorMin, false, "FactorMin"}, {HedgePolicy::ImpactAware, false, "ImpactAware"},
                             {HedgePolicy::ImpactAwareNoCross, true, "ImpactAware_noCross@FrontRisk"}, {HedgePolicy::ImpactAwareLinear, true, "ImpactAware_linear@FrontRisk"}, {HedgePolicy::ImpactAware, true, "ImpactAware@FrontRisk"}};
    std::vector<Vec> books; for (int b = 0; b < n_books; ++b) { Vec q(n); for (size_t j = 0; j < n; ++j) q[j] = std::round(mean_q[j] + sd_q[j] * rng.normal()); books.push_back(q); }
    std::vector<double> front_risk(n_books), notional(n_books); for (int b = 0; b < n_books; ++b) { Vec hf = H.hedge(books[b], HedgePolicy::FrontStack); Vec r(n); for (size_t j = 0; j < n; ++j) r[j] = books[b][j] + hf[j]; front_risk[b] = std::sqrt(H.risk_usd2(r)); double nt = 0; for (double x : books[b]) nt += std::fabs(x) * ls.contract_size * ls.price_level; notional[b] = nt; }
    std::vector<HedgeCell> out; std::vector<Vec> cpm_by_pol(pols.size());
    for (size_t p = 0; p < pols.size(); ++p) {
        HedgeCell c; c.policy = pols[p].name; c.equal_risk = pols[p].equal_risk; c.mean_h.assign(n, 0.0); Vec costs, cpms, risks, objs;
        for (int b = 0; b < n_books; ++b) { const Vec& q = books[b]; Vec h = pols[p].equal_risk ? H.hedge_to_risk(q, pols[p].p, front_risk[b]) : H.hedge(q, pols[p].p); double cost = H.cost_usd(h, H.impact); Vec r(n); for (size_t j = 0; j < n; ++j) r[j] = q[j] + h[j]; double risk = std::sqrt(H.risk_usd2(r));
            costs.push_back(cost); cpms.push_back(notional[b] > 0 ? cost / notional[b] * 1e6 : 0); risks.push_back(risk); objs.push_back(cost + H.gamma * H.tau_days * risk * risk); for (size_t j = 0; j < n; ++j) { c.mean_h[j] += h[j] / n_books; c.mean_abs_h += std::fabs(h[j]) / n_books; } }
        CI ci = bootstrap_mean_ci(costs), cp = bootstrap_mean_ci(cpms); c.cost_usd = ci.mean; c.cost_lo = ci.lo; c.cost_hi = ci.hi; c.cost_per_mn = cp.mean; c.cost_per_mn_lo = cp.lo; c.cost_per_mn_hi = cp.hi; c.risk_usd = mean(risks); c.objective = mean(objs);
        cpm_by_pol[p] = cpms; out.push_back(c);
    }
    size_t fs = 1; for (size_t p = 0; p < pols.size(); ++p) { Vec d; for (int b = 0; b < n_books; ++b) d.push_back(100 * (cpm_by_pol[p][b] - cpm_by_pol[fs][b]) / std::max(cpm_by_pol[fs][b], 1e-9)); CI ci = bootstrap_mean_ci(d); out[p].vs_front_pct = ci.mean; out[p].vs_front_lo = ci.lo; out[p].vs_front_hi = ci.hi; }
    return out;
}

} // namespace cetk
