#pragma once
// The client layer of a commodity swap desk, simulated and calibrated to stated targets (no public single-dealer RFQ
// data exists in commodities either).  Three tiers:
//   consumer   corporate hedgers who pay fixed (buy the swap): airlines, utilities, refiners.  Seasonal: intensity
//              1 + A cos(2 pi (t - peak)); concentrated in the front two years; mostly one-way (buys).
//   producer   corporate hedgers who receive fixed (sell the swap): E&Ps, gas producers; further out the curve; one-way (sells).
//   fund       informed two-way flow: direction follows a hidden state that mirrors the hedging-pressure signal of Kang,
//              Rouwenhorst & Tang (2020), and the curve drifts in that direction whether or not the desk trades (the
//              adverse selection of BBGL 2025 is information, not impact); knows the sign of a scheduled-release jump
//              with probability p_info inside the pre-release window.
// Hit probability on a half-spread delta ($/unit): f(delta) = 1 / (1 + exp(alpha + beta delta)), the logistic pricing
// ladder of the FX dealer papers.  Sizes in contracts.
#include "stats.hpp"
#include "json.hpp"
#include <string>
#include <vector>
#include <cmath>

namespace cetk {

struct ClientTier {
    std::string name; double lambda;      // RFQs per hour per ladder tenor (both sides together)
    double alpha, beta;                   // hit probability parameters, beta per $/unit of half-spread
    double p_buy;                         // probability the request is a client buy (dealer sells the forward)
    double mu;                            // expected move against the dealer over the holding period when this tier trades, $/unit (the widening)
    double p_info;                        // pre-release: probability the client knows the jump sign
    double season_amp, season_peak;       // intensity seasonality amplitude and peak (fraction of year)
    Vec tenor_w;                          // relative intensity by ladder tenor
    double f(double d) const { return 1.0 / (1.0 + std::exp(alpha + beta * d)); }
    double d_riskless() const { double best = 0, bv = -1; for (int i = 1; i <= 400; ++i) { double d = 0.005 * i, v = d * f(d); if (v > bv) { bv = v; best = d; } } return best; }
    void exp_fit(double& A, double& k) const { double d0 = d_riskless(); k = beta * (1 - f(d0)); A = f(d0) * std::exp(k * d0); }   // A e^{-k d} tangent at the riskless optimum
    double intensity(double t_year, size_t tenor) const { double s = 1 + season_amp * std::cos(2 * 3.141592653589793 * (t_year - season_peak)); return lambda * std::max(0.0, s) * (tenor < tenor_w.size() ? tenor_w[tenor] : 1.0); }
};

struct LadderSpec {
    std::vector<int> months = {1, 2, 3, 6, 12};   // ladder tenors (contract months ahead)
    std::vector<double> tenor_years;                  // filled from the calendar
    double contract_size = 1000;                      // units per contract (bbl for CL, MMBtu for NG)
    std::vector<double> sizes = {10, 25, 50};         // RFQ sizes, contracts
    std::vector<double> size_w = {0.5, 0.35, 0.15};
    double price_level = 70;                          // $/unit, for notional conversions
    std::string root = "CL";
};

inline std::vector<ClientTier> default_tiers(const std::string& root) {
    // seasonal peaks: consumers hedge into the driving season (oil) or ahead of winter (gas); producers after the budget season
    double peak = root == "NG" ? 0.6 : 0.2;
    std::vector<ClientTier> t;
    t.push_back({"consumer", 0.6, -1.5, 5.0, 0.80, 0.00, 0.00, 0.5, peak, {1.0, 1.0, 1.0, 0.8, 0.6, 0.3}});
    t.push_back({"producer", 0.3, -1.3, 4.0, 0.20, 0.00, 0.00, 0.3, peak + 0.5, {0.3, 0.4, 0.6, 0.8, 1.0, 1.0}});
    t.push_back({"fund", 0.5, -1.0, 10.0, 0.50, 0.04, 0.60, 0.0, 0.0, {1.0, 1.0, 0.8, 0.6, 0.4, 0.2}});
    return t;
}

// Hidden flow direction for the informed tier: a two-state chain switching with rate `switch_per_day`
struct FlowState { double dir = 1; double switch_per_day = 0.5; double next_switch = 0;
    void init(Rng& rng, double t) { dir = rng.uniform() < 0.5 ? -1 : 1; next_switch = t + rng.exponential(switch_per_day); }
    void advance(Rng& rng, double t_days) { while (t_days >= next_switch) { dir = -dir; next_switch += rng.exponential(switch_per_day); } } };

} // namespace cetk
