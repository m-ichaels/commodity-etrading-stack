#pragma once
// Pre-trade and continuous risk controls for the RFQ engine: position limits per tenor, a factor-risk limit on the
// book, a P&L drawdown kill switch with a cooldown, and the dashboard rows the monitor writes.
#include "quoter.hpp"

namespace cetk {

struct RiskLimits {
    double max_abs_q = 1000;         // contracts per tenor
    double max_risk_usd = 2.0e6;     // daily P&L standard deviation of the book, $
    double max_drawdown_usd = 1.0e6; // from the day's high-water mark
    double cooldown_min = 30;        // no quoting after a kill for this long, then resume
};

struct RiskRow { double t; double risk_usd, pnl, drawdown; double max_abs_q; bool halted; };

struct RiskMonitor {
    RiskLimits lim; double hwm = 0, halted_until = -1; int n_breach_q = 0, n_breach_risk = 0, n_kills = 0; std::vector<RiskRow> rows;
    // pre-trade: may the dealer take this fill?  side +1 = client buys => dealer's q_j falls by z
    bool allow(const Vec& q, int tenor, int side, double z, double t) const {
        if (t < halted_until) return false;
        double after = q[tenor] - side * z; if (std::fabs(after) > lim.max_abs_q && std::fabs(after) > std::fabs(q[tenor])) return false;   // risk-adding beyond the limit
        return true;
    }
    // continuous: called at every mark; returns true when a kill was triggered now
    bool check(double t, double risk_usd, double pnl, double max_abs_q, bool& need_flatten) {
        hwm = std::max(hwm, pnl); double dd = hwm - pnl; bool halted = t < halted_until; bool kill = false;
        if (!halted && dd > lim.max_drawdown_usd) { halted_until = t + lim.cooldown_min * 60; ++n_kills; kill = true; need_flatten = true; }
        if (risk_usd > lim.max_risk_usd) { ++n_breach_risk; need_flatten = true; }
        if (max_abs_q > lim.max_abs_q) ++n_breach_q;
        rows.push_back({t, risk_usd, pnl, dd, max_abs_q, halted || kill});
        return kill;
    }
    void new_day() { hwm = 0; halted_until = -1; }
    Json to_json() const { Json j = Json::object(); j["n_kills"] = n_kills; j["n_breach_risk"] = n_breach_risk; j["n_breach_q"] = n_breach_q; j["limits"] = Json::object(); j["limits"]["max_abs_q"] = lim.max_abs_q; j["limits"]["max_risk_usd"] = lim.max_risk_usd; j["limits"]["max_drawdown_usd"] = lim.max_drawdown_usd; return j; }
};

} // namespace cetk
