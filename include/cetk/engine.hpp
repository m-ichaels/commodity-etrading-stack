#pragma once
// The RFQ engine: the state a commodity swap desk's e-trading stack keeps and the two things a client can do to it,
// ask for a price and hit it.  The same object is driven in-process by the simulator and over TCP by the FIX gateway,
// which is what the identical-fills self-test checks.
//
// State: ladder mids F_j (fair value from the curve), inventory q_j (contracts), the transient-impact overlay of the
// desk's own hedges, scheduled adverse-selection drifts, scheduled release jumps, cash, and the P&L decomposition
//   total = spread + inventory + adverse + hedge_exec + hedge_impact   (exact by construction; asserted in the tests)
//   spread        z * CS * delta per fill
//   inventory     q * CS * dF from the exogenous factor path (and release jumps, and the bridge to a realised close)
//   adverse       q * CS * dF from the informed drift: the hidden direction the informed tier trades ahead of moves the
//                 level of the curve at info_drift_per_day whether or not the desk trades (information, not impact)
//   hedge_exec    - CS * (spread + concave self-impact + cross-impact) paid on the desk's own futures trades
//   hedge_impact  q_after * CS * dF from the desk's own impact (its arrival and its decay)
// Time is seconds inside a quoting day of `day_hours`; the market path is drawn on a one-minute grid up front from the
// factor model so quoters and hedgers are compared on common random numbers.
#include "hedge.hpp"
#include "risk.hpp"
#include "toxicity.hpp"
#include <deque>
#include <functional>

namespace cetk {

struct EventSpec { double t_day; Vec jump; std::string type = "EIA"; };
struct DaySpec { std::string date; double t_year = 0; std::vector<EventSpec> events; double close_change_front = NAN; };   // close_change_front: realised front-month change for the bridge

struct EngineConfig {
    LadderSpec ls; QuoteParams qp; std::vector<ClientTier> tiers; LadderRisk risk; Hedger hedger; RiskLimits limits;
    double hedge_review_min = 30, band_usd = 1e5, event_window_min = 30, release_window_min = 30, jump_instant_frac = 0.5;
    bool tox_route = false; double tox_threshold = 0.6, tox_label_horizon = 1800, tox_label_usd = 0.05;
    bool anticipate = false; double flow_ewma_s = 3600, flow_horizon_mult = 0.5;
    double info_drift_per_day = 0.6;   // $/unit per day on the front, scaled along the curve by the level loadings; sign = hidden direction
    uint64_t market_seed = 1; Vec F0;
    double day_s() const { return qp.day_hours * 3600; }
};

struct HedgeRec { double t; Vec h; double cost_usd; double risk_before, risk_after; std::string why; };

struct PnL { double spread = 0, inventory = 0, adverse = 0, hedge_exec = 0, hedge_impact = 0; double total() const { return spread + inventory + adverse + hedge_exec + hedge_impact; } };

struct EngineStats {
    PnL pnl; double cash = 0, mtm0 = 0; std::vector<double> pnl_5min, risk_path; size_t n_rfq = 0, n_fills = 0, n_rejected = 0, n_externalised = 0;
    Vec rfq_by_tier, hits_by_tier; std::vector<Fill> fills; std::vector<HedgeRec> hedges; double client_contracts = 0, hedged_contracts = 0, notional_traded = 0; double max_abs_q = 0;
    Vec quote_sum, quote_n;   // by tier x {normal, pre_event} x {risk-reducing, risk-adding}
};

class RfqEngine {
public:
    explicit RfqEngine(const EngineConfig& c) : c_(c), n_(c.ls.months.size()), q_(n_, 0.0), F_(c.F0), Finfo_(n_, 0.0), I_(n_, 0.0), rng_(c.market_seed), clf_((int)c.tiers.size()) {
        if (F_.size() != n_) F_.assign(n_, c.ls.price_level); size_t nt = c.tiers.size(); st_.rfq_by_tier.assign(nt, 0); st_.hits_by_tier.assign(nt, 0); st_.quote_sum.assign(nt * 4, 0); st_.quote_n.assign(nt * 4, 0);
        quoter_ = nullptr;
    }
    void set_quoter(Quoter* q) { quoter_ = q; }
    const EngineConfig& config() const { return c_; }
    double t() const { return t_; }
    const Vec& mids() const { return F_; }
    const Vec& inventory() const { return q_; }
    const EngineStats& stats() const { return st_; }
    EngineStats& stats() { return st_; }
    const RiskMonitor& risk_monitor() const { return rm_; }
    const ToxicityClassifier& classifier() const { return clf_; }
    double risk_usd(const Vec& q) const { Vec u(n_); for (size_t j = 0; j < n_; ++j) u[j] = q[j] * c_.ls.contract_size; return std::sqrt(std::max(0.0, c_.risk.variance_units(u))); }

    // ---- day lifecycle -------------------------------------------------------------------------------------------
    void start_day(const DaySpec& day) {
        day_ = day; t_ = 0; ev_i_ = 0; const double DAY = c_.day_s(); M_ = (int)std::ceil(DAY / 60.0); cum_.assign(M_ + 1, Vec(n_, 0.0));
        for (int m = 1; m <= M_; ++m) { cum_[m] = cum_[m - 1]; for (size_t k = 0; k < c_.risk.L.size(); ++k) { double e = std::sqrt(std::max(0.0, c_.risk.lam[k]) / M_) * rng_.normal(); for (size_t j = 0; j < n_; ++j) cum_[m][j] += c_.risk.L[k][j] * e; } }
        F_open_ = F_; next_review_ = c_.hedge_review_min * 60; next_mark_ = 300; rm_.new_day(); if (day_idx_ == 0) { st_.mtm0 = mtm(); last_pnl_ = 0; } ++day_idx_;
        if (day_idx_ == 1) flow_.init(rng_, 0.0);
    }
    void end_day() { advance(c_.day_s()); flush_markouts(); double p = pnl_now(); if (p != last_pnl_) { st_.pnl_5min.push_back(p - last_pnl_); last_pnl_ = p; } }
    const std::vector<EventSpec>& day_events() const { return day_.events; }
    bool pre_event(double t) const { for (auto& e : day_.events) { double te = e.t_day * c_.day_s(); if (te > t && te - t <= c_.event_window_min * 60) return true; } return false; }
    double flow_dir() const { return flow_.dir; }   // hidden informed-flow direction (for the client generator)

    // ---- advance the market to time t (seconds in the day) ----------------------------------------------------------
    void advance(double target) {
        const double DAY = c_.day_s(); target = std::min(target, DAY);
        while (t_ < target) {
            double t_ev = ev_i_ < day_.events.size() ? day_.events[ev_i_].t_day * DAY : 1e18;
            double t_next = std::min({target, next_review_, next_mark_, t_ev, next_markout_due()});
            step_market(t_next - t_); t_ = t_next; process_markouts();
            if (t_ == t_ev) { apply_event(day_.events[ev_i_]); ++ev_i_; }
            if (t_ == next_mark_) { mark(); next_mark_ += 300; }
            if (t_ == next_review_) { review("scheduled"); next_review_ += c_.hedge_review_min * 60; }
        }
    }

    // ---- RFQ: quote and execute ----------------------------------------------------------------------------------
    struct QuoteOut { bool valid = false; double delta = 0, bid = 0, ask = 0, mid = 0; int tier = 0, tenor = 0, side = 0; double z = 0; bool pre = false; double t = 0; };
    QuoteOut quote(int tier, int tenor, int side, double z) {
        QuoteOut o; o.tier = tier; o.tenor = tenor; o.side = side; o.z = z; o.t = t_; o.pre = pre_event(t_); st_.n_rfq++; st_.rfq_by_tier[tier] += 1;
        if (!quoter_ || !rm_.allow(q_, tenor, side, z, t_)) { st_.n_rejected++; return o; }
        double t_year = day_.t_year + t_ / (365.0 * 86400);
        o.delta = quoter_->quote(tier, tenor, side, z, q_eff(), o.pre && c_.qp.calendar, t_year); o.mid = F_[tenor]; o.bid = o.mid - o.delta; o.ask = o.mid + o.delta; o.valid = true;
        { double sq = 0; for (size_t b = 0; b < n_; ++b) sq += c_.risk.Sigma[tenor][b] * q_[b]; int red = (side * sq > 0) ? 0 : 1; size_t cell = tier * 4 + (o.pre ? 2 : 0) + red; st_.quote_sum[cell] += o.delta; st_.quote_n[cell] += 1; }
        return o;
    }
    Fill execute(const QuoteOut& o) {
        Fill f; f.t = t_ + (day_idx_ - 1) * c_.day_s(); f.tier = o.tier; f.tenor = o.tenor; f.side = o.side; f.z = o.z; f.delta = o.delta; f.mid = F_[o.tenor]; f.mid_info = Finfo_[o.tenor]; f.pre_event = o.pre;
        const double CS = c_.ls.contract_size; const ClientTier& tier = c_.tiers[o.tier];
        double px = o.side > 0 ? o.ask : o.bid;                       // client buys at the ask
        st_.cash += o.side * o.z * CS * px; q_[o.tenor] -= o.side * o.z; st_.pnl.spread += o.z * CS * o.delta;
        st_.n_fills++; st_.hits_by_tier[o.tier] += 1; st_.client_contracts += o.z; st_.notional_traded += o.z * CS * f.mid; st_.max_abs_q = std::max(st_.max_abs_q, std::fabs(q_[o.tenor]));
        (void)tier;
        { double at = t_ + (day_idx_ - 1) * c_.day_s(); flow_ewma_ = flow_ewma_ * std::exp(-(at - flow_t_) / c_.flow_ewma_s) + o.side * o.z; flow_t_ = at; }   // EWMA of signed client flow on the absolute clock
        // toxicity score at trade time (labels arrive with the 30-minute mark-out)
        { double skew_signed = 0; if (auto* b = dynamic_cast<BegvQuoter*>(quoter_)) skew_signed = -o.side * b->skew(o.tenor, q_); auto x = clf_.features(o.tier, 0, o.z, skew_signed, 0); f.tox = clf_.score(x); tox_feats_.push_back(x); }
        if (c_.tox_route && f.tox > c_.tox_threshold) { Vec h(n_, 0.0); h[o.tenor] = o.side * o.z; do_hedge(h, "toxic_route"); f.externalised = true; st_.n_externalised++; }
        st_.fills.push_back(f); return f;
    }
    double mtm() const { double m = st_.cash; for (size_t j = 0; j < n_; ++j) m += q_[j] * c_.ls.contract_size * F_[j]; return m; }
    double pnl_now() const { return mtm() - st_.mtm0; }
    double decomposition_residual() const { return pnl_now() - st_.pnl.total(); }
    Json quote_structure() const { Json a = Json::array(); for (size_t i = 0; i < c_.tiers.size(); ++i) for (int c = 0; c < 4; ++c) { Json e = Json::object(); e["tier"] = c_.tiers[i].name; e["pre_event"] = c >= 2; e["risk_adding"] = (c % 2) == 1; e["n"] = (long long)st_.quote_n[i * 4 + c]; e["mean_delta"] = st_.quote_n[i * 4 + c] > 0 ? st_.quote_sum[i * 4 + c] / st_.quote_n[i * 4 + c] : 0.0; a.push(e); } return a; }
    Json markout_table() const {
        Json a = Json::array();
        for (size_t i = 0; i < c_.tiers.size(); ++i) for (int pre = 0; pre < 2; ++pre) { double m[3] = {0, 0, 0}, mi[3] = {0, 0, 0}; size_t cnt = 0; double dsum = 0;
            for (auto& f : st_.fills) if (f.tier == (int)i && (int)f.pre_event == pre && f.markouts.size() == 3) { ++cnt; for (int h = 0; h < 3; ++h) { m[h] += f.markouts[h]; mi[h] += f.markouts_info[h]; } dsum += f.delta; }
            Json e = Json::object(); e["tier"] = c_.tiers[i].name; e["pre_event"] = pre == 1; e["n"] = (long long)cnt; e["mean_delta"] = cnt ? dsum / cnt : 0; e["markout_5m"] = cnt ? m[0] / cnt : 0; e["markout_30m"] = cnt ? m[1] / cnt : 0; e["markout_60m"] = cnt ? m[2] / cnt : 0; e["info_30m"] = cnt ? mi[1] / cnt : 0; a.push(e); }
        return a;
    }

private:
    Vec q_eff() const { if (!c_.anticipate) return q_; Vec q = q_; double decay = std::exp(-std::max(0.0, t_ + (day_idx_ - 1) * c_.day_s() - flow_t_) / c_.flow_ewma_s); for (size_t j = 0; j < n_; ++j) q[j] -= c_.flow_horizon_mult * flow_ewma_ * decay / n_; return q; }
    void step_market(double dt) {
        if (dt <= 0) return; const double CS = c_.ls.contract_size, DAY = c_.day_s();
        int m0 = std::min(M_, (int)(t_ / 60.0)), m1 = std::min(M_, (int)((t_ + dt) / 60.0));
        Vec dexo(n_, 0.0), ddrift(n_, 0.0), dimp(n_, 0.0);
        for (size_t j = 0; j < n_; ++j) dexo[j] = cum_[m1][j] - cum_[m0][j];
        if (!std::isnan(day_.close_change_front)) { double rem = DAY - t_; double target1 = F_open_[0] + day_.close_change_front - jumps_remaining(0); double l1 = c_.risk.L[0][0] != 0 ? c_.risk.L[0][0] : 1; for (size_t j = 0; j < n_; ++j) dexo[j] += (target1 - F_[0]) * dt / std::max(rem, dt) * (c_.risk.L[0][j] / l1); }
        { double l1 = c_.risk.L[0][0] != 0 ? c_.risk.L[0][0] : 1; for (size_t j = 0; j < n_; ++j) ddrift[j] = c_.info_drift_per_day * flow_.dir * dt / DAY * (c_.risk.L[0][j] / l1); }   // the informed direction moves the level
        for (auto& d : event_drifts_) { double a = std::max(0.0, std::min(dt, d.t_end - t_)); dexo[d.tenor] += d.rate * a; Finfo_[d.tenor] += d.rate * a; }
        event_drifts_.erase(std::remove_if(event_drifts_.begin(), event_drifts_.end(), [&](const Drift& d) { return d.t_end <= t_ + dt; }), event_drifts_.end());
        // transient impact decays toward zero (the permanent part was booked separately)
        double kappa = -std::log(std::max(1e-9, 1 - c_.hedger.impact.decay_per_window)) / 3600.0;   // per second, one window = one hour
        for (size_t j = 0; j < n_; ++j) { double dI = I_[j] * (std::exp(-kappa * dt) - 1); I_[j] += dI; dimp[j] += dI; }
        for (size_t j = 0; j < n_; ++j) { F_[j] += dexo[j] + ddrift[j] + dimp[j]; Finfo_[j] += ddrift[j]; st_.pnl.inventory += q_[j] * CS * dexo[j]; st_.pnl.adverse += q_[j] * CS * ddrift[j]; st_.pnl.hedge_impact += q_[j] * CS * dimp[j]; }
        double tdays = (day_idx_ - 1) + (t_ + dt) / DAY; flow_.advance(rng_, tdays);
    }
    double jumps_remaining(size_t j) const { double s = 0; for (size_t e = ev_i_; e < day_.events.size(); ++e) s += day_.events[e].jump[j]; return s; }
    void apply_event(const EventSpec& e) { const double CS = c_.ls.contract_size; double W = c_.release_window_min * 60; for (size_t j = 0; j < n_; ++j) { double J = e.jump[j]; double inst = c_.jump_instant_frac * J; F_[j] += inst; Finfo_[j] += inst; st_.pnl.inventory += q_[j] * CS * inst; if (W > 0 && c_.jump_instant_frac < 1) event_drifts_.push_back({t_ + W, (1 - c_.jump_instant_frac) * J / W, (int)j}); } }
    void mark() {
        double p = pnl_now(); st_.pnl_5min.push_back(p - last_pnl_); last_pnl_ = p; double r = risk_usd(q_); st_.risk_path.push_back(r);
        bool flatten = false; rm_.check(t_, r, p, max_abs_q(), flatten); if (flatten) review("risk_limit", true);
    }
    double max_abs_q() const { double m = 0; for (double x : q_) m = std::max(m, std::fabs(x)); return m; }
    // hedge review: nothing inside the internalisation band; outside it the policy's hedge, and if that still leaves the
    // book outside the band (an impact-aware hedge at a low risk aversion), the cheapest hedge that brings it back inside
    void review(const std::string& why, bool force = false) {
        if (c_.hedger.policy == HedgePolicy::None && !force) return; Vec qe = q_eff(); double r = risk_usd(qe); if (!force && r <= c_.band_usd) return;
        Vec h = force ? c_.hedger.hedge(qe, HedgePolicy::Strip) : c_.hedger.hedge(qe);
        if (!force && c_.band_usd > 0) {
            Vec after(n_); for (size_t j = 0; j < n_; ++j) after[j] = qe[j] + h[j];
            if (risk_usd(after) > c_.band_usd) h = c_.hedger.hedge_to_risk(qe, c_.hedger.policy == HedgePolicy::FrontStack || c_.hedger.policy == HedgePolicy::Strip || c_.hedger.policy == HedgePolicy::FactorMin ? HedgePolicy::ImpactAware : c_.hedger.policy, c_.band_usd);
            else {   // a full hedge overshoots the band: trade only the fraction that brings the book back inside it (every policy ends a review at the same risk)
                double lo = 0, hi = 1; for (int it = 0; it < 40; ++it) { double m = 0.5 * (lo + hi); Vec qm(n_); for (size_t j = 0; j < n_; ++j) qm[j] = qe[j] + m * h[j]; if (risk_usd(qm) > c_.band_usd) lo = m; else hi = m; }
                for (double& x : h) x *= hi;
            }
        }
        bool any = false; for (double x : h) if (std::fabs(x) >= 0.5) any = true; if (!any) return;
        for (double& x : h) x = std::round(x); do_hedge(h, why);
    }
    void do_hedge(const Vec& h, const std::string& why) {
        const double CS = c_.ls.contract_size; HedgeRec rec; rec.t = t_ + (day_idx_ - 1) * c_.day_s(); rec.h = h; rec.why = why; rec.risk_before = risk_usd(q_);
        double cost = c_.hedger.impact.exec_cost(h) * CS; st_.pnl.hedge_exec -= cost;
        // cash: pay pre-trade mids plus the concession; inventory: the hedged contracts
        double paid = 0; for (size_t j = 0; j < n_; ++j) { paid += h[j] * CS * F_[j]; q_[j] += h[j]; } st_.cash -= paid + cost;
        // the desk's own impact moves the mids: permanent part stays, the transient part enters the decaying overlay
        Vec dp = c_.hedger.impact.price_moves(h); for (size_t j = 0; j < n_; ++j) { F_[j] += dp[j]; I_[j] += (1 - c_.hedger.impact.perm_frac) * dp[j]; st_.pnl.hedge_impact += q_[j] * CS * dp[j]; st_.hedged_contracts += std::fabs(h[j]); }
        rec.cost_usd = cost; rec.risk_after = risk_usd(q_); st_.hedges.push_back(rec);
    }
    double next_markout_due() const { const double H[3] = {300, 1800, 3600}; double best = 1e18; for (size_t k = cursor_; k < st_.fills.size(); ++k) { const Fill& f = st_.fills[k]; if (f.markouts.size() < 3) best = std::min(best, f.t - (day_idx_ - 1) * c_.day_s() + H[f.markouts.size()]); } return best; }
    void process_markouts(bool flush = false) {
        const double H[3] = {300, 1800, 3600}; double now = t_ + (day_idx_ - 1) * c_.day_s();
        bool advance_cursor = true;
        for (size_t k = cursor_; k < st_.fills.size(); ++k) {
            Fill& f = st_.fills[k]; size_t h = f.markouts.size();
            while (h < 3 && (now >= f.t + H[h] - 1e-9 || flush)) { f.markouts.push_back(f.side * (F_[f.tenor] - f.mid)); f.markouts_info.push_back(f.side * (Finfo_[f.tenor] - f.mid_info)); ++h;
                if (h == 2) { double y = f.markouts[1] > c_.tox_label_usd ? 1.0 : 0.0; clf_.update(tox_feats_[k], y); clf_.observe_markout(f.tier, f.markouts[1]); if (quoter_) quoter_->on_markout(f); } }
            if (h == 3 && advance_cursor) cursor_ = k + 1; else advance_cursor = false;   // cursor = first fill with an outstanding horizon; later fills are still visited
            if (h < 3 && f.t + H[h] > now + 3600 * 2) break;   // beyond the longest horizon nothing later can be due
        }
    }
    void flush_markouts() { process_markouts(true); cursor_ = st_.fills.size(); }

    EngineConfig c_; size_t n_; Vec q_, F_, Finfo_, I_, F_open_; Rng rng_; ToxicityClassifier clf_; Quoter* quoter_;
    DaySpec day_; int day_idx_ = 0, M_ = 0; std::vector<Vec> cum_; double t_ = 0, next_review_ = 0, next_mark_ = 0, last_pnl_ = 0; size_t ev_i_ = 0, cursor_ = 0;
    struct Drift { double t_end, rate; int tenor; }; std::vector<Drift> event_drifts_;
    EngineStats st_; RiskMonitor rm_; FlowState flow_; double flow_ewma_ = 0, flow_t_ = 0; std::vector<std::vector<double>> tox_feats_;
public:
    void set_limits(const RiskLimits& l) { rm_.lim = l; }
};

// ---- client flow: a scripted day of RFQs, drawn from the tiers with their seasonality, one-way biases and informed
// pre-release trading; the script is what the in-process simulator and the FIX client both replay.
struct Rfq { double t; int tier, tenor, side; double z, u; };
inline std::vector<Rfq> generate_flow(const EngineConfig& c, const DaySpec& day, double flow_dir, Rng& rng) {
    std::vector<Rfq> out; const double DAY = c.day_s(); size_t nt = c.tiers.size(), n = c.ls.months.size();
    double lam_max = 0; for (size_t i = 0; i < nt; ++i) for (size_t j = 0; j < n; ++j) lam_max += c.tiers[i].lambda * (1 + c.tiers[i].season_amp) * (j < c.tiers[i].tenor_w.size() ? c.tiers[i].tenor_w[j] : 1.0) / 3600.0 * 3.0;
    double t = 0;
    while (true) {
        t += rng.exponential(lam_max); if (t >= DAY) break;
        const EventSpec* ev = nullptr; for (auto& e : day.events) { double te = e.t_day * DAY; if (te > t && te - t <= c.event_window_min * 60) { ev = &e; break; } }
        std::vector<double> w; w.reserve(nt * n); double tot = 0; double t_year = day.t_year + t / (365.0 * 86400);
        for (size_t i = 0; i < nt; ++i) for (size_t j = 0; j < n; ++j) { double l = c.tiers[i].intensity(t_year, j) / 3600.0; if (ev && c.tiers[i].p_info > 0) l *= 3.0; w.push_back(l); tot += l; }
        if (rng.uniform() > tot / lam_max) continue;   // thinning
        size_t k = rng.weighted(w); size_t i = k / n, j = k % n; const ClientTier& tier = c.tiers[i];
        int side; double pb = tier.p_buy;
        if (tier.p_info > 0) { pb = 0.5 + 0.3 * flow_dir; if (ev) { double jump_sign = ev->jump[j] > 0 ? 1 : -1; if (rng.uniform() < tier.p_info) pb = jump_sign > 0 ? 1.0 : 0.0; } }   // informed: buy ahead of an upward jump
        side = rng.uniform() < pb ? +1 : -1;
        size_t si = rng.weighted(c.ls.size_w); out.push_back({t, (int)i, (int)j, side, c.ls.sizes[si], rng.uniform()});
    }
    return out;
}

// synthetic days: releases on release_days (every 5th day by default, at t_day) with jumps from the factor model
inline std::vector<DaySpec> synthetic_days(int n_days, const LadderRisk& risk, Rng& rng, double p_event = 0.2, double jump_sigma_mult = 2.0, double t_event = 1.5 / 8, double t_year0 = 0.7) {
    std::vector<DaySpec> days; size_t n = risk.Sigma.size();
    for (int d = 0; d < n_days; ++d) { DaySpec day; day.date = "synthetic-" + std::to_string(d); day.t_year = t_year0 + d / 252.0;
        if (rng.uniform() < p_event) { EventSpec e; e.t_day = t_event; e.jump.assign(n, 0.0); for (size_t k = 0; k < risk.L.size(); ++k) { double f = jump_sigma_mult * std::sqrt(std::max(0.0, risk.lam[k])) * rng.normal(); for (size_t j = 0; j < n; ++j) e.jump[j] += risk.L[k][j] * f; } day.events.push_back(e); }
        days.push_back(day); }
    return days;
}

// ---- in-process run --------------------------------------------------------------------------------------------
struct RunResult {
    std::string quoter, hedger; PnL pnl; double total = 0, residual = 0, pnl_5min_std = 0, risk_usd_mean = 0, internalisation = 0, hedge_cost_per_mn = 0, max_abs_q = 0;
    size_t n_rfq = 0, n_fills = 0, n_rejected = 0, n_externalised = 0, n_hedges = 0; Vec hits_by_tier, rfq_by_tier; Json markouts, quote_structure, risk; Vec pnl_5min;
    size_t n_days = 1;
    // daily P&L per unit of 5-minute P&L variance: the risk-adjusted yardstick, independent of the horizon
    double pnl_per_var() const { return pnl_5min_std > 0 ? total / (double)std::max<size_t>(1, n_days) / (pnl_5min_std * pnl_5min_std) : 0; }
    Json to_json() const { Json j = Json::object(); j["quoter"] = quoter; j["hedger"] = hedger; j["pnl"] = total; Json d = Json::object(); d["spread"] = pnl.spread; d["inventory"] = pnl.inventory; d["adverse"] = pnl.adverse; d["hedge_exec"] = pnl.hedge_exec; d["hedge_impact"] = pnl.hedge_impact; d["residual"] = residual; j["decomposition"] = d; j["pnl_5min_std"] = pnl_5min_std; j["pnl_per_var"] = pnl_per_var(); j["risk_usd_mean"] = risk_usd_mean; j["internalisation_ratio"] = internalisation; j["hedge_cost_per_mn_notional"] = hedge_cost_per_mn; j["max_abs_q"] = max_abs_q; j["n_rfq"] = (long long)n_rfq; j["n_fills"] = (long long)n_fills; j["n_rejected"] = (long long)n_rejected; j["n_externalised"] = (long long)n_externalised; j["n_hedges"] = (long long)n_hedges; Json h = Json::array(); for (size_t i = 0; i < hits_by_tier.size(); ++i) h.push(rfq_by_tier[i] > 0 ? hits_by_tier[i] / rfq_by_tier[i] : 0.0); j["hit_ratio_by_tier"] = h; j["markouts"] = markouts; j["quote_structure"] = quote_structure; j["risk"] = risk; return j; }
};

inline RunResult run_engine(const EngineConfig& cfg, Quoter& quoter, const std::vector<DaySpec>& days, uint64_t flow_seed) {
    RfqEngine eng(cfg); eng.set_quoter(&quoter); eng.set_limits(cfg.limits); Rng crng(flow_seed);
    for (auto& day : days) {
        eng.start_day(day); auto script = generate_flow(cfg, day, eng.flow_dir(), crng);
        for (auto& r : script) { eng.advance(r.t); auto q = eng.quote(r.tier, r.tenor, r.side, r.z); if (!q.valid) continue; if (r.u < cfg.tiers[r.tier].f(q.delta)) eng.execute(q); }
        eng.end_day();
    }
    const EngineStats& s = eng.stats(); RunResult R; R.quoter = quoter.name(); R.hedger = hedge_name(cfg.hedger.policy); R.pnl = s.pnl; R.total = eng.pnl_now(); R.residual = eng.decomposition_residual(); R.pnl_5min = s.pnl_5min; R.pnl_5min_std = stdev(s.pnl_5min); R.n_days = days.size(); R.risk_usd_mean = mean(s.risk_path);
    R.internalisation = s.client_contracts > 0 ? 1 - std::min(1.0, s.hedged_contracts / s.client_contracts) : 0; R.hedge_cost_per_mn = s.notional_traded > 0 ? -s.pnl.hedge_exec / s.notional_traded * 1e6 : 0; R.max_abs_q = s.max_abs_q;
    R.n_rfq = s.n_rfq; R.n_fills = s.n_fills; R.n_rejected = s.n_rejected; R.n_externalised = s.n_externalised; R.n_hedges = s.hedges.size(); R.hits_by_tier = s.hits_by_tier; R.rfq_by_tier = s.rfq_by_tier; R.markouts = eng.markout_table(); R.quote_structure = eng.quote_structure(); R.risk = eng.risk_monitor().to_json();
    return R;
}

} // namespace cetk
