#pragma once
// RFQ quoting on a ladder of swap tenors against the fitted curve as fair value.
//
// Units: prices $/unit (bbl or MMBtu); inventory q_j in contracts, contract_size units each; risk gamma q'Sigma q per
// day with Sigma in ($/unit)^2/day on the ladder and q in units.  Half-spreads delta in $/unit per RFQ.
//
//   Symmetric   delta_i solving f_i(delta) = h*, no skew
//   BEGV        Bergault-Evangelista-Gueant-Vieira (2021) closed-form multi-asset quotes with the factor reduction:
//               Gamma D Gamma = (gamma/2) Sigma, D_j = A_j zbar k_j / e, delta0_j = 1/k_j + Gamma_jj zbar / 2, skew_j = (Gamma q)_j
//   BEGV_adv    + the adverse-selection widening of Barzykin, Bergault, Gueant & Lemmel (2025): tier i is widened by
//               mu_i, its expected move against the dealer over the holding period, so informed flow pays for the drift
//               it trades ahead of
//   +calendar   inside the pre-release window of a scheduled EIA report the informed tier is widened (calendar feature)
//   Oracle      BEGV_adv plus each tier's true expected adverse selection including p_info E|jump| before releases:
//               the closed form given the true client model, the benchmark every inferring quoter is measured against
#include "factors.hpp"
#include "clients.hpp"

namespace cetk {

struct LadderRisk {
    Mat Sigma;   // [n][n] ($/unit)^2 per day at the ladder tenors (K-factor)
    Mat L;       // [K][n] loadings
    Vec lam;     // K eigenvalues
    void build(const FactorModel& fm, const LadderSpec& ls) { std::vector<double> m(ls.months.begin(), ls.months.end()); Sigma = fm.covariance_at(m); L = fm.loadings_at(m); lam.assign(fm.evals.begin(), fm.evals.begin() + fm.K); }
    double variance_units(const Vec& q_units) const { return quad(q_units, Sigma); }   // $^2/day
    Vec factor_exposure(const Vec& q_units) const { Vec e(L.size(), 0.0); for (size_t k = 0; k < L.size(); ++k) for (size_t j = 0; j < q_units.size(); ++j) e[k] += L[k][j] * q_units[j]; return e; }
};

struct QuoteParams {
    double gamma = 5e-7;          // risk aversion, 1/$
    double hit_target = 0.40;
    double event_widen = 0.05;    // $/unit extra for the informed tier inside the pre-release window (others a quarter)
    double day_hours = 8.0;       // quoting day
    bool calendar = true;
};

struct Fill { double t; int tier, tenor, side; double z, delta, mid, mid_info = 0; Vec markouts, markouts_info; bool pre_event = false; double tox = 0; bool externalised = false; };

class Quoter {
public:
    virtual ~Quoter() = default;
    virtual std::string name() const = 0;
    // half-spread ($/unit) for an RFQ: side +1 client buys (dealer sells the forward), -1 client sells; q in contracts
    virtual double quote(int tier, int tenor, int side, double z, const Vec& q, bool pre_event, double t_year) = 0;
    virtual void on_markout(const Fill&) {}
};

class SymmetricQuoter : public Quoter {
public:
    SymmetricQuoter(const std::vector<ClientTier>& tiers, QuoteParams p) { for (auto& t : tiers) { double lo = 0, hi = 5; for (int i = 0; i < 100; ++i) { double m = 0.5 * (lo + hi); (t.f(m) > p.hit_target ? lo : hi) = m; } d0_.push_back(0.5 * (lo + hi)); } }
    std::string name() const override { return "Symmetric"; }
    double quote(int tier, int, int, double, const Vec&, bool, double) override { return d0_[tier]; }
private: Vec d0_;
};

class BegvQuoter : public Quoter {
public:
    BegvQuoter(const std::vector<ClientTier>& tiers, const LadderSpec& ls, const LadderRisk& risk, QuoteParams p, bool adverse = false, bool oracle = false, double event_jump_sigma = 2.0)
        : ls_(ls), p_(p), adverse_(adverse), oracle_(oracle), tiers_(tiers) {
        size_t n = ls.months.size(); double cs = ls.contract_size;
        double zbar = 0; for (size_t s = 0; s < ls.sizes.size(); ++s) zbar += ls.sizes[s] * ls.size_w[s]; zbar_ = zbar * cs;   // units
        for (size_t i = 0; i < tiers.size(); ++i) { double A, k; tiers[i].exp_fit(A, k); A_.push_back(A); k_.push_back(k); }
        // per-tenor total intensity per side per day and the flow-weighted k
        Vec D(n); for (size_t j = 0; j < n; ++j) { double Asum = 0, ksum = 0; for (size_t i = 0; i < tiers.size(); ++i) { double a = A_[i] * tiers[i].intensity(0.0, j) * p.day_hours * 0.5; Asum += a; ksum += k_[i] * a; } double kbar = Asum > 0 ? ksum / Asum : 10; D[j] = std::max(Asum, 1e-6) * zbar_ * kbar * std::exp(-1.0); }
        Mat M = zeros(n, n); for (size_t a = 0; a < n; ++a) for (size_t b = 0; b < n; ++b) M[a][b] = std::sqrt(D[a]) * 0.5 * p.gamma * risk.Sigma[a][b] * std::sqrt(D[b]);
        Mat R = sym_pow(M, 0.5); Gamma_ = zeros(n, n); for (size_t a = 0; a < n; ++a) for (size_t b = 0; b < n; ++b) Gamma_[a][b] = R[a][b] / std::sqrt(D[a] * D[b]);
        for (size_t j = 0; j < n; ++j) jump_abs_.push_back(event_jump_sigma * std::sqrt(std::max(0.0, risk.Sigma[j][j])) * std::sqrt(2.0 / 3.141592653589793));
    }
    std::string name() const override { return oracle_ ? "Oracle" : adverse_ ? (p_.calendar ? "BEGV_adv_cal" : "BEGV_adv") : "BEGV"; }
    double skew(int tenor, const Vec& q) const { double s = 0; for (size_t b = 0; b < q.size(); ++b) s += Gamma_[tenor][b] * q[b] * ls_.contract_size; return s; }   // $/unit; > 0 when long => sell cheaper, buy dearer
    double quote(int tier, int tenor, int side, double z, const Vec& q, bool pre_event, double) override {
        double d0 = 1.0 / k_[tier] + Gamma_[tenor][tenor] * z * ls_.contract_size / 2;
        double d = d0 - side * skew(tenor, q);
        if (adverse_ || oracle_) d += tiers_[tier].mu;   // BBGL: the informed tier pays its expected drift over the holding period
        if (oracle_ && pre_event) d += tiers_[tier].p_info * jump_abs_[tenor];
        if (adverse_ && !oracle_ && p_.calendar && pre_event) d += p_.event_widen * (tiers_[tier].p_info > 0 ? 1.0 : 0.25);
        return std::max(0.0, d);
    }
    const Mat& Gamma() const { return Gamma_; }
protected:
    LadderSpec ls_; QuoteParams p_; bool adverse_, oracle_; std::vector<ClientTier> tiers_; Vec A_, k_, jump_abs_; Mat Gamma_; double zbar_ = 1;
};

} // namespace cetk
