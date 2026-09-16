#pragma once
// Online toxicity scoring from mark-outs (Oomen 2019 price signatures as the data; Cartea,
// Duran-Martin & Sanchez-Betancourt 2023 Bayesian last-layer update as the learner).
//
// Model: P(toxic | x) = sigmoid(w . x), w ~ N(m, P).  Each labelled trade (label = the mark-out at
// the label horizon exceeded a threshold in the client's favour) updates (m, P) with one extended
// Kalman step - the recursive Bayesian logistic regression used as the last layer in that paper.
// Scoring uses the probit approximation sigmoid(m.x / sqrt(1 + pi/8 x'Px)) so uncertainty widens
// the score toward 1/2 for unfamiliar feature vectors.  Labels arrive with a delay (the mark-out
// horizon), so scores at trade time only use trades whose outcome is already known.
// Features: tier one-hots, log size, the skew shown to the client signed by the client's direction
// (a reader trading with the skew is the signature of price reading), and the tier's EWMA of past
// mark-outs (the price signature).
#include "stats.hpp"
#include "json.hpp"
#include <vector>
#include <cmath>

namespace cetk {

class ToxicityClassifier {
public:
    explicit ToxicityClassifier(int n_tiers, double prior_var = 4.0, double process_noise = 1e-4)
        : T_(n_tiers), d_(n_tiers + 4), m_(d_, 0.0), P_(d_, std::vector<double>(d_, 0.0)), ewma_(n_tiers, 0.0), qnoise_(process_noise) {
        for (int i = 0; i < d_; ++i) P_[i][i] = prior_var;
    }
    std::vector<double> features(int tier, int, double z, double skew_signed, int) const {
        std::vector<double> x(d_, 0.0);
        x[tier] = 1.0; x[T_] = 1.0; x[T_ + 1] = std::log1p(z); x[T_ + 2] = skew_signed; x[T_ + 3] = ewma_[tier];
        return x;
    }
    double score(const std::vector<double>& x) const {
        double mu = 0, var = 0;
        for (int i = 0; i < d_; ++i) { mu += m_[i] * x[i]; for (int j = 0; j < d_; ++j) var += x[i] * P_[i][j] * x[j]; }
        return 1.0 / (1.0 + std::exp(-mu / std::sqrt(1.0 + 3.14159265 / 8.0 * std::max(var, 0.0))));
    }
    void update(const std::vector<double>& x, double y) {
        // EKF step for a Bernoulli observation with logistic link
        for (int i = 0; i < d_; ++i) P_[i][i] += qnoise_;
        double mu = 0; for (int i = 0; i < d_; ++i) mu += m_[i] * x[i];
        double p = 1.0 / (1.0 + std::exp(-mu)), r = std::max(p * (1 - p), 1e-4);
        std::vector<double> Px(d_, 0.0); double s = 0;
        for (int i = 0; i < d_; ++i) { for (int j = 0; j < d_; ++j) Px[i] += P_[i][j] * x[j]; s += x[i] * Px[i]; }
        double S = s + 1.0 / r;                  // innovation variance (observation noise 1/(p(1-p)))
        std::vector<double> K(d_); for (int i = 0; i < d_; ++i) K[i] = Px[i] / S;
        double innov = (y - p) / r;              // pseudo-observation residual
        for (int i = 0; i < d_; ++i) m_[i] += K[i] * innov;
        for (int i = 0; i < d_; ++i) for (int j = 0; j < d_; ++j) P_[i][j] -= K[i] * Px[j];
        ++n_updates_; if (y > 0.5) ++n_pos_;
    }
    void observe_markout(int tier, double m) { ewma_[tier] = 0.9 * ewma_[tier] + 0.1 * m; }
    Json diagnostics() const {
        Json j = Json::object(); j["n_labelled"] = (long long)n_updates_; j["n_toxic_labels"] = (long long)n_pos_;
        Json w = Json::array(); for (double v : m_) w.push(v); j["weights"] = w;
        Json e = Json::array(); for (double v : ewma_) e.push(v); j["tier_markout_ewma"] = e;
        return j;
    }
private:
    int T_, d_; std::vector<double> m_; std::vector<std::vector<double>> P_; std::vector<double> ewma_; double qnoise_;
    size_t n_updates_ = 0, n_pos_ = 0;
};

} // namespace cetk
