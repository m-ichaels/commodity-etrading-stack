#pragma once
// Factor risk of the futures curve: PCA of daily price changes on a monthly ladder (level / slope / curvature,
// and for gas a seasonal factor), loadings interpolated to any tenor, covariance at the RFQ ladder.
// The ladder panel is read off fitted curves (one CurveModel per day) so days with a gap in the observed
// contracts still contribute; the window is restricted to days whose gap is small (stated in the results).
#include "curve.hpp"

namespace cetk {

struct FactorModel {
    std::vector<int> months;      // ladder months used for the PCA
    Mat cov;                      // daily covariance of price changes ($^2/day)
    Vec evals;                    // descending
    Mat loadings;                 // [factor][tenor], unit norm, sign-fixed: level positive, slope rising with maturity
    int K = 3; size_t n_days = 0; std::string from, to;
    double explained(int k) const { double tot = 0; for (double e : evals) tot += e; double s = 0; for (int i = 0; i < k && i < (int)evals.size(); ++i) s += evals[i]; return tot > 0 ? s / tot : 0; }

    // X: rows = days, columns = ladder prices
    void fit_panel(const std::vector<Vec>& X, const std::vector<int>& ladder, int k = 3) {
        K = k; months = ladder; size_t n = ladder.size(); std::vector<Vec> D;
        for (size_t d = 1; d < X.size(); ++d) { Vec dx(n); bool ok = true; for (size_t j = 0; j < n; ++j) { dx[j] = X[d][j] - X[d - 1][j]; if (std::isnan(dx[j])) ok = false; } if (ok) D.push_back(dx); }
        n_days = D.size(); Vec mu(n, 0.0); for (auto& x : D) for (size_t j = 0; j < n; ++j) mu[j] += x[j] / std::max<size_t>(1, D.size());
        cov = zeros(n, n); for (auto& x : D) for (size_t a = 0; a < n; ++a) for (size_t b = 0; b < n; ++b) cov[a][b] += (x[a] - mu[a]) * (x[b] - mu[b]) / std::max<size_t>(1, D.size() - 1);
        Mat V; eig_sym(cov, evals, V); loadings = zeros(K, n);
        for (int f = 0; f < K; ++f) { for (size_t j = 0; j < n; ++j) loadings[f][j] = V[j][f];
            double s = 0; if (f == 0) for (double v : loadings[f]) s += v; else if (f == 1) s = loadings[f][n - 1] - loadings[f][0]; else s = loadings[f][n / 2] - 0.5 * (loadings[f][0] + loadings[f][n - 1]);
            if (s < 0) for (double& v : loadings[f]) v = -v; }
    }
    // linear interpolation of a factor's loading to a tenor in months
    double loading_at(int f, double m) const { size_t n = months.size(); if (m <= months[0]) return loadings[f][0]; if (m >= months[n - 1]) return loadings[f][n - 1]; size_t i = 1; while (months[i] < m) ++i; double w = (m - months[i - 1]) / (double)(months[i] - months[i - 1]); return loadings[f][i - 1] * (1 - w) + loadings[f][i] * w; }
    // K-factor covariance at arbitrary tenors ($^2/day)
    Mat covariance_at(const std::vector<double>& tenors) const { size_t n = tenors.size(); Mat S = zeros(n, n); for (int f = 0; f < K; ++f) for (size_t a = 0; a < n; ++a) for (size_t b = 0; b < n; ++b) S[a][b] += evals[f] * loading_at(f, tenors[a]) * loading_at(f, tenors[b]); return S; }
    Mat loadings_at(const std::vector<double>& tenors) const { Mat L = zeros(K, tenors.size()); for (int f = 0; f < K; ++f) for (size_t j = 0; j < tenors.size(); ++j) L[f][j] = loading_at(f, tenors[j]); return L; }
    Json to_json() const { Json j = Json::object(); j["from"] = from; j["to"] = to; j["n_days"] = (long long)n_days; j["months"] = Json(months); j["evals_usd2_per_day"] = Json(evals); Json L = Json::array(); for (auto& l : loadings) L.push(Json(l)); j["loadings"] = L; Json ex = Json::array(); for (int k = 1; k <= K; ++k) ex.push(explained(k)); j["explained"] = ex; Json c = Json::array(); for (auto& r : cov) c.push(Json(r)); j["cov"] = c; return j; }
};

// Ladder panel from fitted curves: for each day fit `spec` and read the model at the ladder months' maturities.
// Days with more than `max_gap` missing months inside the observed range are skipped.
inline std::vector<Vec> ladder_panel(const std::vector<CurveDay>& days, const CurveSpec& spec, const std::vector<int>& ladder, const std::string& root, int max_gap, std::vector<std::string>* dates = nullptr) {
    std::vector<Vec> X;
    for (auto& d : days) {
        if (d.gap_months > max_gap || d.pts.size() < 6) continue;
        // a ladder month beyond the last observed contract is extrapolation: skip the day
        if (ladder.back() > d.pts.back().months) continue;
        CurveModel m; m.fit(d.pts, spec); Vec row(ladder.size());
        for (size_t j = 0; j < ladder.size(); ++j) { auto del = nth_delivery(root, d.ymd, ladder[j]); const CurvePoint* obs = d.at_months(ladder[j]); row[j] = obs ? obs->price : m.predict(years_to_expiry(root, d.ymd, del.first, del.second), season_phase(del.second)); }
        X.push_back(row); if (dates) dates->push_back(d.date);
    }
    return X;
}

// Panel of the observed contracts 1..4 from the EIA front-contract series (long history, no fitting)
inline std::vector<Vec> front_panel(const std::vector<FrontRow>& rows, const std::string& from, const std::string& to) { std::vector<Vec> X; for (auto& r : rows) { if (r.date < from || r.date > to) continue; X.push_back({r.c[0], r.c[1], r.c[2], r.c[3]}); } return X; }

} // namespace cetk
