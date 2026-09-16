#pragma once
// Small statistics toolkit: RNG, descriptive stats, KS / Wasserstein-1 distances, ACF,
// paired bootstrap confidence intervals, OLS, logistic regression.
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <cstdint>
#include <limits>

namespace cetk {

// Deterministic 64-bit RNG (SplitMix64 seed -> xoshiro256**). Same seed => same tape, any platform.
class Rng {
public:
    explicit Rng(uint64_t seed = 42) { seed_with(seed); }
    void seed_with(uint64_t seed) {
        uint64_t z = seed;
        for (auto& si : s_) { z += 0x9E3779B97F4A7C15ULL; uint64_t x = z; x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL; x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL; si = x ^ (x >> 31); }
    }
    uint64_t next() {
        const uint64_t result = rotl(s_[1] * 5, 7) * 9;
        const uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0]; s_[3] ^= s_[1]; s_[1] ^= s_[2]; s_[0] ^= s_[3]; s_[2] ^= t; s_[3] = rotl(s_[3], 45);
        return result;
    }
    double uniform() { return (next() >> 11) * 0x1.0p-53; }             // [0,1)
    double uniform_pos() { double u; do { u = uniform(); } while (u <= 0); return u; }
    double exponential(double rate) { return -std::log(uniform_pos()) / rate; }
    double normal() { // Box-Muller
        double u1 = uniform_pos(), u2 = uniform();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }
    size_t index(size_t n) { return (size_t)(uniform() * n) % n; }
    // Sample index from unnormalised weights.
    size_t weighted(const std::vector<double>& w) {
        double tot = 0; for (double x : w) tot += x;
        double u = uniform() * tot, acc = 0;
        for (size_t i = 0; i < w.size(); ++i) { acc += w[i]; if (u < acc) return i; }
        return w.empty() ? 0 : w.size() - 1;
    }
private:
    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    uint64_t s_[4];
};

inline double mean(const std::vector<double>& v) { return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size(); }
inline double variance(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    double m = mean(v), s = 0; for (double x : v) s += (x - m) * (x - m); return s / (v.size() - 1);
}
inline double stdev(const std::vector<double>& v) { return std::sqrt(variance(v)); }
inline double quantile(std::vector<double> v, double q) {
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(v.begin(), v.end());
    double pos = q * (v.size() - 1); size_t lo = (size_t)pos; double fr = pos - lo;
    return lo + 1 < v.size() ? v[lo] * (1 - fr) + v[lo + 1] * fr : v[lo];
}
inline double median(const std::vector<double>& v) { return quantile(v, 0.5); }
inline double kurtosis(const std::vector<double>& v) { // excess kurtosis
    if (v.size() < 4) return 0.0;
    double m = mean(v), s2 = 0, s4 = 0;
    for (double x : v) { double d = x - m; s2 += d * d; s4 += d * d * d * d; }
    s2 /= v.size(); s4 /= v.size();
    return s2 > 0 ? s4 / (s2 * s2) - 3.0 : 0.0;
}
inline double autocorr(const std::vector<double>& v, size_t lag) {
    if (v.size() <= lag + 1) return 0.0;
    double m = mean(v), num = 0, den = 0;
    for (size_t i = 0; i < v.size(); ++i) den += (v[i] - m) * (v[i] - m);
    for (size_t i = lag; i < v.size(); ++i) num += (v[i] - m) * (v[i - lag] - m);
    return den > 0 ? num / den : 0.0;
}

// Two-sample Kolmogorov-Smirnov distance sup|F_a - F_b|.
inline double ks_distance(std::vector<double> a, std::vector<double> b) {
    if (a.empty() || b.empty()) return 1.0;
    std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
    size_t i = 0, j = 0; double d = 0;
    while (i < a.size() && j < b.size()) {
        double x = std::min(a[i], b[j]);
        while (i < a.size() && a[i] <= x) ++i;
        while (j < b.size() && b[j] <= x) ++j;
        d = std::max(d, std::fabs((double)i / a.size() - (double)j / b.size()));
    }
    return d;
}
// 1-Wasserstein distance between empirical distributions (integral of |F_a - F_b|).
inline double wasserstein1(std::vector<double> a, std::vector<double> b) {
    if (a.empty() || b.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
    // evaluate both quantile functions on a common grid of 512 points
    const int G = 512; double w = 0;
    for (int g = 0; g < G; ++g) {
        double q = (g + 0.5) / G;
        double qa = a[std::min(a.size() - 1, (size_t)(q * a.size()))];
        double qb = b[std::min(b.size() - 1, (size_t)(q * b.size()))];
        w += std::fabs(qa - qb);
    }
    return w / G;
}

struct CI { double mean, lo, hi; size_t n; };
// Percentile bootstrap CI for the mean of a sample (used on per-day paired differences).
inline CI bootstrap_mean_ci(const std::vector<double>& x, int B = 2000, double alpha = 0.05, uint64_t seed = 7) {
    CI ci{mean(x), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), x.size()};
    if (x.size() < 2) return ci;
    Rng rng(seed);
    std::vector<double> ms; ms.reserve(B);
    for (int b = 0; b < B; ++b) {
        double s = 0; for (size_t i = 0; i < x.size(); ++i) s += x[rng.index(x.size())];
        ms.push_back(s / x.size());
    }
    ci.lo = quantile(ms, alpha / 2); ci.hi = quantile(ms, 1 - alpha / 2);
    return ci;
}

// Simple OLS y = a + b x. Returns {a, b, r2}.
struct Ols { double a, b, r2; size_t n; };
inline Ols ols(const std::vector<double>& x, const std::vector<double>& y) {
    size_t n = std::min(x.size(), y.size());
    if (n < 3) return {0, 0, 0, n};
    double mx = 0, my = 0; for (size_t i = 0; i < n; ++i) { mx += x[i]; my += y[i]; } mx /= n; my /= n;
    double sxx = 0, sxy = 0, syy = 0;
    for (size_t i = 0; i < n; ++i) { sxx += (x[i] - mx) * (x[i] - mx); sxy += (x[i] - mx) * (y[i] - my); syy += (y[i] - my) * (y[i] - my); }
    double b = sxx > 0 ? sxy / sxx : 0, a = my - b * mx;
    double r2 = (sxx > 0 && syy > 0) ? (sxy * sxy) / (sxx * syy) : 0;
    return {a, b, r2, n};
}

// Logistic regression by Newton-Raphson with ridge penalty. X rows already include a 1 for the intercept.
inline std::vector<double> logistic_fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y,
                                        double ridge = 1e-3, int iters = 30) {
    size_t n = X.size(); if (n == 0) return {};
    size_t k = X[0].size();
    std::vector<double> w(k, 0.0);
    for (int it = 0; it < iters; ++it) {
        std::vector<double> g(k, 0.0); std::vector<std::vector<double>> H(k, std::vector<double>(k, 0.0));
        for (size_t i = 0; i < n; ++i) {
            double z = 0; for (size_t j = 0; j < k; ++j) z += w[j] * X[i][j];
            double p = 1.0 / (1.0 + std::exp(-z));
            double r = y[i] - p, s = p * (1 - p);
            for (size_t j = 0; j < k; ++j) { g[j] += r * X[i][j]; for (size_t l = 0; l < k; ++l) H[j][l] += s * X[i][j] * X[i][l]; }
        }
        for (size_t j = 0; j < k; ++j) { g[j] -= ridge * w[j]; H[j][j] += ridge; }
        // solve H d = g (Gaussian elimination)
        std::vector<double> d(k, 0.0);
        std::vector<std::vector<double>> A = H; std::vector<double> bvec = g;
        for (size_t c = 0; c < k; ++c) {
            size_t piv = c; for (size_t r = c + 1; r < k; ++r) if (std::fabs(A[r][c]) > std::fabs(A[piv][c])) piv = r;
            std::swap(A[c], A[piv]); std::swap(bvec[c], bvec[piv]);
            if (std::fabs(A[c][c]) < 1e-12) continue;
            for (size_t r = c + 1; r < k; ++r) { double f = A[r][c] / A[c][c]; for (size_t l = c; l < k; ++l) A[r][l] -= f * A[c][l]; bvec[r] -= f * bvec[c]; }
        }
        for (size_t c = k; c-- > 0;) { double s = bvec[c]; for (size_t l = c + 1; l < k; ++l) s -= A[c][l] * d[l]; d[c] = std::fabs(A[c][c]) < 1e-12 ? 0 : s / A[c][c]; }
        double step = 0; for (size_t j = 0; j < k; ++j) { w[j] += d[j]; step += d[j] * d[j]; }
        if (step < 1e-10) break;
    }
    return w;
}
inline double logistic_predict(const std::vector<double>& w, const std::vector<double>& x) {
    double z = 0; for (size_t j = 0; j < w.size() && j < x.size(); ++j) z += w[j] * x[j];
    return 1.0 / (1.0 + std::exp(-z));
}

} // namespace cetk
