#pragma once
// Small dense linear algebra: Gaussian elimination with partial pivoting, weighted ridge least squares,
// Jacobi symmetric eigen-decomposition.  Sizes here are tens, not thousands.
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace cetk {

using Mat = std::vector<std::vector<double>>;
using Vec = std::vector<double>;

inline Mat zeros(size_t n, size_t m) { return Mat(n, Vec(m, 0.0)); }
inline Mat identity(size_t n) { Mat I = zeros(n, n); for (size_t i = 0; i < n; ++i) I[i][i] = 1; return I; }
inline Vec matvec(const Mat& A, const Vec& x) { Vec y(A.size(), 0.0); for (size_t i = 0; i < A.size(); ++i) for (size_t j = 0; j < x.size(); ++j) y[i] += A[i][j] * x[j]; return y; }
inline double dot(const Vec& a, const Vec& b) { double s = 0; for (size_t i = 0; i < a.size() && i < b.size(); ++i) s += a[i] * b[i]; return s; }
inline double quad(const Vec& q, const Mat& S) { double v = 0; for (size_t a = 0; a < q.size(); ++a) for (size_t b = 0; b < q.size(); ++b) v += q[a] * S[a][b] * q[b]; return v; }

// Solve A x = b (A square).  Returns false when singular.
inline bool solve(Mat A, Vec b, Vec& x) {
    size_t n = A.size(); x.assign(n, 0.0);
    for (size_t c = 0; c < n; ++c) {
        size_t piv = c; for (size_t r = c + 1; r < n; ++r) if (std::fabs(A[r][c]) > std::fabs(A[piv][c])) piv = r;
        std::swap(A[c], A[piv]); std::swap(b[c], b[piv]);
        if (std::fabs(A[c][c]) < 1e-300) return false;
        for (size_t r = c + 1; r < n; ++r) { double f = A[r][c] / A[c][c]; if (f == 0) continue; for (size_t j = c; j < n; ++j) A[r][j] -= f * A[c][j]; b[r] -= f * b[c]; }
    }
    for (size_t c = n; c-- > 0;) { double s = b[c]; for (size_t j = c + 1; j < n; ++j) s -= A[c][j] * x[j]; x[c] = s / A[c][c]; }
    return true;
}

// Weighted ridge least squares: minimise sum_i w_i (y_i - G_i . u)^2 + u' P u  (P symmetric PSD, e.g. lambda K or a difference penalty)
inline Vec ridge_solve(const Mat& G, const Vec& y, const Vec& w, const Mat& P) {
    size_t n = G.size(), m = n ? G[0].size() : 0; Mat M = zeros(m, m); Vec v(m, 0.0);
    for (size_t i = 0; i < n; ++i) for (size_t a = 0; a < m; ++a) { double ga = w[i] * G[i][a]; v[a] += ga * y[i]; for (size_t b = 0; b < m; ++b) M[a][b] += ga * G[i][b]; }
    if (!P.empty()) for (size_t a = 0; a < m; ++a) for (size_t b = 0; b < m; ++b) M[a][b] += P[a][b];
    for (size_t a = 0; a < m; ++a) M[a][a] += 1e-12 * (1 + std::fabs(M[a][a]));   // jitter
    Vec u; if (!solve(M, v, u)) throw std::runtime_error("ridge_solve: singular system"); return u;
}

// Jacobi eigen-decomposition of a symmetric matrix: eigenvalues descending, eigenvectors in columns.
inline void eig_sym(Mat A, Vec& evals, Mat& evecs) {
    size_t n = A.size(); evecs = identity(n);
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0; for (size_t i = 0; i < n; ++i) for (size_t j = i + 1; j < n; ++j) off += A[i][j] * A[i][j];
        if (off < 1e-22) break;
        for (size_t p = 0; p < n; ++p) for (size_t q = p + 1; q < n; ++q) {
            if (std::fabs(A[p][q]) < 1e-300) continue;
            double theta = (A[q][q] - A[p][p]) / (2 * A[p][q]); double t = (theta >= 0 ? 1 : -1) / (std::fabs(theta) + std::sqrt(theta * theta + 1)); double c = 1 / std::sqrt(t * t + 1), s = t * c;
            for (size_t k = 0; k < n; ++k) { double akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq; }
            for (size_t k = 0; k < n; ++k) { double apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk; }
            for (size_t k = 0; k < n; ++k) { double vkp = evecs[k][p], vkq = evecs[k][q]; evecs[k][p] = c * vkp - s * vkq; evecs[k][q] = s * vkp + c * vkq; }
        }
    }
    evals.resize(n); for (size_t i = 0; i < n; ++i) evals[i] = A[i][i];
    std::vector<size_t> idx(n); for (size_t i = 0; i < n; ++i) idx[i] = i; std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return evals[a] > evals[b]; });
    Vec ev2(n); Mat V2 = zeros(n, n); for (size_t j = 0; j < n; ++j) { ev2[j] = evals[idx[j]]; for (size_t i = 0; i < n; ++i) V2[i][j] = evecs[i][idx[j]]; }
    evals = ev2; evecs = V2;
}

// Symmetric square root and inverse square root through the eigen-decomposition (PSD input).
inline Mat sym_pow(const Mat& S, double p) { Vec ev; Mat V; eig_sym(S, ev, V); size_t n = S.size(); Mat R = zeros(n, n); for (size_t a = 0; a < n; ++a) for (size_t b = 0; b < n; ++b) { double s = 0; for (size_t k = 0; k < n; ++k) s += V[a][k] * std::pow(std::max(ev[k], 0.0), p) * V[b][k]; R[a][b] = s; } return R; }
inline Mat matmul(const Mat& A, const Mat& B) { size_t n = A.size(), m = B[0].size(), k = B.size(); Mat C = zeros(n, m); for (size_t i = 0; i < n; ++i) for (size_t l = 0; l < k; ++l) { double a = A[i][l]; if (a == 0) continue; for (size_t j = 0; j < m; ++j) C[i][j] += a * B[l][j]; } return C; }

} // namespace cetk
