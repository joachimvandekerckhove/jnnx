#include "sl_math.h"

#include <cmath>
#include <limits>
#include <utility>

namespace jnnx {
namespace sl {

namespace {

constexpr double kJitter = 1e-10;
constexpr double kPi = 3.14159265358979323846;

bool invert_matrix(const std::vector<double>& a, int n, std::vector<double>& inv_out) {
    inv_out.assign(static_cast<size_t>(n * n), 0.0);
    std::vector<double> aug(static_cast<size_t>(n * 2 * n), 0.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            aug[static_cast<size_t>(i * 2 * n + j)] = a[static_cast<size_t>(i * n + j)];
        }
        aug[static_cast<size_t>(i * 2 * n + n + i)] = 1.0;
    }

    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double max_val = std::fabs(aug[static_cast<size_t>(pivot * 2 * n + col)]);
        for (int row = col + 1; row < n; ++row) {
            const double val = std::fabs(aug[static_cast<size_t>(row * 2 * n + col)]);
            if (val > max_val) {
                max_val = val;
                pivot = row;
            }
        }
        if (max_val < 1e-15) {
            return false;
        }
        if (pivot != col) {
            for (int j = 0; j < 2 * n; ++j) {
                std::swap(aug[static_cast<size_t>(pivot * 2 * n + j)],
                          aug[static_cast<size_t>(col * 2 * n + j)]);
            }
        }

        const double diag = aug[static_cast<size_t>(col * 2 * n + col)];
        for (int j = 0; j < 2 * n; ++j) {
            aug[static_cast<size_t>(col * 2 * n + j)] /= diag;
        }
        for (int row = 0; row < n; ++row) {
            if (row == col) {
                continue;
            }
            const double factor = aug[static_cast<size_t>(row * 2 * n + col)];
            for (int j = 0; j < 2 * n; ++j) {
                aug[static_cast<size_t>(row * 2 * n + j)] -=
                    factor * aug[static_cast<size_t>(col * 2 * n + j)];
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            inv_out[static_cast<size_t>(i * n + j)] =
                aug[static_cast<size_t>(i * 2 * n + n + j)];
        }
    }
    return true;
}

bool logdet_and_solve(const std::vector<double>& omega, int p,
                      const std::vector<double>& rhs,
                      double& logdet_out, std::vector<double>& sol_out) {
    std::vector<double> a = omega;
    sol_out = rhs;
    logdet_out = 0.0;

    for (int col = 0; col < p; ++col) {
        int pivot = col;
        double max_val = std::fabs(a[static_cast<size_t>(pivot * p + col)]);
        for (int row = col + 1; row < p; ++row) {
            const double val = std::fabs(a[static_cast<size_t>(row * p + col)]);
            if (val > max_val) {
                max_val = val;
                pivot = row;
            }
        }
        if (max_val < 1e-15) {
            return false;
        }
        if (pivot != col) {
            for (int j = 0; j < p; ++j) {
                std::swap(a[static_cast<size_t>(pivot * p + j)],
                          a[static_cast<size_t>(col * p + j)]);
            }
            std::swap(sol_out[static_cast<size_t>(pivot)],
                      sol_out[static_cast<size_t>(col)]);
            logdet_out -= std::log(max_val);
        } else {
            logdet_out -= std::log(max_val);
        }

        const double diag = a[static_cast<size_t>(col * p + col)];
        logdet_out += std::log(std::fabs(diag));
        for (int row = col + 1; row < p; ++row) {
            const double factor = a[static_cast<size_t>(row * p + col)] / diag;
            for (int j = col; j < p; ++j) {
                a[static_cast<size_t>(row * p + j)] -=
                    factor * a[static_cast<size_t>(col * p + j)];
            }
            sol_out[static_cast<size_t>(row)] -= factor * sol_out[static_cast<size_t>(col)];
        }
    }

    for (int col = p - 1; col >= 0; --col) {
        for (int row = 0; row < col; ++row) {
            sol_out[static_cast<size_t>(row)] -=
                a[static_cast<size_t>(row * p + col)] * sol_out[static_cast<size_t>(col)];
        }
        sol_out[static_cast<size_t>(col)] /=
            a[static_cast<size_t>(col * p + col)];
    }
    return true;
}

void add_jitter(std::vector<double>& mat, int p) {
    for (int i = 0; i < p; ++i) {
        mat[static_cast<size_t>(i * p + i)] += kJitter;
    }
}

}  // namespace

std::vector<std::pair<int, int>> upper_tri_index_pairs(int p) {
    std::vector<std::pair<int, int>> pairs;
    pairs.reserve(static_cast<size_t>(p * (p + 1) / 2));
    for (int i = 0; i < p; ++i) {
        for (int j = i; j < p; ++j) {
            pairs.emplace_back(i, j);
        }
    }
    return pairs;
}

void assemble_L(const std::vector<double>& chol_upper, int p,
                std::vector<double>& L_flat) {
    L_flat.assign(static_cast<size_t>(p * p), 0.0);
    const auto pairs = upper_tri_index_pairs(p);
    for (size_t k = 0; k < pairs.size(); ++k) {
        const int i = pairs[k].first;
        const int j = pairs[k].second;
        L_flat[static_cast<size_t>(i * p + j)] = chol_upper[k];
    }
}

void omega1_from_chol(const std::vector<double>& chol_upper, int p,
                      std::vector<double>& omega1_out) {
    std::vector<double> L_flat;
    assemble_L(chol_upper, p, L_flat);
    omega1_out.assign(static_cast<size_t>(p * p), 0.0);
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j < p; ++j) {
            double sum = 0.0;
            for (int k = 0; k < p; ++k) {
                sum += L_flat[static_cast<size_t>(k * p + i)] *
                       L_flat[static_cast<size_t>(k * p + j)];
            }
            omega1_out[static_cast<size_t>(i * p + j)] = sum;
        }
    }
}

bool omega_total_from_chol(const std::vector<double>& chol_upper, int p,
                           double n_trials,
                           const std::vector<double>& sigma_emu,
                           std::vector<double>& omega_total_out,
                           bool apply_jitter) {
    if (n_trials <= 0.0 || !std::isfinite(n_trials)) {
        return false;
    }

    std::vector<double> omega1;
    omega1_from_chol(chol_upper, p, omega1);

    std::vector<double> omega_sampling(static_cast<size_t>(p * p), 0.0);
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j < p; ++j) {
            omega_sampling[static_cast<size_t>(i * p + j)] =
                n_trials * omega1[static_cast<size_t>(i * p + j)];
        }
    }

    std::vector<double> sigma_sampling;
    if (!invert_matrix(omega_sampling, p, sigma_sampling)) {
        return false;
    }

    std::vector<double> sigma_total(static_cast<size_t>(p * p), 0.0);
    for (int i = 0; i < p * p; ++i) {
        sigma_total[static_cast<size_t>(i)] =
            sigma_sampling[static_cast<size_t>(i)] + sigma_emu[static_cast<size_t>(i)];
    }

    if (apply_jitter) {
        add_jitter(sigma_total, p);
    }

    return invert_matrix(sigma_total, p, omega_total_out);
}

double mvn_logdens_precision(const std::vector<double>& x,
                             const std::vector<double>& mu,
                             const std::vector<double>& omega, int p) {
    std::vector<double> diff(static_cast<size_t>(p), 0.0);
    for (int i = 0; i < p; ++i) {
        diff[static_cast<size_t>(i)] = x[static_cast<size_t>(i)] - mu[static_cast<size_t>(i)];
    }

    std::vector<double> omega_work = omega;
    add_jitter(omega_work, p);

    double logdet = 0.0;
    std::vector<double> sol;
    if (!logdet_and_solve(omega_work, p, diff, logdet, sol)) {
        return -std::numeric_limits<double>::infinity();
    }

    double quad = 0.0;
    for (int i = 0; i < p; ++i) {
        quad += diff[static_cast<size_t>(i)] * sol[static_cast<size_t>(i)];
    }
    return -0.5 * (static_cast<double>(p) * std::log(2.0 * kPi) - logdet + quad);
}

}  // namespace sl
}  // namespace jnnx
