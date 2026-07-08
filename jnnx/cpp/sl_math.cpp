#include "sl_math.h"

#include <cmath>
#include <functional>
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

void add_jitter(std::vector<double>& mat, int p) {
    for (int i = 0; i < p; ++i) {
        mat[static_cast<size_t>(i * p + i)] += kJitter;
    }
}

bool chol_lower_spd(const std::vector<double>& sigma, int p,
                    std::vector<double>& L_out) {
    L_out.assign(static_cast<size_t>(p * p), 0.0);
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k) {
                sum += L_out[static_cast<size_t>(i * p + k)] *
                       L_out[static_cast<size_t>(j * p + k)];
            }
            if (i == j) {
                const double diag = sigma[static_cast<size_t>(i * p + i)] - sum;
                if (diag <= 0.0) {
                    return false;
                }
                L_out[static_cast<size_t>(i * p + j)] = std::sqrt(diag);
            } else {
                const double denom = L_out[static_cast<size_t>(j * p + j)];
                if (std::fabs(denom) < 1e-15) {
                    return false;
                }
                L_out[static_cast<size_t>(i * p + j)] =
                    (sigma[static_cast<size_t>(i * p + j)] - sum) / denom;
            }
        }
    }
    return true;
}

bool chol_logdet_spd(const std::vector<double>& omega, int p, double& logdet_out) {
    std::vector<double> L;
    if (!chol_lower_spd(omega, p, L)) {
        return false;
    }
    logdet_out = 0.0;
    for (int i = 0; i < p; ++i) {
        logdet_out += 2.0 * std::log(L[static_cast<size_t>(i * p + i)]);
    }
    return true;
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
    if (!chol_logdet_spd(omega_work, p, logdet)) {
        return -std::numeric_limits<double>::infinity();
    }

    double quad = 0.0;
    for (int i = 0; i < p; ++i) {
        double row = 0.0;
        for (int j = 0; j < p; ++j) {
            row += omega_work[static_cast<size_t>(i * p + j)] *
                   diff[static_cast<size_t>(j)];
        }
        quad += diff[static_cast<size_t>(i)] * row;
    }
    return -0.5 * (static_cast<double>(p) * std::log(2.0 * kPi) - logdet + quad);
}

bool mvn_sample_precision(const std::vector<double>& mu,
                          const std::vector<double>& omega, int p,
                          const std::function<double()>& normal_draw,
                          std::vector<double>& x_out,
                          bool apply_jitter) {
    if (!normal_draw || static_cast<int>(mu.size()) < p) {
        return false;
    }

    std::vector<double> omega_work = omega;
    if (apply_jitter) {
        add_jitter(omega_work, p);
    }

    std::vector<double> sigma;
    if (!invert_matrix(omega_work, p, sigma)) {
        return false;
    }

    std::vector<double> L;
    if (!chol_lower_spd(sigma, p, L)) {
        return false;
    }

    std::vector<double> z(static_cast<size_t>(p), 0.0);
    for (int i = 0; i < p; ++i) {
        z[static_cast<size_t>(i)] = normal_draw();
    }

    x_out.assign(static_cast<size_t>(p), 0.0);
    for (int i = 0; i < p; ++i) {
        double sum = mu[static_cast<size_t>(i)];
        for (int j = 0; j <= i; ++j) {
            sum += L[static_cast<size_t>(i * p + j)] * z[static_cast<size_t>(j)];
        }
        x_out[static_cast<size_t>(i)] = sum;
    }
    return true;
}

}  // namespace sl
}  // namespace jnnx
