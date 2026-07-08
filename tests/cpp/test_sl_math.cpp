#include "sl_math.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr double kTol = 1e-6;

bool approx_equal(double a, double b, double tol = kTol) {
    return std::fabs(a - b) <= tol * (1.0 + std::max(std::fabs(a), std::fabs(b)));
}

std::vector<double> matmul(const std::vector<double>& a, const std::vector<double>& b,
                           int n) {
    std::vector<double> out(static_cast<size_t>(n * n), 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += a[static_cast<size_t>(i * n + k)] *
                       b[static_cast<size_t>(k * n + j)];
            }
            out[static_cast<size_t>(i * n + j)] = sum;
        }
    }
    return out;
}

}  // namespace

int main() {
    using jnnx::sl::assemble_L;
    using jnnx::sl::mvn_logdens_precision;
    using jnnx::sl::omega1_from_chol;
    using jnnx::sl::omega_total_from_chol;
    using jnnx::sl::upper_tri_index_pairs;

    int failures = 0;
    auto check = [&](bool ok, const std::string& msg) {
        if (!ok) {
            std::cerr << "FAIL: " << msg << std::endl;
            ++failures;
        }
    };

    const int p = 3;
    const std::vector<double> chol_upper = {0.8, 0.1, -0.05, 1.2, 0.2, 0.9};
    std::vector<double> omega1;
    omega1_from_chol(chol_upper, p, omega1);

    std::vector<double> L;
    assemble_L(chol_upper, p, L);
    const std::vector<double> LT = [&]() {
        std::vector<double> out(static_cast<size_t>(p * p), 0.0);
        for (int i = 0; i < p; ++i) {
            for (int j = 0; j < p; ++j) {
                out[static_cast<size_t>(i * p + j)] =
                    L[static_cast<size_t>(j * p + i)];
            }
        }
        return out;
    }();
    const std::vector<double> expected = matmul(LT, L, p);
    for (int i = 0; i < p * p; ++i) {
        check(approx_equal(omega1[static_cast<size_t>(i)], expected[static_cast<size_t>(i)]),
              "omega1_from_chol");
    }

    const std::vector<double> sigma_emu = {
        0.012, 0.001, 0.000,
        0.001, 0.015, 0.002,
        0.000, 0.002, 0.018,
    };
    std::vector<double> omega_total;
    check(omega_total_from_chol(chol_upper, p, 600.0, sigma_emu, omega_total),
          "omega_total_from_chol success");

    const std::vector<double> mu = {0.1, -0.2, 0.05};
    const std::vector<double> x = {0.12, -0.18, 0.04};
    const double logdens = mvn_logdens_precision(x, mu, omega_total, p);
    check(std::isfinite(logdens), "logdens finite");

    const auto pairs = upper_tri_index_pairs(p);
    check(pairs.size() == 6U, "upper_tri pairs size");

    if (failures == 0) {
        std::cout << "All sl_math tests passed." << std::endl;
        return 0;
    }
    return 1;
}
