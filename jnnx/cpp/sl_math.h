#ifndef JNNX_SL_MATH_H_
#define JNNX_SL_MATH_H_

#include <functional>
#include <vector>

namespace jnnx {
namespace sl {

/** Nonlinear column transform codes (baked from obs_transform.json). */
enum JnnxTransform : int {
    kIdentity = 0,
    kLog1p = 1,
    kLog = 2,
    kSqrt = 3,
};

/** Row-major upper-triangle index pairs for dimension p. */
std::vector<std::pair<int, int>> upper_tri_index_pairs(int p);

/** Assemble upper-triangular L (row-major p*p) from flat chol_upper. */
void assemble_L(const std::vector<double>& chol_upper, int p,
                std::vector<double>& L_flat);

/** Per-trial precision Omega1 = L^T L (row-major p*p). */
void omega1_from_chol(const std::vector<double>& chol_upper, int p,
                      std::vector<double>& omega1_out);

/**
 * Sigma_sampling = inv(N * Omega1); Sigma_total = Sigma_sampling + sigma_emu;
 * returns Omega_total = inv(Sigma_total) (row-major p*p).
 * Returns false if any matrix operation fails.
 */
bool omega_total_from_chol(const std::vector<double>& chol_upper, int p,
                           double n_trials,
                           const std::vector<double>& sigma_emu,
                           std::vector<double>& omega_total_out,
                           bool apply_jitter = true);

/**
 * log MVN(x; mu, Omega) with precision Omega (JAGS dmnorm form).
 * Returns -Inf if Omega is not positive definite.
 */
double mvn_logdens_precision(const std::vector<double>& x,
                             const std::vector<double>& mu,
                             const std::vector<double>& omega, int p);

/**
 * Draw x ~ MVN(mu, Omega) where Omega is precision (JAGS dmnorm parameterization).
 * normal_draw supplies i.i.d. standard normals. Returns false on PD failure.
 */
bool mvn_sample_precision(const std::vector<double>& mu,
                          const std::vector<double>& omega, int p,
                          const std::function<double()>& normal_draw,
                          std::vector<double>& x_out,
                          bool apply_jitter = true);

/**
 * Forward observation transform: raw physical summaries -> standardized space.
 * transforms[j] is a JnnxTransform code; mean/scale are StandardScaler params.
 */
bool obs_raw_to_std(const double* raw, double* std_out, int p,
                    const int* transforms,
                    const double* mean, const double* scale);

/**
 * Inverse observation transform: standardized space -> raw physical summaries.
 */
bool obs_std_to_raw(const double* std_in, double* raw_out, int p,
                    const int* transforms,
                    const double* mean, const double* scale);

}  // namespace sl
}  // namespace jnnx

#endif  // JNNX_SL_MATH_H_
