#include "elastic_energy.h"
#include <Eigen/LU>

namespace vbd {

// ============================================================================
// 变形梯度 F = Ds · Dm^{-1}
// ============================================================================
Eigen::Matrix3d compute_deformation_gradient(
    const Eigen::Ref<const Eigen::Matrix<double, 4, 3>>& vertices,
    const Eigen::Ref<const Eigen::Matrix3d>& dm_inv)
{
    Eigen::Matrix3d Ds;
    for (int c = 0; c < 3; ++c)
        Ds.col(c) = vertices.row(c + 1).transpose() - vertices.row(0).transpose();
    return Ds * dm_inv;
}

// ============================================================================
// Neo-Hookean 能量密度
// ============================================================================
double neo_hookean_energy_density(
    const Eigen::Ref<const Eigen::Matrix3d>& F,
    double mu, double lam, double inverted_penalty)
{
    double I_c = (F.transpose() * F).trace();
    double J = F.determinant();

    if (J <= 1e-12) {
        double d = 1.0 - J;
        return inverted_penalty * d * d;
    }

    double log_J = std::log(J);
    return 0.5 * mu * (I_c - 3.0) - mu * log_J + 0.5 * lam * log_J * log_J;
}

// ============================================================================
// 第一 Piola-Kirchhoff 应力 P
// ============================================================================
Eigen::Matrix3d neo_hookean_pk1_stress(
    const Eigen::Ref<const Eigen::Matrix3d>& F,
    double mu, double lam, double inverted_penalty)
{
    double J = F.determinant();

    if (J <= 1e-12) {
        Eigen::Matrix3d FinvT = F.inverse().transpose();
        double penalty_factor = -2.0 * inverted_penalty * (1.0 - J) * J;
        return penalty_factor * FinvT;
    }

    Eigen::Matrix3d FinvT = F.inverse().transpose();
    double log_J = std::log(J);
    return mu * F + (lam * log_J - mu) * FinvT;
}

// ============================================================================
// 材料切线模量 9×9
// ============================================================================
Eigen::Matrix<double, 9, 9> neo_hookean_material_tangent_9x9(
    const Eigen::Ref<const Eigen::Matrix3d>& F,
    double mu, double lam, double inverted_penalty)
{
    double J = F.determinant();
    if (J <= 1e-12) {
        return inverted_penalty * Eigen::Matrix<double, 9, 9>::Identity();
    }

    Eigen::Matrix3d FinvT = F.inverse().transpose();
    double log_J = std::log(J);
    double coeff = mu - lam * log_J;

    Eigen::Matrix<double, 9, 9> C = Eigen::Matrix<double, 9, 9>::Zero();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            int row = i * 3 + j;
            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
                    int col = k * 3 + l;
                    double val = 0.0;

                    // μ · δ_{ik} · δ_{jl}
                    if (i == k && j == l) val += mu;

                    // (λ ln(J) - μ) · F^{-T}_{il} · F^{-T}_{kj}
                    val += coeff * FinvT(i, l) * FinvT(k, j);

                    // λ · F^{-T}_{ij} · F^{-T}_{lk}
                    val += lam * FinvT(i, j) * FinvT(l, k);

                    C(row, col) = val;
                }
            }
        }
    }
    return C;
}

// ============================================================================
// 四面体力 + Hessian 对角块组装
// ============================================================================
void compute_tet_force_and_hessian_contributions(
    const Eigen::Ref<const Eigen::Matrix<double, 4, 3>>& tet_vertices,
    const Eigen::Ref<const Eigen::Matrix3d>& dm_inv,
    double rest_volume,
    double mu, double lam, double inverted_penalty,
    Eigen::Ref<Eigen::Matrix<double, 4, 3>> forces_out,
    Eigen::Ref<Eigen::Matrix<double, 4, 9>> hessian_out)  // 4×9 (每个顶点的 3×3 展平)
{
    // 步骤 1：连续介质物理量
    Eigen::Matrix3d F = compute_deformation_gradient(tet_vertices, dm_inv);
    Eigen::Matrix3d P = neo_hookean_pk1_stress(F, mu, lam, inverted_penalty);
    Eigen::Matrix<double, 9, 9> C_9x9 = neo_hookean_material_tangent_9x9(F, mu, lam, inverted_penalty);

    // 步骤 2：形函数梯度
    Eigen::Matrix3d B = dm_inv.transpose();  // Dm^{-T}
    Eigen::Vector3d g[4];
    g[0] = -(B.col(0) + B.col(1) + B.col(2));
    g[1] = B.col(0);
    g[2] = B.col(1);
    g[3] = B.col(2);

    // 步骤 3：组装力 f_a = -V₀ · P · g_a
    for (int a = 0; a < 4; ++a) {
        forces_out.row(a) = -rest_volume * (P * g[a]).transpose();
    }

    // 步骤 4：组装局部 Hessian 对角块 (展平为 9)
    for (int a = 0; a < 4; ++a) {
        Eigen::Matrix3d H_aa = Eigen::Matrix3d::Zero();
        for (int p = 0; p < 3; ++p) {
            for (int q = 0; q < 3; ++q) {
                double val = 0.0;
                for (int n = 0; n < 3; ++n) {
                    for (int s = 0; s < 3; ++s) {
                        int row_idx = p * 3 + n;
                        int col_idx = q * 3 + s;
                        val += C_9x9(row_idx, col_idx) * g[a](n) * g[a](s);
                    }
                }
                H_aa(p, q) = rest_volume * val;
            }
        }
        // 展平 3×3 → 1×9
        hessian_out(a, 0) = H_aa(0,0); hessian_out(a, 1) = H_aa(0,1); hessian_out(a, 2) = H_aa(0,2);
        hessian_out(a, 3) = H_aa(1,0); hessian_out(a, 4) = H_aa(1,1); hessian_out(a, 5) = H_aa(1,2);
        hessian_out(a, 6) = H_aa(2,0); hessian_out(a, 7) = H_aa(2,1); hessian_out(a, 8) = H_aa(2,2);
    }
}

} // namespace vbd
