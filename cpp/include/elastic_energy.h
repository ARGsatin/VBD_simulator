#pragma once

#include <Eigen/Dense>

namespace vbd {

// 变形梯度
Eigen::Matrix3d compute_deformation_gradient(
    const Eigen::Ref<const Eigen::Matrix<double, 4, 3>>& vertices,
    const Eigen::Ref<const Eigen::Matrix3d>& dm_inv);

// Neo-Hookean 本构函数
double neo_hookean_energy_density(
    const Eigen::Ref<const Eigen::Matrix3d>& F,
    double mu, double lam, double inverted_penalty = 1e8);

Eigen::Matrix3d neo_hookean_pk1_stress(
    const Eigen::Ref<const Eigen::Matrix3d>& F,
    double mu, double lam, double inverted_penalty = 1e8);

Eigen::Matrix<double, 9, 9> neo_hookean_material_tangent_9x9(
    const Eigen::Ref<const Eigen::Matrix3d>& F,
    double mu, double lam, double inverted_penalty = 1e8);

// 四面体组装（力 + Hessian 对角块）
void compute_tet_force_and_hessian_contributions(
    const Eigen::Ref<const Eigen::Matrix<double, 4, 3>>& tet_vertices,
    const Eigen::Ref<const Eigen::Matrix3d>& dm_inv,
    double rest_volume,
    double mu, double lam, double inverted_penalty,
    Eigen::Ref<Eigen::Matrix<double, 4, 3>> forces_out,
    Eigen::Ref<Eigen::Matrix<double, 4, 9>> hessian_out);

} // namespace vbd
