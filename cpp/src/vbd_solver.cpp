#include "vbd_solver.h"
#include "physics_terms.h"
#include <Eigen/LU>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <utility>

namespace vbd {

// ============================================================================
// 工具函数：由特征值裁剪重构 Hessian 为正定
// ============================================================================
static Eigen::Matrix3d make_psd(const Eigen::Matrix3d& H)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(H);
    if (eig.info() != Eigen::Success) return H;

    Eigen::Vector3d vals = eig.eigenvalues();
    if (vals.minCoeff() >= 0.0) return H;

    // 裁剪负特征值并重构
    Eigen::Vector3d clipped = vals.cwiseMax(0.0);
    return eig.eigenvectors() * clipped.asDiagonal() * eig.eigenvectors().transpose();
}

// ============================================================================
// Chebyshev 加速因子
// ============================================================================
static double chebyshev_omega(int iteration, double rho_cheb)
{
    double rho_k = std::pow(rho_cheb, iteration);
    return std::min(0.5, rho_k / (1.0 + rho_k));
}

// ============================================================================
// 更新 CZM 状态（从 Python 移植）
// ============================================================================
static void update_czm_states_inplace(
    MeshData& mesh,
    const std::vector<int>& bottom_nodes,
    const Eigen::VectorXd& internal_pull_z,
    double area, double t_max, double k_czm, double delta_f,
    double z_fep, double dt)
{
    for (int node_id : bottom_nodes) {
        if (!mesh.active_mask(node_id)) continue;
        CZMState state = static_cast<CZMState>(mesh.czm_state(node_id));
        if (state == CZMState::FIXED) {
            // FIXED 阶段：检查是否超过损伤起始阈值
            double z = mesh.vertices(node_id, 2);
            double gap = std::max(z - z_fep, 0.0);
            double traction = k_czm * gap;  // 弹性段牵引力
            if (gap > delta_f || traction > t_max) {
                mesh.czm_state(node_id) = static_cast<int>(CZMState::DAMAGING);
                mesh.damage(node_id) = 0.0;
            }
        }
        else if (state == CZMState::DAMAGING) {
            // DAMAGING 阶段：损伤演化
            double z = mesh.vertices(node_id, 2);
            double gap = std::max(z - z_fep, 0.0);
            double pull = std::abs(internal_pull_z(node_id < static_cast<int>(internal_pull_z.size()) ? node_id : 0));
            double dmg_rate = pull > 0 ? std::min(1.0, pull * dt / (t_max * delta_f)) : 0.0;
            mesh.damage(node_id) = std::min(1.0, mesh.damage(node_id) + dmg_rate);
            if (mesh.damage(node_id) >= 1.0 || gap > 5.0 * delta_f) {
                mesh.czm_state(node_id) = static_cast<int>(CZMState::FREE);
                mesh.damage(node_id) = 1.0;
                mesh.time_free(node_id) = 0.0;
            }
        }
        else if (state == CZMState::FREE) {
            mesh.time_free(node_id) += dt;
        }
    }
}

// ============================================================================
// 核心：solve_until_stable
// ============================================================================
VBDSolveResult solve_until_stable(
    MeshData& mesh,
    const SolverConfig& cfg,
    double e_z,
    int layer_id)
{
    const int nV = mesh.num_vertices;

    // 保存本时间步初态
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> x_prev = mesh.vertices;

    // ── 构建固定掩码 ──
    // 使用 Eigen 向量来存储布尔掩码，std::vector<bool> 性能较差
    std::vector<char> fixed(nV, 0);
    for (int i = 0; i < nV; ++i) {
        fixed[i] = mesh.is_top_fixed(i) ||
                   (static_cast<CZMState>(mesh.czm_state(i)) == CZMState::FIXED) ||
                   !mesh.active_mask(i);
    }

    // ── 图着色分组 ──
    // 找出所有颜色值
    int max_color = 0;
    for (int i = 0; i < nV; ++i)
        if (mesh.colors(i) > max_color) max_color = mesh.colors(i);

    // 为每个颜色建立节点列表
    std::vector<std::vector<int>> color_groups(max_color + 1);
    for (int i = 0; i < nV; ++i)
        color_groups[mesh.colors(i)].push_back(i);

    // ── 构建初始物理项 ──
    LocalPhysicsTerms terms(nV);
    build_local_physics_terms(mesh, cfg, e_z, x_prev, terms);

    // ── 自适应加速度 ──
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> adaptive_accel =
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>::Zero(nV, 3);
    for (int i = 0; i < nV; ++i) {
        if (mesh.active_mask(i)) {
            double inv_m = 1.0 / std::max(mesh.masses(i), 1e-12);
            adaptive_accel.row(i) = cfg.c_init * inv_m * terms.force.row(i);
        }
    }

    // ── Y 向量（惯性参考点） ──
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> Y =
        x_prev + cfg.dt * mesh.velocities + (cfg.dt * cfg.dt) * adaptive_accel;

    // ── 主迭代循环 ──
    double max_dx = 0.0;
    int iterations_done = 0;
    int stable_counter = 0;

    const double dt2 = cfg.dt * cfg.dt;
    const double inv_dt = 1.0 / std::max(cfg.dt, 1e-12);
    const double inv_dt2 = 1.0 / std::max(dt2, 1e-12);

    for (int iteration = 1; iteration <= cfg.max_iters; ++iteration) {
        iterations_done = iteration;
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> x_old_iter = mesh.vertices;

        // 重新计算物理项
        build_local_physics_terms(mesh, cfg, e_z, x_prev, terms);
        max_dx = 0.0;

        // ── 按颜色分组遍历 ──
        for (int c = 0; c <= max_color; ++c) {
            const auto& group = color_groups[c];
            for (int node_id : group) {
                if (fixed[node_id]) continue;

                // FEP 穿透约束
                if (!mesh.is_top_fixed(node_id) && mesh.is_bottom_surface(node_id)) {
                    if (mesh.vertices(node_id, 2) < cfg.z_fep)
                        mesh.vertices(node_id, 2) = cfg.z_fep;
                }

                // 弹性 Hessian
                Eigen::Matrix3d h_elastic = terms.hessian[node_id];

                // 损伤节点正定化
                if (static_cast<CZMState>(mesh.czm_state(node_id)) == CZMState::DAMAGING) {
                    h_elastic = make_psd(h_elastic);
                }

                // 总 Hessian: H = M/dt²·I + H_elastic + (k_d/dt)·H_elastic + ε·I
                Eigen::Matrix3d H_total = inv_dt2 * mesh.masses(node_id) * Eigen::Matrix3d::Identity()
                                         + h_elastic
                                         + (cfg.k_d * inv_dt) * h_elastic
                                         + 1e-9 * Eigen::Matrix3d::Identity();

                // 惯性力
                Eigen::Vector3d f_inertia = -inv_dt2 * mesh.masses(node_id)
                    * (mesh.vertices.row(node_id).transpose() - Y.row(node_id).transpose());

                // 阻尼力
                Eigen::Vector3d f_damp = -(cfg.k_d * inv_dt) * h_elastic
                    * (mesh.vertices.row(node_id).transpose() - x_prev.row(node_id).transpose());

                // 合力
                Eigen::Vector3d f_total = terms.force.row(node_id).transpose() + f_inertia + f_damp;

                // 解 3×3
                Eigen::Vector3d dx = H_total.ldlt().solve(f_total);
                double length = dx.norm();

                // 步长限制
                if (length > 0.01) {
                    dx *= 0.01 / length;
                    length = 0.01;
                }

                mesh.vertices.row(node_id) += dx.transpose();

                // 再次 FEP 检查
                if (!mesh.is_top_fixed(node_id) && mesh.is_bottom_surface(node_id)) {
                    if (mesh.vertices(node_id, 2) < cfg.z_fep)
                        mesh.vertices(node_id, 2) = cfg.z_fep;
                }

                if (length > max_dx) max_dx = length;
            }
        }

        // ── Chebyshev 加速 ──
        if (iteration > 5) {
            double omega = chebyshev_omega(iteration, cfg.rho_cheb);
            for (int i = 0; i < nV; ++i) {
                if (fixed[i]) continue;
                if (static_cast<CZMState>(mesh.czm_state(i)) == CZMState::DAMAGING) continue;
                mesh.vertices.row(i) += omega * (mesh.vertices.row(i) - x_old_iter.row(i));
            }
        }

        // ── 收敛判定 ──
        if (max_dx < cfg.epsilon) {
            stable_counter++;
        } else {
            stable_counter = 0;
        }
        if (stable_counter >= cfg.N_stable) break;
    }

    // ── 后处理：速度更新 ──
    for (int i = 0; i < nV; ++i) {
        if (!fixed[i]) {
            mesh.velocities.row(i) = (mesh.vertices.row(i) - x_prev.row(i)) * inv_dt;
        } else {
            mesh.velocities.row(i).setZero();
        }
    }

    // ── all_free ──
    bool all_free = true;
    for (int i = 0; i < nV; ++i) {
        if (mesh.is_bottom_surface(i) && mesh.active_mask(i)) {
            if (static_cast<CZMState>(mesh.czm_state(i)) != CZMState::FREE) {
                all_free = false;
                break;
            }
        }
    }

    // ── 动能 ──
    double kinetic = 0.0;
    for (int i = 0; i < nV; ++i) {
        if (!fixed[i]) {
            kinetic += 0.5 * mesh.masses(i) * mesh.velocities.row(i).squaredNorm();
        }
    }

    // ── damaging 计数 ──
    int damaging_count = 0;
    for (int i = 0; i < nV; ++i) {
        if (mesh.active_mask(i) && static_cast<CZMState>(mesh.czm_state(i)) == CZMState::DAMAGING)
            damaging_count++;
    }

    return {max_dx, kinetic, iterations_done, stable_counter, all_free, damaging_count};
}

// ============================================================================
// solve_with_lift（带平台提升）
// ============================================================================
VBDSolveResult solve_with_lift(
    MeshData& mesh,
    const SolverConfig& cfg,
    double e_z,
    int layer_id,
    const std::vector<int>& lifting_top)
{
    const int nV = mesh.num_vertices;

    // 保存初态
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> x_prev = mesh.vertices;

    // ── 阶段 1：平台提升剥离 ──
    double lift_distance = 0.0;
    double lift_max = 5.0 * cfg.z_fep;  // 假设 layer_thickness ≈ z_fep

    // 收集底面节点
    std::vector<int> bottom;
    for (int i = 0; i < nV; ++i) {
        if (mesh.is_bottom_surface(i) && mesh.active_mask(i))
            bottom.push_back(i);
    }

    Eigen::VectorXd pull_z(bottom.size());
    pull_z.setConstant(cfg.T_max * 1.05);

    while (lift_distance < lift_max) {
        // 抬升顶部节点
        for (int node_id : lifting_top) {
            if (mesh.active_mask(node_id))
                mesh.vertices(node_id, 2) += cfg.v_lift * cfg.dt;
        }
        lift_distance += cfg.v_lift * cfg.dt;

        // 更新 CZM 状态
        if (!bottom.empty()) {
            update_czm_states_inplace(mesh, bottom, pull_z,
                cfg.node_area, cfg.T_max, cfg.K_czm, cfg.delta_f, cfg.z_fep, cfg.dt);
        }

        // 检查是否全部脱膜
        bool all_free = true;
        for (int b : bottom) {
            if (static_cast<CZMState>(mesh.czm_state(b)) != CZMState::FREE) {
                all_free = false;
                break;
            }
        }
        if (all_free) break;
    }

    // ── 阶段 2：静平衡 ──
    // 构建固定掩码
    std::vector<char> fixed(nV, 0);
    for (int i = 0; i < nV; ++i) {
        fixed[i] = mesh.is_top_fixed(i) ||
                   (static_cast<CZMState>(mesh.czm_state(i)) == CZMState::FIXED) ||
                   !mesh.active_mask(i);
    }

    // 图着色分组
    int max_color = 0;
    for (int i = 0; i < nV; ++i)
        if (mesh.colors(i) > max_color) max_color = mesh.colors(i);

    std::vector<std::vector<int>> color_groups(max_color + 1);
    for (int i = 0; i < nV; ++i)
        color_groups[mesh.colors(i)].push_back(i);

    LocalPhysicsTerms terms(nV);
    build_local_physics_terms(mesh, cfg, e_z, x_prev, terms);

    // 自适应加速度
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> adaptive_accel =
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>::Zero(nV, 3);
    for (int i = 0; i < nV; ++i) {
        if (mesh.active_mask(i)) {
            double inv_m = 1.0 / std::max(mesh.masses(i), 1e-12);
            adaptive_accel.row(i) = cfg.c_init * inv_m * terms.force.row(i);
        }
    }

    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> Y =
        x_prev + cfg.dt * mesh.velocities + (cfg.dt * cfg.dt) * adaptive_accel;

    double max_dx = 0.0;
    int iterations_done = 0;
    int stable_counter = 0;

    const double dt2 = cfg.dt * cfg.dt;
    const double inv_dt = 1.0 / std::max(cfg.dt, 1e-12);
    const double inv_dt2 = 1.0 / std::max(dt2, 1e-12);

    for (int iteration = 1; iteration <= cfg.max_iters; ++iteration) {
        iterations_done = iteration;
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> x_old_iter = mesh.vertices;

        build_local_physics_terms(mesh, cfg, e_z, x_prev, terms);
        max_dx = 0.0;

        for (int c = 0; c <= max_color; ++c) {
            const auto& group = color_groups[c];
            for (int node_id : group) {
                if (fixed[node_id]) continue;

                if (!mesh.is_top_fixed(node_id) && mesh.is_bottom_surface(node_id)) {
                    if (mesh.vertices(node_id, 2) < cfg.z_fep)
                        mesh.vertices(node_id, 2) = cfg.z_fep;
                }

                Eigen::Matrix3d h_elastic = terms.hessian[node_id];
                if (static_cast<CZMState>(mesh.czm_state(node_id)) == CZMState::DAMAGING)
                    h_elastic = make_psd(h_elastic);

                Eigen::Matrix3d H_total = inv_dt2 * mesh.masses(node_id) * Eigen::Matrix3d::Identity()
                                         + h_elastic
                                         + (cfg.k_d * inv_dt) * h_elastic
                                         + 1e-9 * Eigen::Matrix3d::Identity();

                Eigen::Vector3d f_inertia = -inv_dt2 * mesh.masses(node_id)
                    * (mesh.vertices.row(node_id).transpose() - Y.row(node_id).transpose());
                Eigen::Vector3d f_damp = -(cfg.k_d * inv_dt) * h_elastic
                    * (mesh.vertices.row(node_id).transpose() - x_prev.row(node_id).transpose());
                Eigen::Vector3d f_total = terms.force.row(node_id).transpose() + f_inertia + f_damp;

                Eigen::Vector3d dx = H_total.ldlt().solve(f_total);
                double length = dx.norm();
                if (length > 0.01) {
                    dx *= 0.01 / length;
                    length = 0.01;
                }

                mesh.vertices.row(node_id) += dx.transpose();

                if (!mesh.is_top_fixed(node_id) && mesh.is_bottom_surface(node_id)) {
                    if (mesh.vertices(node_id, 2) < cfg.z_fep)
                        mesh.vertices(node_id, 2) = cfg.z_fep;
                }

                if (length > max_dx) max_dx = length;
            }
        }

        if (iteration > 5) {
            double omega = chebyshev_omega(iteration, cfg.rho_cheb);
            for (int i = 0; i < nV; ++i) {
                if (fixed[i]) continue;
                if (static_cast<CZMState>(mesh.czm_state(i)) == CZMState::DAMAGING) continue;
                mesh.vertices.row(i) += omega * (mesh.vertices.row(i) - x_old_iter.row(i));
            }
        }

        if (max_dx < cfg.epsilon) {
            stable_counter++;
        } else {
            stable_counter = 0;
        }
        if (stable_counter >= cfg.N_stable) break;
    }

    // 后处理
    for (int i = 0; i < nV; ++i) {
        if (!fixed[i]) {
            mesh.velocities.row(i) = (mesh.vertices.row(i) - x_prev.row(i)) * inv_dt;
        } else {
            mesh.velocities.row(i).setZero();
        }
    }

    bool all_free = true;
    for (int i = 0; i < nV; ++i) {
        if (mesh.is_bottom_surface(i) && mesh.active_mask(i)) {
            if (static_cast<CZMState>(mesh.czm_state(i)) != CZMState::FREE) {
                all_free = false;
                break;
            }
        }
    }

    double kinetic = 0.0;
    for (int i = 0; i < nV; ++i) {
        if (!fixed[i]) {
            kinetic += 0.5 * mesh.masses(i) * mesh.velocities.row(i).squaredNorm();
        }
    }

    int damaging_count = 0;
    for (int i = 0; i < nV; ++i) {
        if (mesh.active_mask(i) && static_cast<CZMState>(mesh.czm_state(i)) == CZMState::DAMAGING)
            damaging_count++;
    }

    return {max_dx, kinetic, iterations_done, stable_counter, all_free, damaging_count};
}

} // namespace vbd
