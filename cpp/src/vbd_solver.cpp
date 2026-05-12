#include "vbd_solver.h"
#include "physics_terms.h"
#include <Eigen/LU>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <utility>

// GUI 安全模式：编译时强制禁用 OpenMP，避免 QThread 中 vcomp.dll
// 创建的 Win32 工作线程与 Qt 线程管理冲突导致 segfault。
#ifdef VBD_NO_OPENMP
#  undef _OPENMP
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

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
    // 使用索引遍历，确保 local index i 仅访问 local 数组 internal_pull_z，
    // 杜绝全局 node_id 越界访问长度仅为 bottom_nodes.size() 的局部数组。
    for (size_t i = 0; i < bottom_nodes.size(); ++i) {
        int node_id = bottom_nodes[i];
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
            // 使用局部索引 i 安全访问 internal_pull_z（长度 = bottom_nodes.size()）
            double pull = std::abs(internal_pull_z(static_cast<Eigen::Index>(i)));
            double dmg_rate = pull > 0 ? std::min(1.0, pull * dt / (t_max * delta_f)) : 0.0;
            mesh.damage(node_id) = std::min(1.0, mesh.damage(node_id) + dmg_rate);
            double z = mesh.vertices(node_id, 2);
            double gap = std::max(z - z_fep, 0.0);
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
    // 使用 std::vector<char> 来存储布尔掩码，std::vector<bool> 性能较差
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
    const double dx_clip = 0.002;  // 步长截断阈值：2 mm

    for (int iteration = 1; iteration <= cfg.max_iters; ++iteration) {
        iterations_done = iteration;
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> x_old_iter = mesh.vertices;

        // TODO: 未来重构需将弹力与Hessian计算下沉移入着色循环的最内层，
        //       以恢复真正的Gauss-Seidel性能。当前在着色循环外部统一计算全局
        //       物理力（基于上一轮旧坐标），使Gauss-Seidel跌落为Jacobi迭代，
        //       降低了收敛速度，也使"图着色"失去了其核心加速意义。
        // 重新计算物理项
        build_local_physics_terms(mesh, cfg, e_z, x_prev, terms);

        // Thread-local max_dx accumulator for OpenMP reduction equivalent
        std::vector<double> local_max_dx(max_color + 1, 0.0);

        // ── 按颜色分组遍历（OpenMP 并行化）──
        // 注意：不同颜色组之间无数据依赖，可安全并行
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1)
        #endif
        for (int c = 0; c <= max_color; ++c) {
            const auto& group = color_groups[c];
            const int gsize = static_cast<int>(group.size());
            double thread_max_dx = 0.0;

            // 使用整数索引遍历以满足 OpenMP 要求
            for (int idx = 0; idx < gsize; ++idx) {
                int node_id = group[idx];
                if (fixed[node_id]) continue;

                // FEP 穿透约束
                if (!mesh.is_top_fixed(node_id) && mesh.is_bottom_surface(node_id)) {
                    if (mesh.vertices(node_id, 2) < cfg.z_fep)
                        mesh.vertices(node_id, 2) = cfg.z_fep;
                }

                    // 弹性 Hessian（每个线程独立分配的局部变量，线程安全）
                    Eigen::Matrix3d h_elastic = terms.hessian[node_id];

                    // VBD 无条件正定化：保证非线性超弹性收敛的绝对前提
                    h_elastic = make_psd(h_elastic);

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

                // 步长限制收紧至 0.002（2 mm），适配毫米级网格
                if (length > dx_clip) {
                    dx *= dx_clip / length;
                    length = dx_clip;
                }

                mesh.vertices.row(node_id) += dx.transpose();

                // 再次 FEP 检查
                if (!mesh.is_top_fixed(node_id) && mesh.is_bottom_surface(node_id)) {
                    if (mesh.vertices(node_id, 2) < cfg.z_fep)
                        mesh.vertices(node_id, 2) = cfg.z_fep;
                }

                if (length > thread_max_dx) thread_max_dx = length;
            }
            local_max_dx[c] = thread_max_dx;
        }

        // 归约各线程的 max_dx
        max_dx = 0.0;
        for (int c = 0; c <= max_color; ++c) {
            if (local_max_dx[c] > max_dx) max_dx = local_max_dx[c];
        }

        // ── Chebyshev 加速 ──
        if (iteration > 5) {
            double omega = chebyshev_omega(iteration, cfg.rho_cheb);
            #ifdef _OPENMP
            #pragma omp parallel for
            #endif
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
// solve_lift_and_relax（单步提升 + 静平衡求解器）
//
// 控制反转（Inversion of Control）架构：
// 删除了原来的 while (lift_distance < lift_max) 大循环！
// 该函数降级为"单步求解器"，仅执行：
//   Step 1: 单次微小提升 — top 节点提升 v_lift * dt
//   Step 2: 单次 VBD 静平衡松弛（内层迭代至收敛）
//   Step 3: 更新 CZM 底部损伤状态
//   Step 4: 返回单步结果（由 Python 侧接管时间流逝循环）
//
// 这样 C++ 代码不再长时间持有 GIL，Worker 线程可以在每步之间
// 向主线程发射信号，彻底解决界面"未响应"问题。
//
// 关键数值安全措施：
// - 步长截断阈值收紧至 0.002（2 mm），适配毫米级 DLP 打印网格
// - 每步提升量 = v_lift * dt（微米级），形变梯度 F 始终接近单位矩阵
// - 提升后立刻 VBD 松弛，弹性应力即时释放
// ============================================================================
VBDSolveResult solve_lift_and_relax(
    MeshData& mesh,
    const SolverConfig& cfg,
    double e_z,
    int layer_id,
    const std::vector<int>& lifting_top)
{
    const int nV = mesh.num_vertices;

    const double dt2 = cfg.dt * cfg.dt;
    const double inv_dt = 1.0 / std::max(cfg.dt, 1e-12);
    const double inv_dt2 = 1.0 / std::max(dt2, 1e-12);
    const double dx_clip = 0.002;  // 步长截断阈值收紧至 2 mm
    const double lift_step = cfg.v_lift * cfg.dt;

    // ── 收集底面节点（用于 CZM 状态更新与剥离判定）──
    std::vector<int> bottom;
    for (int i = 0; i < nV; ++i) {
        if (mesh.is_bottom_surface(i) && mesh.active_mask(i))
            bottom.push_back(i);
    }

    Eigen::VectorXd pull_z(bottom.size());
    pull_z.setConstant(cfg.T_max * 1.05);

    // ── 图着色分组（每次调用重构建，确保与当前状态一致）──
    int max_color = 0;
    for (int i = 0; i < nV; ++i)
        if (mesh.colors(i) > max_color) max_color = mesh.colors(i);

    std::vector<std::vector<int>> color_groups(max_color + 1);
    for (int i = 0; i < nV; ++i)
        color_groups[mesh.colors(i)].push_back(i);

    // ════════════════════════════════════════════════════════════════════
    // Step 1: 单次微小增量驱动 —— 将 lifting_top 沿 Z 轴提升
    //          cfg.v_lift * cfg.dt（微米级增量）
    // ════════════════════════════════════════════════════════════════════
    for (int node_id : lifting_top) {
        if (mesh.active_mask(node_id))
            mesh.vertices(node_id, 2) += lift_step;
    }

    // ════════════════════════════════════════════════════════════════════
    // Step 2: 保存本步初态 x_prev（提升后的状态）
    // ════════════════════════════════════════════════════════════════════
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> x_prev = mesh.vertices;

    // ════════════════════════════════════════════════════════════════════
    // Step 3: 重建固定掩码
    // ════════════════════════════════════════════════════════════════════
    std::vector<char> fixed(nV, 0);
    for (int i = 0; i < nV; ++i) {
        fixed[i] = mesh.is_top_fixed(i) ||
                   (static_cast<CZMState>(mesh.czm_state(i)) == CZMState::FIXED) ||
                   !mesh.active_mask(i);
    }

    // ════════════════════════════════════════════════════════════════════
    // Step 4: 构建物理项 + 自适应加速度 + Y 向量
    // ════════════════════════════════════════════════════════════════════
    LocalPhysicsTerms terms(nV);
    build_local_physics_terms(mesh, cfg, e_z, x_prev, terms);

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

    // ════════════════════════════════════════════════════════════════════
    // Step 5: 单次内层 VBD 静平衡松弛
    //         系统在承受本次微小的边界拉拔后，立刻通过内部迭代
    //         寻找当前时间步的静平衡。
    // ════════════════════════════════════════════════════════════════════
    double max_dx = 0.0;
    int iterations_done = 0;
    int stable_counter = 0;

    for (int iteration = 1; iteration <= cfg.max_iters; ++iteration) {
        iterations_done = iteration;
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> x_old_iter = mesh.vertices;

        // TODO: 未来重构需将弹力与Hessian计算下沉移入着色循环的最内层，
        //       以恢复真正的Gauss-Seidel性能。当前在着色循环外部统一计算全局
        //       物理力（基于上一轮旧坐标），使Gauss-Seidel跌落为Jacobi迭代，
        //       降低了收敛速度，也使"图着色"失去了其核心加速意义。
        // 重新计算物理项（顶点位置已在上次迭代中更新）
        build_local_physics_terms(mesh, cfg, e_z, x_prev, terms);

        // Thread-local max_dx accumulator
        std::vector<double> local_max_dx(max_color + 1, 0.0);

        // ── 按颜色分组遍历（OpenMP 并行化）──
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1)
        #endif
        for (int c = 0; c <= max_color; ++c) {
            const auto& group = color_groups[c];
            const int gsize = static_cast<int>(group.size());
            double thread_max_dx = 0.0;

            for (int idx = 0; idx < gsize; ++idx) {
                int node_id = group[idx];
                if (fixed[node_id]) continue;

                // FEP 穿透约束
                if (!mesh.is_top_fixed(node_id) && mesh.is_bottom_surface(node_id)) {
                    if (mesh.vertices(node_id, 2) < cfg.z_fep)
                        mesh.vertices(node_id, 2) = cfg.z_fep;
                }

                // 弹性 Hessian（线程局部变量，线程安全）
                Eigen::Matrix3d h_elastic = terms.hessian[node_id];

                // VBD 无条件正定化：保证非线性超弹性收敛的绝对前提
                h_elastic = make_psd(h_elastic);

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

                // 步长限制收紧至 0.002（2 mm），适配毫米级网格
                if (length > dx_clip) {
                    dx *= dx_clip / length;
                    length = dx_clip;
                }

                mesh.vertices.row(node_id) += dx.transpose();

                // 再次 FEP 检查
                if (!mesh.is_top_fixed(node_id) && mesh.is_bottom_surface(node_id)) {
                    if (mesh.vertices(node_id, 2) < cfg.z_fep)
                        mesh.vertices(node_id, 2) = cfg.z_fep;
                }

                if (length > thread_max_dx) thread_max_dx = length;
            }
            local_max_dx[c] = thread_max_dx;
        }

        // 归约各线程的 max_dx
        max_dx = 0.0;
        for (int c = 0; c <= max_color; ++c) {
            if (local_max_dx[c] > max_dx) max_dx = local_max_dx[c];
        }

        // ── Chebyshev 加速 ──
        if (iteration > 5) {
            double omega = chebyshev_omega(iteration, cfg.rho_cheb);
            #ifdef _OPENMP
            #pragma omp parallel for
            #endif
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

    // ════════════════════════════════════════════════════════════════════
    // Step 6: 速度更新
    // ════════════════════════════════════════════════════════════════════
    for (int i = 0; i < nV; ++i) {
        if (!fixed[i]) {
            mesh.velocities.row(i) = (mesh.vertices.row(i) - x_prev.row(i)) * inv_dt;
        } else {
            mesh.velocities.row(i).setZero();
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // Step 7: 更新底部 CZM 损伤状态
    // ════════════════════════════════════════════════════════════════════
    if (!bottom.empty()) {
        update_czm_states_inplace(mesh, bottom, pull_z,
            cfg.node_area, cfg.T_max, cfg.K_czm, cfg.delta_f, cfg.z_fep, cfg.dt);
    }

    // ════════════════════════════════════════════════════════════════════
    // Step 8: 检查剥离退出条件
    // ════════════════════════════════════════════════════════════════════
    bool all_free_flag = true;
    for (int b : bottom) {
        if (static_cast<CZMState>(mesh.czm_state(b)) != CZMState::FREE) {
            all_free_flag = false;
            break;
        }
    }

    // ── 动能计算 ──
    double kinetic = 0.0;
    for (int i = 0; i < nV; ++i) {
        if (mesh.active_mask(i)) {
            kinetic += 0.5 * mesh.masses(i) * mesh.velocities.row(i).squaredNorm();
        }
    }

    // ── damaging 计数 ──
    int damaging_count = 0;
    for (int i = 0; i < nV; ++i) {
        if (mesh.active_mask(i) && static_cast<CZMState>(mesh.czm_state(i)) == CZMState::DAMAGING)
            damaging_count++;
    }

    return {max_dx, kinetic, iterations_done, stable_counter, all_free_flag, damaging_count};
}

} // namespace vbd
