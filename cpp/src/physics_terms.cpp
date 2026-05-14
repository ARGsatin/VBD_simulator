#include "physics_terms.h"
#include "elastic_energy.h"
#include "types.h"
#include <Eigen/LU>
#include <algorithm>

namespace vbd {

// Lamé 参数转换
namespace {
    inline std::pair<double, double> poisson_to_lame(double mu, double kappa) {
        double lam = kappa - (2.0 / 3.0) * mu;
        return {mu, lam};
    }
}

// ============================================================================
// 组装局部物理项（力 + Hessian 对角块）
// ============================================================================
void build_local_physics_terms(
    MeshData& mesh,
    const SolverConfig& cfg,
    double e_z,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>& x_prev,
    LocalPhysicsTerms& out)
{
    const int nV = mesh.num_vertices;
    const int nT = mesh.num_tets;

    // 清零
    out.force.setZero();
    for (int i = 0; i < nV; ++i)
        out.hessian[i].setZero();

    // ---- 重力 + 电场力 ----
    for (int i = 0; i < nV; ++i) {
        if (!mesh.active_mask(i)) continue;
        double m = mesh.masses(i);
        out.force(i, 0) += m * cfg.g_x;
        out.force(i, 1) += m * cfg.g_y;
        out.force(i, 2) += m * cfg.g_z + cfg.q_ion * e_z;
    }

    // ---- Neo-Hookean 超弹性（遍历四面体） ----
    auto [mu_lame, lam_lame] = poisson_to_lame(cfg.mu, cfg.kappa);

    // 临时的单四面体输出
    Eigen::Matrix<double, 4, 3> tet_f;
    Eigen::Matrix<double, 4, 9> tet_h;

    for (int tet_id = 0; tet_id < nT; ++tet_id) {
        if (!mesh.active_tet_mask(tet_id)) continue;

        // 提取四面体顶点坐标
        Eigen::Matrix<double, 4, 3> tet_verts;
        for (int a = 0; a < 4; ++a) {
            int vid = mesh.tets(tet_id, a);
            tet_verts.row(a) = mesh.vertices.row(vid);
        }

        // dm_inv 从展平的 9 向量恢复为 3×3
        Eigen::Matrix3d dm_inv_tet;
        dm_inv_tet << mesh.dm_inv(tet_id, 0), mesh.dm_inv(tet_id, 1), mesh.dm_inv(tet_id, 2),
                      mesh.dm_inv(tet_id, 3), mesh.dm_inv(tet_id, 4), mesh.dm_inv(tet_id, 5),
                      mesh.dm_inv(tet_id, 6), mesh.dm_inv(tet_id, 7), mesh.dm_inv(tet_id, 8);

        // 主动收缩：将 dm_inv 整体除以 c_shrink，使 F = Ds·(Dm^{-1}/c_shrink)
        // 从而在变形梯度中产生基准收缩预应力（而非仅修改体积）
        dm_inv_tet /= cfg.c_shrink;
        double rest_vol = mesh.tet_volumes(tet_id);

        try {
            compute_tet_force_and_hessian_contributions(
                tet_verts, dm_inv_tet, rest_vol, mu_lame, lam_lame, 1e8, tet_f, tet_h);
        } catch (...) {
            // 退化单元 → 线性弹簧回退
            double stiffness = std::max(mu_lame, 1.0) * 1e-4;
            for (int a = 0; a < 4; ++a) {
                int vid = mesh.tets(tet_id, a);
                if (!mesh.active_mask(vid)) continue;
                Eigen::Vector3d d = mesh.vertices.row(vid) - mesh.ideal_vertices.row(vid);
                out.force.row(vid) += -stiffness * d.transpose();
                out.hessian[vid] += stiffness * Eigen::Matrix3d::Identity();
            }
            continue;
        }

        // 分散到各顶点
        for (int a = 0; a < 4; ++a) {
            int vid = mesh.tets(tet_id, a);
            if (!mesh.active_mask(vid)) continue;

            out.force.row(vid) += tet_f.row(a);

            // 展平的 Hessian → 3×3 矩阵
            Eigen::Matrix3d H;
            H << tet_h(a, 0), tet_h(a, 1), tet_h(a, 2),
                 tet_h(a, 3), tet_h(a, 4), tet_h(a, 5),
                 tet_h(a, 6), tet_h(a, 7), tet_h(a, 8);
            out.hessian[vid] += H;
        }
    }

    // ---- CZM 粘聚区 + 流体拖曳（逐顶点） ----
    for (int i = 0; i < nV; ++i) {
        if (!mesh.active_mask(i)) continue;
        if (!mesh.is_current_bottom(i)) continue;

        CZMState state = static_cast<CZMState>(mesh.czm_state(i));
        double z = mesh.vertices(i, 2);
        double gap = std::max(z - cfg.z_fep, 0.0);

        if (state == CZMState::DAMAGING) {
            double softening = std::max(0.0, 1.0 - gap / std::max(cfg.delta_f, 1e-12));
            double traction = (1.0 - mesh.damage(i)) * cfg.T_max * softening;
            out.force(i, 2) -= traction * cfg.node_area;
            out.hessian[i](2, 2) += (1.0 - mesh.damage(i)) * cfg.T_max * cfg.node_area / std::max(cfg.delta_f, 1e-12);
        }
        else if (state == CZMState::FREE) {
            gap = std::max(gap, cfg.d_min);
            if (gap < cfg.d_fluid_max && mesh.time_free(i) < cfg.t_fluid_max) {
                double v_z_imp = (mesh.vertices(i, 2) - x_prev(i, 2)) / std::max(cfg.dt, 1e-12);
                double r4 = std::pow(cfg.fluid_radius, 4);
                double coeff = cfg.C_0 * cfg.eta * r4 / (gap * gap * gap);
                out.force(i, 2) -= coeff * v_z_imp;
                out.hessian[i](2, 2) += coeff / std::max(cfg.dt, 1e-12);
            }
        }
    }
}

} // namespace vbd
