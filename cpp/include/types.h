#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <vector>

namespace vbd {

// ============================================================================
// CZM 状态枚举
// ============================================================================
enum class CZMState : int32_t { FIXED = 0, DAMAGING = 1, FREE = 2 };

// ============================================================================
// 配置参数结构体（从 Python SimulationConfig 映射）
// ============================================================================
struct SolverConfig {
    // 时间步
    double dt = 1e-4;
    int max_iters = 200;
    int N_stable = 5;
    double epsilon = 1e-6;

    // 阻尼
    double k_d = 0.05;

    // Chebyshev
    double rho_cheb = 0.95;

    // 材料（本构）
    double mu = 1e5;
    double kappa = 2e5;
    double c_shrink = 1.0;

    // 电场
    double q_ion = 1.0;

    // 重力
    double g_x = 0.0, g_y = 0.0, g_z = -9.81;

    // CZM
    double T_max = 1e4;
    double K_czm = 1e7;
    double delta_f = 1e-4;
    double node_area = 1e-8;
    double z_fep = 0.0;

    // 流体拖曳
    double C_0 = 1.0;
    double eta = 0.001;
    double fluid_radius = 1e-3;
    double d_min = 1e-6;
    double d_fluid_max = 1e-3;
    double t_fluid_max = 0.1;

    // 平台提升
    double v_lift = 0.0;

    // 自适应加速度
    double c_init = 0.5;
};

// ============================================================================
// 网格数据（通过 pybind11 从 Python 零拷贝映射）
// ============================================================================
struct MeshData {
    // 顶点数据
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> vertices;
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> velocities;
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> ideal_vertices;
    Eigen::Map<Eigen::VectorXd> masses;

    // 掩码
    Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> active_mask;
    Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> is_top_fixed;
    Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> is_bottom_surface;
    Eigen::Map<Eigen::VectorXi> czm_state;

    // 损伤 + 自由时间
    Eigen::Map<Eigen::VectorXd> damage;
    Eigen::Map<Eigen::VectorXd> time_free;

    // 四面体
    Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, 4, Eigen::RowMajor>> tets;
    Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> active_tet_mask;
    // dm_inv: (num_tets, 3, 3) 展平为 (num_tets, 9)
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 9, Eigen::RowMajor>> dm_inv;
    Eigen::Map<Eigen::VectorXd> tet_volumes;

    // 图着色
    Eigen::Map<Eigen::VectorXi> colors;

    int num_vertices;
    int num_tets;

    MeshData(
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> v,
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> vel,
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> iv,
        Eigen::Map<Eigen::VectorXd> m,
        Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> act,
        Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> tf,
        Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> bs,
        Eigen::Map<Eigen::VectorXi> czm,
        Eigen::Map<Eigen::VectorXd> dmg,
        Eigen::Map<Eigen::VectorXd> tf_t,
        Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, 4, Eigen::RowMajor>> tet,
        Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> at,
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 9, Eigen::RowMajor>> dmi,
        Eigen::Map<Eigen::VectorXd> tv,
        Eigen::Map<Eigen::VectorXi> col
    )
        : vertices(v), velocities(vel), ideal_vertices(iv), masses(m),
          active_mask(act), is_top_fixed(tf), is_bottom_surface(bs),
          czm_state(czm), damage(dmg), time_free(tf_t),
          tets(tet), active_tet_mask(at), dm_inv(dmi), tet_volumes(tv),
          colors(col),
          num_vertices(static_cast<int>(v.rows())),
          num_tets(static_cast<int>(tet.rows()))
    {}
};

// ============================================================================
// VBD 求解结果
// ============================================================================
struct VBDSolveResult {
    double max_dx;
    double kinetic_energy;
    int iterations;
    int stable_steps;
    bool all_free;
    int chebyshev_skipped_damaging;
};

// ============================================================================
// 局部物理项（力 + Hessian 对角块）
// ============================================================================
struct LocalPhysicsTerms {
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> force;
    std::vector<Eigen::Matrix3d> hessian;

    LocalPhysicsTerms(int n)
        : force(Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>::Zero(n, 3)),
          hessian(n, Eigen::Matrix3d::Zero())
    {}
};

} // namespace vbd
