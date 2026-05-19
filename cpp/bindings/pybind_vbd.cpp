#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <chrono>

#include "types.h"
#include "vbd_solver.h"
#include "elastic_energy.h"

namespace py = pybind11;
using namespace vbd;

#ifndef VBD_PYBIND_MODULE_NAME
#define VBD_PYBIND_MODULE_NAME hydrogel_vbd_cpp
#endif

// ============================================================================
// Python 侧 solver 类封装：直接操作 MeshState 内存
// ============================================================================

// 包装函数：从 Python NumPy 数组构建 MeshData 并调用 C++ 求解器
static py::dict solve_until_stable_py(
    py::array_t<double> vertices_in,     // (N, 3) 可写
    py::array_t<double> velocities,      // (N, 3) 可写
    py::array_t<double> ideal_vertices,  // (N, 3) 只读
    py::array_t<double> masses,
    py::array_t<int> first_active_layer,
    py::array_t<int> is_top_surface_of_layer,
    py::array_t<bool> active_mask,       // (N,) 只读
    py::array_t<bool> is_top_fixed,      // (N,) 只读
    py::array_t<bool> is_bottom_surface, // (N,) 只读
    py::array_t<bool> is_current_bottom, // (N,) 只读
    py::array_t<int> czm_state,          // (N,) 可写
    py::array_t<double> damage,          // (N,) 可写
    py::array_t<double> time_free,       // (N,) 可写
    py::array_t<int> tets,               // (T, 4) 只读
    py::array_t<bool> active_tet_mask,   // (T,) 只读
    py::array_t<double> dm_inv,          // (T, 9) 只读
    py::array_t<double> tet_volumes,     // (T,) 只读
    py::array_t<int> colors,             // (N,) 只读
    const SolverConfig& cfg,
    double e_z,
    int layer_id)
{
    // 获取缓冲区信息
    auto buf_v = vertices_in.request();
    auto buf_vel = velocities.request();
    auto buf_iv = ideal_vertices.request();
    auto buf_m = masses.request();
    auto buf_fal = first_active_layer.request();
    auto buf_surf = is_top_surface_of_layer.request();
    auto buf_act = active_mask.request();
    auto buf_tf = is_top_fixed.request();
    auto buf_bs = is_bottom_surface.request();
    auto buf_cb = is_current_bottom.request();
    auto buf_czm = czm_state.request();
    auto buf_dmg = damage.request();
    auto buf_tf_t = time_free.request();
    auto buf_tet = tets.request();
    auto buf_at = active_tet_mask.request();
    auto buf_dmi = dm_inv.request();
    auto buf_tv = tet_volumes.request();
    auto buf_col = colors.request();

    int nV = static_cast<int>(buf_v.shape[0]);
    int nT = static_cast<int>(buf_tet.shape[0]);

    // 构建 Eigen::Map（零拷贝）
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> v_map(
        static_cast<double*>(buf_v.ptr), nV, 3);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> vel_map(
        static_cast<double*>(buf_vel.ptr), nV, 3);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> iv_map(
        static_cast<double*>(buf_iv.ptr), nV, 3);
    Eigen::Map<Eigen::VectorXd> m_map(
        static_cast<double*>(buf_m.ptr), nV);
    Eigen::Map<Eigen::VectorXi> fal_map(
        static_cast<int*>(buf_fal.ptr), nV);
    Eigen::Map<Eigen::VectorXi> surf_map(
        static_cast<int*>(buf_surf.ptr), nV);
    Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> act_map(
        static_cast<bool*>(buf_act.ptr), nV);
    Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> tf_map(
        static_cast<bool*>(buf_tf.ptr), nV);
    Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> bs_map(
        static_cast<bool*>(buf_bs.ptr), nV);
    Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> cb_map(
        static_cast<bool*>(buf_cb.ptr), nV);
    Eigen::Map<Eigen::VectorXi> czm_map(
        static_cast<int*>(buf_czm.ptr), nV);
    Eigen::Map<Eigen::VectorXd> dmg_map(
        static_cast<double*>(buf_dmg.ptr), nV);
    Eigen::Map<Eigen::VectorXd> tf_t_map(
        static_cast<double*>(buf_tf_t.ptr), nV);
    Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, 4, Eigen::RowMajor>> tet_map(
        static_cast<int*>(buf_tet.ptr), nT, 4);
    Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> at_map(
        static_cast<bool*>(buf_at.ptr), nT);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 9, Eigen::RowMajor>> dmi_map(
        static_cast<double*>(buf_dmi.ptr), nT, 9);
    Eigen::Map<Eigen::VectorXd> tv_map(
        static_cast<double*>(buf_tv.ptr), nT);
    Eigen::Map<Eigen::VectorXi> col_map(
        static_cast<int*>(buf_col.ptr), nV);

    MeshData mesh(
        v_map, vel_map, iv_map, m_map,
        fal_map, surf_map,
        act_map, tf_map, bs_map, cb_map, czm_map,
        dmg_map, tf_t_map,
        tet_map, at_map, dmi_map, tv_map,
        col_map);

    VBDSolveResult result;
    {
        py::gil_scoped_release release;
        result = vbd::solve_until_stable(mesh, cfg, e_z, layer_id);
    }

    // 构建返回字典
    py::dict out;
    out["max_dx"] = result.max_dx;
    out["kinetic_energy"] = result.kinetic_energy;
    out["iterations"] = result.iterations;
    out["stable_steps"] = result.stable_steps;
    out["all_free"] = result.all_free;
    out["chebyshev_skipped_damaging"] = result.chebyshev_skipped_damaging;
    return out;
}

// solve_lift_and_relax 包装（单步提升 + 单次静平衡）
static py::dict solve_lift_and_relax_py(
    py::array_t<double> vertices_in,
    py::array_t<double> velocities,
    py::array_t<double> ideal_vertices,
    py::array_t<double> masses,
    py::array_t<int> first_active_layer,
    py::array_t<int> is_top_surface_of_layer,
    py::array_t<bool> active_mask,
    py::array_t<bool> is_top_fixed,
    py::array_t<bool> is_bottom_surface,
    py::array_t<bool> is_current_bottom,
    py::array_t<int> czm_state,
    py::array_t<double> damage,
    py::array_t<double> time_free,
    py::array_t<int> tets,
    py::array_t<bool> active_tet_mask,
    py::array_t<double> dm_inv,
    py::array_t<double> tet_volumes,
    py::array_t<int> colors,
    const SolverConfig& cfg,
    double e_z,
    int layer_id,
    const std::vector<int>& lifting_top)
{
    auto buf_v = vertices_in.request();
    auto buf_vel = velocities.request();
    auto buf_iv = ideal_vertices.request();
    auto buf_m = masses.request();
    auto buf_fal = first_active_layer.request();
    auto buf_surf = is_top_surface_of_layer.request();
    auto buf_act = active_mask.request();
    auto buf_tf = is_top_fixed.request();
    auto buf_bs = is_bottom_surface.request();
    auto buf_cb = is_current_bottom.request();
    auto buf_czm = czm_state.request();
    auto buf_dmg = damage.request();
    auto buf_tf_t = time_free.request();
    auto buf_tet = tets.request();
    auto buf_at = active_tet_mask.request();
    auto buf_dmi = dm_inv.request();
    auto buf_tv = tet_volumes.request();
    auto buf_col = colors.request();

    int nV = static_cast<int>(buf_v.shape[0]);
    int nT = static_cast<int>(buf_tet.shape[0]);

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> v_map(
        static_cast<double*>(buf_v.ptr), nV, 3);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> vel_map(
        static_cast<double*>(buf_vel.ptr), nV, 3);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> iv_map(
        static_cast<double*>(buf_iv.ptr), nV, 3);
    Eigen::Map<Eigen::VectorXd> m_map(
        static_cast<double*>(buf_m.ptr), nV);
    Eigen::Map<Eigen::VectorXi> fal_map(
        static_cast<int*>(buf_fal.ptr), nV);
    Eigen::Map<Eigen::VectorXi> surf_map(
        static_cast<int*>(buf_surf.ptr), nV);
    Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> act_map(
        static_cast<bool*>(buf_act.ptr), nV);
    Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> tf_map(
        static_cast<bool*>(buf_tf.ptr), nV);
    Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> bs_map(
        static_cast<bool*>(buf_bs.ptr), nV);
    Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> cb_map(
        static_cast<bool*>(buf_cb.ptr), nV);
    Eigen::Map<Eigen::VectorXi> czm_map(
        static_cast<int*>(buf_czm.ptr), nV);
    Eigen::Map<Eigen::VectorXd> dmg_map(
        static_cast<double*>(buf_dmg.ptr), nV);
    Eigen::Map<Eigen::VectorXd> tf_t_map(
        static_cast<double*>(buf_tf_t.ptr), nV);
    Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, 4, Eigen::RowMajor>> tet_map(
        static_cast<int*>(buf_tet.ptr), nT, 4);
    Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> at_map(
        static_cast<bool*>(buf_at.ptr), nT);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 9, Eigen::RowMajor>> dmi_map(
        static_cast<double*>(buf_dmi.ptr), nT, 9);
    Eigen::Map<Eigen::VectorXd> tv_map(
        static_cast<double*>(buf_tv.ptr), nT);
    Eigen::Map<Eigen::VectorXi> col_map(
        static_cast<int*>(buf_col.ptr), nV);

    MeshData mesh(
        v_map, vel_map, iv_map, m_map,
        fal_map, surf_map,
        act_map, tf_map, bs_map, cb_map, czm_map,
        dmg_map, tf_t_map,
        tet_map, at_map, dmi_map, tv_map,
        col_map);

    VBDSolveResult result;
    {
        py::gil_scoped_release release;
        result = vbd::solve_lift_and_relax(mesh, cfg, e_z, layer_id, lifting_top);
    }

    py::dict out;
    out["max_dx"] = result.max_dx;
    out["kinetic_energy"] = result.kinetic_energy;
    out["iterations"] = result.iterations;
    out["stable_steps"] = result.stable_steps;
    out["all_free"] = result.all_free;
    out["chebyshev_skipped_damaging"] = result.chebyshev_skipped_damaging;
    return out;
}

// ============================================================================
// 模块定义
// ============================================================================
static py::tuple debug_tet_force_hessian_py(
    py::array_t<double> vertices_in,
    py::array_t<double> dm_inv_in,
    double rest_volume,
    double mu,
    double lam)
{
    auto v = vertices_in.unchecked<2>();
    auto dmi = dm_inv_in.unchecked<2>();

    Eigen::Matrix<double, 4, 3> verts;
    Eigen::Matrix3d dm_inv;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 3; ++j)
            verts(i, j) = v(i, j);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            dm_inv(i, j) = dmi(i, j);

    Eigen::Matrix<double, 4, 3> forces;
    Eigen::Matrix<double, 4, 9> hessian_flat;
    compute_tet_force_and_hessian_contributions(
        verts, dm_inv, rest_volume, mu, lam, 1e8, forces, hessian_flat);

    Eigen::Matrix<double, 12, 3, Eigen::RowMajor> hessian_rows;
    for (int a = 0; a < 4; ++a)
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                hessian_rows(a * 3 + r, c) = hessian_flat(a, r * 3 + c);

    return py::make_tuple(forces, hessian_rows);
}

static py::dict debug_single_tet_line_search_py(
    py::array_t<double> vertices_in,
    py::array_t<double> dm_inv_in,
    double rest_volume,
    double mu,
    double lam,
    int node_id,
    py::array_t<double> dx_in,
    double floor_z)
{
    auto v = vertices_in.unchecked<2>();
    auto dmi = dm_inv_in.unchecked<2>();
    auto dx_buf = dx_in.unchecked<1>();

    Eigen::Matrix<double, 4, 3> verts;
    Eigen::Matrix3d dm_inv;
    Eigen::Vector3d dx;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 3; ++j)
            verts(i, j) = v(i, j);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            dm_inv(i, j) = dmi(i, j);
    for (int i = 0; i < 3; ++i)
        dx(i) = dx_buf(i);

    auto energy = [&]() {
        Eigen::Matrix3d F = compute_deformation_gradient(verts, dm_inv);
        return rest_volume * neo_hookean_energy_density(F, mu, lam, 1e8);
    };

    const Eigen::Vector3d x_saved = verts.row(node_id).transpose();
    const double e_before = energy();
    double alpha = 1.0;
    bool accepted = false;
    constexpr int max_trials = 12;
    constexpr double min_alpha = 1.0 / 4096.0;
    constexpr double energy_tol = 1e-10;

    for (int trial = 0; trial < max_trials; ++trial) {
        verts.row(node_id) = (x_saved + alpha * dx).transpose();
        if (verts(node_id, 2) < floor_z) {
            verts(node_id, 2) = floor_z;
        }
        const double e_after = energy();
        if (std::isfinite(e_after) && e_after <= e_before + energy_tol) {
            accepted = true;
            break;
        }
        alpha *= 0.5;
        if (alpha < min_alpha) break;
    }

    if (!accepted) {
        alpha = 0.0;
        verts.row(node_id) = x_saved.transpose();
    }

    py::dict out;
    out["accepted"] = accepted;
    out["alpha"] = alpha;
    out["energy_before"] = e_before;
    out["energy_after"] = energy();
    out["position_after"] = verts.row(node_id).transpose();
    return out;
}

static py::dict result_to_dict(const VBDSolveResult& result)
{
    py::dict out;
    out["max_dx"] = result.max_dx;
    out["kinetic_energy"] = result.kinetic_energy;
    out["iterations"] = result.iterations;
    out["stable_steps"] = result.stable_steps;
    out["all_free"] = result.all_free;
    out["chebyshev_skipped_damaging"] = result.chebyshev_skipped_damaging;
    return out;
}

static double field_event_window_e_z(
    int step,
    int expected_steps,
    double detach_e_z,
    int detach_step,
    const SolverConfig& cfg,
    double peak_e_z,
    int& peak_start)
{
    if ((detach_e_z <= 0.0 && peak_e_z <= 0.0) || expected_steps <= 0) {
        peak_start = 0;
        return 0.0;
    }

    const int detach_pre = std::max(0, cfg.field_detach_pre_steps);
    const int detach_post = std::max(0, cfg.field_detach_post_steps);
    const int peak_steps = std::max(0, cfg.field_peak_window_steps);
    peak_start = peak_steps > 0 ? std::max(1, expected_steps - peak_steps + 1) : 0;

    const bool in_detach_window =
        detach_step > 0
        && (detach_step - detach_pre) <= step
        && step <= (detach_step + detach_post);
    const bool in_peak_window = peak_start > 0 && step >= peak_start;

    double value = 0.0;
    if (in_detach_window) value = std::max(value, std::max(detach_e_z, 0.0));
    if (in_peak_window) value = std::max(value, std::max(peak_e_z, 0.0));
    return value;
}

static int positive_step_count(double total, double max_step)
{
    total = std::max(0.0, total);
    max_step = std::max(0.0, max_step);
    if (total <= 0.0 || max_step <= 0.0) return 0;
    const double ratio = total / max_step;
    return static_cast<int>(std::ceil(ratio - std::max(1e-12, std::abs(ratio) * 1e-12)));
}

static py::dict solve_field_debug_branch_py(
    py::array_t<double> vertices_in,
    py::array_t<double> velocities,
    py::array_t<double> ideal_vertices,
    py::array_t<double> masses,
    py::array_t<int> first_active_layer,
    py::array_t<int> is_top_surface_of_layer,
    py::array_t<bool> active_mask,
    py::array_t<bool> is_top_fixed,
    py::array_t<bool> is_bottom_surface,
    py::array_t<bool> is_current_bottom,
    py::array_t<int> czm_state,
    py::array_t<double> damage,
    py::array_t<double> time_free,
    py::array_t<int> tets,
    py::array_t<bool> active_tet_mask,
    py::array_t<double> dm_inv,
    py::array_t<double> tet_volumes,
    py::array_t<int> colors,
    const SolverConfig& cfg,
    int layer_id,
    double e_z,
    const std::vector<int>& lifting_top,
    int expected_lift_steps,
    int event_window_detach_step,
    double peak_e_z,
    bool continue_to_peak,
    int n_layers)
{
    auto buf_v = vertices_in.request();
    auto buf_vel = velocities.request();
    auto buf_iv = ideal_vertices.request();
    auto buf_m = masses.request();
    auto buf_fal = first_active_layer.request();
    auto buf_surf = is_top_surface_of_layer.request();
    auto buf_act = active_mask.request();
    auto buf_tf = is_top_fixed.request();
    auto buf_bs = is_bottom_surface.request();
    auto buf_cb = is_current_bottom.request();
    auto buf_czm = czm_state.request();
    auto buf_dmg = damage.request();
    auto buf_tf_t = time_free.request();
    auto buf_tet = tets.request();
    auto buf_at = active_tet_mask.request();
    auto buf_dmi = dm_inv.request();
    auto buf_tv = tet_volumes.request();
    auto buf_col = colors.request();

    int nV = static_cast<int>(buf_v.shape[0]);
    int nT = static_cast<int>(buf_tet.shape[0]);

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> v_map(
        static_cast<double*>(buf_v.ptr), nV, 3);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> vel_map(
        static_cast<double*>(buf_vel.ptr), nV, 3);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> iv_map(
        static_cast<double*>(buf_iv.ptr), nV, 3);
    Eigen::Map<Eigen::VectorXd> m_map(static_cast<double*>(buf_m.ptr), nV);
    Eigen::Map<Eigen::VectorXi> fal_map(static_cast<int*>(buf_fal.ptr), nV);
    Eigen::Map<Eigen::VectorXi> surf_map(static_cast<int*>(buf_surf.ptr), nV);
    Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> act_map(static_cast<bool*>(buf_act.ptr), nV);
    Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> tf_map(static_cast<bool*>(buf_tf.ptr), nV);
    Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> bs_map(static_cast<bool*>(buf_bs.ptr), nV);
    Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> cb_map(static_cast<bool*>(buf_cb.ptr), nV);
    Eigen::Map<Eigen::VectorXi> czm_map(static_cast<int*>(buf_czm.ptr), nV);
    Eigen::Map<Eigen::VectorXd> dmg_map(static_cast<double*>(buf_dmg.ptr), nV);
    Eigen::Map<Eigen::VectorXd> tf_t_map(static_cast<double*>(buf_tf_t.ptr), nV);
    Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, 4, Eigen::RowMajor>> tet_map(static_cast<int*>(buf_tet.ptr), nT, 4);
    Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> at_map(static_cast<bool*>(buf_at.ptr), nT);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 9, Eigen::RowMajor>> dmi_map(static_cast<double*>(buf_dmi.ptr), nT, 9);
    Eigen::Map<Eigen::VectorXd> tv_map(static_cast<double*>(buf_tv.ptr), nT);
    Eigen::Map<Eigen::VectorXi> col_map(static_cast<int*>(buf_col.ptr), nV);

    MeshData mesh(
        v_map, vel_map, iv_map, m_map,
        fal_map, surf_map,
        act_map, tf_map, bs_map, cb_map, czm_map,
        dmg_map, tf_t_map,
        tet_map, at_map, dmi_map, tv_map,
        col_map);

    const auto t0 = std::chrono::steady_clock::now();
    const double lift_step = cfg.v_lift * cfg.dt;
    const double lift_max = std::max(0, expected_lift_steps) * std::abs(lift_step);
    const bool has_lift = expected_lift_steps > 0 && !lifting_top.empty() && lift_step != 0.0;
    const bool event_windows = event_window_detach_step > 0 && expected_lift_steps > 0;

    VBDSolveResult result{0.0, 0.0, 0, 0, true, 0};
    VBDSolveResult commit_result = result;
    VBDSolveResult guard_result = result;
    int layer_steps = 0;
    int detach_step = 0;
    int commit_steps = 0;
    int field_applied_steps = 0;
    int total_iterations = 0;
    int max_iter_hits = 0;
    int clipped_steps = 0;
    int peak_start_step = 0;

    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> commit_vertices;
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> commit_velocities;
    Eigen::VectorXi commit_czm_state;
    Eigen::VectorXd commit_damage;
    Eigen::VectorXd commit_time_free;
    bool have_commit = false;

    if (has_lift) {
        for (int i = 0; i < expected_lift_steps; ++i) {
            const int step_index = layer_steps + 1;
            double step_e_z = e_z;
            if (event_windows) {
                step_e_z = field_event_window_e_z(
                    step_index,
                    expected_lift_steps,
                    e_z,
                    event_window_detach_step,
                    cfg,
                    peak_e_z,
                    peak_start_step);
                if (step_e_z > 0.0) field_applied_steps++;
            }

            result = vbd::solve_lift_and_relax(mesh, cfg, step_e_z, layer_id, lifting_top);
            layer_steps++;
            total_iterations += result.iterations;
            if (result.iterations >= cfg.max_iters) max_iter_hits++;
            if (result.max_dx >= cfg.dx_clip * (1.0 - 1e-9)) clipped_steps++;

            if (result.all_free) {
                if (detach_step <= 0) detach_step = layer_steps;
                if (!have_commit) {
                    commit_vertices = mesh.vertices;
                    commit_velocities = mesh.velocities;
                    commit_czm_state = mesh.czm_state;
                    commit_damage = mesh.damage;
                    commit_time_free = mesh.time_free;
                    commit_result = result;
                    commit_steps = layer_steps;
                    have_commit = true;
                }
                if (!continue_to_peak) break;
            }
        }
    } else {
        result = vbd::solve_until_stable(mesh, cfg, e_z, layer_id);
        total_iterations += result.iterations;
        if (result.iterations >= cfg.max_iters) max_iter_hits++;
        if (result.max_dx >= cfg.dx_clip * (1.0 - 1e-9)) clipped_steps++;
    }

    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> guard_vertices = mesh.vertices;
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> guard_velocities = mesh.velocities;
    Eigen::VectorXi guard_czm_state = mesh.czm_state;
    Eigen::VectorXd guard_damage = mesh.damage;
    Eigen::VectorXd guard_time_free = mesh.time_free;
    guard_result = result;

    if (!have_commit) {
        commit_vertices = guard_vertices;
        commit_velocities = guard_velocities;
        commit_czm_state = guard_czm_state;
        commit_damage = guard_damage;
        commit_time_free = guard_time_free;
        commit_result = guard_result;
        commit_steps = layer_steps;
    }

    int return_steps = 0;
    double platform_return_distance = 0.0;
    if (has_lift && layer_id + 1 < n_layers && std::abs(lift_step) > 0.0 && commit_steps > 0) {
        mesh.vertices = commit_vertices;
        mesh.velocities = commit_velocities;
        mesh.czm_state = commit_czm_state;
        mesh.damage = commit_damage;
        mesh.time_free = commit_time_free;

        platform_return_distance = static_cast<double>(commit_steps) * std::abs(lift_step);
        const int planned_return_steps = positive_step_count(platform_return_distance, std::abs(lift_step));
        double remaining = platform_return_distance;
        for (int r = 0; r < planned_return_steps; ++r) {
            const double step_distance = std::min(std::abs(lift_step), remaining);
            SolverConfig down_cfg = cfg;
            down_cfg.v_lift = -step_distance / std::max(std::abs(cfg.dt), 1e-12);
            down_cfg.enable_czm = false;
            commit_result = vbd::solve_lift_and_relax(mesh, down_cfg, 0.0, layer_id, lifting_top);
            return_steps++;
            total_iterations += commit_result.iterations;
            if (commit_result.iterations >= cfg.max_iters) max_iter_hits++;
            if (commit_result.max_dx >= cfg.dx_clip * (1.0 - 1e-9)) clipped_steps++;
            remaining -= step_distance;
        }
        commit_vertices = mesh.vertices;
        commit_velocities = mesh.velocities;
        commit_czm_state = mesh.czm_state;
        commit_damage = mesh.damage;
        commit_time_free = mesh.time_free;
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    py::dict info;
    info["timing_mode"] = event_windows ? "event_windows_v2" : "constant";
    info["expected_steps"] = static_cast<double>(expected_lift_steps);
    info["detach_step"] = static_cast<double>(detach_step);
    info["commit_step"] = static_cast<double>(commit_steps);
    info["guard_step"] = static_cast<double>(layer_steps);
    info["return_steps"] = static_cast<double>(return_steps);
    info["platform_return_distance"] = platform_return_distance;
    info["peak_start_step"] = static_cast<double>(
        peak_start_step > 0 ? peak_start_step :
        (expected_lift_steps > 0 && cfg.field_peak_window_steps > 0
            ? std::max(1, expected_lift_steps - cfg.field_peak_window_steps + 1)
            : 0));
    info["applied_steps"] = static_cast<double>(field_applied_steps);
    info["detach_E_z"] = e_z;
    info["peak_E_z"] = peak_e_z;
    info["cpp_solve_ms"] = elapsed_ms;
    info["python_solve_ms"] = 0.0;
    info["czm_sync_ms"] = 0.0;
    info["return_ms"] = 0.0;
    info["snapshot_ms"] = 0.0;
    info["branch_runner"] = 1.0;

    py::dict out;
    out["commit_vertices"] = commit_vertices;
    out["commit_velocities"] = commit_velocities;
    out["commit_czm_state"] = commit_czm_state;
    out["commit_damage"] = commit_damage;
    out["commit_time_free"] = commit_time_free;
    out["guard_vertices"] = guard_vertices;
    out["guard_velocities"] = guard_velocities;
    out["guard_czm_state"] = guard_czm_state;
    out["guard_damage"] = guard_damage;
    out["guard_time_free"] = guard_time_free;
    out["commit_result"] = result_to_dict(commit_result);
    out["guard_result"] = result_to_dict(guard_result);
    out["commit_steps"] = commit_steps;
    out["executed_steps"] = layer_steps;
    out["total_iterations"] = total_iterations;
    out["max_iter_hits"] = max_iter_hits;
    out["clipped_steps"] = clipped_steps;
    out["lift_max"] = lift_max;
    out["info"] = info;
    return out;
}

PYBIND11_MODULE(VBD_PYBIND_MODULE_NAME, m) {
    m.doc() = "C++ accelerated VBD solver for hydrogel DLP simulation";

    // 配置结构体
    py::class_<SolverConfig>(m, "SolverConfig")
        .def(py::init<>())
        .def_readwrite("dt", &SolverConfig::dt)
        .def_readwrite("max_iters", &SolverConfig::max_iters)
        .def_readwrite("N_stable", &SolverConfig::N_stable)
        .def_readwrite("epsilon", &SolverConfig::epsilon)
        .def_readwrite("k_d", &SolverConfig::k_d)
        .def_readwrite("rho_cheb", &SolverConfig::rho_cheb)
        .def_readwrite("mu", &SolverConfig::mu)
        .def_readwrite("kappa", &SolverConfig::kappa)
        .def_readwrite("c_shrink", &SolverConfig::c_shrink)
        .def_readwrite("q_ion", &SolverConfig::q_ion)
        .def_readwrite("g_x", &SolverConfig::g_x)
        .def_readwrite("g_y", &SolverConfig::g_y)
        .def_readwrite("g_z", &SolverConfig::g_z)
        .def_readwrite("T_max", &SolverConfig::T_max)
        .def_readwrite("K_czm", &SolverConfig::K_czm)
        .def_readwrite("delta_f", &SolverConfig::delta_f)
        .def_readwrite("node_area", &SolverConfig::node_area)
        .def_readwrite("enable_czm", &SolverConfig::enable_czm)
        .def_readwrite("z_fep", &SolverConfig::z_fep)
        .def_readwrite("C_0", &SolverConfig::C_0)
        .def_readwrite("eta", &SolverConfig::eta)
        .def_readwrite("fluid_radius", &SolverConfig::fluid_radius)
        .def_readwrite("d_min", &SolverConfig::d_min)
        .def_readwrite("d_fluid_max", &SolverConfig::d_fluid_max)
        .def_readwrite("t_fluid_max", &SolverConfig::t_fluid_max)
        .def_readwrite("v_lift", &SolverConfig::v_lift)
        .def_readwrite("layer_thickness", &SolverConfig::layer_thickness)
        .def_readwrite("dx_clip", &SolverConfig::dx_clip)
        .def_readwrite("field_detach_pre_steps", &SolverConfig::field_detach_pre_steps)
        .def_readwrite("field_detach_post_steps", &SolverConfig::field_detach_post_steps)
        .def_readwrite("field_peak_window_steps", &SolverConfig::field_peak_window_steps)
        .def_readwrite("c_init", &SolverConfig::c_init);

    // 求解函数
    m.def("solve_until_stable", &solve_until_stable_py,
          py::arg("vertices"), py::arg("velocities"), py::arg("ideal_vertices"),
          py::arg("masses"), py::arg("first_active_layer"), py::arg("is_top_surface_of_layer"), py::arg("active_mask"), py::arg("is_top_fixed"),
          py::arg("is_bottom_surface"), py::arg("is_current_bottom"),
          py::arg("czm_state"), py::arg("damage"),
          py::arg("time_free"), py::arg("tets"), py::arg("active_tet_mask"),
          py::arg("dm_inv"), py::arg("tet_volumes"), py::arg("colors"),
          py::arg("config"), py::arg("e_z"), py::arg("layer_id"),
          "Solve VBD until static equilibrium");

    m.def("solve_lift_and_relax", &solve_lift_and_relax_py,
          py::arg("vertices"), py::arg("velocities"), py::arg("ideal_vertices"),
          py::arg("masses"), py::arg("first_active_layer"), py::arg("is_top_surface_of_layer"), py::arg("active_mask"), py::arg("is_top_fixed"),
          py::arg("is_bottom_surface"), py::arg("is_current_bottom"),
          py::arg("czm_state"), py::arg("damage"),
          py::arg("time_free"), py::arg("tets"), py::arg("active_tet_mask"),
          py::arg("dm_inv"), py::arg("tet_volumes"), py::arg("colors"),
          py::arg("config"), py::arg("e_z"), py::arg("layer_id"),
          py::arg("lifting_top"),
          "Single-step lift + relax (Python side controls time loop)");

    m.def("solve_field_debug_branch", &solve_field_debug_branch_py,
          py::arg("vertices"), py::arg("velocities"), py::arg("ideal_vertices"),
          py::arg("masses"), py::arg("first_active_layer"), py::arg("is_top_surface_of_layer"), py::arg("active_mask"), py::arg("is_top_fixed"),
          py::arg("is_bottom_surface"), py::arg("is_current_bottom"),
          py::arg("czm_state"), py::arg("damage"),
          py::arg("time_free"), py::arg("tets"), py::arg("active_tet_mask"),
          py::arg("dm_inv"), py::arg("tet_volumes"), py::arg("colors"),
          py::arg("config"), py::arg("layer_id"), py::arg("e_z"),
          py::arg("lifting_top"), py::arg("expected_lift_steps"),
          py::arg("event_window_detach_step"), py::arg("peak_e_z"),
          py::arg("continue_to_peak"), py::arg("n_layers"),
          "Run one complete field-debug branch inside C++");

    m.def("debug_tet_force_hessian", &debug_tet_force_hessian_py,
          py::arg("vertices"), py::arg("dm_inv"), py::arg("rest_volume"),
          py::arg("mu"), py::arg("lam"),
          "Debug helper: single-tet elastic force and diagonal Hessian blocks");

    m.def("debug_single_tet_line_search", &debug_single_tet_line_search_py,
          py::arg("vertices"), py::arg("dm_inv"), py::arg("rest_volume"),
          py::arg("mu"), py::arg("lam"), py::arg("node_id"),
          py::arg("dx"), py::arg("floor_z"),
          "Debug helper: single-tet elastic backtracking line search");
}
