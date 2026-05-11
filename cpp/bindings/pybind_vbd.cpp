#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "types.h"
#include "vbd_solver.h"

namespace py = pybind11;
using namespace vbd;

// ============================================================================
// Python 侧 solver 类封装：直接操作 MeshState 内存
// ============================================================================

// 包装函数：从 Python NumPy 数组构建 MeshData 并调用 C++ 求解器
static py::dict solve_until_stable_py(
    py::array_t<double> vertices_in,     // (N, 3) 可写
    py::array_t<double> velocities,      // (N, 3) 可写
    py::array_t<double> ideal_vertices,  // (N, 3) 只读
    py::array_t<double> masses,          // (N,) 只读
    py::array_t<bool> active_mask,       // (N,) 只读
    py::array_t<bool> is_top_fixed,      // (N,) 只读
    py::array_t<bool> is_bottom_surface, // (N,) 只读
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
    auto buf_act = active_mask.request();
    auto buf_tf = is_top_fixed.request();
    auto buf_bs = is_bottom_surface.request();
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
    Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> act_map(
        static_cast<bool*>(buf_act.ptr), nV);
    Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> tf_map(
        static_cast<bool*>(buf_tf.ptr), nV);
    Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> bs_map(
        static_cast<bool*>(buf_bs.ptr), nV);
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
        act_map, tf_map, bs_map, czm_map,
        dmg_map, tf_t_map,
        tet_map, at_map, dmi_map, tv_map,
        col_map);

    VBDSolveResult result = vbd::solve_until_stable(mesh, cfg, e_z, layer_id);

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
    py::array_t<bool> active_mask,
    py::array_t<bool> is_top_fixed,
    py::array_t<bool> is_bottom_surface,
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
    auto buf_act = active_mask.request();
    auto buf_tf = is_top_fixed.request();
    auto buf_bs = is_bottom_surface.request();
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
    Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> act_map(
        static_cast<bool*>(buf_act.ptr), nV);
    Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> tf_map(
        static_cast<bool*>(buf_tf.ptr), nV);
    Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> bs_map(
        static_cast<bool*>(buf_bs.ptr), nV);
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
        act_map, tf_map, bs_map, czm_map,
        dmg_map, tf_t_map,
        tet_map, at_map, dmi_map, tv_map,
        col_map);

    VBDSolveResult result = vbd::solve_lift_and_relax(mesh, cfg, e_z, layer_id, lifting_top);

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
PYBIND11_MODULE(vbd_solver_cpp, m) {
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
        .def_readwrite("z_fep", &SolverConfig::z_fep)
        .def_readwrite("C_0", &SolverConfig::C_0)
        .def_readwrite("eta", &SolverConfig::eta)
        .def_readwrite("fluid_radius", &SolverConfig::fluid_radius)
        .def_readwrite("d_min", &SolverConfig::d_min)
        .def_readwrite("d_fluid_max", &SolverConfig::d_fluid_max)
        .def_readwrite("t_fluid_max", &SolverConfig::t_fluid_max)
        .def_readwrite("v_lift", &SolverConfig::v_lift)
        .def_readwrite("c_init", &SolverConfig::c_init);

    // 求解函数
    m.def("solve_until_stable", &solve_until_stable_py,
          py::arg("vertices"), py::arg("velocities"), py::arg("ideal_vertices"),
          py::arg("masses"), py::arg("active_mask"), py::arg("is_top_fixed"),
          py::arg("is_bottom_surface"), py::arg("czm_state"), py::arg("damage"),
          py::arg("time_free"), py::arg("tets"), py::arg("active_tet_mask"),
          py::arg("dm_inv"), py::arg("tet_volumes"), py::arg("colors"),
          py::arg("config"), py::arg("e_z"), py::arg("layer_id"),
          "Solve VBD until static equilibrium");

    m.def("solve_lift_and_relax", &solve_lift_and_relax_py,
          py::arg("vertices"), py::arg("velocities"), py::arg("ideal_vertices"),
          py::arg("masses"), py::arg("active_mask"), py::arg("is_top_fixed"),
          py::arg("is_bottom_surface"), py::arg("czm_state"), py::arg("damage"),
          py::arg("time_free"), py::arg("tets"), py::arg("active_tet_mask"),
          py::arg("dm_inv"), py::arg("tet_volumes"), py::arg("colors"),
          py::arg("config"), py::arg("e_z"), py::arg("layer_id"),
          py::arg("lifting_top"),
          "Single-step lift + relax (Python side controls time loop)");
}
