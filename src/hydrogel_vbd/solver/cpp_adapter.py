# -*- coding: utf-8 -*-
"""C++ 加速求解器适配层。

将 SimulationConfig 和 MeshState 转换为 C++ pybind11 模块所需的格式，
提供与 PythonReferenceVBDSolver 相同的接口。

若 C++ 模块不可用（未编译或平台不兼容），则优雅降级。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from hydrogel_vbd.core.config import SimulationConfig
from hydrogel_vbd.core.state import MeshState
from hydrogel_vbd.solver.vbd_solver import VBDSolveResult

_CPP_AVAILABLE = False
_CPP_IMPORT_ERROR: str | None = None

DX_CLIP_MAX = 2.0e-3
DX_CLIP_MIN = 5.0e-6
DX_CLIP_LAYER_FRACTION = 0.5
DX_CLIP_LIFT_STEP_FACTOR = 2.0


def _prefer_newer_cpp_build_dir() -> None:
    """Prefer freshly built development binaries over stale deployed pyd files."""
    try:
        project_root = Path(__file__).resolve().parents[3]
    except IndexError:
        return
    src_dir = project_root / "src"
    cpp_dir = project_root / "cpp"
    if not cpp_dir.exists():
        return
    build_pyds = []
    for build_dir in cpp_dir.glob("build*/Release"):
        build_pyds.extend(build_dir.glob("hydrogel_vbd_cpp*.pyd"))
    if not build_pyds:
        return
    for build_pyd in sorted(build_pyds, key=lambda p: p.stat().st_mtime):
        src_pyd = src_dir / build_pyd.name
        if not src_pyd.exists() or build_pyd.stat().st_mtime > src_pyd.stat().st_mtime:
            build_path = str(build_pyd.parent)
            if build_path not in sys.path:
                sys.path.insert(0, build_path)


_prefer_newer_cpp_build_dir()

# ── QThread 安全：在加载 C++ 模块前禁用 OpenMP 多线程 ──
# MSVC 的 vcomp.dll 在 QThread 内创建 Win32 工作线程会导致 segfault。
# OMP_NUM_THREADS 必须在 vcomp.dll 加载前设置（DLL 加载时缓存此值）。
# 设置 HYDROGEL_VBD_OMP=1 可恢复多线程（CLI/批处理场景）。
# 默认加载 _qt 变体（无 OpenMP），仅当 HYDROGEL_VBD_OMP=1 时加载标准变体。
_USE_OMP = os.environ.get("HYDROGEL_VBD_OMP") == "1"
if not _USE_OMP:
    os.environ.setdefault("OMP_NUM_THREADS", "1")

_CPP_AVAILABLE = False
_CPP_IMPORT_ERROR: str | None = None

if _USE_OMP:
    # CLI/批处理：加载标准 OpenMP 多线程版本
    try:
        import hydrogel_vbd_cpp  # type: ignore[import-untyped]
        _CPP_AVAILABLE = True
        _CPP_MODULE_NAME = "hydrogel_vbd_cpp"
    except ImportError as exc:
        _CPP_IMPORT_ERROR = str(exc)
else:
    # GUI：优先加载无 OpenMP 版本（QThread 安全），回退到标准版本
    try:
        import hydrogel_vbd_cpp_qt as _cpp_mod  # type: ignore[import-untyped]
        hydrogel_vbd_cpp = _cpp_mod  # 统一别名
        _CPP_AVAILABLE = True
        _CPP_MODULE_NAME = "hydrogel_vbd_cpp_qt"
    except ImportError:
        try:
            import hydrogel_vbd_cpp  # type: ignore[import-untyped]
            _CPP_AVAILABLE = True
            _CPP_MODULE_NAME = "hydrogel_vbd_cpp"
        except ImportError as exc:
            _CPP_IMPORT_ERROR = str(exc)


def is_cpp_available() -> bool:
    """检查 C++ solver 模块是否已编译并可导入。"""
    return _CPP_AVAILABLE


def get_import_error() -> str | None:
    """获取 C++ 模块导入失败的错误信息（仅调试/日志用途）。"""
    return _CPP_IMPORT_ERROR


def cpp_module_info() -> str:
    """Return the loaded C++ module name and binary path for diagnostics."""
    if not _CPP_AVAILABLE:
        return f"unavailable: {_CPP_IMPORT_ERROR}"
    module_path = getattr(hydrogel_vbd_cpp, "__file__", "")
    return f"{_CPP_MODULE_NAME} {module_path}"


def solver_dx_clip(cfg: SimulationConfig) -> float:
    """Return a per-call displacement clip scaled to the print layer size."""
    layer_thickness = abs(float(getattr(cfg, "layer_thickness", 0.0)))
    lift_step = abs(
        float(getattr(cfg, "v_lift", 0.0)) * float(getattr(cfg, "dt", 0.0))
    )
    scaled = max(
        DX_CLIP_MIN,
        DX_CLIP_LAYER_FRACTION * layer_thickness,
        DX_CLIP_LIFT_STEP_FACTOR * lift_step,
    )
    return min(DX_CLIP_MAX, scaled)


def refresh_availability() -> bool:
    """编译后重新检测 C++ 模块（支持热加载，无需重启进程）。

    若模块之前导入失败，编译后调用此函数可重新尝试导入。
    若模块已导入，则 reload 以使用最新编译的版本。

    Returns
    -------
    bool
        True 表示 C++ 模块当前可用。
    """
    global _CPP_AVAILABLE, _CPP_IMPORT_ERROR, hydrogel_vbd_cpp
    import importlib

    if _USE_OMP:
        try:
            if "hydrogel_vbd_cpp" in sys.modules:
                importlib.reload(sys.modules["hydrogel_vbd_cpp"])
            else:
                import hydrogel_vbd_cpp
            _CPP_AVAILABLE = True
            _CPP_IMPORT_ERROR = None
            _CPP_MODULE_NAME = "hydrogel_vbd_cpp"
        except ImportError as exc:
            _CPP_AVAILABLE = False
            _CPP_IMPORT_ERROR = str(exc)
    else:
        try:
            if "hydrogel_vbd_cpp_qt" in sys.modules:
                importlib.reload(sys.modules["hydrogel_vbd_cpp_qt"])
            else:
                import hydrogel_vbd_cpp_qt as _cpp_mod
            hydrogel_vbd_cpp = _cpp_mod  # 统一别名
            _CPP_AVAILABLE = True
            _CPP_IMPORT_ERROR = None
            _CPP_MODULE_NAME = "hydrogel_vbd_cpp_qt"
        except ImportError:
            try:
                import hydrogel_vbd_cpp
                _CPP_AVAILABLE = True
                _CPP_IMPORT_ERROR = None
                _CPP_MODULE_NAME = "hydrogel_vbd_cpp"
            except ImportError as exc:
                _CPP_AVAILABLE = False
                _CPP_IMPORT_ERROR = str(exc)
    return _CPP_AVAILABLE


def _build_cpp_config(cfg: SimulationConfig) -> Any:
    """将 Python SimulationConfig 转换为 C++ SolverConfig。"""
    cpp_cfg = hydrogel_vbd_cpp.SolverConfig()
    cpp_cfg.dt = cfg.dt
    cpp_cfg.max_iters = cfg.max_iters
    cpp_cfg.N_stable = cfg.N_stable
    cpp_cfg.epsilon = cfg.epsilon
    cpp_cfg.k_d = cfg.k_d
    cpp_cfg.rho_cheb = cfg.rho_cheb
    cpp_cfg.mu = cfg.mu
    cpp_cfg.kappa = cfg.kappa
    cpp_cfg.c_shrink = cfg.c_shrink
    cpp_cfg.q_ion = cfg.q_ion
    cpp_cfg.g_x = cfg.g[0]
    cpp_cfg.g_y = cfg.g[1]
    cpp_cfg.g_z = cfg.g[2]
    cpp_cfg.T_max = cfg.T_max
    cpp_cfg.K_czm = cfg.K_czm
    cpp_cfg.delta_f = cfg.delta_f
    cpp_cfg.node_area = cfg.node_area
    cpp_cfg.enable_czm = cfg.enable_czm
    cpp_cfg.z_fep = cfg.z_fep
    cpp_cfg.C_0 = cfg.C_0
    cpp_cfg.eta = cfg.eta
    cpp_cfg.fluid_radius = cfg.fluid_radius
    cpp_cfg.d_min = cfg.d_min
    cpp_cfg.d_fluid_max = cfg.d_fluid_max
    cpp_cfg.t_fluid_max = cfg.t_fluid_max
    cpp_cfg.v_lift = cfg.v_lift
    cpp_cfg.layer_thickness = cfg.layer_thickness
    cpp_cfg.dx_clip = solver_dx_clip(cfg)
    cpp_cfg.c_init = cfg.c_init
    return cpp_cfg


# ── 数组校验（在传给 C++ Eigen::Map 前检查，防止静默崩溃）──

def _check_shape(name: str, arr: np.ndarray, expected: tuple) -> None:
    if arr.shape != expected:
        raise ValueError(f"{name}: expected shape {expected}, got {arr.shape}")


def _check_contiguous(name: str, arr: np.ndarray) -> None:
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError(
            f"{name} is not C-contiguous (strides={arr.strides}). "
            f"Use np.ascontiguousarray() before passing to C++."
        )


def _ensure_mesh_int32_array(mesh: MeshState, attr: str) -> np.ndarray:
    """确保 C++ 会原地写入的整型数组持久挂在 mesh 上。"""
    arr = getattr(mesh, attr)
    prepared = np.ascontiguousarray(arr, dtype=np.int32)
    if prepared is not arr:
        setattr(mesh, attr, prepared)
    return prepared


def _validate_arrays(
    vertices: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    first_active_layer: np.ndarray,
    is_top_surface_of_layer: np.ndarray,
    active_vertex_mask: np.ndarray,
    is_top_fixed: np.ndarray,
    is_bottom_surface: np.ndarray,
    is_current_bottom: np.ndarray,
    czm_state: np.ndarray,
    damage: np.ndarray,
    time_free: np.ndarray,
    tets: np.ndarray,
    active_tet_mask: np.ndarray,
    dm_inv: np.ndarray,
    tet_volumes: np.ndarray,
    colors: np.ndarray,
) -> tuple[int, int]:
    """校验所有传入 C++ 的数组，防止 Eigen::Map 静默崩溃。

    Returns (nV, nT) on success.
    """
    nV = vertices.shape[0]
    nT = tets.shape[0]

    # ── 形状一致性 ──
    _check_shape("vertices", vertices, (nV, 3))
    _check_shape("velocities", velocities, (nV, 3))
    _check_shape("masses", masses, (nV,))
    _check_shape("first_active_layer", first_active_layer, (nV,))
    _check_shape("is_top_surface_of_layer", is_top_surface_of_layer, (nV,))
    _check_shape("active_vertex_mask", active_vertex_mask, (nV,))
    _check_shape("is_top_fixed", is_top_fixed, (nV,))
    _check_shape("is_bottom_surface", is_bottom_surface, (nV,))
    _check_shape("is_current_bottom", is_current_bottom, (nV,))
    _check_shape("czm_state", czm_state, (nV,))
    _check_shape("damage", damage, (nV,))
    _check_shape("time_free", time_free, (nV,))
    _check_shape("tets", tets, (nT, 4))
    _check_shape("active_tet_mask", active_tet_mask, (nT,))
    _check_shape("tet_volumes", tet_volumes, (nT,))
    _check_shape("colors", colors, (nV,))
    # dm_inv: (T,3,3) 和 (T,9) 内存布局等价，都接受
    if dm_inv.ndim == 3:
        if dm_inv.shape != (nT, 3, 3):
            raise ValueError(f"dm_inv shape must be ({nT}, 3, 3), got {dm_inv.shape}")
    elif dm_inv.ndim == 2:
        if dm_inv.shape != (nT, 9):
            raise ValueError(f"dm_inv shape must be ({nT}, 9), got {dm_inv.shape}")
    else:
        raise ValueError(f"dm_inv must be 2D or 3D, got {dm_inv.ndim}D")

    # ── C 连续检查（Eigen::Map<..., RowMajor> 要求）──
    for name, arr in [
        ("vertices", vertices), ("velocities", velocities),
        ("masses", masses), ("first_active_layer", first_active_layer),
        ("is_top_surface_of_layer", is_top_surface_of_layer),
        ("active_vertex_mask", active_vertex_mask),
        ("is_top_fixed", is_top_fixed), ("is_bottom_surface", is_bottom_surface),
        ("is_current_bottom", is_current_bottom),
        ("czm_state", czm_state), ("damage", damage),
        ("time_free", time_free), ("tets", tets),
        ("active_tet_mask", active_tet_mask),
        ("dm_inv", dm_inv), ("tet_volumes", tet_volumes),
        ("colors", colors),
    ]:
        _check_contiguous(name, arr)

    # 可写 float64 数组（C++ 原地修改，dtype 不可静默转换）
    for name, arr in [
        ("vertices", vertices), ("velocities", velocities),
        ("damage", damage), ("time_free", time_free),
    ]:
        if arr.dtype != np.float64:
            raise TypeError(f"{name} must be float64 (C++ writes in-place), got {arr.dtype}")

    return nV, nT


def _current_bottom_mask(mesh: MeshState, layer_id: int) -> np.ndarray:
    """Mask for the interface that is currently attached to the FEP."""
    mask = np.zeros(mesh.vertices.shape[0], dtype=bool)
    bottom_nodes = mesh.bottom_nodes(int(layer_id))
    if bottom_nodes.size:
        mask[bottom_nodes] = True
    elif not np.any(mesh.is_top_surface_of_layer >= 0):
        mask = np.asarray(mesh.is_bottom_surface, dtype=bool).copy()
    mask &= mesh.active_vertex_mask
    return np.ascontiguousarray(mask, dtype=bool)


def _hit_clipped_max_iters(result_dict: dict[str, Any], config: SimulationConfig) -> bool:
    dx_clip = solver_dx_clip(config)
    return (
        int(result_dict["iterations"]) >= int(config.max_iters)
        and float(result_dict["max_dx"]) >= dx_clip * (1.0 - 1e-9)
    )


def solve_until_stable(
    mesh: MeshState,
    config: SimulationConfig,
    e_z: float,
    layer_id: int,
) -> VBDSolveResult:
    """C++ 加速的静力平衡求解（与 PythonReferenceVBDSolver 接口一致）。

    Parameters
    ----------
    mesh : MeshState
        当前网格状态（原地修改）。
    config : SimulationConfig
        物理/求解器参数。
    e_z : float
        均匀电场强度 (V/m)。
    layer_id : int
        当前层编号。

    Returns
    -------
    VBDSolveResult
    """
    if not _CPP_AVAILABLE:
        raise ImportError(
            f"C++ solver 不可用。请编译 C++ 模块后重试。\n"
            f"导入错误: {_CPP_IMPORT_ERROR}"
        )

    # ── Dtype 标准化（numpy 默认 int64，但 C++ 需要 int32/bool）──
    # czm_state 是 C++ 原地写入状态，必须持久挂回 mesh，不能用临时拷贝。
    _czm = _ensure_mesh_int32_array(mesh, "czm_state")
    _tets = np.ascontiguousarray(mesh.tets, dtype=np.int32)
    _colors = np.ascontiguousarray(mesh.colors, dtype=np.int32)
    _first_active_layer = np.ascontiguousarray(mesh.first_active_layer, dtype=np.int32)
    _surface_layers = np.ascontiguousarray(mesh.is_top_surface_of_layer, dtype=np.int32)
    _bs = np.ascontiguousarray(mesh.is_bottom_surface, dtype=bool)
    _current_bottom = _current_bottom_mask(mesh, layer_id)

    # ── 预检：数组 shape/contiguity ──
    _validate_arrays(
        mesh.vertices, mesh.velocities, mesh.masses,
        _first_active_layer, _surface_layers,
        mesh.active_vertex_mask, mesh.is_top_fixed,
        _bs, _current_bottom, _czm, mesh.damage, mesh.time_free,
        _tets, mesh.active_tet_mask,
        mesh.dm_inv, mesh.tet_volumes, _colors,
    )

    cpp_cfg = _build_cpp_config(config)

    result_dict = hydrogel_vbd_cpp.solve_until_stable(
        mesh.vertices,
        mesh.velocities,
        mesh.ideal_vertices,
        mesh.masses,
        _first_active_layer,
        _surface_layers,
        mesh.active_vertex_mask,
        mesh.is_top_fixed,
        _bs,
        _current_bottom,
        _czm,
        mesh.damage,
        mesh.time_free,
        _tets,
        mesh.active_tet_mask,
        mesh.dm_inv,
        mesh.tet_volumes,
        _colors,
        cpp_cfg,
        e_z,
        layer_id,
    )

    return VBDSolveResult(
        x=mesh.vertices,
        v=mesh.velocities,
        iterations=result_dict["iterations"],
        max_dx=result_dict["max_dx"],
        kinetic_energy=result_dict["kinetic_energy"],
        stable_steps=result_dict["stable_steps"],
        all_free=result_dict["all_free"],
        chebyshev_skipped_damaging=result_dict["chebyshev_skipped_damaging"],
    )


def solve_lift_and_relax(
    mesh: MeshState,
    config: SimulationConfig,
    e_z: float,
    layer_id: int,
    lifting_top: np.ndarray,
) -> VBDSolveResult:
    """C++ 加速的单步提升-静平衡求解（单次调用，不再内嵌循环）。

    Parameters
    ----------
    mesh : MeshState
        当前网格状态（原地修改）。
    config : SimulationConfig
        物理/求解器参数。
    e_z : float
        均匀电场强度 (V/m)。
    layer_id : int
        当前层编号。
    lifting_top : np.ndarray, shape (K,)
        平台提升阶段需要抬升的顶层节点索引。

    Returns
    -------
    VBDSolveResult
    """
    if not _CPP_AVAILABLE:
        raise ImportError(
            f"C++ solver 不可用。请编译 C++ 模块后重试。\n"
            f"导入错误: {_CPP_IMPORT_ERROR}"
        )

    # ── Dtype 标准化（numpy 默认 int64，C++ 需要 int32/bool）──
    # czm_state 是 C++ 原地写入状态，必须持久挂回 mesh，不能用临时拷贝。
    _czm = _ensure_mesh_int32_array(mesh, "czm_state")
    _tets = np.ascontiguousarray(mesh.tets, dtype=np.int32)
    _colors = np.ascontiguousarray(mesh.colors, dtype=np.int32)
    _first_active_layer = np.ascontiguousarray(mesh.first_active_layer, dtype=np.int32)
    _surface_layers = np.ascontiguousarray(mesh.is_top_surface_of_layer, dtype=np.int32)
    _bs = np.ascontiguousarray(mesh.is_bottom_surface, dtype=bool)
    _current_bottom = _current_bottom_mask(mesh, layer_id)

    # ── 预检：数组 shape/contiguity ──
    _validate_arrays(
        mesh.vertices, mesh.velocities, mesh.masses,
        _first_active_layer, _surface_layers,
        mesh.active_vertex_mask, mesh.is_top_fixed,
        _bs, _current_bottom, _czm, mesh.damage, mesh.time_free,
        _tets, mesh.active_tet_mask,
        mesh.dm_inv, mesh.tet_volumes, _colors,
    )

    cpp_cfg = _build_cpp_config(config)
    lifting_top_list = [int(x) for x in lifting_top]
    x_before = mesh.vertices.copy()
    v_before = mesh.velocities.copy()

    result_dict = hydrogel_vbd_cpp.solve_lift_and_relax(
        mesh.vertices,
        mesh.velocities,
        mesh.ideal_vertices,
        mesh.masses,
        _first_active_layer,
        _surface_layers,
        mesh.active_vertex_mask,
        mesh.is_top_fixed,
        _bs,
        _current_bottom,
        _czm,
        mesh.damage,
        mesh.time_free,
        _tets,
        mesh.active_tet_mask,
        mesh.dm_inv,
        mesh.tet_volumes,
        _colors,
        cpp_cfg,
        e_z,
        layer_id,
        lifting_top_list,
    )
    if _hit_clipped_max_iters(result_dict, config):
        mesh.vertices[:] = x_before
        mesh.velocities[:] = v_before
        lift_step = float(config.v_lift) * float(config.dt)
        valid_top = np.asarray(lifting_top, dtype=int)
        valid_top = valid_top[
            (valid_top >= 0)
            & (valid_top < mesh.vertices.shape[0])
            & mesh.active_vertex_mask[valid_top]
        ]
        mesh.vertices[valid_top, 2] += lift_step
        mesh.velocities[valid_top] = 0.0

    return VBDSolveResult(
        x=mesh.vertices,
        v=mesh.velocities,
        iterations=result_dict["iterations"],
        max_dx=result_dict["max_dx"],
        kinetic_energy=result_dict["kinetic_energy"],
        stable_steps=result_dict["stable_steps"],
        all_free=result_dict["all_free"],
        chebyshev_skipped_damaging=result_dict["chebyshev_skipped_damaging"],
    )
