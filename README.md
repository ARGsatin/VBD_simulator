# 🧪 水凝胶 DLP VBD 仿真框架

> **Hydrogel DLP VBD Simulation Framework**  
> 基于 VBD（Vertex Block Descent）的水凝胶数字光处理打印多物理场仿真系统

---

## 📖 项目概述

本仓库提供了一个**以 Python 为主的参考框架**，用于电场辅助水凝胶 DLP 打印过程的仿真。
系统集成了多种物理效应（超弹性、内聚力断裂、流体拖曳、电场提升、重力等），
通过 VBD 求解器进行实时或近实时的求解。

### 核心流程

1. **全局共形四面体网格构建** — 含共享层间界面节点的多层网格
2. **逐层激活与保形** — 离型膜碰撞处理、继承变形几何、防穿透插值
3. **局部力与 Hessian 装配** — 惯性、超弹刚度、阻尼、CZM 软化、流体吸力、电场提升
4. **VBD 求解** — 基于图着色的顶点批次 3×3 Newton 局部迭代
5. **CZM 损伤状态机** — `FIXED → DAMAGING → FREE` 三态转换
6. **电场调控** — 支持标量 PID、底部 Z 向误差反演、RMS 守门与短窗口施加
7. **多格式输出** — NPZ 状态快照、VTU 可视化、CSV 报告、JSON 回放、`M150 E...` G-code

---

## 📁 项目结构

```
VBD_simulator/
├── README.md                          # 本文件
├── pyproject.toml                     # Python 项目配置与依赖
├── run_gui.py                         # 图形界面启动脚本
│
├── configs/                           # 配置文件
│   ├── config.yaml                    # 主配置（物理/求解器/PID 参数）
│   ├── electrode_config.json          # 电极参数
│   ├── material_hydrogel.json         # 水凝胶材料参数
│   ├── printer_dlp.json               # DLP 打印机工艺参数
│   └── solver_vbd.json                # VBD 求解器参数
│
├── src/hydrogel_vbd/                  # 主代码包
│   ├── __init__.py                    # 包初始化
│   │
│   ├── core/                          # 核心层：配置、状态、主循环
│   │   ├── config.py                  # 仿真配置数据类（YAML 加载）
│   │   ├── state.py                   # 仿真状态管理（网格/顶点数据）
│   │   └── main_loop.py               # 主仿真循环（逐层求解）
│   │
│   ├── control/                       # 控制模块
│   │   ├── field_controller.py        # PID 电场控制器
│   │   └── voltage_optimizer.py       # 电压优化器
│   │
│   ├── evaluation/                    # 评估模块
│   │   ├── metrics.py                 # 评估指标计算
│   │   └── shape_error.py             # 形状误差计算
│   │
│   ├── physics/                       # 物理层：力模型 + 本构 + 弹性能量
│   │   ├── aggregate.py               # 力聚合与 Hessian 装配
│   │   ├── czm.py                     # 内聚力模型（损伤断裂）
│   │   ├── electric.py                # 电场力
│   │   ├── fluid_drag.py              # 流体拖曳力
│   │   ├── gravity.py                 # 重力
│   │   ├── local_terms.py             # 局部物理项（弹性 + 惯性）
│   │   ├── peel.py                    # 剥离力
│   │   ├── surface_tension.py         # 表面张力
│   │   ├── hydrogel_model.py          # 水凝胶本构模型（固化度相关）
│   │   └── elastic_energy.py          # 四面体超弹性能量计算
│   │
│   ├── geometry/                      # 几何处理模块
│   │   ├── conformal_pipeline.py      # 保形网格管道
│   │   ├── layer_activator.py         # 逐层激活器
│   │   └── stl_mesher.py              # STL → 四面体网格
│   │
│   ├── gui/                           # 图形界面（PySide6）
│   │   ├── main_window.py             # 主窗口（4 步工作流 + 自动编译集成）
│   │   ├── mesh_viewer.py             # 3D 网格可视化（matplotlib 工具栏 + 边界面提取）
│   │   └── simulation_worker.py       # 仿真工作线程（QThread）
│   │
│   ├── io/                            # 输入输出模块
│   │   ├── gcode_exporter.py          # G-code 导出（含 M150 电场指令）
│   │   ├── npz_state.py               # NPZ 状态快照存取
│   │   ├── report_writer.py           # CSV/JSON 报告写入
│   │   └── vtk_writer.py              # VTK/VTU 可视化输出
│   │
│   └── solver/                        # 求解器模块
│       ├── constraints.py             # 约束处理（狄利克雷边界）
│       ├── cpp_adapter.py             # C++ 求解器适配层（自动检测/降级）
│       ├── cpp_builder.py             # C++ 自动编译模块（一键构建）
│       ├── graph_coloring.py          # 图着色（VBD 顶点分组）
│       └── vbd_solver.py              # VBD 主求解器（Chebyshev 半隐式）
│
├── cpp/                               # C++ 加速模块（可选）
│   ├── CMakeLists.txt                 # CMake 构建配置
│   ├── include/                       # 头文件
│   │   ├── types.h                    # 数据类型定义
│   │   ├── elastic_energy.h           # 弹性能量声明
│   │   ├── physics_terms.h            # 物理项声明
│   │   └── vbd_solver.h               # VBD 求解器声明
│   ├── src/                           # 实现文件
│   │   ├── elastic_energy.cpp
│   │   ├── physics_terms.cpp
│   │   └── vbd_solver.cpp
│   └── bindings/                      # pybind11 绑定
│       └── pybind_vbd.cpp
│
├── tests/                             # 测试文件（31 个测试）
│   ├── test_conformal_architecture.py
│   ├── test_io_and_main_loop.py
│   ├── test_lift_and_gui.py
│   ├── test_models_solver_control.py
│   ├── test_multiphysics_vbd_pid.py
│   ├── test_numerical_guards.py        # 数值鲁棒性测试（NaN/Inf 防护）
│   ├── test_package_and_configs.py
│   └── test_state_and_activation.py
│
├── docs/                              # 文档
│   ├── architecture/                  # 架构文档
│   │   ├── 技术栈.md
│   │   └── 伪代码.md
│   ├── guide/                         # 使用指南
│   └── superpowers/                   # 开发计划
│       └── plans/
│
├── assets/                            # 静态资源
│   └── test_models/                   # 测试用 STL 模型
│       ├── 长方体(1).STL
│       └── demo7(1).STL
│
└── outputs/                           # 仿真输出目录
```

---

## ⚙️ 配置说明

主配置文件 `configs/config.yaml` 集中管理所有物理参数、求解器参数和控制参数：

| 参数组 | 关键参数 | 说明 |
|--------|----------|------|
| **物理/材料** | `g`, `rho`, `mu`, `kappa`, `c_shrink` | 重力、密度、拉梅常数、收缩因子 |
| **CZM 断裂** | `T_max`, `K_czm`, `delta_f`, `eta`, `d_min` | 内聚强度、刚度、失效距离、损伤指数 |
| **流体拖曳** | `d_fluid_max`, `t_fluid_max`, `fluid_radius` | 最大阻尼距离、时间、作用半径 |
| **求解器** | `dt`, `epsilon`, `max_iters`, `N_stable` | 时间步长、收敛容差、最大迭代 |
| **PID 控制** | `c_init`, `err_target`, `K_p`, `K_i`, `K_d` | 初始固化度、目标误差、PID 增益 |
| **电场** | `q_ion`, `E_max`, `field_control_mode`, `field_regularization` | 离子电荷密度、最大电场强度、控制模式、反演正则 |
| **电场短窗口** | `field_detach_pre_steps`, `field_detach_post_steps`, `field_peak_window_steps` | 脱膜窗口与最高点窗口的施加步数 |
| **打印工艺** | `layer_thickness`, `z_fep`, `v_lift`, `build_axis` | 层厚、离型膜位置、提升速度、构建方向（0=X/1=Y/2=Z） |

---

## ⚡ 电场调控策略

当前实现以 **Z 向均匀电场 `E_z`** 作为第一阶段控制对象，不做 x/y 补偿，也不做多电极 FEM 电场分布求解。底层求解器仍接收单个标量 `e_z`，电场力采用现有等效体力模型：

```text
f_z = q_ion * E_z
```

### 控制模式

`SimulationConfig.field_control_mode` 支持以下模式：

| 模式 | 行为 |
|------|------|
| `scalar_pid` | 原有标量 PID 路径，根据形状误差调节单个 `E_z` |
| `bottom_z` | 使用底部节点 Z 向下垂误差反推出单个 `E_z` |
| `bottom_z_guarded` | 同时评估 baseline 与 candidate，仅在全局 RMS 不恶化时采用 candidate |

### Bottom-Z 反演

`BottomZFieldController` 的输入为当前层底部节点、目标顶点、仿真顶点和 `SimulationConfig`。控制器计算底部节点 Z 向下垂：

```text
sag_i = max(target_z_i - sim_z_i - err_target, 0)
```

然后根据 PID 项得到期望 Z 向补偿力，其中 D 项按 `dt` 计算：

```text
force_desired = K_p * sag + K_i * integral + K_d * (sag - sag_prev) / dt
```

v1 映射矩阵使用均匀电场假设：

```text
B_z = q_ion * ones((N_bottom, 1))
```

再通过带 Tikhonov 正则的最小二乘求单个 `E_z`，并裁剪到：

```text
0 <= E_z <= E_max
```

因此该控制器不会产生负向电场；当底部无节点、误差为负、或结构已被抬升到目标以上时，输出为 0。

### RMS 守门

`bottom_z_guarded` 和 GUI 的“电场调试对比”遵循保守守门原则：

```text
candidate_rms <= baseline_rms * (1 + rms_guard_tolerance) + 1e-12
```

默认容忍度为 1%。同时会比较 `max_error` 和改善状态，避免为了局部底部抬升而造成整体形状显著拉长或失真。若 candidate 未通过守门，则自动回退 no-field / scalar PID 结果。

每层结果会记录关键调试指标，例如：

- `field_no_field_rms`
- `field_with_field_rms`
- `field_no_field_max_error`
- `field_with_field_max_error`
- `field_guard_passed`
- `field_guard_reason`
- `field_effective_mode`
- `field_detach_E_z`
- `field_peak_E_z`
- `field_window_applied_steps`

### V2 短窗口施加

为避免电场在整个上提过程中持续拉伸结构，GUI field-debug 路径采用短窗口策略：

1. **脱膜窗口**：以 no-field 分支推断出的脱膜步为中心，在 `field_detach_pre_steps` 到 `field_detach_post_steps` 范围内施加 `E_detach`。
2. **最高点窗口**：在提升末端 `field_peak_window_steps` 步施加 `E_peak`，主要用于 guard 对比和形态检查。

`E_detach` 和 `E_peak` 分别由对应时刻的底部 Z 向误差自适应计算；两个窗口重叠时取较大的正向电场。窗口外 `E_z = 0`。

### GUI 调试对比

勾选 **“电场调试对比”** 后，每层会从同一初始状态分别运行：

- no-field baseline
- with-field candidate

然后逐层显示/记录 no-field 与 with-field 的 RMS、max error、底部 Z 误差、电场强度、守门状态和最终采用分支。若同时勾选 C++ 加速，GUI 会优先使用直接 C++ adapter 执行分支；失败时回退 Python 求解器。

该模式用于验证电场是否真正改善形状，计算量大于普通单路径仿真。

---

## 🚀 快速开始

### 环境要求

- Python **3.10+**
- 核心依赖：`numpy`, `matplotlib`, `scipy>=1.9`, `trimesh`, `rtree`
- GUI（可选）：`pyside6`
- 网格生成（可选）：`gmsh>=4.11`
- C++ 加速（可选）：MSVC 2022 + CMake ≥ 3.18 + pybind11

### 安装

```powershell
# 克隆仓库
git clone https://github.com/ARGsatin/VBD_simulator.git
cd VBD_simulator

# 可选：创建虚拟环境
python -m venv venv
venv\Scripts\activate

# 安装依赖
pip install -e .
```

### 运行仿真

```powershell
# 方式一：命令行动态引入
python -c "import sys; sys.path.insert(0,'src'); from hydrogel_vbd.core.main_loop import run_demo; run_demo(layers=3, output='outputs/demo')"

# 方式二：直接运行主循环
python src/hydrogel_vbd/core/main_loop.py --layers 3

# 方式三：启动图形界面
python run_gui.py
```

### 运行测试

```powershell
# 运行全部测试（31 个）
python -m pytest tests/ -v

# 运行单个测试文件
python -m pytest tests/test_state_and_activation.py -v
```

---

## 📊 输出文件

仿真输出保存在 `outputs/<名称>/` 目录下，包含：

| 子目录/文件 | 格式 | 说明 |
|-------------|------|------|
| `states/` | `.npz` | 每层求解后的完整状态快照 |
| `vtk/` | `.vtu` | ParaView 可读的网格可视化文件 |
| `reports/` | `.csv`, `.json` | 误差指标、收敛历史、PID 日志 |
| `gcode/` | `.gcode` | 含 `M150 E...` 电场指令的打印 G-code |

CSV 报告格式：
```csv
layer_id, err_avg, E_z, kinetic_energy, stable_steps, max_dx, all_free
```

---

## 🔌 C++ 加速模块

项目包含可选的 C++17/Eigen/pybind11 加速核心，可将关键计算路径加速数十倍。
编译后的 `.pyd` 模块会被 Python 自动加载；若未编译则透明回退到纯 Python 实现。

### 环境依赖（编译所需）

| 工具 | 版本要求 | 安装方式 |
|------|----------|----------|
| **MSVC 编译器** | VS 2022 (v19.4+) | [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)，勾选"使用 C++ 的桌面开发"工作负载 |
| **CMake** | ≥ 3.18 | `winget install Kitware.CMake` 或 [cmake.org](https://cmake.org/download/) |
| **Python 开发库** | 3.10+ | 随 Python 安装包自带（`python.org` 版本已包含 `python3.lib`） |
| **pybind11** | ≥ 2.11 | `pip install pybind11`（编译前会自动安装） |
| **Eigen** | 3.4.0 | CMake 自动从 GitLab 下载（无需手动安装） |
| **OpenMP** | 随 MSVC 附带 | MSVC 编译器自带的 `/openmp` 支持 |

> **注意**：使用 `python.org` 安装的 Python 即可正常编译。避免使用 Microsoft Store 版 Python（缺少 `python3.lib` 可能导致链接失败）。

### 自动编译（推荐）

点击 GUI 的 **"运行仿真"** 按钮时，系统会自动检测 C++ 模块是否已编译：

1. **若 .pyd 存在** → 直接加载 C++ 加速求解器，日志显示 `[info] 使用 C++ 加速求解器`
2. **若 .pyd 不存在但编译工具就绪** → 自动在后台线程执行 `cmake configure + build`，日志实时输出编译进度，完成后自动继续仿真
3. **若缺少编译工具** → 日志提示缺失项（如 "未找到 MSVC 编译器"），回退到 Python 参考求解器

```text
# 自动编译时的 GUI 日志示例
[build] C++ 求解器未编译，开始自动编译 …
[build] 前置条件就绪，正在后台编译 …
[build]   检查 pybind11 ... [OK]
[build]   cmake configure ... [OK]
[build]   cmake build (Release) ...
[build]   部署 .pyd -> src/ ...
[build] ✓ 编译成功 (37.0s)
[info] 使用 C++ 加速求解器
```

### 手动编译

如需手动编译（例如调试构建或更换编译器），执行以下命令：

```powershell
# 确保已在项目 venv 中
venv\Scripts\activate
pip install pybind11

# 进入 cpp 目录，创建 build 子目录
cd cpp
rm -r -force build          # 清理旧构建（如有）
mkdir build && cd build

# CMake 配置（指定 Visual Studio 2022 生成器 + x64 架构）
cmake .. -G "Visual Studio 17 2022" -A x64 `
  -Dpybind11_DIR="$env:VIRTUAL_ENV\Lib\site-packages\pybind11\share\cmake\pybind11" `
  -DPython_ROOT_DIR="$env:VIRTUAL_ENV"

# 编译 Release 版本
cmake --build . --config Release

# 将产物复制到 src/ 目录（Python 自动发现）
copy Release\hydrogel_vbd_cpp.*.pyd ..\src\
```

编译成功后，产物为 `src/hydrogel_vbd_cpp.cp3XX-win_amd64.pyd`（`cp3XX` 对应 Python 版本）。

### 踩坑记录

| 问题 | 原因 | 解决 |
|------|------|------|
| pybind11 下载超时 | GitHub `git clone` 在墙内不稳定 | CMakeLists 已配置 Gitee 镜像 + `find_package` 回退 |
| SHA256 哈希不匹配 | GitLab 重新生成 tarball | 去掉 `URL_HASH` 校验 |
| `Eigen::Map` 类型错误 | 函数签名要求 `Map` 但传入 `Matrix` | 改用 `Eigen::Ref`（兼容 Matrix/Map） |
| CMake 找到 MSYS2 Python | PATH 中 MSYS2 优先级高于 venv | 显式指定 `-DPython_ROOT_DIR` |
| Python 库未找到 | pybind11 旧版 `FindPythonLibs` 不支持 3.13 | 设置 `PYBIND11_FINDPYTHON ON` 使用现代 FindPython |
| `.pyd` 无法导入 | 模块名 `vbd_solver_cpp` 与 Python 端 `hydrogel_vbd_cpp` 不一致 | 统一使用 `hydrogel_vbd_cpp` |
| Unicode 编码错误 | cmake 输出含中文，GBK 无法解码 | subprocess 显式 `encoding="utf-8", errors="replace"` |

---

## 🧩 架构设计

### 设计原则

- **模块化**：每个物理效应为独立的力模块，易于扩展和替换
- **配置驱动**：所有参数集中于 `config.yaml`，便于调参和复现
- **Python 优先**：参考实现用 Python 编写，接口为 C++ 移植预留
- **测试覆盖**：31 个测试覆盖核心路径（状态管理、求解收敛、力向量、IO 往返、数值稳定性）

### 关键类

| 类 | 模块 | 职责 |
|----|------|------|
| `SimulationConfig` | `core.config` | 全局参数数据中心 |
| `MeshState` | `core.state` | 网格顶点/四面体/掩码管理 |
| `PythonReferenceVBDSolver` | `solver.vbd_solver` | Chebyshev 半隐式 VBD 求解 |
| `FieldController` | `control.field_controller` | PID 电场调节 |
| `LayerActivator` | `geometry.layer_activator` | 新层激活与保形 |

---

## 📝 技术栈

- **语言**：Python 3.10+（核心），C++17（加速）
- **数值计算**：NumPy, SciPy
- **网格处理**：trimesh, PyVista
- **GUI**：PySide6（Qt for Python），含自动编译集成
- **C++ 绑定**：pybind11，支持编译后热加载
- **线性代数**：Eigen 3.4（C++ 端）
- **构建系统**：CMake + MSBuild，GUI 内一键自动编译
- **测试**：pytest

---

## 📄 许可

待定
