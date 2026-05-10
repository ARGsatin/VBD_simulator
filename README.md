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
6. **PID 控制电场** — 评估底部节点平均垂度并自动调节 `E_z`
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
│   ├── config.py                      # 仿真配置数据类（YAML 加载）
│   ├── state.py                       # 仿真状态管理（网格/顶点数据）
│   ├── main_loop.py                   # 主仿真循环（逐层求解）
│   │
│   ├── control/                       # 控制模块
│   │   ├── field_controller.py        # PID 电场控制器
│   │   └── voltage_optimizer.py       # 电压优化器
│   │
│   ├── evaluation/                    # 评估模块
│   │   ├── metrics.py                 # 评估指标计算
│   │   └── shape_error.py             # 形状误差计算
│   │
│   ├── forces/                        # 物理力模型（9 种力）
│   │   ├── aggregate.py               # 力聚合与 Hessian 装配
│   │   ├── czm.py                     # 内聚力模型（损伤断裂）
│   │   ├── electric.py                # 电场力
│   │   ├── fluid_drag.py              # 流体拖曳力
│   │   ├── gravity.py                 # 重力
│   │   ├── local_terms.py             # 局部物理项（弹性 + 惯性）
│   │   ├── peel.py                    # 剥离力
│   │   └── surface_tension.py         # 表面张力
│   │
│   ├── geometry/                      # 几何处理模块
│   │   ├── conformal_pipeline.py      # 保形网格管道
│   │   ├── layer_activator.py         # 逐层激活器
│   │   └── stl_mesher.py              # STL → 四面体网格
│   │
│   ├── gui/                           # 图形界面（PySide6）
│   │   ├── main_window.py             # 主窗口
│   │   └── mesh_viewer.py             # 网格可视化组件
│   │
│   ├── io/                            # 输入输出模块
│   │   ├── gcode_exporter.py          # G-code 导出（含 M150 电场指令）
│   │   ├── npz_state.py               # NPZ 状态快照存取
│   │   ├── report_writer.py           # CSV/JSON 报告写入
│   │   └── vtk_writer.py              # VTK/VTU 可视化输出
│   │
│   ├── material/                      # 材料模块
│   │   └── hydrogel_model.py          # 水凝胶本构模型（固化度相关）
│   │
│   └── solver/                        # 求解器模块
│       ├── constraints.py             # 约束处理（狄利克雷边界）
│       ├── elastic_energy.py          # 超弹性能量计算
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
├── tests/                             # 测试文件（29 个测试）
│   ├── test_conformal_architecture.py
│   ├── test_io_and_main_loop.py
│   ├── test_lift_and_gui.py
│   ├── test_models_solver_control.py
│   ├── test_multiphysics_vbd_pid.py
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
├── 测试模型/                          # 测试用 STL 模型
│   ├── 长方体(1).STL
│   └── demo7(1).STL
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
| **电场** | `q_ion`, `E_max` | 离子电荷密度、最大电场强度 |
| **打印工艺** | `layer_thickness`, `z_fep`, `v_lift` | 层厚、离型膜位置、提升速度 |

---

## 🚀 快速开始

### 环境要求

- Python **3.10+**
- 依赖包：`numpy`, `scipy`, `trimesh`, `pyvista`, `pyside6`, `pyyaml`, `pytest`

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
# 方式一：直接调用函数
python -c "import sys; sys.path.insert(0,'src'); from hydrogel_vbd.main_loop import run_demo; run_demo(layers=3, output='outputs/demo')"

# 方式二：作为模块运行
python -m hydrogel_vbd.main_loop --layers 3 --output outputs/demo

# 方式三：启动图形界面
python run_gui.py
```

### 运行测试

```powershell
# 运行全部测试（29 个）
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

### 编译

```powershell
cd cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
cmake --install .
```

编译后，Python 端会自动检测并使用 C++ 加速版本；若未检测到则回退到纯 Python 实现。

---

## 🧩 架构设计

### 设计原则

- **模块化**：每个物理效应为独立的力模块，易于扩展和替换
- **配置驱动**：所有参数集中于 `config.yaml`，便于调参和复现
- **Python 优先**：参考实现用 Python 编写，接口为 C++ 移植预留
- **测试覆盖**：29 个测试覆盖核心路径（状态管理、求解收敛、力向量、IO 往返）

### 关键类

| 类 | 模块 | 职责 |
|----|------|------|
| `SimulationConfig` | `config` | 全局参数数据中心 |
| `MeshState` | `state` | 网格顶点/四面体/掩码管理 |
| `VbdSolver` | `solver.vbd_solver` | Chebyshev 半隐式 VBD 求解 |
| `FieldController` | `control.field_controller` | PID 电场调节 |
| `LayerActivator` | `geometry.layer_activator` | 新层激活与保形 |

---

## 📝 技术栈

- **语言**：Python 3.10+（核心），C++17（加速）
- **数值计算**：NumPy, SciPy
- **网格处理**：trimesh, PyVista
- **GUI**：PySide6（Qt for Python）
- **C++ 绑定**：pybind11
- **线性代数**：Eigen 3.4（C++ 端）
- **测试**：pytest

---

## 📄 许可

待定
