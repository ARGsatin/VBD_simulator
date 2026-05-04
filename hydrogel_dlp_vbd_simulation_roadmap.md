# 基于 VBD 的电场辅助水凝胶 DLP 打印仿真系统技术路径

## 0. 项目目标

构建一个面向水凝胶 DLP 打印的逐层仿真—评估—补偿闭环系统。

系统目标是：在每一打印循环中，利用 VBD 求解器预测当前已固化/半固化水凝胶结构在重力、剥离力、流体阻力、界面作用和电场力下的变形；再将预测形状与理想 CAD/切片轮廓比较；最后反推出下一打印循环需要施加的补偿电场，以减少悬垂结构、桥拱塌陷和层间累积误差。

核心闭环：

```text
理想模型 / 当前层切片
        ↓
逐层网格激活
        ↓
VBD 多物理场形变预测
        ↓
预测形状 vs 理想形状
        ↓
误差提取
        ↓
电场补偿计算
        ↓
下一层电场控制指令
        ↓
继续下一打印循环
```

---

## 1. 总体技术判断

VBD 不应被视为完整多物理场求解器，而应作为快速、稳定的弹性形变预测内核。

VBD 适合负责：

- 已固化/半固化水凝胶结构的弹性形变预测；
- 每层新增材料后的整体结构松弛；
- 重力、剥离力、电场力作用下的短时间形状演化；
- 固定迭代预算下的快速前向预测；
- 补偿控制器中的 forward simulator。

VBD 不直接负责：

- 电场分布求解；
- 离子迁移、电渗、电泳等电化学过程；
- DLP 光固化反应动力学；
- 树脂槽完整流场；
- 真实粘附断裂和剥离破坏；
- 电极控制策略优化。

这些物理效应需要在 VBD 外部建模，然后转化为 VBD 可处理的外力、能量项、材料参数或边界条件。

---

## 2. 推荐系统模块

建议将完整系统拆成六个模块。

```text
1. GeometryPipeline
   STL → slicing → tet mesh → layer activation map

2. MaterialModel
   hydrogel density, modulus, Poisson ratio, curing degree, damping

3. ForceModel
   gravity, peel force, surface tension, fluid drag, electric force

4. VBDSolver
   local 3x3 solve, vertex coloring, substep, damping, constraints

5. ShapeEvaluator
   simulated shape vs target shape, feature error extraction

6. FieldController
   error → desired compensation force → electrode voltage command
```

数据流：

```text
CAD / STL
  ↓
GeometryPipeline
  ↓
Layered Tet Mesh
  ↓
MaterialModel + ForceModel
  ↓
VBDSolver
  ↓
Predicted Deformed Shape
  ↓
ShapeEvaluator
  ↓
FieldController
  ↓
Electric Field Command / G-code Extension
```

---

## 3. 与已有 VBD 仓库的关系

github repository： https://github.com/andrewleachtx/vbd

已有 VBD 仓库可借鉴的核心包括：

- 四面体网格数据结构；
- 顶点位置、速度、四面体连接关系；
- `vertex2tets` 邻接关系；
- `Dm_inverses` 和 `tet_volumes`；
- 顶点图着色 `colors` / `color_ranges`；
- 局部弹性能量梯度和 Hessian；
- 每个顶点的 3×3 局部求解；
- CPU 端 VBD 主循环。

但是已有仓库不适合直接用于最终系统，因为它缺少：

- 逐层网格激活；
- 水凝胶固化度模型；
- 电场力模型；
- 剥离力模型；
- 流体阻力模型；
- 完整阻尼；
- 稳定边界条件系统；
- 打印过程状态管理；
- 电场补偿控制器；
- 实验标定接口。

因此建议策略是：保留 VBD 局部求解思想，重构数据结构和仿真主循环。

---

## 4. 核心数学框架

### 4.1 VBD 变分能量

基础隐式欧拉变分形式：

```math
G(x)=\frac{1}{2h^2}\|x-y\|_M^2+E_{elastic}(x)
```

其中：

```math
y=x^t+h v^t+h^2 a_{ext}
```

扩展到水凝胶 DLP 打印：

```math
G(x)=
\frac{1}{2h^2}\|x-y\|_M^2
+E_{elastic}(x)
+E_{adhesion}(x)
+E_{contact}(x)
+E_{field}(x)
```

工程实现中，电场、剥离、流体等也可以不写成显式能量，而是直接加入局部力：

```math
f_i=f_{inertia,i}
+f_{elastic,i}
+f_{peel,i}
+f_{fluid,i}
+f_{surface,i}
+f_{electric,i}
```

每个顶点的 VBD 局部更新仍为：

```math
H_i \Delta x_i = f_i
```

```math
x_i \leftarrow x_i+\Delta x_i
```

### 4.2 外力统一形式

对每个顶点：

```math
a_{ext,i}=g+a_{peel,i}+a_{fluid,i}+a_{surface,i}+a_{electric,i}
```

然后：

```math
y_i=x_i^t+h v_i^t+h^2 a_{ext,i}
```

或者直接在局部力中加入：

```math
f_{external,i}=m_i a_{ext,i}
```

---

## 5. 电场力建模路径

### 5.1 最小可行模型

第一阶段使用经验线性模型：

```math
f_{electric,i}=\alpha_i U d_i
```

其中：

- `alpha_i`：材料/位置相关的电场响应系数；
- `U`：电压或电极控制量；
- `d_i`：电场作用方向；
- `f_electric,i`：等效顶点体力。

如果使用多个电极：

```math
f_{electric,i}=\sum_m \alpha_{i,m} U_m d_{i,m}
```

矩阵形式：

```math
f_{electric}=B U
```

其中：

- `B` 是电极—力映射矩阵；
- `U` 是电极电压向量。

### 5.2 进阶模型

后续可升级为介电泳或电液耦合模型：

```math
f_{electric}\propto \nabla |E|^2
```

或：

```math
\nabla \cdot (\epsilon \nabla \phi)=0
```

```math
E=-\nabla \phi
```

```math
f_{electric}=f(E,c,\epsilon,\nabla \epsilon)
```

建议开发顺序：

```text
Level 1: f_elec = alpha * U * direction
Level 2: f_elec = alpha(x, curing_degree) * E(x)
Level 3: f_elec ∝ grad(|E|^2)
Level 4: electric field + ion migration + fluid + solid coupling
```

---

## 6. 电场补偿控制器

### 6.1 误差定义

仿真后得到预测形状：

```math
x_{sim,k}
```

理想目标位置：

```math
x_{target,k}
```

几何误差：

```math
e_k=x_{target,k}-x_{sim,k}
```

可提取的关键误差指标：

- 悬垂端最大下垂量；
- 桥拱跨中位移；
- 支撑腿倾角误差；
- 脚部水平位移；
- 层间错位；
- 最大节点误差；
- RMS 几何误差；
- Hausdorff 距离。

### 6.2 误差到期望补偿力

最小版本：

```math
f_{des,k}=K_p e_k+K_d(e_k-e_{k-1})
```

也可以加入速度误差：

```math
f_{des,k}=K_p(x_{target}-x_{sim})+K_d(v_{target}-v_{sim})
```

### 6.3 期望补偿力到电压

如果电场力写成：

```math
f_{electric}=B U
```

则求解：

```math
\min_U \|BU-f_{des}\|^2+\lambda\|U\|^2
```

约束：

```text
U_min ≤ U ≤ U_max
|U_k - U_{k-1}| ≤ ΔU_max
allowed_polarity ∈ {positive, negative, off}
thermal_limit ≤ limit
electrolysis_limit ≤ limit
```

初始版本可使用无约束最小二乘：

```math
U=(B^T B+\lambda I)^{-1}B^T f_{des}
```

工程版本建议使用带约束优化：

```text
scipy.optimize.lsq_linear
cvxpy quadratic programming
custom projected gradient
```

---

## 7. 逐层打印仿真主循环

建议主循环如下：

```python
for layer_id in range(num_layers):

    # 1. 获取当前层理想切片
    target_slice = geometry.get_slice(layer_id)

    # 2. 激活当前层网格
    mesh.activate_layer(layer_id)

    # 3. 更新材料参数
    material.update_curing_degree(layer_id, exposure_time)
    material.update_lame_parameters()

    # 4. 构造边界条件
    constraints = build_constraints(
        platform_fixed=True,
        sacrificial_layer=True,
        interface_contact=True
    )

    # 5. 计算外力
    f_gravity = force_model.gravity(mesh, material)
    f_peel = force_model.peel(mesh, peel_params)
    f_fluid = force_model.fluid_drag(mesh, process_params)
    f_surface = force_model.surface_tension(mesh, surface_params)
    f_electric = force_model.electric(mesh, field_command)

    f_total = f_gravity + f_peel + f_fluid + f_surface + f_electric

    # 6. VBD 仿真
    x_sim, v_sim = vbd_solver.step(
        mesh=mesh,
        material=material,
        forces=f_total,
        constraints=constraints,
        dt=dt,
        substeps=substeps,
        iterations=iterations
    )

    # 7. 形状误差评估
    error_metrics = evaluator.compare(
        x_sim=x_sim,
        x_target=target_geometry[layer_id]
    )

    # 8. 计算下一层补偿电场
    next_field_command = controller.compute(
        error=error_metrics,
        previous_command=field_command
    )

    # 9. 保存状态
    state.save(layer_id, x_sim, v_sim, error_metrics, next_field_command)

    # 10. 更新下一层控制指令
    field_command = next_field_command
```

---

## 8. 数据结构设计

### 8.1 MeshState

```python
class MeshState:
    vertices: np.ndarray          # shape: (N, 3)
    prev_vertices: np.ndarray     # shape: (N, 3)
    velocities: np.ndarray        # shape: (N, 3)
    tets: np.ndarray              # shape: (T, 4)
    vertex2tets: list[list[int]]
    tet_volumes: np.ndarray
    Dm_inv: np.ndarray            # shape: (T, 3, 3)
    colors: np.ndarray            # shape: (N,)
    color_ranges: list[tuple[int, int]]
    layer_id_per_vertex: np.ndarray
    active_vertex_mask: np.ndarray
    active_tet_mask: np.ndarray
    boundary_flags: np.ndarray
```

### 8.2 MaterialState

```python
class MaterialState:
    density: float
    young_modulus: np.ndarray
    poisson_ratio: float
    mu: np.ndarray
    lam: np.ndarray
    damping: np.ndarray
    curing_degree: np.ndarray
    peel_stress_crit: float
    electric_response_alpha: np.ndarray
```

### 8.3 ForceState

```python
class ForceState:
    gravity: np.ndarray
    peel: np.ndarray
    fluid: np.ndarray
    surface: np.ndarray
    electric: np.ndarray
    total: np.ndarray
```

### 8.4 FieldCommand

```python
class FieldCommand:
    voltage: np.ndarray
    polarity: np.ndarray
    duration: float
    start_time: float
    electrode_ids: list[int]
```

### 8.5 LayerResult

```python
class LayerResult:
    layer_id: int
    x_sim: np.ndarray
    v_sim: np.ndarray
    error_metrics: dict
    field_command_next: FieldCommand
    max_deformation: float
    rms_error: float
    success: bool
```

---

## 9. 材料模型

### 9.1 固化度

水凝胶模量随固化度变化：

```math
E(\phi)=E_{min}+(E_{max}-E_{min})\phi^n
```

其中：

- `phi = 0`：未固化；
- `phi = 1`：完全固化；
- `n`：拟合指数。

### 9.2 Lamé 参数

由杨氏模量和泊松比转换：

```math
\mu=\frac{E}{2(1+\nu)}
```

```math
\lambda=\frac{E\nu}{(1+\nu)(1-2\nu)}
```

水凝胶近似不可压缩时：

```text
nu ≈ 0.45 ~ 0.49
```

注意：当 `nu` 接近 0.5 时，`lambda` 会很大，容易导致数值刚性增强。第一版可使用 `nu = 0.45`。

### 9.3 阻尼

第一版建议使用 Kelvin-Voigt 或 Rayleigh damping 的简化形式：

```math
f_{damping,i}=-c_i v_i
```

或 VBD 论文中的局部刚度阻尼形式：

```math
f_{damping,i}= -\frac{k_d}{h} H_{elastic,i}(x_i-x_i^t)
```

```math
H_i \leftarrow H_i+\frac{k_d}{h}H_{elastic,i}
```

---

## 10. 剥离力模型

### 10.1 实验标定

通过标准试样获得剥离力—位移曲线：

```text
1. 打印标准水凝胶试样
2. 保留底部黏附
3. 使用 Instron 或自制传感器拉伸剥离
4. 记录 force-displacement curve
5. 多次实验取峰值剥离力
6. 转换为单位面积临界剥离应力
```

```math
\sigma_{crit}=\frac{F_{peak}}{A_{contact}}
```

### 10.2 仿真中的等效剥离力

对界面顶点：

```math
f_{peel,i}=p_{peel} A_i n_i
```

其中：

- `p_peel`：等效剥离压强或拉应力；
- `A_i`：顶点关联面积；
- `n_i`：界面法向。

可设置：

```math
p_{peel}=\beta \sigma_{crit}
```

其中 `beta` 为与平台速度、离型膜状态相关的经验系数。

---

## 11. 几何处理与网格生死管理

### 11.1 前处理

输入：

```text
STL model
layer height
exposure time
material parameters
printer parameters
electrode geometry
```

输出：

```text
tet mesh
layer activation map
target vertices per layer
JSON configuration
```

建议工具：

```text
Slicing: trimesh / pyvista / custom slicer
Tetrahedralization: TetGen / Gmsh
Visualization: VTK / PyVista / ParaView
Graph coloring: networkx / custom greedy coloring
```

### 11.2 Layer Activation

每个 tet 和 vertex 增加激活层编号：

```python
vertex_layer_id[i]
tet_layer_id[j]
```

每一层开始时：

```python
active_vertex_mask = vertex_layer_id <= current_layer
active_tet_mask = tet_layer_id <= current_layer
```

对未激活顶点和四面体：

```text
不参与 VBD 更新
不参与力计算
不输出为当前实体
```

---

## 12. VBD 求解器实现要点

### 12.1 CPU 版优先

第一版建议先实现 CPU 版，方便调试和验证。

核心流程：

```python
for substep in range(substeps):

    compute_initial_guess()

    for iter in range(iterations):

        for color in colors:

            parallel_for vertex in color:

                if vertex is fixed:
                    continue

                f_i = inertial_force + external_force
                H_i = inertial_hessian

                for tet in vertex2tets[i]:
                    if tet is active:
                        f_elastic, H_elastic = compute_elastic_gradient_hessian(...)
                        f_i += f_elastic
                        H_i += H_elastic

                if det(H_i) > eps:
                    dx = inv(H_i) @ f_i
                    x_new[i] = x[i] + dx
                else:
                    x_new[i] = x[i]

            copy x_new[color] to x[color]

    update_velocity()
```

### 12.2 外力接入位置

外力有两种接入方式。

方式 A：进入 `y_i`

```math
y_i=x_i^t+h v_i^t+h^2 a_{ext,i}
```

适合：

- 重力；
- 稳定体力；
- 电场等效加速度。

方式 B：直接进入局部力 `f_i`

```math
f_i \leftarrow f_i + f_{external,i}
```

适合：

- 剥离力；
- 流体阻力；
- 表面张力；
- 经验控制力。

工程实现中可同时使用，但要避免重复计入同一物理力。

### 12.3 固定边界

第一版固定边界直接跳过：

```python
if fixed_vertex[i]:
    x_new[i] = x[i]
    continue
```

适用于：

- 打印平台粘附区域；
- 牺牲层；
- 手动指定的约束点。

---

## 13. 输出与可视化

每层输出：

```text
states/layer_0001.npz
states/layer_0002.npz
...
```

包含：

```python
x_sim
v_sim
active_mask
error_metrics
electric_force
field_command
```

可视化文件：

```text
VTU: deformed mesh
PLY: ideal vs simulated surface
VTS: electric force vector field
CSV: layer-wise error metrics
JSON/YAML: summary report
```

建议可视化指标：

```text
deformation magnitude
signed z-displacement
electric force arrows
peel force distribution
curing degree
active layer ID
RMS error per layer
maximum overhang sagging
```

---

## 14. G-code / 打印控制指令扩展

在标准 G-code 中插入电场控制注释或自定义命令：

```gcode
;LAYER: 12
G1 Z0.600
;E_FIELD: ELECTRODE=LEFT, VOLTAGE=2.0, POLARITY=POS, DURATION=0.8
M106 S255
G4 P800
;E_FIELD: OFF
```

建议系统中保留两类输出：

```text
1. simulation_field_commands.json
2. compensated_print.gcode
```

`simulation_field_commands.json` 用于仿真复现；`compensated_print.gcode` 用于实际打印。

---

## 15. 最小可行系统 MVP

第一阶段只做最小闭环，不做完整多物理场。

### 15.1 MVP 功能

```text
1. 输入一个拱桥 STL
2. 生成四面体网格
3. 按层激活网格
4. 使用 VBD 预测重力下垂
5. 加入简化剥离面力
6. 加入简化电场体力
7. 对比理想形状与预测形状
8. 计算下一层补偿电压
9. 输出每层误差曲线和 VTU 可视化
```

### 15.2 MVP 可忽略内容

```text
1. 完整流体求解
2. 完整电化学模型
3. 离子迁移
4. 温度场
5. 自碰撞
6. 复杂断裂
7. 高性能 CUDA
```

### 15.3 MVP 成功标准

对比：

```text
Case A: no electric field
Case B: constant electric field
Case C: closed-loop compensated electric field
```

核心指标：

```math
\eta=\frac{\delta_{no-field}-\delta_{comp}}{\delta_{no-field}}
```

其中：

- `delta_no-field`：无电场最大下垂量；
- `delta_comp`：补偿电场下最大下垂量；
- `eta`：补偿改善率。

另外记录：

```text
RMS geometry error
maximum nodal deviation
arch midpoint displacement
leg angle error
layer-wise accumulated error
print success/failure
```

---

## 16. 推荐开发阶段

### Stage 1: VBD 弹性核验证

目标：

```text
实现或复用 VBD CPU 求解器
完成简单梁/拱桥在重力下的变形预测
```

交付：

```text
beam_gravity_demo.vtu
arch_sagging_demo.vtu
error_curve.csv
```

### Stage 2: 逐层激活

目标：

```text
让模型不是一次性出现，而是按 DLP 层厚逐层生长
```

交付：

```text
layer_activation_demo
active_tet_mask visualization
```

### Stage 3: 剥离力模型

目标：

```text
加入等效剥离面力
支持实验剥离力参数输入
```

交付：

```text
peel_force_model.py
peel_calibration.json
```

### Stage 4: 电场体力模型

目标：

```text
加入 f_elec = alpha * U * direction
支持多电极输入
```

交付：

```text
electric_force_model.py
electrode_config.json
```

### Stage 5: 形状评估器

目标：

```text
计算仿真形状与理想形状之间的几何误差
```

交付：

```text
shape_evaluator.py
error_metrics.csv
error_colormap.vtu
```

### Stage 6: 电场补偿控制器

目标：

```text
由误差计算下一层电压
```

交付：

```text
field_controller.py
field_commands.json
```

### Stage 7: 实验标定与打印验证

目标：

```text
用实际水凝胶材料参数修正模型
对比无电场、固定电场、闭环电场打印结果
```

交付：

```text
calibrated_material.json
experiment_report.csv
comparison_figures
```

---

## 17. 推荐目录结构

```text
hydrogel_vbd_sim/
├── configs/
│   ├── material_hydrogel.json
│   ├── printer_dlp.json
│   ├── electrode_config.json
│   └── solver_vbd.json
├── data/
│   ├── stl/
│   ├── meshes/
│   ├── calibration/
│   └── experiments/
├── src/
│   ├── geometry/
│   │   ├── slicer.py
│   │   ├── tet_mesher.py
│   │   └── layer_activator.py
│   ├── material/
│   │   ├── hydrogel_model.py
│   │   └── curing_model.py
│   ├── forces/
│   │   ├── gravity.py
│   │   ├── peel.py
│   │   ├── fluid_drag.py
│   │   ├── surface_tension.py
│   │   └── electric.py
│   ├── solver/
│   │   ├── vbd_solver.py
│   │   ├── elastic_energy.py
│   │   ├── graph_coloring.py
│   │   └── constraints.py
│   ├── evaluation/
│   │   ├── shape_error.py
│   │   └── metrics.py
│   ├── control/
│   │   ├── field_controller.py
│   │   └── voltage_optimizer.py
│   ├── io/
│   │   ├── vtk_writer.py
│   │   ├── npz_state.py
│   │   └── gcode_exporter.py
│   └── main_loop.py
├── outputs/
│   ├── states/
│   ├── vtk/
│   ├── reports/
│   └── gcode/
└── README.md
```

---

## 18. 建议本地 Agent 的执行任务清单

### Task 1: 创建项目骨架

```text
创建上述目录结构
创建 configs/*.json 模板
创建 main_loop.py 主入口
```

### Task 2: 实现 MeshState

```text
支持 vertices, tets, active masks, velocities, layer ids
支持保存/读取 .npz
```

### Task 3: 实现 VBD CPU Solver

```text
实现 local 3x3 solve
实现 tetrahedral elastic gradient/hessian
实现 vertex coloring loop
实现 fixed vertex constraint
```

### Task 4: 实现 LayerActivator

```text
根据 layer_id 激活顶点和四面体
支持新增层初始速度和初始位置设置
```

### Task 5: 实现 ForceModel

```text
gravity
peel equivalent surface force
electric equivalent body force
optional fluid drag
```

### Task 6: 实现 ShapeEvaluator

```text
计算 target vs simulated 的 nodal error
输出 max error, RMS error, arch midpoint sagging, leg angle
```

### Task 7: 实现 FieldController

```text
先实现 PD 控制
再实现 least-squares voltage solver
```

### Task 8: 实现 VTK 输出

```text
导出 deformed mesh
导出 error colormap
导出 electric force vector field
```

### Task 9: 实现实验标定接口

```text
读取 peel force CSV
计算 sigma_crit
写入 material_hydrogel.json
```

### Task 10: 实现 G-code 扩展

```text
读取标准 G-code
按 layer 插入 ;E_FIELD 指令
输出 compensated_print.gcode
```

---

## 19. 当前最重要的开发原则

1. 先做可运行闭环，不追求完整多物理精度。
2. 先 CPU 后 GPU。
3. 先经验电场体力，后电化学机理。
4. 先拱桥/悬臂梁简单结构，后复杂模型。
5. 先仿真指标闭环，后接入真实打印机。
6. 所有材料参数必须可 JSON 配置。
7. 所有层结果必须可回放、可可视化、可量化比较。
8. 不要把电场补偿直接写死为 `-K * error`，应保留 `force → voltage` 的反演层。
9. 剥离力必须实验标定，否则仿真很容易偏离实际。
10. VBD 是求解核心，不是完整系统；系统价值在于 VBD + 打印过程 + 电场控制的闭环集成。

---

## 20. 一句话总结

本项目应实现一个以 VBD 为弹性形变预测内核的水凝胶 DLP 打印闭环仿真平台：通过逐层激活四面体网格，加入重力、剥离力、流体阻力和电场等效体力，预测每层固化后的实际形状，再根据与理想模型的几何误差反推出下一层补偿电场，最终生成可视化结果和可执行的电场增强打印指令。
