# codex/perf-czm-lift-field-debug

本分支用于系统性定位并降低以下三类开销叠加后的仿真耗时：

- 开启 CZM 后，每个提升步需要额外同步脱膜状态。
- 提高提升距离后，lift 步数和平台回程步数线性增加。
- 开启电场调试对比后，每层会分别运行 no-field 与 with-field 分支。

当前阶段优先保持物理语义不变，不牺牲每层 no-field/with-field 精确守门；本分支先完成可观测性和低风险减负，为后续 C++ 热路径下沉提供数据依据。

## 已实施的提速措施

### 1. 每层性能诊断 CSV

开启 GUI 的“输出求解器诊断 CSV”后，现在会同时写出：

- `outputs/gui/reports/solver_diagnostics.csv`
- `outputs/gui/reports/performance_diagnostics.csv`

`performance_diagnostics.csv` 用于解释“慢在哪里”，核心字段如下：

| 字段 | 含义 |
| --- | --- |
| `layer_id` | 层号，从 0 开始 |
| `mode` | 执行模式，例如 `single_path`、`field_debug`、`cpp_subprocess` |
| `elapsed_s` | 当前层总耗时 |
| `lift_steps` | 当前层 lift 分支步数 |
| `return_steps` | 平台回程步数 |
| `no_field_ms` | field-debug 中 no-field 分支耗时 |
| `with_field_ms` | field-debug 中 with-field 分支耗时 |
| `cpp_solve_ms` | C++ 求解调用累计耗时 |
| `czm_sync_ms` | Python 侧 CZM 状态同步累计耗时 |
| `render_ms` | 帧推送/渲染信号相关耗时 |
| `solver_steps` | 当前层实际求解调用次数 |
| `solver_iterations` | 当前层迭代总数 |
| `rms_error` | 当前层最终 RMS 误差 |
| `max_error` | 当前层最终最大形状误差 |
| `guard_reason` | field-debug 守门结果原因 |

这些字段的目标不是替代物理指标，而是把耗时拆成 lift、return、CZM、field-debug 分支和渲染几块，便于决定下一步优化优先级。

### 2. field-debug 单帧推送

之前开启“电场调试对比”时，GUI 会推送 no-field baseline 帧，再推送 selected 帧。现在只推送最终选中的 selected 帧：

- 保留每层 no-field 与 with-field 指标对比。
- 保留 RMS/max-error 守门逻辑。
- 减少 GUI 渲染和数组拷贝开销。
- 避免用户误以为当前画面在显示未采用分支。

### 3. field-debug 减少重复拷贝

field-debug 分支仍然需要克隆同层状态来保证 no-field/with-field 从同一起点运行，但本分支做了两点低风险减负：

- 去掉主循环进入 field-debug 前的一层多余 mesh 预拷贝。
- 分支内部统一使用现有 `SimulationWorker._deep_copy_mesh()`，避免通用 `copy.deepcopy()` 带来的额外 Python 对象遍历开销。

这不会改变物理计算结果，只减少分支启动和 commit/guard 快照成本。

### 4. C++ 子进程路径补充性能拆分

普通 C++ 子进程路径现在也会把以下字段回传到 `LayerResult.error_metrics`，并写入性能 CSV：

- `solver_return_steps`
- `solver_platform_return_distance`
- `perf_lift_steps`
- `perf_return_steps`
- `perf_cpp_solve_ms`
- `perf_czm_sync_ms`
- `perf_render_ms`
- `perf_snapshot_ms`

这样可以直接比较“非 field-debug 的 C++ 子进程路径”和“field-debug 的 Python 控制循环 + C++ adapter 路径”之间的差异。

### 5. 标准慢场景矩阵

`hydrogel_vbd.solver.diagnostics.default_performance_benchmark_matrix()` 定义了默认性能基准矩阵：

- 模型：`assets/test_models/demo7.STEP`
- 层数：32
- CZM：开/关
- 提升高度：默认 `5mm`
- 电场调试：开/关

组合数为 4 个，用于覆盖当前最容易拖慢仿真的主要因素。

### 6. C++ field-debug branch runner

field-debug 的直接 C++ adapter 路径现在优先调用 `solve_field_debug_branch`，将单个 no-field/with-field 分支的 lift 循环、event window 电场、commit/guard 快照和平台回程合并到一次 C++ 调用中。GUI 侧仍然负责每层双分支对比、RMS/max 守门和 selected mesh 选择，因此物理守门语义保持不变。

性能 CSV 对应的 `perf_cpp_solve_ms`、`perf_no_field_ms`、`perf_with_field_ms` 仍可用于观察耗时；`LayerResult.error_metrics` 额外记录 `field_branch_runner_enabled`、`field_cpp_module`、`field_cpp_openmp_enabled` 和 `field_cpp_threads`，用于确认本次是否走到了新热路径。

## 建议的测试指标

### 物理正确性指标

每次提速后至少检查：

- `rms_error` 不恶化，尤其是 field-debug 的 selected 分支结果。
- `max_error` 不出现异常突增。
- CZM 状态转移仍符合预期，不能因为提速跳过真实脱膜状态。
- field-debug 守门结果 `guard_reason` 与原逻辑一致。
- 平台回程后最终帧仍是层结束状态，而不是最高点 guard 状态。

### 性能指标

建议按层统计并比较：

- 总耗时：`elapsed_s`
- lift 成本：`lift_steps`
- 平台回程成本：`return_steps`
- field-debug 分支成本：`no_field_ms`、`with_field_ms`
- C++ 求解成本：`cpp_solve_ms`
- CZM 同步成本：`czm_sync_ms`
- GUI 帧推送成本：`render_ms`

判断瓶颈时优先看比例：

- `czm_sync_ms / elapsed_s` 高：优先考虑把 CZM 状态更新下沉到 C++。
- `return_steps` 接近 `lift_steps`：优先优化平台回程策略。
- `with_field_ms + no_field_ms` 接近 2 倍单路径：优先做 C++ branch runner 或批处理。
- `render_ms` 高：继续减少中间帧推送或做 GUI 侧节流。

## 验证命令

本分支当前通过的主回归集合：

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_solver_diagnostics.py tests\test_lift_and_gui.py tests\test_models_solver_control.py tests\test_io_and_main_loop.py tests\test_multiphysics_vbd_pid.py -q
```

最近一次验证结果：

```text
111 passed
```

针对性能诊断本身的快速测试：

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_solver_diagnostics.py -q
```

针对 GUI/Worker 和 C++ 子进程性能字段的快速测试：

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_lift_and_gui.py::WorkerLiftControlTests tests\test_lift_and_gui.py::CppSubprocessRuntimeTests tests\test_lift_and_gui.py::CppAdapterStateWritebackTests -q
```

## 如何读取一次 GUI 实跑结果

1. 在 GUI 中勾选“输出求解器诊断 CSV”。
2. 按需要组合开启：
   - 使用 C++ 加速求解器
   - 电场调试对比
   - 禁用/启用 CZM
   - 固定 5mm 提升高度
3. 仿真结束后查看：
   - `outputs/gui/reports/solver_diagnostics.csv`
   - `outputs/gui/reports/performance_diagnostics.csv`
4. 优先按 `elapsed_s` 排序找最慢层，再看同层的 `return_steps`、`cpp_solve_ms`、`czm_sync_ms`、`no_field_ms`、`with_field_ms`。

## 后续优化方向

### A. C++ CZM 状态同步

如果 `czm_sync_ms` 占比高，下一步应把当前 Python 侧的：

- 恢复 bottom CZM 状态
- 计算局部物理项
- `update_czm_states`
- 回写 `result.all_free`

合并进 C++ 热路径，减少每步 Python 往返和数组构造。

### B. C++ field-debug branch runner

如果 field-debug 的 `no_field_ms + with_field_ms` 是主瓶颈，建议新增 C++ branch runner：

- 输入同一层初始 mesh snapshot。
- 在 C++ 内部运行 no-field 和 with-field 两个分支。
- 内部处理短窗口电场、commit/guard 快照、平台回程和 CZM 更新。
- 一次性返回 selected mesh、guard metrics 和性能计数。

这能减少 Python 每步 adapter 调用、数组验证和重复快照成本。

### C. 平台回程策略

如果 `return_steps` 对总耗时贡献很高，需要单独评估平台回程是否必须与 lift 使用同等步长。可选方向：

- 给回程设置独立最大步数。
- 回程阶段禁用不必要的物理项。
- 将回程作为几何位移加一次稳定求解，而不是完整逐步求解。

这类改动会更接近物理语义变更，必须用 `rms_error`、`max_error` 和 CZM 状态回归守门。

## 当前边界

- 本分支尚未声称 32 层实跑已达到 2x 提速。
- 当前主要完成的是可观测性、单帧推送和低风险减拷贝。
- 后续是否下沉 C++，应由 `performance_diagnostics.csv` 的真实占比决定。
