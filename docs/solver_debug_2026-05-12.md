# C++ 求解器崩溃修复记录

**日期**: 2026-05-12  
**目标**: 解决 C++ 求解器在 GUI QThread 中 segfault 导致闪退的问题，同时不影响 Python 求解器

---

## 一、架构背景

- 项目：水凝胶 DLP VBD 模拟器
- 双求解器：Python（稳定但慢） + C++ pybind11（快 3-5x 但不稳定）
- GUI：PySide6 + QThread 工作线程
- C++ 编译：MSVC + OpenMP + Eigen + pybind11
- 关键设计：C++ 使用 `Eigen::Map` 零拷贝访问 NumPy 数组内存

## 二、原始问题

1. **GUI 工具栏布局**：按钮拥挤、参数面板过窄、13 个控件挤在一行
2. **CJK 中文字体**：matplotlib 3D 视图中中文显示为方框，Qt 控件中文依赖系统默认
3. **C++ 求解器崩溃**：从 QThread 调用时静默 segfault，GUI 整个闪退，无任何报错

## 三、完成的修改

### 3.1 GUI 视觉改进

#### 左侧面板布局重构 (`main_window.py`)

**问题**: 13 个控件（层厚、层数、分辨率、提示、自定义分辨率、网格算法、C++ 开关）挤在一个 `QHBoxLayout` 中，460px 宽度下必然截断。

**修改**: 拆分为 3 行逻辑分组：
- Row 1: 层厚 + 层数 + 分辨率 + 自动提示
- Row 2: 自定义分辨率复选框 + 网格算法下拉
- Row 3: C++ 加速求解器复选框

面板宽度 440/520 → 480/620，spinbox 120 → 140px，网格算法文字缩短（不影响功能，用的是 `currentData()`）。

#### 工具栏 (`main_window.py`)

- `_btn_stop_anim` 补了 `setFixedWidth(80)`（之前宽度不固定）
- 文件操作按钮和仿真控制按钮之间加了竖分隔线 `QFrame.VLine`
- 所有按钮保持 28px 高度、彩色样式已存在

#### CJK 中文字体

**matplotlib 3D 视图** (`mesh_viewer.py`)：

- 原始代码：遍历 `font_manager.ttflist` 搜索字体名，然后设 `rcParams["font.family"] = font_name`
- **无效的原因**：`font.family` 应设为泛类名（如 `sans-serif`），具体字体名应写入 `font.sans-serif` 列表。原始 API 用法错误，字体设置被静默忽略。
- 修复：改用 `font_manager.fontManager.addfont()` 按文件路径直接注册（`C:/Windows/Fonts/msyh.ttc` 等），然后设 `rcParams["font.sans-serif"]` 列表。

**Qt 控件** (`main_window.py` `launch_gui()`)：

- 新增：`app.setFont(QFont(["Microsoft YaHei", "SimHei", ...]))` 确保 Qt 控件使用中文字体族。

### 3.2 C++ 求解器崩溃修复

#### 诊断过程

1. **dtype 不匹配**（`cpp_adapter.py`）：
   - numpy 在 64 位 Windows 上默认 `int64`，但 C++ `Eigen::VectorXi` 需要 `int32`
   - `tets`、`czm_state`、`colors` 三个数组需要转换
   - 修复：在 `solve_until_stable()` 和 `solve_lift_and_relax()` 中自动归一化 dtype

2. **数组预检验证**（`cpp_adapter.py`）：
   - 新增 `_validate_arrays()`：检查形状一致性、C-contiguity（Eigen::Map 要求）
   - 可写 float64 数组检查 dtype（不能静默转换，因为 C++ 原地写入）

3. **OpenMP + QThread 冲突**（最困难的问题）：
   - MSVC 的 `vcomp.dll` 在 QThread 中创建 Win32 工作线程
   - `#pragma omp parallel for` 执行时触发 segfault
   - 修复方案：设置 `OMP_NUM_THREADS=1` 在**仅** `cpp_adapter.py` 导入 C++ 模块之前
   - `HYDROGEL_VBD_OMP=1` 环境变量可恢复多线程（CLI/批处理场景）

4. **子进程隔离**（最终方案，新建 `cpp_subprocess.py`）：
   - C++ 求解器在独立 `multiprocessing.Process` 中运行
   - 即使 C++ segfault，只杀死子进程，GUI 主进程不受影响
   - 崩溃时自动回退到 Python 求解器
   - 通信协议：Pipe + pickle（mesh ~500KB，序列化 ~5ms）

#### 新增文件

**`src/hydrogel_vbd/solver/cpp_subprocess.py`**
- `_worker_run()`: 子进程入口，接收 mesh+config，执行完整仿真
- `_run_simulation()`: 逐层 VBD 求解循环（提升→静平衡→CZM 更新→帧推送）
- `CppSubprocessSolver`: 主进程侧管理器，spawn 子进程并迭代接收消息
- 消息类型：`_LogMsg`、`_ProgressMsg`、`_FrameMsg`、`_DoneMsg`、`_ErrorMsg`

#### 修改的 C++ 文件

**`cpp/CMakeLists.txt`**
- 新增第二个编译目标 `hydrogel_vbd_cpp_qt`：不链接 OpenMP，定义 `VBD_NO_OPENMP` 宏
- 标准目标 `hydrogel_vbd_cpp` 保留（用于 CLI/批处理）

**`cpp/src/vbd_solver.cpp`**
- 新增 `#ifdef VBD_NO_OPENMP` → `#undef _OPENMP` 守卫

**`src/hydrogel_vbd/solver/cpp_builder.py`**
- `build()` 部署步骤改为同时复制两个 .pyd 文件

**`src/hydrogel_vbd/solver/cpp_adapter.py`**
- GUI 模式优先加载 `hydrogel_vbd_cpp_qt`（无 OpenMP），回退到 `hydrogel_vbd_cpp`
- CLI 模式（`HYDROGEL_VBD_OMP=1`）：加载标准版本

### 3.3 SimulationWorker 修改

**`src/hydrogel_vbd/gui/simulation_worker.py`**

- 新增 `_run_cpp_subprocess()`: 序列化 mesh+config，启动子进程，迭代接收帧/进度/结果
- 新增 `_restore_mesh_copy()`: 子进程失败后恢复原始网格，用于 Python 重试
- `_run_layers()` C++ 路径：先尝试子进程 → 成功则返回 → 失败则 log 警告并回退到 Python 路径
- **Python 路径代码一行未改**（`solver = PythonReferenceVBDSolver(...) if not self._use_cpp else None` 及以下所有代码保持 git 原版）
- trace 增加 `f.flush()` 确保崩溃前数据写入磁盘
- `__init__` 保存 `self._mesh_original` 引用

---

## 四、遇到的问题（按时间顺序）

### 问题 1: trace 文件显示 layer_0_lift_start 后无任何输出

**现象**: 子进程在第一次 `cpp_solve_lift_and_relax` 调用时崩溃，无任何 Python 异常。  
**原因**: C++ segfault 直接杀死子进程（`faulthandler` 在子进程中启用了但输出到 stderr，未捕获到）。  
**解决**: 将 faulthandler 输出重定向到 `cpp_subprocess_crash.log` 文件。

### 问题 2: `sub_progress(int,int,int) only accepts 3 argument(s), 4 given!`

**现象**: 子进程成功运行，但主进程处理 `_ProgressMsg` 时崩溃。  
**原因**: `_ProgressMsg` 原来有 4 个字段（layer, total_layers, step, msg），但 Qt 信号只接受 3 个 int。  
**解决**: 将 `_ProgressMsg` 改为 3 字段（layer, percentage, step），移除 msg 字段。

### 问题 3: `frame_ready(QVariantMap) only accepts 1 argument(s), 3 given!`

**现象**: 同上，`_FrameMsg` 传了 3 个参数但信号是 `Signal(dict)`。  
**原因**: 帧消息需要打包成字典 `{vertices, tets, active_mask, title}`。  
**解决**: 在 `_run_cpp_subprocess` 消息处理中将 FrameMsg 属性打包为 dict；`_FrameMsg` 增加 `active_mask` 字段。

### 问题 4: `AttributeError: property 'masses' of 'MeshState' object has no setter`

**现象**: 子进程中 `setattr(mesh, 'masses', value)` 失败。  
**原因**: `MeshState.masses` 是 `@property`（只读），底层存储是 `node_mass`。  
**解决**: 序列化时使用 `node_mass` 代替 `masses`。

### 问题 5: `TypeError: '<=' not supported between instances of 'NoneType' and 'int'`

**现象**: `activator.activate_with_inheritance` 崩溃，`first_active_layer` 为 None。  
**原因**: 子进程中使用 `object.__new__(MeshState)` 绕过 `__post_init__`，导致许多字段未初始化。  
**解决**: 改用正常 `MeshState(**{required_fields})` 构造，让 `__post_init__` 自动填充默认值；然后 `setattr` 覆盖实际数据。

### 问题 6: `update_czm_states() missing 7 required positional arguments`

**现象**: 子进程中 CZM 更新调用失败。  
**原因**: 子进程的控制循环是 Worker 原版的简化拷贝，`update_czm_states` 调用签名错误（传了 `(mesh, config)` 而非逐个参数）。  
**解决**: 补全参数（`bottom`, `internal_pull_z`, `area`, `t_max`, `k_czm`, `delta_f`, `z_fep`, `dt`），加入 `all_free` 检查和最大步数限制（`MAX_STEPS_PER_LAYER=5000`）。

### 问题 7: C++ 求解器运行 1400 步后 GUI 闪退

**现象**: 子进程成功运行了 1400+ 步 C++ 求解器调用，`max_dx` 收敛到 1e-18，但 GUI 最终闪退。  
**原因**: `lift_max = 2.9975`（约 3mm 提升距离），每步 1e-6m，需要约 3000 步/层，9 层共约 27000 步。可能触发内存泄漏或资源耗尽。  
**解决**: 未完全解决。每层有 `MAX_STEPS_PER_LAYER=5000` 安全上限。

### 问题 8: Python 求解器也卡住/闪退

**现象**: 不勾选 C++ 时，Python 求解器在第一层卡住或闪退。回退到 git 原版后问题依旧。  
**原因**: 高度怀疑是 `launch_gui()` 中全局 `OMP_NUM_THREADS=1` 环境变量影响了 numpy 的 BLAS（MKL 读取该变量，限制为单线程），导致 Python 矩阵运算极慢（看起来像卡死）。  
**解决**: 已从 `launch_gui()` 移除 `OMP_NUM_THREADS=1`；仅在 `cpp_adapter.py` 导入 C++ 模块前设置。

### 问题 9: GUI 文件全部 git checkout 后 Python 求解器仍不正常

**现象**: 还原 `main_window.py`、`mesh_viewer.py`、`simulation_worker.py` 到 git 原版后，Python 求解器仍然有问题。  
**分析**: 如果原版代码也出问题，可能原因：① 测试时的 mesh 参数不同；② `cpp_adapter.py` 的 import 路径变化影响了模块加载；③ 外部因素（系统资源、Python 环境变化）。  
**状态**: 悬而未决，需要进一步调试。

---

## 五、未解决问题

### 5.1 Python 求解器不稳定

**现象**: 即使还原到 git 原版，Python 求解器有时卡住或闪退。  
**需要排查**:
1. 检查 `outputs/gui/worker_trace.log` 确认崩溃/卡住的位置
2. 对比修改前后的 git diff，确认还原是否完整
3. 尝试在不同 mesh 模型上复现
4. 检查 Python 环境是否有变化（numpy 版本、BLAS 库等）

### 5.2 C++ 求解器在子进程中仍可能 segfault

**现象**: 子进程隔离让 GUI 不崩溃，但 C++ 求解器在约 285 次调用后仍有概率 segfault。  
**需要排查**:
1. 启用子进程内 faulthandler 输出到文件，捕获 C 级别栈回溯
2. 检查 `cpp_subprocess_crash.log` 
3. 可能的根因：内存越界访问（`lifting_top` 索引、`dm_inv` 形状、非连续数组）
4. 考虑在 C++ 代码中添加 `Eigen::Map` 构造前的数组有效性检查

### 5.3 提升步数过多

**现象**: `lift_max = 2.9975`（约 3mm）配合 `lift_step = v_lift * dt = 1e-6m`，每层需要约 3000 步。  
**分析**:
- 这是控制反转设计的自然结果（每步 = 一个时间步 dt 的平台位移）
- 用户如果觉得太慢，可以调大 `v_lift` 或 `dt`
- C++ 单次调用约 2ms，每层约 6 秒（含 10 次内迭代/步 = 30K 次内迭代）

### 5.4 子进程停止信号未接通

**现状**: 子进程中有每 20 步检查 `conn.poll()` 的逻辑，但主进程侧**尚未实现**发送 "stop" 指令。用户点击"停止"按钮无法终止子进程。  
**需要**: 在 `CppSubprocessSolver` 中添加 `send_stop()` 方法，在 Worker 的 `request_stop()` 中调用。

---

## 六、文件修改清单

| 文件 | 状态 | 变动 |
|------|------|------|
| `main_window.py` | 已修改 | 布局重构（3行）、工具栏分隔线、CJK Qt字体、spinbox 140px |
| `mesh_viewer.py` | 已修改 | matplotlib CJK 字体修复（文件路径注册） |
| `simulation_worker.py` | 已修改 | 子进程 C++ 路径 + Python 重试 + trace flush |
| `cpp_adapter.py` | 已修改 | dtype 归一化、数组验证、OMP_NUM_THREADS、_qt 变体导入 |
| `cpp_subprocess.py` | **新建** | 子进程仿真执行 + 主进程管理器 |
| `cpp/CMakeLists.txt` | 已修改 | 双目标编译（标准 + _qt 无 OpenMP） |
| `cpp/src/vbd_solver.cpp` | 已修改 | VBD_NO_OPENMP 预处理守卫 |
| `cpp_builder.py` | 已修改 | 双 .pyd 部署 |

---

## 七、经验教训

1. **Python 和 C 扩展的线程交互非常脆弱**。OpenMP 在 QThread 中创建线程会导致不可预测的 segfault。最安全的方案是进程隔离。
2. **环境变量影响范围难追踪**。`OMP_NUM_THREADS=1` 在进程级别设置，会影响所有使用 OpenMP 的库（包括 numpy BLAS）。应在最小作用域设置。
3. **pybind11 + Eigen::Map 零拷贝**虽然高效，但对数据类型（int32 vs int64）、内存布局（C-contiguous）和形状非常敏感。必须在 Python 侧做充分的预检验证。
4. **信号签名必须精确匹配**。Qt Signal 的参数类型和数量在编译时（通过 MOC）确定，运行时多传或少传参数都会崩溃。
5. **逐层还原调试策略是有效的**。当不确定哪个改动引起问题时，用 git 逐步还原直到问题消失，然后逐个加回改动。
