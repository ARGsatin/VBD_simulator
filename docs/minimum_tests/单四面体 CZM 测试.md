# 单四面体 CZM 测试

  CZM（Cohesive Zone
  Model，内聚力模型）模拟水凝胶从离型膜（FEP）上剥离的过程。这是一个三态状态机：

      FIXED(0)          DAMAGING(1)          FREE(2)
     (粘附在FEP上) → (界面损伤演化) → (完全脱粘)

---
##   一、状态机的数学定义

  每个底面节点根据 gap（节点到 FEP 的 Z 距离）和 拉力 在这三个状态之间转换：

```
  state = CZMState(mesh.czm_state[node_id])
  gap = max(vertex_z - z_fep, 0.0)          ← 节点到 FEP 的距离
```

  三条转换路径：

  转换: FIXED → DAMAGING
  触发条件: gap > δ_f 或 k_czm·gap > T_max
  含义: 间隙超过失效位移 或 弹性牵引力超过强度
  ────────────────────────────────────────
  转换: DAMAGING → FREE
  触发条件: damage ≥ 1.0 或 gap > 5·δ_f
  含义: 损伤累积到上限 或 间隙过大直接脱粘
  ────────────────────────────────────────
  转换: FREE →（永远 FREE）
  触发条件: —
  含义: 不可逆

  损伤演化（仅在 DAMAGING 状态）：

  $$\text{dmg-rate} = \min\left(1.0,;\frac{\text{pull} \cdot dt}{T_{max} \cdot
  \delta_f}\right)$$

  $$\text{damage} \leftarrow \min(1.0,;\text{damage} + \text{dmg-rate})$$

---
##   二、测试 1：FIXED 状态不动

  设置：所有底部节点紧贴 FEP（gap=0），施加 5250N 的拉力。

     z_fep = 0.0
     vertex_z = 0.0
     gap = max(0.0 - 0.0, 0.0) = 0.0
     traction = 1e8 × 0.0 = 0.0
     0.0 > 1e-4? No   0.0 > 5000? No
     → 保持 FIXED ✓

```
  mesh.vertices[:, 2] = 0.0        # 顶点在 FEP 上
  update_czm_states(mesh, bottom, ...,
      z_fep=0.0, dt=0.01)

  # → 所有 bottom 节点仍为 FIXED (state=0)
```



---
##   三、测试 2：FIXED → DAMAGING（间隙触发）

  设置：底部节点抬高 0.001m >> δ_f = 1e-4 m。

```
 gap = 0.001 - 0.0 = 0.001 m
 0.001 > 1e-4? Yes →
 FIXED → DAMAGING
 damage 初始化为 0.0

求解前 (FIXED):        求解后 (DAMAGING):
 v3                     v3
 /|\                    /|\
/ | \                  / | \
```

   /  |  \                /  |  \
  v0──v1──v2             v0──v1──v2
    FEP ──────────        FEP ──────────
    ↑ 粘附                  ↑ 间隙 0.001m, 损伤开始

  验证：

```
  mesh.vertices[0:3, 2] = 0.001   # 抬高 1mm
  update_czm_states(mesh, bottom, ..., delta_f=1e-4, z_fep=0.0)
  assert mesh.czm_state[0] == CZMState.DAMAGING   # = 1
  assert mesh.damage[0] == 0.0                     # 刚进入损伤, damage=0
```



---
##   四、测试 3：DAMAGING → FREE（损伤累积触发）

  设置：damage 已累积到 0.99，再施加一次大拉力。

     dmg_rate = min(1.0, 5250×0.01 / (5000×1e-4))
              = min(1.0, 52.5 / 0.5)
              = min(1.0, 105.0) = 1.0
     damage_new = min(1.0, 0.99 + 1.0) = 1.0
     damage ≥ 1.0? Yes →
     DAMAGING → FREE
     damage 强制设为 1.0, time_free 重置为 0.0

  验证：

```
  mesh.czm_state[:] = CZMState.DAMAGING
  mesh.damage[:] = 0.99              # 只差 1%
  update_czm_states(mesh, bottom, ..., dt=0.01)
  assert mesh.czm_state[0] == CZMState.FREE  # → state=2
  assert mesh.damage[0] == 1.0
  assert mesh.time_free[0] == 0.0            # 刚脱粘, 计时归零
```



---
##   五、测试 4：DAMAGING → FREE（大间隙触发）

  设置：damage 只有 0.1，但间隙巨大（0.01m >> 5×1e-4 = 0.005m）。

  即使没有拉力（pull_z=0），间隙本身也可以触发脱粘：

     gap = 0.01 = 100 × δ_f >> 5 × δ_f = 0.005
     gap > 5*δ_f? Yes →
     DAMAGING → FREE (直接跳转)

  验证：

```
  mesh.czm_state[:] = CZMState.DAMAGING
  mesh.damage[:] = 0.1               # 低损伤
  mesh.vertices[0:3, 2] = 0.01       # 1cm 间隙!
  update_czm_states(mesh, bottom, ..., delta_f=1e-4)
  assert mesh.czm_state[0] == CZMState.FREE
```



---
##   六、测试 5：FREE 状态永久保持

  已脱粘的节点不会重新粘附。time_free 每次递增 dt，用于后续流体拖曳计算。

     FREE → (永远保持 FREE)
     time_free += dt

```
  mesh.time_free[:] = 2.0            # 已脱粘 2 秒
  update_czm_states(mesh, bottom, ..., dt=0.01)
  assert mesh.time_free[0] == 2.01   # +0.01 ✓
  assert mesh.czm_state[0] == CZMState.FREE  # 不変 ✓
```



---
##   七、完整状态转换图

    time ──────────────────────────────────────────────────────────→
    
    ┌──────────┐       gap > δ_f        ┌───────────┐
    │  FIXED   │ ──────────────────────→│ DAMAGING  │
    │ state=0  │  或 traction > T_max   │ state=1   │
    │ damage=0 │                        │ damage:   │
    └──────────┘                        │ 0→0.5→0.99│
                                        └─────┬─────┘
                                              │
                           damage≥1.0  或     │
                           gap > 5·δ_f       │
                                              ↓
                                        ┌───────────┐
                                        │   FREE    │
                                        │ state=2   │
                                        │ damage=1  │
                                        │ time_free │ ← 每秒 +dt
                                        │  递增     │
                                        └───────────┘

---
##   八、测试 6：数值边界

  test_empty_bottom_nodes：空数组 →
  函数直接返回，不崩溃。这对仿真中某些层没有底面节点时是必要的防护。

  test_damage_clipped_at_one：初始 damage=1.5（超限）→ min(1.0, 1.5 + dmg_rate) = 1.0 →
   被钳位到 1.0 → DAMAGING → FREE。下界钳位确保了即使数据被外部污染也能自我纠正。

  test_invalid_czm_state_silent（xfail）：状态码 = 3（无效值）→
  当前代码静默无操作。函数里只有 if FIXED / elif DAMAGING / elif FREE 三个分支，没有
  else。这是一个已知缺陷——如果内存被意外写入非法状态码，节点会永远卡住。

---
##   九、与 elastic_energy 测试的关系

  ┌──────────┬─────────────────────┬──────────────────┐
  │          │ elastic_energy 测试 │     CZM 测试     │
  ├──────────┼─────────────────────┼──────────────────┤
  │ 验证对象 │ 连续性介质力学      │ 离散状态机       │
  ├──────────┼─────────────────────┼──────────────────┤
  │ 数学基础 │ ∂Ψ/∂F 偏导数        │ if/elif 条件分支 │
  ├──────────┼─────────────────────┼──────────────────┤
  │ 验证方法 │ 有限差分            │ 手算状态轨迹     │
  ├──────────┼─────────────────────┼──────────────────┤
  │ 数值风险 │ NaN/Inf 的传播      │ 除零、非法状态码 │
  ├──────────┼─────────────────────┼──────────────────┤
  │ 复杂度   │ 分析公式            │ 枚举状态组合     │
  └──────────┴─────────────────────┴──────────────────┘