# 方案 C：界面方程重构实施方案

**项目**：drop_time_ad — 1D 球形液滴蒸发隐式 Newton 求解器
**基准分支**：`claude/fix-boiling-point-divergence-geKQy`（包含 Fixes 1–4，commit `a27e4cb`）
**参考文献**：Millán-Merino 等 (2021)，Section 2.2，Eqs.(15)–(19)
**文档日期**：2026-03-03

---

## 第一部分：当前代码实现与文献策略对比

### 1.1 文献策略（Millán-Merino 2021）

Millán-Merino 在 Section 2.2 给出了隔离液滴界面的完整边界条件体系：

**Eq.(15) — 燃料组分界面平衡**：

```
ṁ(Y_fuel,g − Y_fuel,l)|_a = (J_fuel,g − J_fuel,l)|_a
```

**Eq.(16) — 惰性组分 Stefan 条件**（N₂ 和 O₂ 各一条）：

```
ṁ · Y_N₂,s = J_N₂,g |_a      →  确定 Y_N₂,s
ṁ · Y_O₂,s = J_O₂,g |_a      →  确定 Y_O₂,s
```

**Eq.(17) — 能量平衡**（确定 ṁ）：

```
ṁ · ΣL_i = k_g ∂T/∂r|_a − k_l ∂T/∂r|_a − Σ J_l,i · L_i |_a
```

**Eq.(19) — 汽液平衡（VLE）**（确定 Ts）：

```
Y_fuel,g,s = psat(Ts) / P × M_fuel / M_mix    (Clausius-Clapeyron)
```

**文献策略要点**：
- ṁ 始终由**能量平衡**（Eq.17）确定，L_v 在任何温度下均有限且非零
- Ts 由**汽液平衡**（Eq.19）确定，psat(Ts) 相对于 Ts 的导数始终有限
- 惰性组分界面浓度由**Stefan 条件**（Eq.16）自洽确定，不依赖远场比例估算
- 无需 EVAP/SAT 模式切换，所有方程在整个温度范围内一致成立

### 1.2 当前代码实现

#### 1.2.1 mpp 确定策略

**当前文件**：`physics/interface_bc.py`，函数 `_build_mpp_row`

**EVAP 模式（默认）**：使用燃料组分 Stefan 条件（对应 Eq.15 简化形式）：

```
F_mpp = j_corr_b − mpp × ΔY_eff = 0
ΔY_eff = Y_fuel,l,b − Y_fuel,g,eq,b
```

**SAT 模式（接近沸点后触发）**：使用能量平衡（对应 Eq.17）：

```
F_mpp = q_g + q_l + q_diff − mpp × L_v = 0
```

当前代码通过 `use_sat_regime` 标志在两种模式之间切换，切换逻辑在 `assembly/residual_global.py` 的 `_update_iface_regime()` 中。

#### 1.2.2 界面组分填充策略

**当前文件**：`properties/p2_equilibrium.py`，函数 `_project_background`

```python
# 按远场摩尔分数比例分配剩余空间给背景组分
y_bg_total = 1.0 - y_fuel_s    # Y_fuel,s 占据后的剩余
y_N2_s = y_bg_total × X_N2_far / (X_N2_far + X_O2_far)   # 等比分配
y_O2_s = y_bg_total × X_O2_far / (X_N2_far + X_O2_far)
```

该策略不包含 Stefan 流信息，完全由远场组分比例决定。

#### 1.2.3 Ts 确定策略

- **EVAP 模式**：`_build_Ts_row` 通过能量平衡方程隐式确定 Ts
- **SAT 模式**：`_build_Ts_sat_row` 将 Ts 直接钉住至 `Ts_pin = Tbub - Ts_sat_eps_K`

Ts 与 VLE 条件（psat(Ts)）之间的一致性由 `interface_equilibrium()` 的 `Ts_eff` 保障，但这是单向传递（不作为残差方程的一行）。

#### 1.2.4 全局 DOF 结构

当前界面 DOF 为 3 个：`{Ts, mpp, Rd}`，闭合组分 Y_N₂ 不在未知量向量中，通过重建得到：

```python
# core/layout.py, _reconstruct_closure()
Y_N₂[cell] = 1.0 - Y_fuel[cell] - Y_O₂[cell]
closure = np.clip(closure, 1.0e-8, 1.0)   # Fix 2 已修改下界
```

### 1.3 当前实现与文献策略的差异总结

| 维度 | 文献（Millán-Merino） | 当前代码 | 差异影响 |
|---|---|---|---|
| **mpp 方程** | 能量平衡（Eq.17，永不退化） | EVAP: Stefan 条件（ΔY 退化）；SAT: 能量平衡 | **主要根因**：接近沸点时 Jacobian 奇异 |
| **Ts 方程** | VLE（Eq.19，直接约束） | EVAP: 能量平衡；SAT: 钉住 Tbub | 间接导致 mpp/Ts 耦合不清晰 |
| **Y_N₂,s** | Stefan 条件 for N₂（Eq.16，自洽） | 按远场比例估算（`_project_background`） | **次要根因**：高 Stefan 数时不自洽，Y_N₂→0 |
| **模式切换** | 无（一套方程始终成立） | EVAP/SAT 二态切换 + 多个阈值参数 | 引入不连续性，阈值依赖工况 |
| **DOF 数量** | 4（Ts, ṁ, Rd, Y_N₂,s）| 3（Ts, mpp, Rd） | Y_N₂,s 不在全局 Newton 迭代中 |

---

## 第二部分：当前代码基准状态分析

### 2.1 已实施的临时修复（Fixes 1–4，commit `a27e4cb`）

| 修复 | 文件 | 内容 | 性质 |
|---|---|---|---|
| Fix 1 | `properties/gas.py` | `Y_min=1e-8` floor 传入 Cantera；`D_floor` 从 1e-9 提升至 1e-6 m²/s | **治标** |
| Fix 2 | `core/layout.py` | 闭合重建下界 `0.0 → 1e-8` | **治标** |
| Fix 3 | `assembly/residual_global.py` | `Ts ≥ Tbub - 10K` 时强制切换 SAT 模式 | **缓解主因** |
| Fix 4 | `cases/claude_tests_case.yaml` | `sat_tol_enter: 0.15`；`T_sat_enter_margin: 10.0` | **辅助 Fix 3** |

### 2.2 Fixes 1–4 的局限性

1. **依赖经验参数**：`T_sat_enter_margin=10.0 K` 对当前工况有效，但对其他燃料/压力需重新标定
2. **EVAP 模式仍有退化风险**：若 Fix 3 未及时触发（如温度快速跳变），Stefan 条件退化依然可能发生
3. **模式切换不连续性**：EVAP→SAT 切换在 `mpp` 残差行突变（Stefan→能量平衡），牛顿迭代可能在切换点附近收敛变慢
4. **界面组分不自洽**：SAT 模式下 Y_N₂,s 仍由 `_project_background` 估算，不满足 Stefan 条件；在高传质速率（B≫1）情况下造成系统性误差

### 2.3 当前界面方程结构（基准）

```
EVAP 模式：
  row_Ts   → 能量平衡: q_g + q_l + q_diff - mpp·L_v = 0  (确定 Ts)
  row_mpp  → Stefan:  j_corr_b - mpp·ΔY_eff = 0           (确定 mpp, ⚠️退化)
  row_Rd   → Rd 演化方程                                    (确定 Rd)

SAT 模式（Fix 3 触发）：
  row_Ts   → 钉住:    Ts - Ts_pin = 0
  row_mpp  → 能量平衡: q_g + q_l + q_diff - mpp·L_v = 0   (确定 mpp, 安全)
  row_Rd   → Rd 演化方程
```

---

## 第三部分：修改目标与终点描述

### 3.1 物理目标

参考 Millán-Merino 框架，在不依赖模式切换的前提下，建立一套在全温度范围内非退化的界面方程组：

1. **ṁ 始终由能量平衡确定**（Eq.17）——L_v 有限且非零，Jacobian 系数 ∂F/∂mpp = −L_v 永不为零
2. **Ts 始终由 VLE 条件确定**（Eq.19）——Y_fuel,s(Ts) 相对于 Ts 单调，Jacobian 系数有限
3. **Y_N₂,s 由 Stefan 条件显式确定**（Eq.16-N₂）——作为新 DOF 纳入全局 Newton 迭代，替代比例估算

### 3.2 代码终态描述

**全局未知量向量扩展（+1 DOF）**：

```
当前：[Tg×Ng | Yg_red×Ng | Tl×Nl | Yl_red×Nl | Ts | mpp | Rd]
目标：[Tg×Ng | Yg_red×Ng | Tl×Nl | Yl_red×Nl | Ts | mpp | Rd | Y_N2_s]
                                                                  ↑ 新增
```

**界面方程（统一，无模式切换）**：

| 行 | 方程名 | 数学形式 | 确定量 | 退化风险 |
|---|---|---|---|---|
| `row_Ts` | VLE 条件 | `Y_fuel,s,eq(Ts) − Y_fuel,s = 0` | Ts | 无（psat 始终有导数） |
| `row_mpp` | 能量平衡 | `q_g + q_l + q_diff − mpp·L_v = 0` | mpp | 无（L_v > 0 恒成立） |
| `row_Y_N2_s` | Stefan for N₂ | `mpp·Y_N₂,s + j_N₂,g·A = 0` | Y_N₂,s | 无（mpp > 0 时恒成立） |
| `row_Rd` | Rd 演化 | 不变 | Rd | 不变 |

**体积单元 Y_N₂ 保持闭合重建**（不变）：

```python
# 体积单元（cells 0 到 Ng-1）：仍为闭合量
Y_N₂[cell] = 1.0 - Y_fuel[cell] - Y_O₂[cell]

# 界面面值（新 DOF）：由全局 Newton 迭代确定
Y_N₂_s = u[idx_Y_N2_s]
```

**模式切换机制**：

- 删除 `use_sat_regime` 分支
- 删除 `_update_iface_regime()` 中的温度 override（Fix 3 可退役）
- `_build_Ts_sat_row`、`_build_Ts_row`（旧版）可删除

---

## 第四部分：分步修改路线（大纲式）

### Step 0：前置准备（基准确认）

**目标**：冻结基准，确保可回退

- 确认 Fixes 1–4 已提交（`git log` 确认 commit `a27e4cb`）
- 在基准上新建工作子分支（可选）
- 准备最小化测试夹具：3 气相单元 + 2 液相单元，已知解析解

**验收**：`git status` 干净；现有测试套件（`pytest tests/`）全部通过

---

### Step 1：扩展 `core/layout.py`——添加 `Y_N2_s` DOF

**修改文件**：`core/layout.py`

**修改要点**：
1. 在 `build_layout()` 中，界面标量块（Ts, mpp, Rd）之后追加第 4 个界面标量 `Y_N2_s`
2. 在 `VarEntry.kind` 枚举中增加 `"Y_N2_s"`
3. 添加辅助方法 `idx_Y_N2_s() → int` 返回 `Y_N2_s` 在全局向量中的索引
4. 更新 `block_slices` 中的 `"Y_N2_s"` 键

**关联修改**：
- `apply_u_to_state(u, state, layout)` 中增加 `state.Y_N2_s = float(u[layout.idx_Y_N2_s()])`
- `pack_u(state, layout)` 中增加 `u[layout.idx_Y_N2_s()] = state.Y_N2_s`
- 初始猜测：`state.Y_N2_s_init = 1 - Y_fuel_s_init - Y_O2_s_init`（沿用旧 `_project_background` 估算值）

**验收**：
- `layout.size == old_size + 1`
- `layout.has_block("Y_N2_s") == True`
- `pack_u(unpack_u(u)) ≈ u`（往返测试）
- 现有布局单元测试通过

---

### Step 2：扩展 `core/types.py`——State 添加 `Y_N2_s` 字段

**修改文件**：`core/types.py`（或 `State` 数据类所在文件）

**修改要点**：

```python
@dataclass
class State:
    ...
    Y_N2_s: float = 0.0   # 界面 N₂ 质量分数（新界面 DOF）
```

**验收**：`State` 可序列化/反序列化（`io/` 模块检查）；旧快照文件（`.npz`/`.csv`）加载时 `Y_N2_s` 默认为 0，不报错

---

### Step 3：重构 `physics/interface_bc.py`——核心界面方程修改

这是方案 C 的核心步骤，分为三个子步骤。

#### Step 3a：重写 Ts 行——改为 VLE 约束

**当前**：`_build_Ts_row` 使用能量平衡确定 Ts
**目标**：VLE 残差 `F_Ts = Y_fuel,s,eq(Ts) − Y_fuel,s = 0`

实现要点：
- `Y_fuel,s,eq(Ts) = psat(Ts)/P × M_fuel / [Y_fuel,s/M_fuel + Y_N2_s/M_N2 + Y_O2_s/M_O2]⁻¹`（Raoult）
- 解析 Jacobian 对角：`∂F_Ts/∂Ts ≈ (1/P)×(dPsat/dTs)×(M_fuel/M_mix)`
- 对 Y_N2_s 和 Y_O2_s 的偏导：`∂F_Ts/∂Y_N2_s`（M_mix 依赖 Y_N2_s，有耦合）
- 或直接依赖 FD Jacobian 自动捕捉所有偏导（`jacobian_mode: mfpc_sparse_fd`）

#### Step 3b：重写 mpp 行——改为能量平衡（最优先）

**当前 EVAP**：Stefan 条件（退化）
**目标**：能量平衡（原 Ts 行内容迁移）

```
F_mpp = q_g + q_l + q_diff - mpp × L_v = 0

Jacobian:
  ∂F/∂mpp = -L_v                           (永不为零)
  ∂F/∂Tg[0] = -k_g × A_if / dr_g           (气相热传导系数)
  ∂F/∂Tl[0] = -k_l × A_if / dr_l           (液相热传导系数)
  ∂F/∂Ts    = (k_g/dr_g + k_l/dr_l) × A_if (汇总)
```

**实施顺序**：本步骤（3b）是消除 ΔY 退化的**最关键修改**，应最优先实施，可先于 3a 和 3c 单独测试。

#### Step 3c：新增 Y_N₂_s 行——Stefan 条件 for N₂

```
F_{Y_N2_s} = mpp × Y_N2_s + j_{N2,g} × A_if = 0

j_{N2,g} = −ρ_g × D_N2 / dr_g × (Y_N2_cell0 − Y_N2_s) × A_if

Jacobian:
  ∂F/∂Y_N2_s = mpp + ρ_g × D_N2 / dr_g × A_if   (注：Y_N2_cell0 是闭合量，不是DOF)
  ∂F/∂mpp    = Y_N2_s
  ∂F/∂Y_fuel[cell0], ∂F/∂Y_O2[cell0] 通过闭合重建间接耦合
```

**注意**：`Y_N2_cell0 = 1 - Y_fuel[cell0] - Y_O2[cell0]` 是闭合重建量（非 DOF），但 `Y_N2_s` 是新 DOF（界面面值）。两者在物理上不同：前者是体积单元体均值，后者是界面面值。

正则化处理（防止 mpp=0 时方程退化）：

```
F_{Y_N2_s} = mpp × Y_N2_s + j_{N2,g} × A_if + ε × (Y_N2_s − Y_N2_s_bg) = 0
```

其中 ε = 1e-8，`Y_N2_s_bg` 为背景值（远场估算）。

#### Step 3d：清理模式切换代码

- 删除 `_build_Ts_sat_row`（钉住方程）
- 删除 `use_sat_regime` 分支及所有相关条件判断
- 保留 `build_interface_coeffs()` 为单一路径（方案 C 路径）

**验收**（Step 3 整体）：
- 给定自洽解 `(Ts*, mpp*, Rd*, Y_N2_s*)` 时，四行残差 < 1e-12
- Jacobian 对角线在 Ts ∈ [300 K, Tbub] 全程有限且非零
- 接近沸点时（Ts = Tbub - 0.1 K）mpp 行条件数不超过 10⁶（与当前相比改善 3 个数量级）

---

### Step 4：修改 `properties/p2_equilibrium.py`——收窄 `_project_background` 职责

**修改目标**：`_project_background` 不再为 Y_N₂,s 赋值

**修改要点**：
1. `interface_equilibrium()` 返回的 `Yg_eq` 中，N₂ 位置赋 `0.0`（或 `nan`），调用方从 `state.Y_N2_s` 读取
2. `_project_background` 仅处理 O₂（若 O₂ 也不扩展，则可保留对 O₂ 的估算；若一并扩展则删除整函数）
3. 初始猜测阶段（第一步时间步）：用旧 `_project_background` 估算值初始化 `state.Y_N2_s`

**过渡策略**：初期可保留 `_project_background` 输出作为 `Y_N2_s` 的预测值（给 Newton 迭代提供好的初值），通过 `state.Y_N2_s` 传递；Newton 迭代会自动纠正到满足 Stefan 条件的真实值。

**验收**：`interface_equilibrium()` 在高 Stefan 数情况下（B = 5.77）的 N₂ 浓度预测与 quasi-steady 解析解误差 < 5%

---

### Step 5：简化 `assembly/residual_global.py`——移除模式切换逻辑

**修改要点**：
1. 删除 `_update_iface_regime()` 中的 `T_sat_enter_margin` override 逻辑（Fix 3 退役）
2. 删除 `_update_iface_regime()` 函数主体（或保留为空函数用于诊断日志）
3. 删除所有 `use_sat_regime` 标志的传递路径
4. 保留 Tbub 计算（用于 VI bounds 上界 `ts_upper_mode: "tbub_last"` 和诊断输出）
5. 保留 `interface_diag.csv` 中的 `regime` 字段输出（固定输出 `"option_c"`，便于调试）

**验收**：`_update_iface_regime` 不再影响 `interface_bc` 的方程路径；诊断 CSV 正常写出

---

### Step 6：YAML 与配置清理

**修改文件**：`cases/claude_tests_case.yaml` 及其他测试 YAML

**可移除参数**（Fix 3/4 的临时参数）：

```yaml
# 以下参数在 Option C 后不再需要：
# sat_tol_enter: 0.15
# sat_tol_exit: 0.10
# T_sat_enter_margin: 10.0
```

**可新增参数**（Option C 稳健性调节）：

```yaml
interface:
  stefan_N2_reg: 1.0e-8     # Y_N2_s Stefan 行的正则化系数（防 mpp=0 时退化）
  Y_N2_s_init_mode: "bg"    # 初始猜测策略: "bg"(远场估算) | "stefan_quasi_steady"
```

**验收**：YAML 加载、解析无报错；配置项有完整文档注释

---

### Step 7：整体集成测试与验收

#### 7.1 单元级验收

| 测试 | 验收标准 |
|---|---|
| `test_layout` | `layout.size = old+1`；`Y_N2_s` 块存在；pack/unpack 往返精度 < 1e-15 |
| `test_interface_bc_option_c` | 在自洽解处四行残差 < 1e-12；Jacobian 对角线全程 ≠ 0 |
| `test_equilibrium_no_N2_fill` | `Yg_eq[N₂_idx] = 0.0`（不再由 `_project_background` 填充） |
| `test_stefan_N2_consistency` | Stefan 条件 for N₂ 验证：`mpp×Y_N2_s + j_N2_g×A = 0` 精度 < 1e-10 |

#### 7.2 集成级验收

| 测试 | 验收标准 |
|---|---|
| 完整蒸发算例（T1） | 成功运行至 t_end，不发散；`scalars.csv` 中 Ts 单调上升 |
| 近沸点区段（step 100-160） | `mpp` 行 Jacobian 有效对角系数 > 1e-3（接近 L_v 量级） |
| 质量守恒检查 | `mass_closure_err < 1e-8` 全程成立 |
| Stefan 数检查 | `B = mpp×Rd/(ρ_g×D_N2)` 随时间平滑变化，无跳变 |

#### 7.3 回归验证（对比基准 Fixes 1–4）

| 阶段 | 验收标准 |
|---|---|
| 低温阶段（Ts < Tbub - 30 K） | Ts、mpp、Rd 曲线与基准吻合，相对误差 < 1% |
| 接近沸点（Ts ∈ [Tbub-10K, Tbub]） | 新代码平滑过渡（无模式切换跳变），基准代码可能振荡 |
| 沸点后（Ts = Tbub） | mpp 由能量供给平滑决定，无奇异 |

---

## 第五部分：风险与注意事项

### 5.1 主要实施风险

| 风险 | 描述 | 缓解策略 |
|---|---|---|
| Y_N₂_s 与闭合重建的双重存在 | 体积单元 Y_N₂ 是闭合量；界面面值 Y_N₂_s 是 DOF，两者在物理和数值上需严格隔离 | 代码审查：确保所有读取 Y_N₂ 的地方正确区分来源 |
| FD Jacobian 着色冲突 | 新 DOF Y_N₂_s 与近界面 Y_N₂ 闭合量之间可能存在结构上的着色图连通 | 审查 `jacobian_mode: mfpc_sparse_fd` 的稀疏模式；确认 Y_N₂_s 着色与体积 Y_N₂ 着色不冲突 |
| mpp=0 时 Y_N₂_s 行退化 | 若 ṁ=0（蒸发停止），`F = mpp × Y_N2_s + j_N2_g = 0` 蜕化为 `j_N2_g = 0`，Jacobian ∂F/∂Y_N2_s = 0 | 加入正则化项；或在 mpp < ε 时将 Y_N₂_s 固定为闭合重建值 |
| VLE 方程 Ts 行耦合复杂 | M_mix 依赖 Y_N₂_s 和 Y_O₂_s，导致 ∂F_Ts/∂Y_N₂_s 非零 | 先用 FD Jacobian 自动处理，后续可实现解析 Jacobian |
| 初始猜测质量 | Y_N₂_s 初值若偏差过大，Newton 迭代可能无法收敛 | 使用 `_project_background` 或 quasi-steady Stefan 公式提供好的初值 |

### 5.2 实施优先级建议

```
优先级 1（消除主因）：Step 3b    ← mpp 行改为能量平衡（最高回报/最低风险）
优先级 2（消除次因）：Step 3c    ← 新增 Y_N₂_s DOF + Stefan 行
优先级 3（结构完整）：Steps 1, 2 ← layout 和 State 扩展（Step 3b 的前提）
优先级 4（清理）：Steps 3a, 4, 5, 6 ← VLE Ts 行、equilibrium 清理、模式切换删除
```

**最小可行版本（MVP）**：仅实施 Steps 1+2+3b，将 mpp 方程改为能量平衡，保留 Ts 旧方程和 Y_N₂_s 近似（临时兼容），验证近沸点发散消失。此后再逐步完成 Steps 3a、3c、4、5、6。

---

## 参考文献

- Millán-Merino A, Fernández-Tarrazo E, Sánchez-Sanz M, Williams FA. *Numerical analysis of the autoignition of isolated droplets*. Combust. Flame, 2021. Eqs.(15)–(19).
