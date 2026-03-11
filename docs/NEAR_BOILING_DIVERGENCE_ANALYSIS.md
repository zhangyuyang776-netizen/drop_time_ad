# 近沸点位置发散问题根因分析报告

**项目**：drop_time_ad — 1D 球形液滴蒸发隐式 Newton 求解器
**案例**：`cases/claude_tests_case.yaml`（正十二烷/空气，T∞=1500 K，P=1 atm）
**发散位置**：时间步 step 133/134，Ts ≈ 488.7 K，Tbub ≈ 489 K
**分析日期**：2026-03-03

---

## 1. 现象描述

### 1.1 发散表现

仿真在液滴温度升高至接近沸点时（psat/P ≈ 0.958）突然发散：

| 量 | u_attempt（Newton前） | u_final（Newton后） | 状态 |
|---|---|---|---|
| Ts | 488.684 K | 5.14×10⁻⁵ K | ❌ 非物理 |
| mpp | 0.2019 kg/m²s | 2.25×10⁻⁵ | 错误 |
| Rd | 9.924×10⁻⁵ m | 17.45 m | ❌ 非物理 |

SNES 状态：`snes_reason=-6`（线搜索失败），`snes_iter=0`，`ksp_reason=2`（GMRES "收敛"但到错误方向）。

### 1.2 关键诊断数据（`out/T1-T4_fixed_out/`）

- **`interface_diag.csv`，step 132**：`Ts=488.684 K`，`psat/P=0.9580`，`regime=evap`，`DeltaY_eff≈0.007`
- **`failure_report.json`**：`ksp_it=23`，fieldsplit 预条件器设置失败，子块奇异
- **`u_attempt.csv` vs `u_final.csv`**：单次 Newton 步将 Ts 推至 0 K、Rd 推至 17 m

---

## 2. 根本原因分析

### 2.1 主因：Stefan 条件在接近沸点时的雅可比奇异性

#### 2.1.1 当前 mpp 方程（EVAP 模式）

当前代码（`physics/interface_bc.py`，`_build_mpp_row`）使用燃料组分的 Stefan 条件确定质量通量 mpp：

```
F_mpp = j_corr_b − mpp × ΔY_eff = 0

其中：ΔY_eff = Y_fuel,l,b − Y_fuel,g,eq,b
```

#### 2.1.2 退化机制

当液滴温度接近沸点时，相平衡（Raoult 定律）要求气相界面燃料质量分数：

```
Y_fuel,g,eq = psat(Ts)/P × M_fuel/M_mix → 1  （当 Ts → Tbub）
```

而液相界面燃料质量分数 `Y_fuel,l ≈ 1`，因此：

```
ΔY_eff = Y_fuel,l − Y_fuel,g,eq → 0  （当 Ts → Tbub）
```

**雅可比矩阵中 mpp 行的对角系数为 `−ΔY_eff ≈ −0.007`**，比其他方程行的系数小 2~3 个数量级，导致：

1. fieldsplit 预条件器中 mpp 块的条件数急剧恶化
2. GMRES 在 23 步内"收敛"，但收敛到错误的更新方向（`ksp_reason=2` 并非真正收敛）
3. 单次 Newton 步大小不受控：Ts 被推至 0 K，Rd 被推至 17 m

**这是发散的直接、主要原因**。

#### 2.1.3 量化

在 step 132，各量的数值如下：

```
ΔY_eff = 0.007       →  mpp 行 Jacobian 对角 ≈ 0.007
j_corr_b ≈ 0.142     →  右端项正常
比值 j_corr_b/ΔY_eff ≈ 20  → mpp ≈ 20 kg/m²s（正确量级）

但 Jacobian 行比值约 1/0.007 ≈ 143
其他行（Tg, Ts）对角系数 O(1~10)
→  条件数比 ≈ 143/10 ≈ 14（只看这两行）
```

考虑 FD Jacobian 扰动 `fd_eps=1e-8`：当 `ΔY_eff ~ O(fd_eps)` 时，扰动列与基态列的差被舍入误差淹没，雅可比列变为零向量，系统完全奇异。

---

### 2.2 次因：闭合组分 Y_N₂ 在近界面单元被压至零

#### 2.2.1 物理机制

由于 Stefan 流（对流）的存在，近界面气相的 N₂ 被向外对流推移。准稳态分析给出界面附近 N₂ 径向分布：

```
Y_N₂(r) = Y_N₂(R_d) × exp[B×(1−R_d/r)]

B = ṁ×R_d / (ρ_g × D_N₂) ≈ 5.77  （step 132 数据）
```

在 R_d ≈ 99 μm，特征长度 δ ≈ 17 μm，意味着 N₂ 浓度在 R_d+δ ≈ 116 μm 处就已下降一个数量级。**当网格分辨率不足时，近界面单元（cells 2–38）均可出现 Y_N₂ ≈ 0**。

#### 2.2.2 数值累积机制

在 `core/layout.py` 的 `_reconstruct_closure()` 中：

```python
Y_N₂ = 1 - Y_fuel - Y_O₂
closure = np.clip(closure, 0.0, 1.0)   # Fix 2 改为 1e-8
```

当 Y_fuel + Y_O₂ 数值积累超过 1.0 时（FD 有限差分扰动或每步截断误差），N₂ 被截断为 0（Fix 2 后为 1e-8）。

#### 2.2.3 后续链式影响

```
Y_N₂ → 0
  → Cantera 计算 D_N₂ → 0（扩散系数为零）
  → FD Jacobian 计算 (j_N₂ 列) 变为零列
  → 全局 Jacobian 新增奇异列
  → GMRES 求解质量进一步恶化
```

---

### 2.3 主次因联系

**两个根因均源于同一物理事件（Ts → Tbub），但各自独立作用于 Jacobian**：

```
Ts → Tbub
 ├─→ ΔY_eff = Y_fuel,l - Y_fuel,g,eq → 0
 │     → mpp 行 Jacobian 对角 → 0          ← 主因（直接奇异）
 │
 └─→ 强 Stefan 流 (B >> 1)
       → Y_N₂ 近界面 → 0
       → D_N₂ → 0                           ← 次因（加剧条件数）
```

**关键澄清**：次因（Y_N₂=0）并非由"错误的 Newton 步"引起，而是在 `u_attempt`（Newton 步执行之前）就已存在。两者是**并行的、同源的**先验条件，共同恶化了 Jacobian 质量，最终导致 Newton 步失败。

---

## 3. 已实施的临时修复（Fixes 1–4）

以下修复已提交至分支 `claude/fix-boiling-point-divergence-geKQy`，commit `a27e4cb`：

| 修复编号 | 文件 | 内容 | 性质 |
|---|---|---|---|
| Fix 1 | `properties/gas.py` | Y_min=1×10⁻⁸ floor 传入 Cantera；D_floor 从 1×10⁻⁹ 提高至 1×10⁻⁶ m²/s | **治标**：防 Cantera 因 Y_N₂=0 崩溃 |
| Fix 2 | `core/layout.py` | 闭合重建下界 0.0 → 1×10⁻⁸ | **治标**：防 Y_N₂ 在状态向量中为精确零 |
| Fix 3 | `assembly/residual_global.py` | 当 Ts ≥ Tbub - 10 K 时强制切换 SAT 模式（能量平衡驱动 mpp） | **缓解主因**：提前切换，避开 ΔY→0 区域 |
| Fix 4 | `cases/claude_tests_case.yaml` | `sat_tol_enter: 0.04 → 0.15`；新增 `T_sat_enter_margin: 10.0` | **辅助 Fix 3**：更早触发 SAT 模式 |

### 3.1 修复的局限性

- Fix 3 依赖 `T_sat_enter_margin=10.0 K` 这一经验参数——对于不同工况（燃料、压力），此阈值可能需要调整
- Fix 3 触发后使用 SAT 模式（Ts 钉住 + 能量平衡 mpp），但模式切换本身引入状态不连续性
- 界面组分（Y_N₂,s、Y_O₂,s）仍由 `_project_background` 比例分配，在高 Stefan 数情况下不自洽
- **根本机制未修复**：Stefan 条件确定 mpp 的方程在 EVAP 模式下仍存在退化风险

---

## 4. 根本修复思路（方案 C 概述）

彻底消除退化的根本方法是**改变界面方程角色**：

| 量 | 当前确定方程 | 根本修复后确定方程 |
|---|---|---|
| ṁ (mpp) | Stefan 条件（ΔY 退化） | **能量平衡**（L_v 永不为零） |
| Ts | 能量平衡 | **VLE 条件**（psat(Ts)=P·Y_fuel,s） |
| Y_N₂,s | 比例分配（非物理） | **Stefan 条件 for N₂**（新 DOF） |

此方案与 Millán-Merino (2021) Eqs.(15)–(19) 的物理框架完全一致，详见 `docs/OPTION_C_IMPLEMENTATION_PLAN.md`。

---

## 5. 参考文献

- Millán-Merino A, Fernández-Tarrazo E, Sánchez-Sanz M, Williams FA. *Numerical analysis of the autoignition of isolated droplets*. Combust. Flame, 2021. (Eqs. 15–19, Section 2.2)
- PETSc SNES/KSP documentation: `ksp_reason=2` (converged by relative tolerance, not by absolute residual reduction)
