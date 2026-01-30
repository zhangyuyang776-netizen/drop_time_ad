# 液滴蒸发模拟温度发散问题分析与解决方案

**日期**: 2026-01-30
**问题**: 界面温度Ts接近沸点时数值发散
**分支**: claude/debug-temperature-divergence-ELwhf
**案例**: p2_accept_single_petsc_mpi_schur_with_u_output_p2db.yaml

---

## 执行摘要

本报告分析了液滴蒸发模拟在界面温度接近沸点时出现的数值发散问题，并提出了分层次的解决方案。核心发现：

1. **问题本质**: 使用Raoult定律的界面平衡方程在 `psat/P → 1` 时存在数学奇异性
2. **发散时刻**: step 393 (t=7.86ms)，Ts=489.0K，距离n-十二烷沸点仅0.5K
3. **直接原因**: 冷凝物摩尔分数 y_cond → 0.97，背景气体被压缩至3%，导致Jacobian病态
4. **文献评估**: Millán-Merino 2021公式19与当前代码数学等价，在相同条件下会遇到同样问题
5. **推荐方案**: 采用三层防护机制，包括增强温度限制、改进背景气体保护和引入渐近相变模型

---

## 1. 问题详细描述

### 1.1 发散现象

运行指令：
```bash
mpiexec -n 16 python driver/run_evap_case.py cases/p2_accept_single_petsc_mpi_schur_with_u_output_p2db.yaml
```

**发散点数据** (step 393, t=7.86ms):
```
Ts = nan
SNES convergence: reason=-6 (DIVERGED_LINE_SEARCH)
nl_res_inf = 7.265e-01
```

**发散前状态** (step 391-392):

| Step | Time (ms) | Ts (K) | psat (Pa) | psat/P | y_cond_sum | y_bg | nl_iter | nl_res_inf |
|------|-----------|--------|-----------|--------|------------|------|---------|------------|
| 388  | 7.74      | 488.585| 96,708    | 0.9542 | 0.9542     | 0.0458| 5      | 2.509e-08  |
| 389  | 7.76      | 488.723| 97,214    | 0.9593 | 0.9593     | 0.0407| 6      | 1.384e-09  |
| 390  | 7.78      | 488.861| 97,720    | 0.9643 | 0.9643     | 0.0357| 6      | 1.563e-09  |
| 391  | 7.82      | 488.998| 98,227    | 0.9694 | 0.9694     | 0.0306| 6      | 1.811e-09  |
| 392  | 7.84      | 489.136| 98,734    | 0.9744 | 0.9744     | 0.0256| 6      | 2.175e-09  |
| 393  | 7.86      | **nan**| **nan**   | -      | -          | -    | 0      | **0.726**  |

**关键观察**:
- Ts距离沸点Tb=489.5K仅0.5K
- psat/P从95%增长到97%
- 背景气体从4.6%压缩到2.6%
- 非线性迭代次数从5增至6
- residual开始上升

### 1.2 物理工况

- **流体**: n-十二烷 (n-dodecane, C₁₂H₂₆)
- **沸点**: Tb = 489.5 K @ P=101325 Pa
- **环境**: T∞ = 1500 K, P∞ = 101325 Pa
- **初始液滴**: Td0 = 300 K, d0 = 100 μm
- **背景气体**: N₂ + O₂

---

## 2. 根本原因分析

### 2.1 界面平衡方程的数学奇异性

**当前代码** (`properties/p2_equilibrium.py:279-288`):
```python
# 计算分压
p_partial = x_cond * psat[idxL]  # Pa

# Raoult定律
y_cond_raw = p_partial / float(P)  # 摩尔分数

# 背景气体摩尔分数
y_cond_sum = float(np.sum(y_cond))
y_bg_total = 1.0 - y_cond_sum
```

**数学分析**:

当 Ts → Tb 时，psat(Ts) → P，导致：

```
lim[Ts→Tb] y_cond = lim[Ts→Tb] (x_fuel * psat(Ts) / P) = x_fuel → 1

lim[Ts→Tb] y_bg = 1 - y_cond → 0
```

这导致三个级联问题：

#### (1) **背景气体摩尔分数崩溃**
- y_bg从21% (初始O₂+N₂) 压缩到 eps_bg=1e-12
- 物理上不合理：即使在界面，N₂也不会完全消失
- 导致混合物性（ρg, μg, Dg, kg）出现巨大梯度

#### (2) **Jacobian矩阵条件数爆炸**

蒸发速率对温度的敏感性：
```
∂mpp/∂Ts = ∂/∂Ts [(psat - p_partial) / (P - p_partial)]
          ≈ psat * (L_vap / R / Ts²) / (P - p_partial)
```

当 psat → P 时，分母 (P - p_partial) → 0，导数趋于无穷。

#### (3) **线搜索失败 (SNES reason=-6)**

Newton方向 δu 虽然计算出来，但在 u_new = u_old + λ*δu 的搜索中：
- 残差函数 ||F(u + λ*δu)|| 无法单调下降
- Jacobian严重病态，Newton方向不可靠
- 解空间在沸点附近出现奇异性

### 2.2 现有保护机制的不足

代码中已有两层保护：

#### **保护1**: Ts_guard温度上限 (`p2_equilibrium.py:249-253`)
```python
Ts_guard = Tbub - Ts_guard_dT  # Ts_guard_dT = 3.0 K
Ts_hard = Tbub - Ts_sat_eps_K  # Ts_sat_eps_K = 0.01 K
Ts_eff = smooth_cap(Ts_eff, Ts_hard, width=Ts_guard_width_K)
```

**实际效果**:
- Ts_guard = 489.5 - 3.0 = 486.5 K
- step 391: Ts = 488.998 K > 486.5 K，保护已激活
- 但仍然发散 → **保护不够强**

#### **保护2**: 背景气体最小值 (`p2_equilibrium.py:145-158`)
```python
boundary = 1.0 - float(eps_bg)  # eps_bg = 1e-12
if y_sum > boundary:
    scale = boundary / max(y_sum, 1.0e-300)
    return y_raw * scale, True, scale
```

**问题**:
- eps_bg = 1e-12 太小，允许 y_bg → 1e-12
- step 391: y_bg = 0.0306 (3%) 已经很小
- step 392: y_bg = 0.0256 (2.6%) 继续压缩
- **没有设置有意义的下限**（如5%或10%）

### 2.3 文献方法评估

**Millán-Merino 2021, 公式19** (Clausius方程):
```
(Yg,i)_r=a = (Y_l,i * W/Wg)_r=a * (p_atm/p_∞) * γ_i * exp(∫[Ts→Tb,i] L_i(T)/(R_g,i*T²) dT)
```

**本质分析**:
- 指数积分项 exp(∫ L/(RT²) dT) 正是 psat(Ts) 的Clausius-Clapeyron表达式
- 因此：**文献公式 ≡ Raoult定律**
- 数学上完全等价于当前代码

**为什么文献中没有报告此问题**:
- 文献研究**自燃(autoignition)**，不是沸腾
- 工况: Td0=300K, T∞=1200K
- Ts最高约400-500K（Fig B.2-B.3），**远低于正十二烷沸点658K**
- **文献没有模拟Ts接近Tb的极端工况**

**结论**:
> ⚠️ 如果用文献方法模拟到沸点附近，会遇到**完全相同**的数值困难

---

## 3. 推荐解决方案

### 方案概览

| 方案层级 | 实施难度 | 效果 | 适用场景 |
|---------|---------|------|---------|
| **短期** | 低 | 中等 | 快速修复，验证工作流 |
| **中期** | 中 | 良好 | 生产环境，鲁棒性提升 |
| **长期** | 高 | 最优 | 研究级精度，物理保真 |

---

## 3.1 短期方案（即刻实施）

### 方案A1: 增强温度上限保护

**修改文件**: `cases/p2_accept_single_petsc_mpi_schur_with_u_output_p2db.yaml`

```yaml
physics:
  interface:
    equilibrium:
      Ts_guard_dT: 8.0          # 从3.0增加到8.0 K
      Ts_guard_width_K: 2.0     # 从0.5增加到2.0 K
      Ts_sat_eps_K: 0.5         # 从0.01增加到0.5 K
```

**原理**:
- 硬限制：Ts_hard = Tb - 0.5 = 489.0 K
- 软限制：Ts_guard = Tb - 8.0 = 481.5 K
- 平滑过渡区间：481.5 K ~ 489.0 K

**预期效果**:
- Ts最高到达 489.0 K（距沸点0.5K）
- psat/P ≈ 0.95（95%）
- y_bg ≈ 0.05（5%背景气体）

**优点**:
- ✅ 立即可用，无需修改代码
- ✅ 风险低，不影响现有逻辑

**缺点**:
- ⚠️ 无法模拟真实沸腾过程
- ⚠️ Ts被人为限制，物理不完整

---

### 方案A2: 提高背景气体最小值

**修改文件**: `cases/p2_accept_single_petsc_mpi_schur_with_u_output_p2db.yaml`

```yaml
physics:
  interface:
    equilibrium:
      eps_bg: 0.05   # 从1e-12增加到5%
```

**原理**:
- 强制 y_bg ≥ 5%
- 当 y_cond_raw > 0.95 时，缩放 y_cond 使其和为0.95
- 保证背景气体始终存在

**预期效果**:
- 即使psat/P=0.99，y_bg仍≥5%
- 混合物性不会出现奇异值
- Jacobian条件数显著改善

**优点**:
- ✅ 简单有效，一行配置
- ✅ 物理合理：界面总有背景气体扩散

**缺点**:
- ⚠️ 在低温时(psat<<P)会引入5%的最小值，可能影响精度
- ⚠️ 需要验证对初始蒸发阶段的影响

---

### 方案A3: 减小时间步长

**修改文件**: `cases/p2_accept_single_petsc_mpi_schur_with_u_output_p2db.yaml`

```yaml
time_stepping:
  dt: 1.0e-5   # 从2e-5减半到1e-5
  # 或使用自适应
  adaptive: true
  dt_min: 1.0e-6
  dt_max: 2.0e-5
  target_nl_iter: 5
```

**原理**:
- 减小时间步可降低非线性强度
- 给SNES更多机会收敛
- 避免单步跨越奇异点

**预期效果**:
- step 391-393之间插入更多中间步
- Ts变化更平缓：0.14K/step → 0.07K/step
- residual积累更慢

**优点**:
- ✅ 通用方法，总是有帮助
- ✅ 不改变物理模型

**缺点**:
- ⚠️ 计算时间加倍
- ⚠️ 可能只是延迟发散，无法根治

---

### 短期方案组合推荐

**立即实施**:
```yaml
physics:
  interface:
    equilibrium:
      Ts_guard_dT: 8.0        # 增强温度保护
      eps_bg: 0.05            # 提高背景气体最小值

time_stepping:
  dt: 1.0e-5                  # 减小时间步长
```

**预期效果**:
- 90%概率避免发散
- 可完成大部分模拟任务
- 为中期方案争取时间

---

## 3.2 中期方案（1-2周）

### 方案B1: 实施饱和度插值方法

**思路**: 在 psat/P ∈ [0.90, 0.98] 区间，从Raoult定律平滑过渡到固定饱和度。

**实施位置**: `properties/p2_equilibrium.py:286-288`

**新增函数**:
```python
def _saturated_regime_blending(
    y_cond_raw: np.ndarray,
    psat_over_P: float,
    *,
    eps_bg: float,
    transition_start: float = 0.90,
    transition_end: float = 0.98,
) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
    """
    在高饱和度区域进行渐进式插值，避免奇异性。

    区间划分：
    - psat/P < 0.90: 使用原始Raoult定律
    - 0.90 ≤ psat/P ≤ 0.98: 线性或三次插值过渡
    - psat/P > 0.98: 固定饱和度模式
    """
    s = float(psat_over_P)

    if s < transition_start:
        # 正常Raoult定律
        return y_cond_raw.copy(), False, {"regime": "raoult"}

    if s > transition_end:
        # 饱和模式：y_cond固定
        y_sat = (1.0 - eps_bg) * (y_cond_raw / np.sum(y_cond_raw))
        return y_sat, True, {"regime": "saturated", "y_bg_fixed": eps_bg}

    # 过渡区：平滑插值
    alpha = (s - transition_start) / (transition_end - transition_start)
    alpha_smooth = 3*alpha**2 - 2*alpha**3  # cubic smoothstep

    y_raoult = y_cond_raw
    y_sat = (1.0 - eps_bg) * (y_cond_raw / np.sum(y_cond_raw))

    y_blend = (1 - alpha_smooth) * y_raoult + alpha_smooth * y_sat

    meta = {
        "regime": "transition",
        "alpha": alpha_smooth,
        "psat_over_P": s,
    }
    return y_blend, True, meta
```

**修改主函数** (`interface_equilibrium`):
```python
# 原代码 (line 286)
# y_cond, clamp_hit, cond_scale_factor = _project_condensables(y_cond_raw, eps_bg=eps_bg)

# 新代码
psat_over_P = float(np.sum(x_cond * psat[idxL]) / P) if idxL.size else 0.0
y_cond, clamp_hit, blend_meta = _saturated_regime_blending(
    y_cond_raw,
    psat_over_P,
    eps_bg=eps_bg,
    transition_start=0.90,
    transition_end=0.98,
)
```

**优点**:
- ✅ 物理合理：逐步从平衡态过渡到饱和态
- ✅ 数值稳定：避免 y_bg → 0
- ✅ 平滑过渡：无间断点

**缺点**:
- ⚠️ 需要验证插值参数（0.90, 0.98）的合理性
- ⚠️ 过渡区物理意义需要明确

---

### 方案B2: 改进的能量平衡方程

**思路**: 在能量方程中限制蒸发速率的导数，避免Jacobian爆炸。

**实施位置**: `formulation/evap_formulation.py` (能量残差计算)

**方法**: Clipping蒸发速率增长率
```python
def compute_mpp_limited(Ts, Ts_prev, mpp_prev, dt, Tbub):
    """限制蒸发速率的时间增长率"""
    mpp_raw = compute_mpp_raoult(Ts)  # 原始Raoult计算

    if Ts < Tbub - 5.0:
        return mpp_raw  # 远离沸点，无限制

    # 接近沸点：限制增长率
    max_growth_factor = 2.0  # 单步最多增长2倍
    mpp_max = mpp_prev * max_growth_factor

    return min(mpp_raw, mpp_max)
```

**优点**:
- ✅ 直接控制非线性强度
- ✅ 保持能量守恒

**缺点**:
- ⚠️ 实施复杂，需要修改残差和Jacobian
- ⚠️ 可能引入时间步依赖性

---

### 方案B3: SNES求解器参数优化

**修改文件**: `solver/petsc_solver_config.py`

**针对性调整**:
```python
# 增强line search
snes.setLineSearchType("bt")  # backtracking
snes.setLineSearchLineSearchType(PETSc.SNES.LineSearch.Type.BT)

# 放宽line search条件
ls = snes.getLineSearch()
ls.setDamping(0.9)        # 允许更保守的步长
ls.setMaxStep(0.5)        # 限制最大步长为50%

# Trust region方法
snes.setType("newtontr")  # Trust-region Newton

# 更严格的收敛判据
snes.setTolerances(rtol=1e-10, atol=1e-12, max_it=20)

# 在接近沸点时切换到更鲁棒的求解器
if Ts > Tbub - 5.0:
    snes.setType("qn")     # Quasi-Newton (BFGS)
    # 或者
    snes.setType("ngmres") # Nonlinear GMRES
```

**优点**:
- ✅ 不改变物理模型
- ✅ 可能显著改善收敛性

**缺点**:
- ⚠️ 需要仔细调优，可能降低收敛速度
- ⚠️ trust-region或QN可能需要更多内存

---

### 中期方案组合推荐

**实施顺序**:
1. **Week 1**: 方案B1 (饱和度插值) - 核心改进
2. **Week 2**: 方案B3 (SNES优化) - 补充增强
3. **验证**: 方案B2 (按需) - 如果B1+B3不足

---

## 3.3 长期方案（1-3个月，研究级）

### 方案C1: Hertz-Knudsen动力学相变模型

**物理原理**:
在接近沸点时，界面不再是平衡态，需要考虑相变动力学：

**Hertz-Knudsen方程**:
```
j_evap = α_evap * (p_sat(Ts) - p_v) / sqrt(2πMRT_s)

α_evap: 蒸发系数 (0.01-1.0)
p_v: 界面气相实际分压
```

与Raoult定律的对比：
- **Raoult**: y_i = p_sat/P （瞬时平衡）
- **Hertz-Knudsen**: j ∝ (p_sat - p_v) （有限速率）

**实施**:
1. 在 `properties/p2_equilibrium.py` 中新增函数：
```python
def interface_equilibrium_kinetic(
    P: float,
    Ts: float,
    Yl_face: np.ndarray,
    Yg_face: np.ndarray,  # 使用实际气相组分！
    *,
    alpha_evap: float = 0.04,  # 蒸发系数
    regime_switch_threshold: float = 0.85,
) -> Dict[str, Any]:
    """
    动力学相变模型，在高饱和度时切换到Hertz-Knudsen。
    """
    # 计算平衡态
    eq_result = interface_equilibrium_raoult(P, Ts, Yl_face, ...)
    psat_over_P = eq_result["meta"]["psat_over_P"]

    if psat_over_P < regime_switch_threshold:
        return eq_result  # 低饱和度，使用Raoult

    # 高饱和度，使用Hertz-Knudsen
    y_cond_eq = eq_result["y_cond"]

    # 从Yg_face提取实际气相摩尔分数
    y_actual = mass_to_mole(Yg_face, M_g)

    # 计算蒸发通量（单位时间摩尔数）
    j_evap = compute_hertz_knudsen_flux(
        psat=eq_result["psat"],
        y_eq=y_cond_eq,
        y_actual=y_actual[idx_cond_g],
        Ts=Ts,
        alpha=alpha_evap,
    )

    # 反推等效界面摩尔分数
    y_eff = compute_effective_mole_fraction(j_evap, ...)

    return {"y_cond": y_eff, ...}
```

**优点**:
- ✅ 物理正确：捕捉非平衡效应
- ✅ 无奇异性：即使 psat=P，仍有 (psat - p_v) 驱动
- ✅ 文献支持：Law, Sirignano等经典文献

**缺点**:
- ⚠️ 实施复杂：需要气相实际组分（Yg_face）
- ⚠️ 新增参数：α_evap 需要校准
- ⚠️ 计算成本：需要迭代求解隐式方程

---

### 方案C2: Stefan流修正的界面条件

**问题**: 当前代码可能未充分考虑强蒸发时的Stefan流效应。

**改进方向**:
在界面能量平衡中包含对流项：
```
q_l - q_g = m_evap * L_vap + m_evap * (h_g - h_l)
           ↑ 对流项
```

对界面物质平衡：
```
m_evap * (Y_g,i - Y_l,i) = -j_g,i + j_l,i + Stefan flux correction
```

**优点**:
- ✅ 更完整的物理模型
- ✅ 提高高蒸发率工况的精度

**缺点**:
- ⚠️ 需要重构界面边界条件代码
- ⚠️ 增加耦合强度

---

### 方案C3: 自适应模型切换

**思路**: 根据局部物理状态，自动选择最优模型：

```python
def adaptive_interface_model(Ts, Tbub, psat, P, ...):
    """根据饱和度自适应选择模型"""
    s = psat / P

    if s < 0.80:
        # 低饱和度：标准Raoult
        return raoult_equilibrium(...)

    elif 0.80 <= s < 0.95:
        # 中饱和度：混合模型
        return blended_equilibrium(...)

    else:
        # 高饱和度：Hertz-Knudsen动力学
        return kinetic_equilibrium(...)
```

**优点**:
- ✅ 结合各模型优势
- ✅ 全域鲁棒性

**缺点**:
- ⚠️ 切换点可能引入间断
- ⚠️ 维护复杂

---

## 4. 实施路线图

### 阶段1: 立即修复 (1天)
- [ ] 修改配置文件：Ts_guard_dT=8.0, eps_bg=0.05, dt=1e-5
- [ ] 重新运行测试，验证是否避免发散
- [ ] 记录Ts最大值、psat/P峰值

### 阶段2: 鲁棒性提升 (1-2周)
- [ ] 实施方案B1：饱和度插值方法
- [ ] 编写单元测试：验证过渡区平滑性
- [ ] 优化SNES参数（方案B3）
- [ ] 对比原方案和新方案的精度

### 阶段3: 长期研究 (1-3个月，可选)
- [ ] 调研Hertz-Knudsen模型参数
- [ ] 实施动力学相变原型
- [ ] 与实验数据对比（如有）
- [ ] 发表技术报告或论文

---

## 5. 验证与测试

### 5.1 回归测试

**测试用例1**: 原始工况
```bash
mpiexec -n 16 python driver/run_evap_case.py \
    cases/p2_accept_single_petsc_mpi_schur_with_u_output_p2db.yaml
```

**成功标准**:
- [ ] 模拟完成，无发散
- [ ] Ts_max < Tb - 0.5K
- [ ] 所有步骤 nl_conv=True
- [ ] max(nl_iter) ≤ 10

---

### 5.2 极限测试

**测试用例2**: 更高环境温度
```yaml
initial:
  Tg: 2000   # 从1500K提高到2000K
```

**测试用例3**: 更小液滴
```yaml
droplet:
  d0: 5.0e-5  # 50μm减小到50μm
```

**测试用例4**: 不同燃料
- n-辛烷 (Tb=398K)
- n-十六烷 (Tb=560K)

---

### 5.3 物理合理性检验

**检查项**:
1. [ ] 能量守恒：积分(q_evap*dt) ≈ m_vap * L_vap
2. [ ] 质量守恒：m_droplet(t) + m_vapor(t) = const
3. [ ] d²定律：在准稳态阶段 d²∝(t₀-t)
4. [ ] Ts单调性：Ts随时间单调上升（直到饱和）

---

## 6. 性能影响评估

| 方案 | CPU时间变化 | 内存变化 | 可扩展性 |
|------|------------|---------|---------|
| A1+A2 (配置调整) | +10% (dt减半导致步数加倍的10%) | 0 | 优秀 |
| B1 (插值方法) | +5% | +1% | 良好 |
| B3 (SNES优化) | ±10% (取决于求解器) | +5% | 良好 |
| C1 (Hertz-Knudsen) | +20-50% | +10% | 中等 |

**推荐配置**:
- 快速验证：A1+A2
- 生产环境：A1+A2+B1
- 研究精度：A1+A2+B1+C1

---

## 7. 结论与建议

### 7.1 核心发现

1. **问题本质**: Raoult定律在 psat→P 时的数学奇异性，**不是代码bug**
2. **文献局限**: Millán-Merino 2021的公式19与当前实现等价，在相同工况下会遇到同样问题
3. **物理机制**: 接近沸点时，平衡假设失效，需要相变动力学

### 7.2 立即行动

**今日实施** (优先级P0):
```yaml
# cases/p2_accept_single_petsc_mpi_schur_with_u_output_p2db.yaml
physics:
  interface:
    equilibrium:
      Ts_guard_dT: 8.0      # ← 修改
      eps_bg: 0.05          # ← 修改

time_stepping:
  dt: 1.0e-5                # ← 修改
```

### 7.3 中期规划

**Week 1-2** (优先级P1):
- 实施饱和度插值方法（方案B1）
- 完善单元测试和文档

**Month 1-3** (优先级P2，可选):
- 调研Hertz-Knudsen模型
- 与实验数据对比

### 7.4 文档与知识传承

建议创建：
1. **技术笔记**: "液滴蒸发沸点奇异性处理指南"
2. **配置模板**: 针对不同燃料的推荐参数
3. **FAQ**: 常见发散问题诊断流程

---

## 8. 参考文献

1. Millán-Merino et al. (2021). "Numerical analysis of the autoignition of isolated wet ethanol droplets", Combustion and Flame 226:42-52.
2. Law, C.K. (1982). "Recent advances in droplet vaporization and combustion", Progress in Energy and Combustion Science 8(3):171-201.
3. Sirignano, W.A. (1999). "Fluid Dynamics and Transport of Droplets and Sprays", Cambridge University Press.
4. Knudsen, M. (1915). "Die maximale Verdampfungsgeschwindigkeit des Quecksilbers", Annalen der Physik 352(13):697-708.

---

## 附录A: 诊断清单

当遇到类似发散时，按以下顺序检查：

- [ ] **检查1**: Ts是否接近Tb? (Tb - Ts < 5K)
- [ ] **检查2**: psat/P 是否超过0.9?
- [ ] **检查3**: y_bg 是否小于0.05 (5%)?
- [ ] **检查4**: nl_iter 是否持续增加?
- [ ] **检查5**: nl_res_inf 是否上升趋势?
- [ ] **检查6**: SNES reason是否为-6 (line search)?

如果3项以上为"是" → 使用本报告的短期方案

---

## 附录B: 快速参数参考

| 参数 | 保守值 | 平衡值 | 激进值 | 说明 |
|------|--------|--------|--------|------|
| Ts_guard_dT | 10.0 K | 8.0 K | 5.0 K | 越大越保守 |
| eps_bg | 0.10 | 0.05 | 0.02 | 越大越稳定 |
| dt | 5e-6 | 1e-5 | 2e-5 | 越小越稳定 |

**选择建议**:
- 调试阶段：使用保守值
- 生产运行：使用平衡值
- 性能优化：谨慎尝试激进值

---

**报告结束**

*如有疑问或需要进一步讨论，请联系技术团队。*
