# P2 多组分液相：物性混合与界面平衡（定稿说明）

> 目的：在**理想气体（气相）**与**理想溶液（液相）**假设下，给出一套**可实现、可自洽、带护栏**的多组分液滴物性混合与界面平衡方案，用于替换 CoolProp 并避免“界面温度到沸点就发散”。

---

## 0. 适用范围与核心假设

### 0.1 气相
- 气相为理想气体：  
  \[
  f_{G,i} = x_{G,i}\,P \quad (\phi_i^G = 1)
  \]
- 气相热物性/输运物性由 **Cantera** 计算（本文件不改动这部分）。

### 0.2 液相（混合物）
- 液相为理想溶液：活动系数 \(\gamma_i = 1\)。
- 液相体性质（\(\rho, c_p, \mu, k, \sigma\) 等）采用“混合规则”（见第 2 节）。
- 每个组分的纯物性关联式均为 **单变量温度函数**（自变量只有 \(T\)）：
  - \(p_i^{sat}(T)\)、\(\Delta h_{vap,i}(T)\)、\(\rho_i(T)\)、\(c_{p,i}(T)\)、\(\mu_i(T)\)、\(k_i(T)\)、\(\sigma_i(T)\)。

> 压力 \(P\) 的作用：只进入 **界面平衡/泡点约束**（第 3 节）。液相 bulk 物性在 P2 版本不显式依赖压力。

---

## 1. 记号与转换

- \(P\)：界面处总压（Pa）
- \(T_s\)：求解器给出的界面温度（K）
- \(T_s^{eff}\)：施加沸点/泡点护栏后的有效界面温度（K）
- \(i\)：挥发组分（condensables）索引，\(i=1\ldots N_v\)
- 液相**质量分数**：\(Y_{l,i}\)，液相**摩尔分数**：\(x_{l,i}\)
- 界面气相**摩尔分数**：\(x_{g,i}\)；界面气相**质量分数**：\(Y_{g,i}\)
- 分子量：\(M_i\)

### 1.1 液相 \(Y \leftrightarrow x\) 转换（必须统一）
由质量分数得到摩尔分数：
\[
x_{l,i} = \frac{Y_{l,i}/M_i}{\sum_j Y_{l,j}/M_j}
\]
由摩尔分数得到质量分数：
\[
Y_{l,i} = \frac{x_{l,i} M_i}{\sum_j x_{l,j} M_j}
\]

> 注意：界面平衡（Raoult）使用的是 **摩尔分数 \(x_{l,i}\)**，不是质量分数。

---

## 2. 多组分液相 bulk 物性混合规则（P2 定稿版）

目标：**稳定、连续、易实现**。优先采用不会产生负值/奇异的混合形式。

> 输入：\(T\)、液相组分 \(Y_l\) 或 \(x_l\)  
> 输出：\(\rho_l, c_{p,l}, \mu_l, k_l, \sigma_l\)（以及可选的 \(h_l\)）

### 2.1 密度 \(\rho_l(T, \mathbf{Y}_l)\)
推荐使用**比体积加权**（比直接线性加权密度更稳）：
\[
\boxed{\;\frac{1}{\rho_l} = \sum_i \frac{Y_{l,i}}{\rho_i(T)}\;}
\]
- 需要：每个组分的 \(\rho_i(T)\) 纯物性关联式。

### 2.2 比热 \(c_{p,l}(T, \mathbf{Y}_l)\)
推荐使用**质量分数线性加权**（热容是广延量，质量基更自然）：
\[
\boxed{\;c_{p,l} = \sum_i Y_{l,i}\,c_{p,i}(T)\;}
\]

### 2.3 黏度 \(\mu_l(T, \mathbf{x}_l)\)
推荐使用**对数混合**（保证正值、数值稳定）：
\[
\boxed{\;\ln \mu_l = \sum_i x_{l,i}\,\ln \mu_i(T)\;}
\quad\Rightarrow\quad
\mu_l = \exp\!\left(\sum_i x_{l,i}\ln\mu_i(T)\right)
\]
> 这是 P2 的默认规则；更精确的混合（Grunberg–Nissan 等）可作为后续扩展点。

### 2.4 导热系数 \(k_l(T, \mathbf{Y}_l)\)
P2 推荐先用**质量分数线性加权**：
\[
\boxed{\;k_l = \sum_i Y_{l,i}\,k_i(T)\;}
\]
> 后续若需要更准确，可替换为对数混合或对应态混合法，但先别自找麻烦。

### 2.5 表面张力 \(\sigma_l(T, \mathbf{x}_l)\)（若模型用到）
P2 推荐使用**摩尔分数线性加权**：
\[
\boxed{\;\sigma_l = \sum_i x_{l,i}\,\sigma_i(T)\;}
\]
> 若后续发现界面张力对结果敏感，再引入更复杂规则（如 Macleod–Sugden）。

### 2.6 液相焓 \(h_l(T,\mathbf{Y}_l)\)（可选）
若求解器需要液相绝对焓，P2 推荐：
- 使用参考温度 \(T_{ref}\)（在配置中给定，例如 298.15 K）
- 以纯组分 \(c_{p,i}(T)\) 积分得到相对焓：
\[
h_i(T) = h_i(T_{ref}) + \int_{T_{ref}}^{T} c_{p,i}(\theta)\,d\theta
\]
- 混合焓：
\[
\boxed{\;h_l = \sum_i Y_{l,i}\,h_i(T)\;}
\]

---

## 3. 多组分界面平衡（理想气体 + Raoult）与“泡点护栏”

当多个组分的饱和蒸气压都很大时，按平衡推出来的界面气相组分可能满足：
\[
\sum_i x_{g,i} > 1
\]
这在物理上不可能，数值上会导致闭合组分变负、通量公式奇异，从而发散。

**解决办法不是随便改组分，而是约束界面温度进入模型有效域。**

### 3.1 Raoult 定律（理想溶液）
对挥发组分 \(i\)：
\[
\boxed{\;x_{g,i}^\* = \frac{x_{l,i}\,p_i^{sat}(T_s^{eff})}{P}\;}
\]

### 3.2 混合物“泡点温度”定义（关键）
在给定 \(P\) 与液相组成 \(\mathbf{x}_l\) 下，泡点 \(T_{bub}\) 定义为：
\[
\boxed{\;\sum_i x_{l,i}\,p_i^{sat}(T_{bub}) = P\;}
\]

### 3.3 泡点护栏：把 \(T_s\) 约束在 \(T_{bub}\) 以下
P2 定稿采用：
\[
\boxed{\;T_s^{eff} = \min\!\left(T_s,\;T_{bub}(P,\mathbf{x}_l)-\Delta T_{guard}\right)\;}
\]
- \(\Delta T_{guard}\) 建议默认 0.1–1.0 K（可配置）。
- 可选“平滑护栏”（softplus/tanh）以改善牛顿迭代的导数连续性；P2 默认先用硬护栏，稳定优先。

### 3.4 计算界面气相组成与闭合
1) 计算挥发组分未截断值：
\[
x_{g,i}^\* = \frac{x_{l,i}\,p_i^{sat}(T_s^{eff})}{P}
\]
2) 总和：
\[
S = \sum_i x_{g,i}^\*
\]
3) **数值兜底护栏**：
- 若 \(S > 1-\epsilon\)，比例缩放：
\[
x_{g,i} \leftarrow x_{g,i}^\*\cdot\frac{1-\epsilon}{S}
\]
- 否则 \(x_{g,i}=x_{g,i}^\*\)。
- 推荐 \(\epsilon = 10^{-12}\sim 10^{-8}\)。

4) 闭合惰性/其余气体：
\[
x_{g,inert} = 1 - \sum_i x_{g,i}
\]
并要求 \(x_{g,inert}\ge 0\)。

### 3.5 界面气相质量分数（用于质量基通量）
\[
Y_{g,i} = \frac{x_{g,i} M_i}{\sum_k x_{g,k} M_k}
\]
定义挥发组分的质量分数向量：
\[
\mathbf{y}_{cond} = \left\{ Y_{g,i} \right\}_{i=1..N_v}
\]

---

## 4. 多组分汽化潜热与界面能量项（定稿建议）

- 组分汽化潜热：\(\Delta h_{vap,i}(T_s^{eff})\)

若有组分蒸发通量 \( \dot{m}_i \)：
\[
\boxed{\;q_{lat} = \sum_i \dot{m}_i\,\Delta h_{vap,i}(T_s^{eff})\;}
\]

若仅有总蒸发通量 \(\dot{m}\)，用 \(y_{cond,i}\) 分配：
\[
\dot{m}_i = \dot{m}\,y_{cond,i}
\Rightarrow
q_{lat}=\dot{m}\sum_i y_{cond,i}\Delta h_{vap,i}(T_s^{eff})
\]
定义等效潜热：
\[
\boxed{\;\Delta h_{vap,eff}=\sum_i y_{cond,i}\Delta h_{vap,i}(T_s^{eff})\;}
\]

---

## 5. 数值护栏与截断（必须定稿）

### 5.1 组成护栏
- \(Y_{l,i}\ge 0\)，\(\sum_i Y_{l,i}=1\)（允许小误差，需归一化）。
- \(x_{l,i}\ge 0\)，\(\sum_i x_{l,i}=1\)。

### 5.2 温度护栏
所有关联式有有效温区 \([T_{min},T_{max}]\)，先 clamp：
\[
T \leftarrow \mathrm{clip}(T,\;T_{min}+\delta,\;T_{max}-\delta)
\]
\(\delta\) 建议 1e-6～1e-3 K。

### 5.3 泡点求解（1D 求根）
\[
F(T)=\sum_i x_{l,i}p_i^{sat}(T)-P
\]
- 优先二分法，需 bracket \(F(T_{lo})<0<F(T_{hi})\)。
- 找不到 bracket 必须打诊断并进入保守退化（例如只用缩放兜底）。

### 5.4 \(\sum x_g\) 护栏
执行第 3.4 的缩放，保证 \(\sum x_{g,i}\le 1-\epsilon\)。

### 5.5 诊断 meta（建议保留）
- `meta.T_bub`, `meta.Ts_eff`, `meta.sum_xg`, `meta.scaled`, `meta.guard_reason`

---

## 6. 对外接口建议（与现有代码对接）

### 6.1 液相物性
`compute_liquid_mixture_props(T, Y_l, species_list, params_db) -> dict`
- 返回：`rho_l, cp_l, mu_l, k_l, sigma_l`（可选 `h_l`）
- 同时返回 `x_l`（避免重复转换）

### 6.2 界面平衡
`compute_interface_equilibrium_multiliq(Ts, P, x_l, condensables, params_db) -> dict`
- 返回：`Ts_eff`, `xg_cond`, `Yg_cond(y_cond)`, `psat_i`, `hvap_eff`, `meta`

---

## 7. 已知局限与扩展点（先写清楚）

- 默认 \(\gamma_i=1\)，未引入 UNIFAC 等。
- 未引入 Poynting 修正（压力项），P2 不启用。
- 未引入非理想气体（\(\phi_i^G\)），P2 不启用。
- 扩展后仍应保留“泡点护栏”作为防发散底盘。

---

## 8. 最小验收（建议作为单元测试）

1) 泡点求解正确：  
\[
\left|\sum_i x_{l,i}p_i^{sat}(T_{bub})-P\right| < tol
\]
2) 对任意 \(T_s\)：  
- \(T_s^{eff} \le T_{bub}-\Delta T_{guard}\)  
- \(\sum_i x_{g,i}\le 1-\epsilon\)  
- \(x_{g,i}\ge 0\)，且闭合 \(x_{g,inert}\ge 0\)
3) 温区扫描输出无 NaN/inf，随 \(T\) 连续变化。

---

**至此：多组分物性混合 + 多组分界面平衡与护栏（P2 版）已定稿。**
