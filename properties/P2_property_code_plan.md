# P2 物性计算代码实现方案（基于现有 liquid.py / equilibrium.py 接口）

> 目标：**运行时彻底去掉 CoolProp**，仍保持你现在求解器/接口层调用方式基本不变；并把“表面温度 <= 沸点”“y_cond 不爆炸”的护栏机制固化为可测试的代码契约。

---

## 0. 现状接口（你已经有了，不要再推倒重来）

你上传的两份旧代码已经把“上层如何调用”基本写死了：

- `build_liquid_model(cfg, *, liquid_names, gas_liq_map)`：构造液相物性模型（后面会被求解器反复用）【turn27:12†liquid.py†L17-L24】。
- `compute_liquid_props(model, T, P, Yl, ...)`（具体函数名/签名以你文件为准，但核心是：输入 `T,P,液相组成`，输出 `rho_l, cp_l, k_l, h_l` 等，并支持多组分混合）。
- `build_equilibrium_model(cfg, *, gas_names, liq_names, ...)`：构造界面平衡模型；内部有 `sat_source/psat_model` 的选择逻辑（原本支持 CoolProp / custom）【turn29:14†coolprop_usage.md†L8-L23】【turn29:14†coolprop_usage.md†L52-L73】。
- `compute_interface_equilibrium_full(eq_model, Ts, P, Yl_face, Yg_far, ...)`：给出界面平衡（`y_cond, Yg_eq, psat, hvap` 等）。

**实现 P2 的正确姿势：保持这些入口函数名字、返回字段尽量不变。**你现在的 baseline 工具链也依赖这些字段名来写 `baseline_coolprop.npz`【turn29:1†_baseline_utils.py†L7-L27】，meta 里记录了 `sat_source/psat_model/species/pressures/temperatures`【turn29:10†baseline_coolprop_meta.json†L22-L27】【turn29:10†baseline_coolprop_meta.json†L44-L80】。

---

## 1. P2 总体策略

### 1.1 代码结构（推荐）

新增一个 **P2 物性包**，不要继续把所有相关函数塞进 `liquid.py` / `equilibrium.py` 里：

```
properties/
  p2_liquid_db.py          # 读 YAML 参数库 + 基本校验
  p2_pure_models.py        # 纯组分: rho, cp, k, mu, psat, hvap ...
  p2_mix_rules.py          # 多组分混合规则
  p2_equilibrium.py        # 多组分界面平衡 (Raoult + bubble-point guard + clamp)
  p2_safeguards.py         # Ts_cap / y_cond clamp / NaN 处理
  data/
    liquid_props_db.yaml   # 物性参数库（你要的“定稿约束文件”的代码侧落地）
```

然后：
- `liquid.py` 只负责 **适配器**：把原先 CoolProp 的实现换成调用 `properties/p2_*`。
- `equilibrium.py` 同理：保留外壳和返回结构，内部改成 P2。

### 1.2 关键原则

1. **运行时不依赖 CoolProp**（但允许用你生成的 CoolProp baseline 离线拟合参数库）。
2. **P 只进入 psat/Tb/bubble-point**：其他液相物性先按“只与 T 相关”的模型做（你之前已经决定了）。
3. **护栏必须可测试**：
   - `Ts_eff <= Tb(P, x_l) - ΔT_guard`
   - `sum(y_cond) <= 1 - eps_bg`
   - 任何物性/平衡输出必须 finite（无 NaN/inf），否则走降级分支并把 `meta.fallback_reason` 写清楚。

---

## 2. P2 纯组分物性模型（实现约束）

### 2.1 输入/输出约束

所有纯组分模型统一签名（建议）：

```python
# p2_pure_models.py

def rho_l(T, coeffs) -> float

def cp_l(T, coeffs) -> float

def k_l(T, coeffs) -> float

def mu_l(T, coeffs) -> float

def h_l(T, coeffs, T_ref=298.15) -> float

def psat(T, coeffs) -> float  # Pa

def hvap(T, coeffs) -> float  # J/kg
```

并且统一规则：
- `T` 单位 K。
- psat 输出 Pa。
- hvap 输出 J/kg（不是 J/mol）。

### 2.2 相关式类型（先做够用的，别上来就写 20 种）

你要的不是“百科全书”，是“可控可测可收敛”的物性核。

最低配组合（建议作为 P2.0）：

- `psat`: Antoine（log10 或 ln 形式，具体在 YAML 里声明）
- `hvap`: Watson（用 `Tc, hvap_ref, T_ref`）
- `rho_l`: 多项式 / 指数式（先用 baseline 拟合，后续再替换为文献相关式）
- `cp_l`: 多项式（同上）
- `k_l`: 多项式（同上）
- `mu_l`: Andrade（ln(mu)=A + B/T + C ln T）或直接多项式（先跑通）

**注意**：这套模型的价值不在“绝对精度”，而在“导数平滑 + 有界 + 绝不爆炸”。

---

## 3. 多组分液相物性混合（你说“用混合策略即可”，那就别再纠结）

### 3.1 建议混合规则（P2 默认）

假设液相为理想溶液（activity=1），以质量分数 `Yl` 为输入：

- **密度**（体积分数加和 / 比容加和）：
  \[\rho_{mix}^{-1} = \sum_i \frac{Y_i}{\rho_i(T)}\]

- **比热**（质量分数线性）：
  \[c_{p,mix} = \sum_i Y_i c_{p,i}(T)\]

- **导热系数**（质量分数线性，先这样，别上来就折腾复杂 mixing）：
  \[k_{mix} = \sum_i Y_i k_i(T)\]

- **黏度**（对数混合，避免跨数量级时炸裂）：
  \[\ln \mu_{mix} = \sum_i x_i \ln \mu_i(T)\]
  这里 `x_i` 用 `Y -> X` 转换（需要 `M_i`）。

- **焓/内能**：用 `cp(T)` 积分得到 `h_i(T)`，再线性混合：
  \[h_{mix} = \sum_i Y_i h_i(T)\]

### 3.2 实现细节

- 混合前先 `sanitize(Yl)`：
  - NaN/inf -> 0
  - clip 到 [0,1]
  - 归一化
- 任何纯组分模型输出不 finite 时：
  - 触发降级：用最近合法温度点的值（或用 DB 提供的常数 fallback）
  - 同时记录 `meta.fallback_reason`

---

## 4. 多组分界面平衡（重点：你担心的 “psat1+psat2 > P”）

### 4.1 理想溶液 + Raoult（基础）

界面处：

- 先把液相质量分数 `Yl_face` 转为摩尔分数 `x_l`。
- 对每个可凝组分 i：

\[ p_i^{sat}(T_s) \] 由 `psat_i(T)` 给出

\[ y_i^{cond} = \frac{x_i p_i^{sat}(T_s)}{P} \]

这一步就会出现你说的情况：某些温度下 `\sum y_i^{cond} > 1`。

### 4.2 两层护栏（必须同时存在）

#### (A) bubble-point 温度上限（根本约束）

定义 bubble-point：

\[ f(T)=\sum_i x_i p_i^{sat}(T) - P = 0 \]

- 用 bracket + 二分（最稳，别搞牛顿）求 `T_bub(P, x)`。
- 取

\[ T_s^{eff} = \min(T_s,\; T_{bub} - \Delta T_{guard}) \]

这样 **从源头上** 避免 `\sum x_i psat_i > P`。

#### (B) condensable clamp（数值护栏）

即使 (A) 做了，仍要保留 clamp（因为你实际求解里 Ts 会抖，物性会被裁剪，数值会出幺蛾子）：

- 令 `y_cond_raw` 按 Raoult 计算。
- 若 `sum(y_cond_raw) > 1-eps_bg`，则整体缩放：

\[ y_i^{cond} = y_i^{raw} \frac{1-eps_{bg}}{\sum_j y_j^{raw}} \]

- 把剩余的 `eps_bg` 分给背景气体（按远场 `Yg_farfield` 或指定 `background_idx` 分配）。

这套“缩放 + 背景”你现在 equilibrium 里其实已经在做雏形了（condensable guard / farfield seed 逻辑）【turn29:14†coolprop_usage.md†L1-L7】。

### 4.3 输出（保持你现有字段）

`compute_interface_equilibrium_full` 至少输出：

- `y_cond`（摩尔分数，长度 = N_cond）
- `Yg_eq`（质量分数，长度 = N_gas）
- `psat`（每组分 psat_i(Ts_eff)，Pa）
- `hvap`（每组分 hvap_i(Ts_eff)，J/kg）
- `meta`：
  - `Ts_eff, Tbub, guard_active(bool), clamp_active(bool)`
  - `psat_source="p2db"` / `hvap_source="p2db"`
  - `fallback_reason`（如果走了降级）

---

## 5. 需要 Codex 落实的具体改动清单（逐步可验收）

### Step P2-A：落地 `liquid_props_db.yaml` + 读库

**改动**
- 新建 `properties/p2_liquid_db.py`：
  - `load_liquid_db(path)` 读取 YAML
  - 校验每个 species 至少包含：`M, Tc, psat_model, hvap_model, rho_model, cp_model, k_model, mu_model`
  - 校验 `T_valid` 范围

**验收**
- 单元测试：缺字段 -> 抛明确异常；多余字段允许存在。

### Step P2-B：纯组分模型

**改动**
- 新建 `properties/p2_pure_models.py`
- 每个模型实现：Antoine / Watson / poly 等（先支持你 DB 里用到的类型）

**验收**
- `T` 扫描（比如 250-600K）输出全部 finite。
- `psat(T)` 单调递增（至少在 `T_valid` 内）。

### Step P2-C：混合规则

**改动**
- 新建 `properties/p2_mix_rules.py`
- 提供 `mix_props(T, P, Yl, species_list, db)` 返回 `rho_l, cp_l, k_l, mu_l, h_l`

**验收**
- 退化测试：纯组分 `Y=[1,0,...]` 时等于纯组分值。
- `Y` 微扰不产生 NaN。

### Step P2-D：界面平衡

**改动**
- 新建 `properties/p2_equilibrium.py`
  - `bubble_point_T(P, x_l, psat_funcs, T_lo, T_hi)` 二分
  - `interface_equilibrium(P, Ts, Yl_face, ...)` -> `Ts_eff, y_cond, Yg_eq, meta`

**验收**
- 随机 `Ts`（甚至给到超沸点很多）也必须满足：
  - `Ts_eff <= Tbub - ΔT_guard`
  - `sum(y_cond) <= 1-eps_bg`
  - 输出全 finite

### Step P2-E：改造 `liquid.py`

**改动**
- 保留 `build_liquid_model` 外壳【turn27:12†liquid.py†L17-L24】
- 增加 backend 选项：`backend: p2db`
- 将原 `_pure_liquid_props` 等 CoolProp 调用替换为 `p2_mix_rules`。

**验收**
- 跑一个你最小蒸发算例（单组分），数值能推进。
- `import CoolProp` 在运行路径下不再出现（可用 grep 验证）。

### Step P2-F：改造 `equilibrium.py`

**改动**
- `sat_source` 新增取值：`p2db`
- 原来 `sat_source==custom` 的 DB/模型加载逻辑，可以直接复用结构，但底层换成 `p2_liquid_db`。
- `compute_interface_equilibrium_full` 内部调用 `p2_equilibrium.interface_equilibrium`。

**验收**
- 用你现成的 baseline 生成脚本跑一遍（把 case 里的 `sat_source` 改成 `p2db`），检查：
  - 生成 `baseline_*.npz` 不报错
  - meta 里记录的 `psat_source` 不再是 coolprop

---

## 6. 参考参数库（只为让 P2 先跑起来，不要拿去写论文）

你要求“正庚烷 + 正十二烷做多组分测试”。这里给你一个**能跑通**的 YAML 框架。

> 数值系数先用“从 CoolProp baseline 拟合”生成，别在这一步纠结文献精度。

### 6.1 YAML 模板（示例）

```yaml
# properties/data/liquid_props_db.yaml
version: 1
T_ref: 298.15
species:
  n-Heptane:
    M: 0.100205  # kg/mol (示例)
    Tc: 540.0    # K (示例)
    T_valid: [250.0, 600.0]

    psat:
      model: antoine_log10_Pa
      coeffs: [10.0, 2000.0, -50.0]   # 占位: A,B,C

    hvap:
      model: watson
      hvap_ref: 3.0e5   # J/kg 占位
      T_ref: 298.15
      exponent: 0.38

    rho:
      model: poly
      coeffs: [700.0, -0.5, 0.0]      # rho = a0 + a1*T + a2*T^2

    cp:
      model: poly
      coeffs: [2000.0, 1.0, 0.0]

    k:
      model: poly
      coeffs: [0.12, -1e-4, 0.0]

    mu:
      model: andrade
      coeffs: [-3.0, 800.0, 0.0]      # ln(mu)=A + B/T + C*ln(T)

  n-Dodecane:
    M: 0.170334  # kg/mol (示例)
    Tc: 658.0
    T_valid: [250.0, 700.0]

    psat:
      model: antoine_log10_Pa
      coeffs: [10.5, 3000.0, -60.0]

    hvap:
      model: watson
      hvap_ref: 2.5e5
      T_ref: 298.15
      exponent: 0.38

    rho:
      model: poly
      coeffs: [750.0, -0.6, 0.0]

    cp:
      model: poly
      coeffs: [2200.0, 1.2, 0.0]

    k:
      model: poly
      coeffs: [0.13, -1e-4, 0.0]

    mu:
      model: andrade
      coeffs: [-2.5, 1200.0, 0.0]
```

### 6.2 强烈建议：用 baseline 自动拟合出一版“能对齐 CoolProp 的系数”

你已经有 `baseline_coolprop.npz`，里面包含 `rho_l/cp_l/k_l/h_l/psat_l/hvap_l/eq_*` 等网格数据【turn29:1†_baseline_utils.py†L7-L18】。

下一步最省事的做法：

- 写一个离线脚本 `tools/fit_liquid_db_from_baseline.py`
  - 对每个 species：
    - 在 `T_valid` 内对 `rho,cp,k,mu` 做多项式或分段多项式拟合
    - `psat` 拟合 Antoine（或直接拟合 ln(psat) 的多项式）
    - `hvap` 拟合 Watson（需要 `Tc`，`hvap_ref` 可以拟合）
  - 输出 `liquid_props_db.yaml`

这样 P2 立刻就能做到：
- 运行时无 CoolProp
- 数值形态和你现有结果一致（至少在你选的 T/P 范围内）

---

## 7. 你还没问但必须做的事情（不做就等着继续爆炸）

1. **缓存**：`psat/hvap/rho/cp/k/mu` 都是纯函数，按 `(species, T)` 做 LRU cache，能显著减少开销。
2. **温度裁剪策略统一**：
   - 液相物性：`T_eval = clip(T, T_valid)`
   - psat/hvap：同上
   - equilibrium：先 bubble-point guard 再 clip
3. **meta 诊断字段**：把 `Ts_eff, Tbub, clamp_active, fallback_reason` 写进 `diag` 或输出文件。你后面排错全靠它。

---

## 8. 与现有 case/yaml 的对接（最小改动）

你的 baseline meta 里已经记录了 `sat_source="coolprop"`【turn29:10†baseline_coolprop_meta.json†L22-L24】。

P2 建议新增配置：

```yaml
physics:
  liquid:
    backend: p2db
    db_file: properties/data/liquid_props_db.yaml

  interface:
    equilibrium:
      sat_source: p2db
      psat_model: p2db
      Ts_guard_dT: 0.5
      eps_bg: 1.0e-12
```

这样你可以在不改动求解器主体的前提下，切换到 P2 物性核。

---

## 9. 完成标志（P2 验收）

- [ ] 任意合理算例下，求解不再因为 `psat/hvap/y_cond` 产生 NaN/inf 而发散。
- [ ] 单组分：`Ts_eff` 永远不超过 `Tb(P) - ΔT_guard`。
- [ ] 多组分：`sum(y_cond) <= 1-eps_bg` 永远成立，并且 `Tbub` 逻辑可追踪。
- [ ] 运行路径无 `import CoolProp`。

你要把这件事交给 Codex，就按上面 Step P2-A~F 一步步让它交付。别再让它“先跑起来再说”，那只会把 bug 埋得更深。
