# P3 工作路线：把 P2 物性内核接入主求解器，并彻底拔除 CoolProp（含验收）

> 前提：你已通过 P2-B（纯组分模型）验收。现在进入 P3：**系统集成 + 路径收敛 + 工程清理 + 端到端验收**。  
> 目标不是“能跑一次”，而是：**代码库中不再存在 CoolProp 依赖与旧饱和路径，同时数值行为在沸点/泡点附近稳定**。

---

## P3 总目标（必须同时满足）

1. **CoolProp 完全拔除**
   - 代码中无 `import CoolProp`、无 `PropsSI`、无任何 CoolProp 相关分支（包括配置项残留的“coolprop”路径）。
   - CI/本地测试在“未安装 CoolProp”的环境中仍能通过。

2. **新液相物性与界面平衡接入主流程**
   - `liquid.py`、`equilibrium.py` 的“对外接口”保持兼容（或提供明确的适配层），求解器/输出层无需大改。
   - 界面处 `psat/hvap/latent` **同源一致**（同一个模型/同一个 DB），并记录到 diag。

3. **端到端行为验收**
   - 在“安全温区”（远离沸点/泡点）与 P0 baseline（原 CoolProp baseline）在合理误差内一致（误差阈值写死）。
   - 在“接近沸点/泡点”的应力测试中：**不发散**、`Ts_eff <= Tb/Tbub - guard`、`sum(y_cond) <= 1-eps`。

---

## P3-A 接入层：把 P2 DB 变成项目“唯一物性入口”

### 修改方案
1. **确立单一入口：`p2_liquid_db.py`**
   - DB 加载、参数校验、模型分发、缓存都在这里完成。
   - 对外暴露两个稳定接口（名字可按你现有风格调整）：
     - `get_pure_props(species, T) -> dict`
     - `get_mixture_props(species_list, Y_l, T) -> dict`
     - `interface_equilibrium(Ts, P, Y_l or x_l, condensables, ...) -> dict`

2. **`liquid.py` 只做适配，不做物性**
   - 保留原来的函数名/返回字段结构（最小改动）。
   - 内部改为调用 `p2_liquid_db` 的对应接口。
   - 删除旧的 `liquid_sat_db.py` / `saturation_models.py` 的引用（以及任何 “sat_source=custom/coolprop” 的分支）。

3. **缓存与性能**
   - 对纯物性 `props(species, T)` 做 LRU cache。
   - 对 `Tb(P)`/`Tbub(P, x_l)` 做缓存（key 可以使用 `(P, tuple(round(x_i, n)))`），避免 Newton 内重复求根拖垮性能。

### 验收方案
- **接口兼容验收**：原本调用 `liquid.py` 的位置不改（或仅改 import），能拿到同名字段（例如 `rho, mu, cp, k, h` 等）。
- **DB 校验验收**：故意删掉 YAML 某个必要系数，加载时必须 fail-fast（抛清晰错误），不能悄悄给默认值。
- **缓存验收**：同一 `species,T` 重复调用计时显著下降（简单 profile 即可）。

---

## P3-B 界面平衡接管：让 equilibrium.py 只认 P2 路径

### 修改方案
1. **把 equilibrium.py 的 psat/hvap 计算全部换成 P2 DB**
   - 严禁再从任何旧模块（或 CoolProp）混入 `psat` 或 `hvap`。
   - `latent` 项一律使用与 `psat` 同源的 `hvap`（你的 P4 验收逻辑沿用）。

2. **单组分护栏：`Tb(P)`**
   - `Tb(P)` 由 `psat(T)=P` 求根得到（二分法为主）。
   - 施加：`Ts_eff = min(Ts, Tb - dT_guard)` 或平滑护栏（可选）。
   - `y_cond = min(psat/P, 1-eps)`。

3. **多组分护栏：`Tbub(P, x_l)`**
   - `Tbub` 定义：`sum(x_l,i * psat_i(Tbub)) = P`。
   - 施加：`Ts_eff = min(Ts, Tbub - dT_guard)`。
   - Raoult：`xg_i = x_l,i * psat_i(Ts_eff)/P`。
   - 若 `sum(xg) > 1-eps`（浮点误差/越界外推），执行比例缩放兜底。

4. **诊断字段统一输出**
   - 在 `interface_diag.csv` 中确保写入（最少）：
     - `psat_source`, `hvap_source`, `latent_source`
     - `Ts`, `Ts_eff`, `Tb` 或 `Tbub`
     - `sum_xg`/`sum_ycond`, `scaled`（是否触发缩放/护栏）
   - `*_source` 必须全为 `"p2"`（或你定义的统一标识）。

### 验收方案
- **强一致验收**：`psat_source==hvap_source==latent_source=="p2"`（单元测试 + 运行后检查 diag）。
- **沸点/泡点护栏验收**：
  - 构造 `Ts = Tb + 20K`（单组分）/ `Ts = Tbub + 20K`（多组分），检查输出 `Ts_eff <= Tb/Tbub - dT_guard`。
  - 检查 `sum(y_cond) <= 1-eps`。
- **数值稳健验收**：扫描温区（例如 300–(Tb+30)K）输出无 NaN/inf，且量级合理（正值）。

---

## P3-C 彻底拔除 CoolProp 与旧饱和路径（“删干净”阶段）

### 修改方案
1. **移除依赖**
   - 删除 `requirements`/`environment.yml` 中 CoolProp（如果有）。
   - 删除代码中的 `import CoolProp`、`PropsSI` 调用。
   - 删除配置项里关于 `coolprop` 的可选值（或保留但直接报错：不允许）。

2. **清理旧文件与旧分支**
   - 清理/删除：
     - `liquid_sat_db.py`
     - `saturation_models.py`
     - 任何“自定义饱和模型”的旧试验路径（如果你决定统一走 P2 DB）
   - 若短期不能删文件：至少从运行路径彻底断开，并在 import 处显式提示“deprecated”。

3. **全库静态扫描**
   - `ripgrep -n "CoolProp|PropsSI|coolprop"` 必须为空（或只出现在文档的历史说明里）。

### 验收方案
- **无 CoolProp 环境验收**（最硬核，也最有说服力）：
  - 在一个干净环境卸载 CoolProp（或用 CI job 不安装），运行：
    - `python -m compileall .`
    - 全套 `pytest`
    - 至少一个端到端算例
  - 若任何地方隐式 import CoolProp，直接炸出来。

---

## P3-D 配置收敛：让用户只看到“一个正确的选项”

### 修改方案
1. **配置项简化**
   - 物性与饱和模型只保留：
     - `liquid_props_db: path/to/liquid_props_db.yaml`
     - `enabled: true/false`（如果你要支持切换）
   - 删除或弃用：
     - `sat_source`
     - `psat_model`
     - `hvap_source`
   - 若必须保留兼容：解析时映射到 P2，旧值触发 warning，但不再产生分支。

2. **默认行为写死**
   - 默认启用 P2 DB。
   - 未提供 DB 路径则直接报错（避免 silent fallback）。

### 验收方案
- **配置兼容测试**：旧 YAML 仍能加载，但输出 warning 并映射到 P2；新 YAML 必须无 warning。
- **错误提示测试**：缺 DB 或 DB 缺字段时报错信息清晰，能定位到物种/字段名。

---

## P3-E 端到端验收：对齐 P0 baseline + 做“沸点应力测试”

### 修改方案
1. **对齐基线（安全区）**
   - 选择 P0 baseline 覆盖的温区（例如 300–460 K，1/5/10 atm）。
   - 用 P2 计算同一网格上的：
     - `psat(T)`
     - `hvap(T)`
     - `rho(T)`, `cp(T)`, `mu(T)`, `k(T)`（如果 baseline 有）
   - 给出误差阈值（建议先宽后紧）：
     - `psat`: 相对误差 < 10–20%（初版）
     - `hvap`: 相对误差 < 5–10%
     - `rho/cp/mu/k`: 相对误差 < 5–15%
   - 后续你可以用拟合把阈值收紧。

2. **沸点/泡点应力测试（关键）**
   - 单组分：对每个 P 求 Tb(P)，取 `Ts = Tb+{1,5,20}K`，验证护栏、y_cond、数值稳定。
   - 多组分（NC7/NC12）：取几个 `x_l`（例如 0.1/0.5/0.9），对每个 P 求 Tbub(P,x_l)，同样取 `Ts = Tbub+{1,5,20}K`。
   - 记录 diag，检查 `Ts_eff` 和 `sum_ycond`。

3. **真实算例（至少 2 个）**
   - 单组分：NC12（你最关心的）
   - 多组分：NC7/NC12（任意配比）
   - 目标：时间推进到你之前容易炸的时刻（例如 50 ms 或更早），确保不因界面发散而中断。

### 验收方案
- **基线对齐验收**：生成 `p2_vs_baseline.csv`，在阈值内判定 PASS/FAIL。
- **应力验收**：输出 `stress_guard_report.csv`，逐条检查：
  - `Ts_eff <= Tb/Tbub - guard`
  - `sum_ycond <= 1-eps`
  - `scaled` 是否按预期触发
- **端到端验收**：两个算例都能跑到指定时间，无 `NaN/inf`，无“界面平衡异常”中断。

---

## P3-F 工程收尾：文档与迁移说明（别偷懒）

### 修改方案
- 写 `docs/property_model_p2.md`：
  - DB schema
  - 有效温区与护栏
  - 常见错误与诊断字段解释
- 写 `MIGRATION_P2.md`：
  - 从旧配置迁移到新配置的映射规则
  - 已删除/弃用的参数清单

### 验收方案
- 新人只看文档能配置跑通 NC12 单组分算例（你不算新人，但你会忘）。

---

## 建议的交付节奏（按 commit/PR 切分）

1. PR1：P3-A 接入层（liquid.py 适配 + DB loader/校验 + cache）
2. PR2：P3-B 界面平衡接管（equilibrium.py 统一走 P2 + diag）
3. PR3：P3-C 拔除 CoolProp + 清理旧模块 + 静态扫描测试
4. PR4：P3-D 配置收敛 + 迁移文档
5. PR5：P3-E 端到端与应力测试脚本 + 报表输出

---

**P3 的本质**：不是“再写几条公式”，而是把项目从“多条遗留路径叠成的泥潭”变成“唯一可靠路径”。  
你都说了不想新旧共存，那就按这个路线干，别心软。  
