# P4 详细工作路线：把 P2 物性替换“做成定稿版本”（修改方案 + 验收方案）

> 你说 P3 已完成，那就默认：  
> - 主流程已接到 `p2_liquid_db.py`（或等价入口）  
> - `equilibrium.py/liquid.py` 已统一走 P2 路径  
> - CoolProp 已彻底移除、旧饱和路径已清理  
>
> **P4 的工作不是“再接一次”，而是把替换做成“可交付的定稿”：系数来源明确、误差可控、应力测试稳定、回归可复现、诊断可追踪。**

---

## P4 交付标准（P4 完成的定义）

1. **数据库系数定稿**：NC7/NC12（至少）从“smoke-test 占位系数”升级为 **有来源/可追溯** 的参数集（Cuoci/Yaws 或基线拟合），并写入引用信息。
2. **误差闭环**：对 P0 CoolProp baseline（已生成的 `baseline_coolprop.npz`）给出对齐报告与阈值判定（PASS/FAIL）。
3. **应力稳健**：靠近沸点/泡点的应力测试（单组分 Tb、多组分 Tbub）满足：  
   - `Ts_eff <= Tb/Tbub - guard`  
   - `sum(y_cond) <= 1-eps`  
   - 无 NaN/inf、无负值物性
4. **端到端回归**：至少 2 个算例（NC12 单组分 + NC7/NC12 多组分）能推进到指定时刻且输出诊断一致。
5. **诊断可验收**：`interface_diag.csv`（或等价输出）里 `psat_source/hvap_source/latent_source` 一致为 `"p2"`，并记录 `Tb/Tbub, Ts_eff, sum_ycond, scaled, guard_reason`。

---

## P4-A 物性数据库“定稿化”：系数来源、单位、有效域、引用

### 修改方案
1. **扩展 `liquid_props_db.yaml` schema（不改核心计算，只改数据结构）**
   - 为每个 property 增加（或强制）字段：
     - `Tmin/Tmax`（有效温区）
     - `units`（若全局 units 已有，则 property 可省略）
     - `source`（字符串：`"Cuoci2024" / "Yaws" / "fitted_to_coolprop_baseline"`）
     - `ref`（引用条目：DOI/书名/章节/表号/URL 标识，不要求可点击，但要可追溯）
   - 为物种增加：
     - `MW, Tc, Pc, Tb_1atm, T_valid`（至少要能支撑 psat/hvap/rho）

2. **系数获取路线二选一（建议先走 A1，再 A2）**
   - **A1：基线拟合（快且闭环）**
     - 写脚本 `scripts/p4_fit_db_from_baseline.py`
     - 输入：`baseline_coolprop.npz` + 初始 `liquid_props_db.yaml`（模型选择已定：psat/hvap/rho/cp/mu/k/sigma）
     - 输出：`liquid_props_db_fitted.yaml`（把 coeffs 填满，并写 `source: fitted_to_coolprop_baseline`）
     - 你只需要拟合到“安全温区”（P0 baseline 的 300–460 K、1/5/10 atm），先把端到端跑稳。
   - **A2：Cuoci/Yaws 参数替换（最终版）**
     - 把 A1 的拟合系数逐项替换为 Cuoci 文献/其引用数据库给出的系数
     - 同时把 `source/ref` 更新为真实来源
     - 仍保留 `baseline_coolprop.npz` 用于回归对比（它不是依赖，只是金标准之一）

3. **严格的 DB 校验**
   - `p2_liquid_db.py` 加载时：缺字段直接 fail-fast（不允许 silent fallback）
   - 单位不一致（比如 MW kg/mol vs g/mol）：直接报错，别让数值悄悄漂

### 验收方案
- **P4-A1（拟合）验收**：
  - 生成 `liquid_props_db_fitted.yaml`
  - 跑 P2-B 的纯组分验收（你已经有），必须 PASS
- **P4-A2（来源参数）验收**：
  - 对每个 property：随机抽点 T，计算输出必须 finite 且 >0（psat>0）
  - `T` 超出有效域：必须触发统一的 `clip + meta.guard_reason`（或抛异常，按你现有策略）
- **DB 引用验收**：
  - YAML 中每个 property 均有 `source` 与 `ref`（至少 NC7/NC12 完整覆盖）

---

## P4-B “对齐报告”：P2 vs P0 baseline 的系统性误差评估

### 修改方案
1. 新增脚本 `scripts/p4_compare_to_baseline.py`
   - 输入：`baseline_coolprop.npz` + 当前 `liquid_props_db.yaml`
   - 输出：  
     - `out/p4_baseline_compare/p2_vs_coolprop.csv`
     - `out/p4_baseline_compare/summary.json`（每项最大/均方误差、阈值判定）
   - 逐项比对：`psat, hvap, rho, cp, mu, k`（以 baseline 可用字段为准）

2. 误差阈值（建议先宽后紧，但要写死）
   - 初版阈值建议：
     - `psat`：相对误差 RMS < 20%，max < 50%
     - `hvap`：RMS < 10%，max < 20%
     - `rho/cp/mu/k`：RMS < 15%，max < 30%
   - A2 完成后再收紧（例如 psat RMS < 10%）

### 验收方案
- 运行脚本生成报告
- `summary.json` 中 `PASS==true`（所有项都在阈值内）
- 报告写明：物种、温区、压力点、样本数、每项误差统计

---

## P4-C 界面平衡“应力测试”定稿：Tb/Tbub、护栏、缩放、诊断

### 修改方案
1. 新增应力测试脚本 `scripts/p4_stress_interface_guard.py`
   - 单组分（NC7、NC12）：
     - 对 P = 1/5/10 atm，求 Tb(P)
     - 设置 `Ts = Tb + {1, 5, 20}K`，调用 interface equilibrium
   - 多组分（NC7/NC12）：
     - 选 `x_nc7 = {0.1, 0.5, 0.9}`（或质量分数等价组）
     - 对 P = 1/5/10 atm，求 Tbub(P, x_l)
     - 设置 `Ts = Tbub + {1, 5, 20}K`

2. 强制输出诊断到 CSV：
   - `species/mixture_id, P, Ts, Tb/Tbub, Ts_eff, sum_ycond, scaled, guard_reason`
   - 并记录 `psat_source/hvap_source/latent_source`（必须一致为 p2）

3. 若你在 P3 仍用硬 `min()` 护栏且 Newton 出现震荡（可选增强）
   - 增加“平滑护栏”模式（softplus/tanh），但必须可开关：
     - 默认：硬护栏（稳定优先）
     - 可选：平滑护栏（导数连续，收敛更友好）

### 验收方案
- `stress_guard_report.csv` 逐条检查：
  - `Ts_eff <= Tb/Tbub - dT_guard`
  - `sum_ycond <= 1-eps`
  - `sum_ycond` 不为 NaN/inf
  - `scaled` 只在必要时触发（靠近边界点可能触发）
- `*_source` 三项一致验收：
  - `psat_source==hvap_source==latent_source=="p2"`

---

## P4-D 端到端回归：真实算例跑稳 + 输出对齐

### 修改方案
1. 新增回归驱动脚本 `scripts/p4_regression_run.py`
   - 支持一键跑 2–3 个标准算例：
     - Case1：NC12 单组分（你最关心发散的）
     - Case2：NC7/NC12 多组分（任意配比，建议 50/50）
   - 输出统一到 `out/p4_regression/<case_name>/`
     - `scalars.csv`
     - `interface_diag.csv`
     - （可选）关键场量快照

2. 关键判据（不看曲线“感觉”，看硬指标）
   - 运行推进到指定时刻（例如 10 ms/50 ms，按你最易炸的点设）
   - SNES/牛顿不出现“界面平衡导致的异常中断”
   - `interface_diag.csv` 中：
     - `Ts_eff` 一直存在且在护栏内
     - `sum_ycond` 从不超过 1-eps
     - 三个 source 一直是 p2

3. （可选）与旧版本结果对齐
   - 你既然已经删了 CoolProp 路径，就不要在运行时对比
   - 对比只能用 **已保存的历史输出**（比如你之前某次跑通的 scalars.csv 作为 golden）

### 验收方案
- 回归脚本返回码为 0（PASS）
- 输出目录下生成 `regression_summary.json`：
  - 达到的最终时间
  - 是否触发过 guard
  - 最大 `sum_ycond`
  - 是否出现 NaN/inf
- 若启用 golden compare：关键标量（例如 Rd、Ts_eff、m_dot）在容差内

---

## P4-E “无 CoolProp 环境”最终审计：确保替换是真替换

### 修改方案
1. 增加一个 CI / 本地脚本 `scripts/p4_no_coolprop_audit.sh`
   - 执行：
     - `python -m pip uninstall -y CoolProp`（或在干净 env 不安装）
     - `python -m compileall .`
     - `pytest -q`
     - `ripgrep -n "CoolProp|PropsSI|coolprop" .`（必须为空或只在文档历史中）
2. 把 `baseline_coolprop.npz` 留作回归金标准，但：
   - 不允许任何脚本在 P4 依赖 CoolProp 重新生成 baseline（可另放到 optional tools 目录并标注）

### 验收方案
- 上述脚本全 PASS
- grep 结果为 0（或符合白名单规则）

---

## P4-F 文档与交付：你未来的自己也能用

### 修改方案
1. 文档 `docs/liquid_property_model_p2.md`
   - 公式与变量（引用你已定稿的 spec）
   - DB schema（字段、单位、有效域）
   - 护栏策略（Tb/Tbub/缩放兜底）
   - `interface_diag.csv` 字段解释与验收点

2. 迁移说明 `MIGRATION_COOLPROP_TO_P2.md`
   - 旧配置到新配置映射
   - 已删除参数清单
   - 常见报错与解决

### 验收方案
- 新人（或未来你）只看文档能：
  - 配一份 NC12 单组分 YAML 跑起来
  - 读懂 diag 里每个字段为什么存在

---

## 建议的提交拆分（强制你别把所有改动揉成一坨）

1. PR1：P4-A（DB schema + 拟合脚本 A1 + 校验）
2. PR2：P4-B（baseline compare 报告）
3. PR3：P4-C（应力测试脚本 + 护栏/诊断补全）
4. PR4：P4-D（端到端回归驱动 + summary）
5. PR5：P4-E（无 CoolProp 审计脚本 + CI 任务）
6. PR6：P4-F（文档与迁移说明）

---

## 最终一句话（不客气但实用）

P3 只是“把新路接上并把旧路删掉”。  
P4 才是“让你敢在论文和生产算例里说：CoolProp 已被替换且结果可复现”。  
按上面的验收把证据堆起来，你才算真的完成替换。  
