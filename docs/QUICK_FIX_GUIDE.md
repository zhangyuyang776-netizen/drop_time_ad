# 温度发散问题快速修复指南

## 问题症状
✗ 模拟在 Ts 接近沸点时发散
✗ SNES reason=-6 (line search failure)
✗ psat/P > 0.95
✗ y_bg < 0.05 (背景气体被压缩)

## 立即修复（5分钟）

### 步骤1: 修改配置文件

编辑您的case配置文件（如 `cases/p2_accept_single_petsc_mpi_schur_with_u_output_p2db.yaml`）：

```yaml
physics:
  interface:
    equilibrium:
      Ts_guard_dT: 8.0        # 原值: 3.0
      Ts_sat_eps_K: 0.5       # 原值: 0.01
      Ts_guard_width_K: 2.0   # 原值: 0.5
      eps_bg: 0.05            # 原值: 1e-12

time_stepping:
  dt: 1.0e-5                  # 原值: 2e-5 (减半)
```

### 步骤2: 重新运行

```bash
mpiexec -n 16 python driver/run_evap_case.py cases/your_case.yaml
```

### 步骤3: 验证修复

检查输出日志：
- ✓ 所有步骤 `nl_conv=True`
- ✓ `Ts_max < Tb - 0.5K`
- ✓ `nl_iter ≤ 10`
- ✓ 没有 `reason=-6` 错误

## 参数说明

| 参数 | 作用 | 值域 |
|------|------|------|
| `Ts_guard_dT` | 距沸点的安全距离 | 5-10 K |
| `eps_bg` | 背景气体最小摩尔分数 | 0.02-0.10 |
| `dt` | 时间步长 | 1e-6 ~ 2e-5 |

## 调优建议

### 如果仍然发散：
- 增大 `Ts_guard_dT` → 10.0 K
- 增大 `eps_bg` → 0.10
- 减小 `dt` → 5e-6

### 如果运行太慢：
- 减小 `Ts_guard_dT` → 6.0 K
- 减小 `eps_bg` → 0.03
- 增大 `dt` → 1.5e-5（谨慎）

## 预期效果

| 指标 | 修改前 | 修改后 |
|------|--------|--------|
| Ts最大值 | 489.0 K (发散) | 489.0 K (稳定) |
| psat/P 峰值 | 0.974 (发散) | 0.950 (稳定) |
| y_bg 最小值 | 0.026 | 0.050 |
| 完成率 | 失败@7.86ms | 成功完成 |

## 物理影响

⚠️ **注意**: 这些参数会轻微改变物理行为：

1. **Ts_guard_dT=8K**: 界面温度被限制在 Tb-0.5K，无法真实模拟沸腾
2. **eps_bg=0.05**: 背景气体最少5%，可能高估扩散效应

✓ 对于大多数应用（蒸发、燃烧），这些影响可接受
✗ 如需精确模拟沸腾过程，参见主报告的中长期方案

## 下一步

- [ ] 短期方案成功 → 继续使用当前设置
- [ ] 需要更高精度 → 实施中期方案（饱和度插值）
- [ ] 研究级需求 → 考虑Hertz-Knudsen动力学模型

详见：`docs/temperature_divergence_analysis_and_solutions.md`

---

**最后更新**: 2026-01-30
**相关commit**: 930fce9
