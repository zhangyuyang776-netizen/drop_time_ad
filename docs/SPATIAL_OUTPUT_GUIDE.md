# 空间结果输出功能指南

本指南介绍液滴蒸发模拟的统一空间结果输出功能。

## 概述

新的输出系统提供：

1. **统一输出目录结构**：所有结果统一存放在 `3D_out/` 下
2. **每步存储 u 向量和网格坐标**：支持动网格的完整重现
3. **映射文件**：描述如何从 u 向量解析各个物理量
4. **后处理工具**：将二进制 npz 文件转换为人类可读的 CSV 格式

---

## 目录结构

运行模拟后，输出目录结构如下：

```
<output_root>/
  <case_id>/
    <timestamp>/
      scalars/
        scalars.csv           # 标量时间历史
      config.yaml             # 运行配置副本
      run.log                 # 运行日志
      3D_out/                 # 空间输出目录（NEW）
        mapping.json          # u 向量布局描述（每个运行一份）
        steps/
          step_000000_time_0.000000e+00s.npz
          step_000005_time_5.000000e-06s.npz
          step_000010_time_1.000000e-05s.npz
          ...
        post_csv/             # 后处理输出（可选）
          step_000000_time_...csv
          step_000005_time_...csv
          ...
```

**示例**（使用 p3_accept 配置）：
```
../out/p3_acceptance/
  p3_accept_single_petsc_mpi_schur_with_u_output/
    20260116_204758/
      scalars/
      config.yaml
      run.log
      3D_out/              # 所有空间输出在这里
        mapping.json
        steps/
```

### 关键文件说明

- **mapping.json**：描述 u 向量的内存布局
  - 包含每个字段的偏移量、大小、形状
  - 包含物种列表、网格尺寸等元数据
  - 用于后处理时正确解析 u 向量

- **steps/*.npz**：每个时间步的二进制数据文件
  - `step_id`：时间步编号（整数）
  - `t`：物理时间（秒，浮点数）
  - `u`：完整的未知向量（float64 一维数组）
  - `r_g`、`r_l`：气相和液相的网格中心坐标
  - `rf_g`、`rf_l`：气相和液相的网格面坐标
  - `iface_f`：气液界面的面索引

---

## 配置方法

### 1. 在 YAML 配置文件中启用

在你的案例配置文件中添加 `output` 部分：

```yaml
# 在配置文件中启用 u 向量输出
output:
  u_enabled: true    # 启用 u 向量输出
  u_every: 5         # 每 5 个时间步写一次（减少文件数量）
```

**参数说明**：

- `u_enabled`：是否启用 u 向量输出（默认：`false`）
- `u_every`：输出频率，每 N 个时间步写一次（默认：`1`）

### 2. 使用环境变量强制启用

```bash
export DROPLET_WRITE_U=1
python driver/run_evap_case.py cases/your_case.yaml
```

环境变量会覆盖配置文件中的 `u_enabled` 设置。

### 3. 示例配置文件

参考 `cases/p3_accept_single_petsc_mpi_schur_with_u_output.yaml` 获取完整示例。

这个配置基于 PETSc MPI 并行求解器，包含：
- 动网格支持（`include_Rd: true`）
- 非线性求解器（Newton-Krylov）
- Fieldsplit Schur 预条件
- 完整的空间输出配置

---

## 运行模拟

```bash
# 使用示例配置运行
python driver/run_evap_case.py cases/p3_accept_single_petsc_mpi_schur_with_u_output.yaml

# 或者使用环境变量临时启用
DROPLET_WRITE_U=1 python driver/run_evap_case.py cases/case_001.yaml
```

运行完成后，检查输出目录：

```bash
# 找到最新的运行目录
RUN_DIR=$(ls -td ../out/p3_acceptance/p3_accept_single_petsc_mpi_schur_with_u_output/*/ | head -1)
ls -lh ${RUN_DIR}/3D_out/
```

你应该看到：
- `mapping.json`：在第一个时间步后立即生成
- `steps/`：包含多个 `.npz` 文件

---

## 后处理：转换为 CSV

### 基本用法

```bash
# 找到最新的运行目录
RUN_DIR=$(ls -td ../out/p3_acceptance/p3_accept_single_petsc_mpi_schur_with_u_output/*/ | head -1)

# 转换 npz 文件为 CSV
python scripts/postprocess_u_to_csv.py \
    --run-dir ${RUN_DIR}/3D_out
```

这将：
1. 读取 `mapping.json` 了解 u 向量布局
2. 扫描 `steps/*.npz` 文件
3. 将每个 npz 文件转换为对应的 CSV 文件
4. 输出到 `<run-dir>/3D_out/post_csv/`

### 高级选项

#### 指定时间范围

```bash
python scripts/postprocess_u_to_csv.py \
    --run-dir ${RUN_DIR}/3D_out \
    --t-start 1.0e-6 \
    --t-end 5.0e-6
```

只处理时间在 `[1e-6, 5e-6]` 秒范围内的文件。

#### 使用 stride 减少文件数量

```bash
python scripts/postprocess_u_to_csv.py \
    --run-dir ${RUN_DIR}/3D_out \
    --stride 2
```

每隔 2 个文件转换一次（减少输出 CSV 数量）。

#### 自定义输出目录

```bash
python scripts/postprocess_u_to_csv.py \
    --run-dir ${RUN_DIR}/3D_out \
    --out-dir /path/to/custom/output
```

#### 组合使用

```bash
python scripts/postprocess_u_to_csv.py \
    --run-dir ${RUN_DIR}/3D_out \
    --t-start 0 \
    --t-end 1.0e-5 \
    --stride 2 \
    --out-dir results_csv
```

---

## CSV 文件格式

每个 CSV 文件对应一个时间步，包含：

### 文件头（注释行，以 `#` 开头）

```
# step_id=5
# t=5.000000e-06
# Ts=3.000000e+02
# mpp=1.234567e-03
# Rd=9.876543e-05
#
```

### 表头

```
phase,r,T,Y_N2,Y_NC12H26,Y_n-Dodecane,...
```

### 数据行

- **气相**：每行一个网格点
  ```
  gas,1.000100e-04,1.100000e+03,9.900000e-01,1.000000e-02,
  gas,1.000200e-04,1.095000e+03,9.905000e-01,9.500000e-03,
  ...
  ```

- **液相**：紧接气相之后
  ```
  liq,5.000000e-05,3.001000e+02,,,1.000000e+00
  liq,6.000000e-05,3.002000e+02,,,1.000000e+00
  ...
  ```

**注意**：
- 如果某个物种在某相中不存在，对应列为空
- `r` 是径向坐标（米）
- `T` 是温度（开尔文）
- `Y_*` 是质量分数（无量纲，0-1）

---

## 技术细节

### mapping.json 格式

```json
{
  "version": 1,
  "endianness": "little",
  "dtype": "float64",
  "ordering": "C",
  "total_size": 150,
  "blocks": [
    {
      "name": "Tg",
      "offset": 0,
      "size": 100,
      "shape": [100]
    },
    {
      "name": "Yg",
      "offset": 100,
      "size": 100,
      "shape": [1, 100]
    },
    {
      "name": "Tl",
      "offset": 200,
      "size": 10,
      "shape": [10]
    },
    ...
  ],
  "meta": {
    "Ng": 100,
    "Nl": 10,
    "Ns_g_full": 2,
    "Ns_g_eff": 1,
    "species_g_full": ["N2", "NC12H26"],
    "species_g_reduced": ["NC12H26"],
    "species_g_closure": "N2",
    ...
  }
}
```

**关键字段**：
- `blocks`：u 向量的分段描述
  - `offset`：在 u 数组中的起始位置
  - `size`：占用的元素数量
  - `shape`：重塑后的形状（用于多维数组）
- `meta`：网格和物种信息
  - `Ng`、`Nl`：气相和液相网格点数
  - `Ns_g_eff`：气相有效物种数（不包括闭合物种）
  - `species_g_closure`：闭合物种名称

### npz 文件内容

使用 NumPy 加载：

```python
import numpy as np

data = np.load("step_000005_time_5.000000e-06s.npz")
print(data.files)  # ['step_id', 't', 'u', 'r_g', 'r_l', 'rf_g', 'rf_l', 'iface_f']

step_id = int(data['step_id'])
t = float(data['t'])
u = data['u']  # Shape: (total_size,)
r_g = data['r_g']  # Shape: (Ng,)
r_l = data['r_l']  # Shape: (Nl,)
```

---

## 性能建议

### 磁盘空间管理

- **每个 npz 文件大小**：约 `8 * total_size + 8 * (Ng + Nl)` 字节
  - 例如：`total_size=150`, `Ng=100`, `Nl=10` → 约 2 KB/文件
  - 10000 步 → 约 20 MB

- **减少文件数量**：
  - 使用 `u_every` 控制输出频率
  - 仅在需要详细分析的时间段启用输出

### 并行运行

- u 向量输出是 **rank0-only**（仅主进程写文件）
- mapping.json 也是 rank0-only
- 不会产生重复文件或并行竞争

---

## 常见问题

### Q: 如何只输出特定时间段的数据？

**方法 1**：在模拟中途启用环境变量

```bash
# 先运行到感兴趣的时间段
python driver/run_evap_case.py cases/case.yaml

# 然后重启，使用检查点（如果支持）+ 启用输出
DROPLET_WRITE_U=1 python driver/run_evap_case.py cases/case.yaml --restart
```

**方法 2**：使用后处理的时间过滤

```bash
# 只转换特定时间段
python scripts/postprocess_u_to_csv.py \
    --run-dir 3D_out/.../run_xxx \
    --t-start 5.0e-5 \
    --t-end 8.0e-5
```

### Q: 为什么 CSV 中某些物种列是空的？

气相和液相可能有不同的物种。CSV 使用所有物种的并集作为列，某相不存在的物种对应列为空。

### Q: 如何验证 mapping.json 是否正确？

运行测试：

```bash
pytest tests/test_u_output_and_postprocess_smoke.py -v
```

测试会验证：
- mapping 的总大小与 u 向量长度一致
- 后处理脚本能正确解析 npz 文件
- CSV 输出格式正确

---

## 示例工作流程

### 1. 配置并运行模拟

```bash
# 创建配置文件（或使用示例）
cp cases/p3_accept_single_petsc_mpi_schur_with_u_output.yaml cases/my_case.yaml

# 编辑配置
vim cases/my_case.yaml

# 运行模拟
python driver/run_evap_case.py cases/my_case.yaml
```

### 2. 检查输出

```bash
# 找到运行目录
RUN_DIR=$(ls -td ../out/*/my_case/*/ | head -1)
echo "Run directory: $RUN_DIR"

# 检查文件
ls -lh ${RUN_DIR}/3D_out/mapping.json
ls -lh ${RUN_DIR}/3D_out/steps/ | head -20
```

### 3. 后处理

```bash
# 转换所有文件
python scripts/postprocess_u_to_csv.py --run-dir ${RUN_DIR}/3D_out

# 或者只转换部分文件（每隔 10 个）
python scripts/postprocess_u_to_csv.py \
    --run-dir ${RUN_DIR}/3D_out \
    --stride 10

# 检查 CSV 输出
ls -lh ${RUN_DIR}/3D_out/post_csv/ | head -10
head -20 ${RUN_DIR}/3D_out/post_csv/step_000000_time_*.csv
```

### 4. 分析数据

使用 Python/pandas 分析 CSV：

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取单个 CSV
# 假设 RUN_DIR 已经设置（例如从环境变量或上面的 bash 命令）
csv_file = f'{RUN_DIR}/3D_out/post_csv/step_000100_time_1.000000e-05s.csv'
df = pd.read_csv(csv_file, comment='#')

# 分离气相和液相
gas = df[df['phase'] == 'gas']
liq = df[df['phase'] == 'liq']

# 绘制温度分布
plt.figure()
plt.plot(gas['r'], gas['T'], 'r-', label='Gas')
plt.plot(liq['r'], liq['T'], 'b-', label='Liquid')
plt.xlabel('Radius (m)')
plt.ylabel('Temperature (K)')
plt.legend()
plt.show()
```

---

## 相关文件

- 核心实现：`io/writers.py`
- 主循环集成：`driver/run_evap_case.py`
- 后处理脚本：`scripts/postprocess_u_to_csv.py`
- 测试：`tests/test_u_output_and_postprocess_smoke.py`
- 示例配置：`cases/p3_accept_single_petsc_mpi_schur_with_u_output.yaml`

---

## 支持与反馈

如有问题或建议，请查看：
- 测试文件中的示例用法
- 示例配置文件中的注释
- 后处理脚本的 `--help` 输出
