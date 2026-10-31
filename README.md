# RHF-NN：无刷新神经网络相对论 Hartree-Fock（PKA1 / PKO2）

用神经网络直接变分球形原子核的一体密度矩阵，求解密度依赖相对论
Hartree-Fock（DDRHF，PKA1/PKO2）与 RMF（PKDD）基态。训练过程中
**不求解 Dirac 本征方程、不刷新正能子空间、不重置优化器**；SCF 程序
（Core-1204）只在训练结束后用于外部验证。

奇核、配对与阻塞的 RHFB 扩展是独立仓库
[RHFB-NN](https://github.com/MoChen-5678/RHFB-NN)。

## 方法概要

- **变分对象**：每个 `(q, kappa)` 块由网络输出固定完整 Dirac 径向网格上的
  占据因子 `Q`，密度矩阵 `P = Q f Q^T`；QR 严格保证正交。
- **能量泛函**：完整 PKA1 Hartree + Fock（sigma、omega、rho 时空分量、
  rho-tensor、rho 矢量-张量交叉、pion-PV、Coulomb）、接触补偿和密度依赖
  重排项；Fock 非局域密度 `R(r,r')` 不做局域化。
- **损失**：`E/A` + 全空间 Euler-Lagrange 驻定残差 `||(1-QQ^T) H[P] Q||^2`
  + 逆 Hamiltonian 分支约束，`H[P] = dE/dP` 由自动微分得到。
- **无海近似（硬约束）**：网络输出在 QR 前乘以上一轮平均场的正能投影
  `Lambda_+ = [1 + sign(H - W)]/2`（`W = -M`）。sign 函数用行列式缩放的
  Newton 迭代计算，只需稠密求逆，不做本征分解。
- **批量 Fock**：所有块对与 Dirac 分量的交换收缩按 (种类, kappa, 分量)
  批量逐元素计算（`dpl_rhf/rhf/fock_batched.py`）；原逐对循环保留为参考实现。

## 精度说明

所有能量与半径均不含质心修正；参考值为 Core-1204 RHF SCF（同 PKA1 泛函），
仅在训练结束后对照。训练中本征求解、子空间刷新和 Dirac 求解次数均为 0。

**严格无刷新训练与 Core-1204 对比（PKA1）**

| 核 | 训练轮数 | DPL E/A (MeV) | Core E/A (MeV) | ΔE/A (keV) | ΔR_m (fm) | 占据能级 MAE (MeV) | 非局域 R 相对 L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| O16 | 300 | -7.347631 | -7.348947 | +1.3 | 0.0049 | 0.053 | 0.95% |
| Ca40 | 300 | -8.350150 | -8.337794 | -12.4 | 0.0138 | 0.268 | 2.54% |
| Ca48 | 300 | -8.499959 | -8.494940 | -5.0 | 0.0176 | 0.394 | 3.36% |
| Sn132 | 500 | -8.284419 | -8.304924 | +20.5 | -0.0101 | — | — |
| Pb208 | 500 | -7.813766 | -7.839894 | +26.1 | -0.0145 | — | — |

- Sn132、Pb208 为 2026-10-31 无海修复后的 500 轮结果（A100）。末轮驻定残差
  仍在下降（0.131 / 0.146 MeV），能量尚未完全收敛，需要更多轮数；能级与
  非局域密度的 Core 对比尚未对这两次结果重做。
- 非局域 R 相对 L2 为全部 `(q, kappa)` 块四个 Dirac 分量 `R(r,r')` 的聚合误差。

**O16 能量分解（300 轮）**

| 量 | DPL-RHF | Core-1204 | 差 |
| --- | ---: | ---: | ---: |
| E/A (MeV) | -7.34763081 | -7.34894668 | 0.00131588 |
| 物质 RMS 半径 (fm) | 2.738206 | 2.733324 | 0.004882 |
| 动能 (MeV) | 189.732511 | 190.878185 | -1.145674 |
| Hartree 总能量 (MeV) | -247.803218 | -248.993839 | 1.190622 |
| Fock 总能量 (MeV) | -59.491387 | -59.467492 | -0.023895 |

在同一 Core 收敛密度上直接评估本程序泛函，总能量差 4.0e-6 MeV/A、净 Fock
差 3.0e-5 MeV，即泛函与 Hamiltonian 实现本身与 Core 闭合；上表残差来自
网络变分尚未完全收敛。

**数值实现一致性**

| 检查 | 结果 |
| --- | ---: |
| 批量 Fock 与逐对循环：能量 / Hamiltonian 最大差 | < 1e-12 MeV |
| 批量 Fock 与逐对循环：网络梯度相对差 | ~1e-15 |
| 无海投影子（Newton sign）与精确谱投影最大差 | ~1e-15 |
| `tr Lambda_+ - N/2`（W 位于 Dirac 能隙） | ~1e-14 |
| 修复后训练中占据态 Dirac 海权重 | Pb208 ≤ 1.3e-3，Sn132 ≤ 2.8e-5 |

## 2026-10-31 更新

**修复 Sn132 / Pb208 发散（变分塌缩进 Dirac 海）。** 旧版无海约束只是软
罚函数 `relu(1-s)^2`：在边界斜率为零、对全部占据态取平均，而能量随海
分量线性下降。Pb208 在约 150 轮后高 `|kappa|` 块混入 5–8% 海分量，最终
E/A 塌到 -1098 MeV。现改为上述硬投影，并在最佳 checkpoint 选择中惩罚
海权重。相同配置（`energy_weight=2`, `inverse_weight=10`, 500 轮）的对比：

| 核 | 修复前 E/A (MeV) | 修复后 E/A (MeV) | 最大海权重 | Core-1204 |
| --- | ---: | ---: | ---: | ---: |
| Pb208 | -1098（塌缩） | -7.814 | 1.3e-3 | -7.840 |
| Sn132 | -197（塌缩） | -8.284 | 2.8e-5 | -8.305 |
| O16（300 轮） | -931（lr=1e-2 时 40 轮塌缩） | -7.34860 | — | -7.34895 |

Pb208/Sn132 在 500 轮时驻定残差仍在下降，尚需更多轮数收敛。

**性能。** 批量 Fock 与原循环逐项一致（能量差 <1e-12 MeV，网络梯度相对差
~1e-15），并去掉了大核默认的计算图主存卸载：

| 情形 | 原版 | 现版 |
| --- | ---: | ---: |
| Ca48 每轮（T4，含二阶驻定反传） | 3.1 s / 5.7 GiB | 0.54 s / 2.1 GiB |
| Pb208 每轮（含二阶驻定反传） | 108 s（6 卡 + 主存卸载） | 1.4 s（单张 T4，7.5 GiB） |

完整训练每轮另含无海投影和逆 Hamiltonian 线性求解（FP64 求逆）；在 A100 /
Titan V 上这部分开销很小。

## 安装

```bash
conda env create -f environment-rhf.yml
conda activate dpl-rhf
pip install -e .
```

需要 PyTorch（CUDA 版用于 GPU）、SciPy、SymPy；全部计算为 float64。

## 训练

```bash
# PKA1 RHF，O16
dpl-no-refresh train-rhf --model PKA1 --z 8 --n 8 --epochs 500 \
  --device cuda --out outputs/no_refresh/PKA1/O16

# PKA1 RHF，Pb208（单卡约 8 GiB）
dpl-no-refresh train-rhf --model PKA1 --z 82 --n 126 --epochs 1500 \
  --energy-weight 2.0 --inverse-weight 10 --device cuda \
  --out outputs/no_refresh/PKA1/Pb208

# PKDD RMF
dpl-no-refresh train-rmf --model PKDD --z 8 --n 8 --epochs 500 \
  --device cuda --out outputs/no_refresh/PKDD/O16
```

常用选项（`dpl-no-refresh train-rhf --help` 查看全部）：

| 选项 | 默认 | 说明 |
| --- | --- | --- |
| `--no-sea-projection / --no-no-sea-projection` | 开 | 硬无海投影；关闭仅用于复现旧版塌缩 |
| `--graph-offload {none,memory,disk,recompute}` | `none` | 小显存卡可把计算图放到主存或磁盘，梯度完全相同 |
| `--resume` | 关 | 从 `training_checkpoint.pt` 续训，包括无海投影子 |
| `--refresh-interval` | 0 | 必须为 0，否则报错 |

训练日志 `history.csv` 应始终满足：

```text
eigensolve_count = 0
refresh_count = 0
training_dirac_solves = 0
dirac_sea_weight ≈ 0（一轮滞后，通常 < 1e-3）
no_sea_trace_error ≈ 0（tr Lambda_+ = N/2，说明 W 位于 Dirac 能隙内）
```

## Core-1204 验证

```bash
dpl-no-refresh compare-core --kind rhf \
  --result outputs/no_refresh/PKA1/O16 --model PKA1 \
  --z 8 --n 8 --core-dir ../Core-1204 \
  --out outputs/no_refresh/PKA1/O16/core
```

RHF Hamiltonian 的严格定义与闭合检查见
[`docs/RHF_HAMILTONIAN_DEFINITION_AND_AUDIT.md`](docs/RHF_HAMILTONIAN_DEFINITION_AND_AUDIT.md)。

## 测试

```bash
python -m pytest -q dpl_rhf/tests/test_no_refresh_solver.py dpl_rhf/tests/test_density_matrix_rhf.py
```

包括：无海投影与谱投影一致（且不调用 `eigh`）、投影后网络因子海权重为零、
批量 Fock 与逐对循环的能量/Hamiltonian/梯度一致、零本征求解初始化等。

## 文档

- [无刷新求解器原理与工作流程](docs/NO_REFRESH_THEORY_AND_WORKFLOW.md)
  （第 4.1 节：塌缩原因与硬无海投影；第 4.2 节：批量 Fock）
- [RHF 无刷新文献与公式审计](docs/RHF_NO_REFRESH_LITERATURE_AUDIT_20260819.md)
- [PKA1 完整通道审计](docs/PKA1_COMPLETE_CHANNEL_AUDIT.md)

训练输出、检查点、参考文献 PDF 不纳入版本库（见 `.gitignore`）。
