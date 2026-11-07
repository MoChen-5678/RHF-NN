# RHF-NN

球形原子核的神经网络变分求解程序，支持 RHF（PKA1 / PKO2）和 RMF（PKDD）。
主入口采用无子空间刷新的密度矩阵训练。原始数值实现保存在 `dpl_rhf/`。

## 项目结构

```text
dpl_rhf/
├── __main__.py   # 统一命令入口
├── cli/          # 训练、评估、绘图、批量运行与参考对比
├── rhf/          # RHF 泛函、Dirac 算符、交换项与训练器
├── backends/     # RMF/RHF 计算后端
├── functionals/  # 能量泛函接口与 PKDD 实现
├── models/       # 神经网络
├── training/     # 变分训练与梯度计算
├── io/           # 输出与数据格式
├── legacy/       # 原始历史实现
└── tests/        # 已有测试
examples/rhf/    # 可直接运行的示例
scripts/         # 安装脚本
pyproject.toml   # 唯一的依赖与打包配置
```

`outputs/`、`backups/`、`runtime/` 和本地参考资料不上传 GitHub。
配对与阻塞扩展 [RHFB-NN](https://github.com/MoChen-5678/RHFB-NN)
是独立项目，本地 `variants/blocked-rhfb/` 不属于本仓库的整理范围。

## 安装

需要 Python 3.10 或更新版本。在项目根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

也可执行 `conda env create -f environment.yml`，然后 `conda activate dpl-rhf`。
GPU 训练需要当前环境中的 PyTorch 支持 CUDA；CPU 运行使用 `--device cpu`。

## 运行

```bash
# RHF：PKA1，O16
python -m dpl_rhf train-rhf --model PKA1 --z 8 --n 8 \
  --epochs 500 --device cuda --out outputs/PKA1/O16

# RMF：PKDD，O16
python -m dpl_rhf train-rmf --model PKDD --z 8 --n 8 \
  --epochs 500 --device cuda --out outputs/PKDD/O16

# 查看参数
python -m dpl_rhf train-rhf --help
python -m dpl_rhf train-rmf --help
```

安装后也可用 `dpl-no-refresh` 替代 `python -m dpl_rhf`。
RHF 续训使用 `--resume`；主存或磁盘卸载使用
`--graph-offload memory` 或 `--graph-offload disk`。
训练保持 `--refresh-interval 0`，硬无海投影默认开启。

## 参考对比

Core-1204 由用户单独提供，训练完成后执行：

```bash
python -m dpl_rhf compare-core --kind rhf \
  --result outputs/PKA1/O16 --model PKA1 --z 8 --n 8 \
  --core-dir ../Core-1204 --out outputs/PKA1/O16/core
```

## 历史程序

历史实现集中在 `dpl_rhf/legacy/`，例如：

```bash
python -m dpl_rhf.legacy.dpl_rmf --help
python -m dpl_rhf.legacy.fieldspace_rmf --help
```

原来的根目录转发脚本已移除；导入历史类时使用完整模块路径，
例如 `from dpl_rhf.legacy.rmf_functional import PKDDRMFFunctional`。
历史算法与主入口分别保留，不应混用训练配置。

授权说明见 [LICENSES.md](LICENSES.md)。
