# EquiMotion

EquiMotion 面向 **Car-following Trajectory Dataset（车辆跟驰轨迹数据集）** 的
自动驾驶跟驰车辆未来纵向轨迹预测任务。给定每段轨迹前 10 秒的跟驰自动驾驶车辆
（FAV）与前车（LV）状态，以及预测区间内可见的 LV 运动，模型预测 FAV 后续
20 秒的纵向位置。

仓库提供完整训练集与测试集、模型结构、训练与推理代码、数据检查和本地评测实现。
仓库不提供预训练权重或预测结果，所有模型文件与推理产物均由使用者本地生成。

> 数据边界：训练只读取 `data/train.csv`；测试推理只读取 `data/test.csv` 中公开
> 可见的前 100 步 FAV 状态与完整 LV 轨迹，不读取测试集未来 FAV 标签。

## 数据与任务

数据以 0.1 秒为时间间隔，每个 `Segment_ID` 包含 300 步：

- 历史观测：前 100 步，共 10 秒；
- 预测区间：后 200 步，共 20 秒；
- 训练集：4,517 个片段；
- 测试集：1,937 个片段。

模型输入包含 LV/FAV 的位置、速度、加速度、车辆身份、车辆类型、空间间隙、
空间车头时距和速度差。预测目标为后 200 步 FAV 纵向位置 `Pos_FAV`。

## 方法

EquiMotion 由三个连续阶段组成：

1. **未来 LV 条件状态空间编码器**：联合编码 FAV/LV 历史和已知 LV 未来运动，
   输出受物理约束的速度与位置滚动预测。
2. **身份与间隙双域校正**：通过车辆身份轨迹偏置、相对间隙分支和有界门控，
   融合直接位置预测与跟驰间隙预测。
3. **ComfortGuard**：使用三阶差分 Tikhonov 投影抑制 jerk，并以仅依赖可见 LV
   轨迹的常量位置平移保护预测安全性。

ComfortGuard 求解

```text
z* = argmin_z ||z - y||_2^2 + lambda ||D^3 z||_2^2,
```

其中 `D^3` 是三阶有限差分算子。常量位置平移不改变速度、加速度或 jerk。

## 仓库结构

```text
EquiMotion/
|-- data/                       # 完整 train.csv 与 test.csv（Git LFS）
|-- reports/model_card.json     # 任务、数据边界与结构元数据
|-- scripts/                    # 训练、推理、评测、数据校验入口
|-- src/equimotion/
|   |-- training.py             # 网络结构、特征、损失和训练流程
|   |-- release.py              # 推理与 ComfortGuard
|   |-- metrics.py              # 本地评分公式
|   `-- data.py                 # 数据读取与完整性检查
|-- tests/
`-- artifacts/                  # 本地生成文件，Git 默认忽略
```

## 安装

需要 Python 3.10 或更高版本，并先安装 [Git LFS](https://git-lfs.com/)。

```bash
git clone https://github.com/Git-beauti/EquiMotion.git
cd EquiMotion
git lfs pull
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## 数据校验

```bash
equimotion-validate
```

## 训练

建立张量缓存并在固定 train-only 留出集上训练：

```bash
equimotion-train --rebuild-cache --epochs 35 --device auto
```

默认 checkpoint 写入 `artifacts/equimotion_trained.pt`，该目录不会提交到 Git。

完成模型选择后，可使用所选 checkpoint 初始化全量训练：

```bash
equimotion-train \
  --full-train --epochs 3 --batch-size 128 \
  --learning-rate 1e-5 --min-learning-rate 1e-6 --weight-decay 0 \
  --architecture hybrid-gap-idtraj --target-smooth-lambda 1000 \
  --init-checkpoint artifacts/equimotion_trained.pt --reuse-init-normalization \
  --trainable-prefixes "gap_correction_head.,gap_position_gate.,fav_trajectory_bias.,lv_trajectory_bias." \
  --hidden-channels 128 --loss-mode metric --jerk-loss-weight 0.001 \
  --gap-aux-weight 0.2 --max-position-residual 12 --device cpu \
  --checkpoint artifacts/equimotion_full_train.pt
```

## 推理

仓库不附带 checkpoint，必须显式传入本地训练得到的模型：

```bash
equimotion-predict \
  --checkpoint artifacts/equimotion_full_train.pt \
  --output artifacts/predictions.csv \
  --device auto
```

推理报告和中间数组同样写入 `artifacts/`，不会进入版本控制。

## 本地评测

评测文件包含 `Segment_ID,Time_Index,Pos_FAV` 三列，每个有未来真值的片段
对应 200 行：

```bash
equimotion-evaluate \
  --predictions artifacts/validation_predictions.csv \
  --report artifacts/evaluation.json
```

评测实现覆盖 Accuracy、Safety、Comfort 与加权总分。归一化阈值可通过命令行
配置，因此本地结果应作为同一数据划分和同一阈值下的方法比较，而非外部平台分数。

## 许可证

代码采用 MIT License。数据许可证与来源边界见 `data/README.md` 和
`DATA_LICENSE.md`；本仓库的 MIT License 不对数据进行重新授权。

