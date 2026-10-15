# 实验代码

从项目根目录运行以下入口。数据放在 `data/`，输出放在 `outputs/`。

| 目录 | 用途 | 入口 |
| --- | --- | --- |
| `chapter2/noiseEnhance/` | 含噪稀疏观测增强、四算法对比 | `python -m experiments.chapter2.noiseEnhance.run_focused_experiments` |
| `latent/` | 潜变量维度扫描、离线分析、消融、点云图 | `python experiments/latent/latentSweep.py model=MMGNet datamodule.model_name=MMGN` |
| `simulate/` | 浮标扰动仿真与 TCN 补偿 | `python -m experiments.simulate.train_tcn` |

第二章论文生成工具位于 `docs/thesis/chapter2/tools/`。第三章文稿及工具位于 `docs/thesis/chapter3/`。

`latent_offline.py` 的 `CKPT_DICT` 沿用原先各维度指向同一 checkpoint 的设置；做不同维度比较前需自行选定对应训练权重。目录整理没有改变实验参数或重算已有结果。
