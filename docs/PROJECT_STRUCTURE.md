# 项目目录说明

整理日期：2026-10-15。

## 目录布局

```text
MMGN/
├── main.py                     # Hydra 训练入口
├── data_interface.py           # Lightning DataModule
├── model_interface.py          # Lightning 训练/评估接口
├── utils.py
├── requirements.txt
├── configs/                    # 数据、模型、训练器、日志等配置
├── models/                     # MMGN、FFN、SIREN、ResMLP、LNO、MARE
├── dataloaders/                # 数据读取、采样、归一化
├── experiments/
│   ├── chapter2/noiseEnhance/   # 噪声建模、增强算法、四算法对比
│   ├── latent/                 # 潜变量扫描、消融与可视化
│   └── simulate/               # 浮标扰动数据生成与 TCN 补偿
├── scripts/
│   ├── data/                   # CMEMS 下载、格式转换、掩膜、元信息
│   ├── evaluation/             # checkpoint 性能评估
│   ├── plotting/               # 流场帧与误差图
│   └── train/                  # FFN / MLP_SIR / MMGN 批量训练
├── matlab/                     # MATLAB 图表脚本
├── data/
│   ├── raw/cmems/              # 从 scripts/ 移入的 5 个原始 NetCDF
│   ├── current/                # 现有 bohai.npz 与元信息
│   ├── processed/              # 转换脚本今后生成的数据，按需创建
│   ├── simulated/              # 原 simulate/sim_data，含 index.csv
│   └── benchmarks/             # MATLAB 读取的性能对比 CSV
├── outputs/
│   ├── training/               # 原 logs/，完整保留 checkpoint 与日志
│   ├── latent/                 # latent_sweep / latent_offline / latent_tsne_plots
│   ├── simulate/               # runs_tcn 与 eval_out
│   ├── chapter2/               # 噪声增强 metrics 与 figures
│   ├── chapter3/work_inspect/   # 文稿检查文本、媒体与公式拼图
│   ├── matlab/                 # res / resAll / model_efficiency.png
│   └── figures/                # 通用绘图脚本输出
└── docs/
    ├── assets/                 # 原 README 图片
    ├── thesis/
    │   ├── chapter2/           # 第二章全部文稿版本、整改表与 tools/
    │   └── chapter3/           # 第三章全部文稿版本与 tools/
    ├── PROJECT_STRUCTURE.md
    ├── RESULTS.md
    └── directory_migration.csv # 本次文件/目录移动记录
```

## 使用约定

- 命令默认从项目根目录运行。批量训练脚本会自动定位项目根目录。
- 核心 Python 模块留在原处，保持模型配置 `_target_` 和已有 checkpoint 的模块引用稳定。
- 新增实验代码放入 `experiments/<实验名>/`；新工具按用途放入 `scripts/`。
- 原始数据放入 `data/raw/`；转换后的新数据放入 `data/processed/`。现有 `data/current/bohai.npz` 保留原名及内容。
- 默认图表、指标和训练输出分别写入对应 `outputs/` 子目录。显式传入的相对输出参数仍相对当前工作目录。
- 第二章生成工具从 `outputs/chapter2/` 读取图表与指标，文稿写入 `docs/thesis/chapter2/`。
- MATLAB 脚本通过自身文件位置定位 `data/benchmarks/` 和 `outputs/matlab/`。
- 论文版本按章节归档，不推断版本优劣、不删除旧版。原先放在 chapter2 的两个“第三章”文稿已归入第三章。
- `.gitignore` 排除本地原始数据、训练日志、常见缓存和 IDE 状态。原来已纳入版本管理的仿真数据、TCN 参考权重和报告仍可纳入版本管理，未执行取消跟踪或提交。
- 已有日志、Hydra 快照和 checkpoint 内的历史路径保持原样，作为运行记录；重新运行时使用当前配置或显式指定新路径。

## 常用入口

```bash
# 单次重建模型训练
python main.py model=MMGNet datamodule.model_name=MMGN

# 批量训练
bash scripts/train/MMGN.sh
bash scripts/train/FFN.sh
bash scripts/train/MLP_SIR.sh

# checkpoint 性能评估：--ckpt 必填，避免误用旧机器上的路径
python scripts/evaluation/eval_ckpt.py --ckpt outputs/training/MMGN_task1_s_0.05/checkpoints/last.ckpt

# 第二章实验（--quick 可缩小规模；使用单独输出目录保留正式结果）
python -m experiments.chapter2.noiseEnhance.run_focused_experiments --quick --out outputs/chapter2_quick_check

# 第二章文稿生成（会写入对应文稿文件）
python docs/thesis/chapter2/tools/generate_chapter2_integrated_docx.py
python docs/thesis/chapter2/tools/generate_chapter2_rectified_docx.py

# 潜变量扫描
python experiments/latent/latentSweep.py model=MMGNet datamodule.model_name=MMGN

# 浮标仿真：按需生成数据、训练、评估
python -m experiments.simulate.simulate_dataset --help
python -m experiments.simulate.train_tcn --help
python -m experiments.simulate.eval_report --out_dir outputs/simulate/check_eval

# 输入 NetCDF 必须显式指定；默认写入 data/processed/current/
python scripts/data/convert_nc_to_npz.py --input data/raw/cmems/SRtest.nc
python scripts/data/convert_cmems_current_to_npz_uv.py --input data/raw/cmems/SRtest.nc

# 绘图：默认写入 outputs/figures/
python scripts/plotting/plot_npz_timestep.py --input data/current/bohai.npz --step 10
python scripts/plotting/errorPlot.py --run-dir outputs/training/MMGN_task1_s_0.25
```

以上是入口说明，不表示本次整理执行过完整训练或重新生成了文稿。

## 原有依赖与待准备材料

- 数值实验需要 NumPy、SciPy、Pandas、Matplotlib；训练另需 PyTorch、Lightning、Hydra 等。按运行平台先安装合适的 PyTorch，再参考根目录 `requirements.txt`。论文工具另外需要 `python-docx`、`lxml`。
- 第三章 `build_chapter3_cn.py`、`inspect_docx.py` 仍需要 `docs/thesis/chapter3/submission.docx`，原目录中没有该输入文件。
- 部分 Word 公式转换工具仍依赖 Windows Office 的 `MML2OMML.XSL`，这是原工具的环境要求。
- `latent_offline.py` 原本把多个 latent 维度指向同一 checkpoint。本次只修正目录位置，具体对比时应自行指定对应权重。

## 迁移与保留情况

新旧路径详见 [迁移清单](directory_migration.csv)。清单按执行顺序记录；部分子目录先迁移，再迁移其父目录。所有原始数据、权重、图表和文稿均保留。代码只调整位置、导入和文件路径，并修正批量训练对不存在的 `model=MMGN` 配置的引用。

## 本次验证

- 整理前共记录 2,115 个文件，迁移后均找到对应文件；34 个源码、配置或说明文件进行了路径相关修改，原始数据、权重、图表和文稿内容未变。小文件核对 SHA-256，大文件核对同一文件标识、大小与修改时间；IDE 自动更新的 `.idea/workspace.xml` 不参与内容一致性判断。
- 49 个 Python 文件通过语法检查，全部 Hydra YAML 通过解析，3 个 Shell 脚本通过语法检查。
- 用记录程序替代 Python，对 36 条批量训练命令进行空跑，确认从项目外启动时工作目录正确、模型配置文件存在。
- 在可用的 Python 运行时中，第二章模块成功导入并读取 `bohai.npz`，数据形状为 `(731, 69, 117)`；两份第二章文稿生成工具成功导入并找到指标、图表及数据目录。
- 仿真数据生成脚本的模块入口和直接脚本入口均通过 `--help` 检查；对第二章、latent 和 checkpoint 评估入口执行了默认路径检查。
- 未执行完整训练、重新生成实验图表或 Word 文稿，也未运行 MATLAB。当前可用环境缺少 PyTorch、Hydra、Matplotlib 等完整实验依赖。
