#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"



# Task 1
python -u main.py datamodule.model_name=MMGN datamodule.task=task1 datamodule.sampling_rate=[0.050,0.050] model=MMGNet
python -u main.py datamodule.model_name=MMGN datamodule.task=task1 datamodule.sampling_rate=[0.250,0.250] model=MMGNet
python -u main.py datamodule.model_name=MMGN datamodule.task=task1 datamodule.sampling_rate=[0.50,0.50]  model=MMGNet

# Task 2
python -u main.py datamodule.model_name=MMGN datamodule.task=task2 datamodule.sampling_rate=[0.01,0.05]  model=MMGNet
python -u main.py datamodule.model_name=MMGN datamodule.task=task2 datamodule.sampling_rate=[0.125,0.25] model=MMGNet
python -u main.py datamodule.model_name=MMGN datamodule.task=task2 datamodule.sampling_rate=[0.25,0.50]  model=MMGNet

# Task 3y
python -u main.py datamodule.model_name=MMGN datamodule.task=task3 datamodule.sampling_rate=[0.050,0.050] model=MMGNet
python -u main.py datamodule.model_name=MMGN datamodule.task=task3 datamodule.sampling_rate=[0.250,0.250] model=MMGNet
python -u main.py datamodule.model_name=MMGN datamodule.task=task3 datamodule.sampling_rate=[0.50,0.50]  model=MMGNet

# Task 4
python -u main.py datamodule.model_name=MMGN datamodule.task=task4 datamodule.sampling_rate=[0.01,0.05]  model=MMGNet
python -u main.py datamodule.model_name=MMGN datamodule.task=task4 datamodule.sampling_rate=[0.125,0.25] model=MMGNet
python -u main.py datamodule.model_name=MMGN datamodule.task=task4 datamodule.sampling_rate=[0.25,0.50]  model=MMGNet
