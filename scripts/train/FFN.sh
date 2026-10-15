#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Task 1
python -u main.py datamodule.model_name=FFN_P datamodule.task=task1 datamodule.sampling_rate=[0.050,0.050] model=FFN_P
python -u main.py datamodule.model_name=FFN_P datamodule.task=task1 datamodule.sampling_rate=[0.250,0.250] model=FFN_P
python -u main.py datamodule.model_name=FFN_P datamodule.task=task1 datamodule.sampling_rate=[0.50,0.50]  model=FFN_P

# Task 2
python -u main.py datamodule.model_name=FFN_P datamodule.task=task2 datamodule.sampling_rate=[0.01,0.05]  model=FFN_P
python -u main.py datamodule.model_name=FFN_P datamodule.task=task2 datamodule.sampling_rate=[0.125,0.25] model=FFN_P
python -u main.py datamodule.model_name=FFN_P datamodule.task=task2 datamodule.sampling_rate=[0.25,0.50]  model=FFN_P

# Task 1
python -u main.py datamodule.model_name=FFN_G datamodule.task=task1 datamodule.sampling_rate=[0.050,0.050] model=FFN_G
python -u main.py datamodule.model_name=FFN_G datamodule.task=task1 datamodule.sampling_rate=[0.250,0.250] model=FFN_G
python -u main.py datamodule.model_name=FFN_G datamodule.task=task1 datamodule.sampling_rate=[0.50,0.50]  model=FFN_G

# Task 2
python -u main.py datamodule.model_name=FFN_G datamodule.task=task2 datamodule.sampling_rate=[0.01,0.05]  model=FFN_G
python -u main.py datamodule.model_name=FFN_G datamodule.task=task2 datamodule.sampling_rate=[0.125,0.25] model=FFN_G
python -u main.py datamodule.model_name=FFN_G datamodule.task=task2 datamodule.sampling_rate=[0.25,0.50]  model=FFN_G
