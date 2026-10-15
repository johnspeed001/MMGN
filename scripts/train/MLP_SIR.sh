#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Task 1
python -u main.py datamodule.model_name=ResMLP datamodule.task=task1 datamodule.sampling_rate=[0.050,0.050] model=ResMLP
python -u main.py datamodule.model_name=ResMLP datamodule.task=task1 datamodule.sampling_rate=[0.250,0.250] model=ResMLP
python -u main.py datamodule.model_name=ResMLP datamodule.task=task1 datamodule.sampling_rate=[0.50,0.50]  model=ResMLP

# Task 2
python -u main.py datamodule.model_name=ResMLP datamodule.task=task2 datamodule.sampling_rate=[0.01,0.05]  model=ResMLP
python -u main.py datamodule.model_name=ResMLP datamodule.task=task2 datamodule.sampling_rate=[0.125,0.25] model=ResMLP
python -u main.py datamodule.model_name=ResMLP datamodule.task=task2 datamodule.sampling_rate=[0.25,0.50]  model=ResMLP


# Task 1
python -u main.py datamodule.model_name=SIREN datamodule.task=task1 datamodule.sampling_rate=[0.050,0.050] model=SirenNet
python -u main.py datamodule.model_name=SIREN datamodule.task=task1 datamodule.sampling_rate=[0.250,0.250] model=SirenNet
python -u main.py datamodule.model_name=SIREN datamodule.task=task1 datamodule.sampling_rate=[0.50,0.50]  model=SirenNet

# Task 2
python -u main.py datamodule.model_name=SIREN datamodule.task=task2 datamodule.sampling_rate=[0.01,0.05]  model=SirenNet
python -u main.py datamodule.model_name=SIREN datamodule.task=task2 datamodule.sampling_rate=[0.125,0.25] model=SirenNet
python -u main.py datamodule.model_name=SIREN datamodule.task=task2 datamodule.sampling_rate=[0.25,0.50]  model=SirenNet
