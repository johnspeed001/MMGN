# -*- coding: utf-8 -*-
from pathlib import Path

import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from typing import Optional
from dataloaders.Baselineloder import (get_Baseline, Baseline, vizBaseline)
from dataloaders.MMGNloder import (get_MMGN, MMGN, vizMMGN)


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = "data/gst_data.npz",
                 model_name: str = "ResMLP",
                 pre_method: list = ["zscore", "zscore"],
                 reduce_dim: list = [],
                 task: str = "task1",
                 sampling_rate: list = [0., 0.],
                 n_train_val: list = [128, 32],
                 b_train_val: list = [16, 4],
                 num_workers: int = 4,
                 ):
        super().__init__()

        data_path = Path(data_dir).expanduser()
        if not data_path.is_absolute() and not data_path.exists():
            data_path = Path(__file__).resolve().parent / data_path
        self.data_dir = str(data_path.resolve())
        self.model_name = model_name
        self.pre_method = pre_method
        self.reduce_dim = reduce_dim
        self.task = task
        self.sampling_rate = sampling_rate
        self.n_train_val = n_train_val
        self.n_train, self.n_val = n_train_val
        self.b_train, self.b_val = b_train_val
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        """
        期望的 dataloaders.* 返回值（你已按掩膜改过）：
        - Baseline: get_Baseline(...) -> grid_lat, grid_lon, time_t, temprature, indices, mask, normalizer
        - MMGN    : get_MMGN(...)    -> grid_lat, grid_lon, temprature, indices, mask_3d, normalizer
        """
        if self.model_name in {"ResMLP", "SIREN", "FFN_P", "FFN_G", "MARE"}:
            grid_lat, grid_lon, time_t, temprature, indices, mask, normalizer = \
                get_Baseline(self.data_dir, self.n_train, self.pre_method, self.reduce_dim, self.task,
                             self.sampling_rate)

            # 训练/验证/测试数据集（都带 mask）
            self.train_data = Baseline(temprature, grid_lat, grid_lon, time_t, indices, mask)
            self.val_data = vizBaseline(temprature[:self.n_val, :, :], grid_lat, grid_lon, time_t[:self.n_val],
                                        mask[:self.n_val])
            self.test_data = vizBaseline(temprature, grid_lat, grid_lon, time_t, mask)

            # 释放显存/内存
            del grid_lat, grid_lon, temprature, indices, mask, time_t


        elif self.model_name in {"MMGN", "MMGN_NEW"}:
            grid_lat, grid_lon, temprature, indices, mask_3d, normalizer = \
                get_MMGN(self.data_dir, self.n_train, self.pre_method, self.reduce_dim, self.task, self.sampling_rate)
            self.train_data = MMGN(temprature, grid_lat, grid_lon, indices, mask_3d)
            self.val_data = vizMMGN(temprature[:self.n_val, :, :], grid_lat, grid_lon, mask_3d[:self.n_val])
            self.test_data = vizMMGN(temprature, grid_lat, grid_lon, mask_3d)

            # 释放显存/内存
            del grid_lat, grid_lon, temprature, indices, mask_3d

        elif self.model_name in {"LNO"}:
            grid_lat, grid_lon, time_t, temprature, indices, mask, normalizer = \
                get_Baseline(self.data_dir, self.n_train, self.pre_method, self.reduce_dim, self.task,
                             self.sampling_rate)
            H, W = grid_lat.shape[0], grid_lon.shape[1]
            from dataloaders.Baselineloder import LNOTrain, LNOViz
            self.train_data = LNOTrain(temprature, indices, mask, H, W)
            self.val_data = LNOViz(temprature[:self.n_val, :, :], indices[:self.n_val, :], mask[:self.n_val, :, :], H,
                                   W)
            self.test_data = LNOViz(temprature, indices, mask, H, W)
            del grid_lat, grid_lon, temprature, indices, mask, time_t
            return normalizer



        else:
            raise ValueError(f"Unknown model_name: {self.model_name}")

        return normalizer

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.b_train,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.b_val,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.b_train,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False
        )
