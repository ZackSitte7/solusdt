# -*- coding: utf-8 -*-
"""金融时序交叉验证 —— Purged K-Fold + Embargo。

为什么不能用普通 KFold:
  金融标签是**重叠**的(相邻 bar 的前瞻收益窗口互相覆盖), 若随机折分, 验证集的标签
  信息会通过相邻训练样本泄漏进训练集, 指标虚高。
做法(López de Prado):
  1. 按时间切成 K 个连续折;
  2. 每折验证时, 从训练集中挖掉与验证集**标签区间重叠**的样本(purge);
  3. 再在验证集两侧各挖掉 embargo 根, 阻断序列自相关残留。
"""
from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np

import config as C


class PurgedKFold:
    """时序 Purged K-Fold。样本按时间升序, 索引即时间序。"""

    def __init__(self, n_splits: int = None, embargo: int = None, horizon: int = None):
        self.n_splits = C.CV_N_SPLITS if n_splits is None else n_splits
        self.embargo = C.CV_EMBARGO_BARS if embargo is None else embargo
        self.horizon = C.HORIZON if horizon is None else horizon

    def split(self, n_samples: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        idx = np.arange(n_samples)
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1

        start = 0
        for fs in fold_sizes:
            v_start, v_end = start, start + fs           # 验证集 [v_start, v_end)
            # 标签区间: 样本 i 的标签覆盖 [i, i+horizon] -> 需 purge 与之重叠的训练样本
            purge_lo = max(0, v_start - self.horizon)
            purge_hi = min(n_samples, v_end + self.horizon)
            test = idx[v_start:v_end]
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[purge_lo:purge_hi] = False        # purge 标签重叠
            # embargo: 验证集后侧再挖一段
            train_mask[v_end: min(n_samples, v_end + self.embargo)] = False
            train = idx[train_mask]
            start += fs
            if len(train) and len(test):
                yield train, test


def time_split(n_samples: int) -> Tuple[slice, slice, slice]:
    """按时间顺序切 训练 / OOF / OOC 三段。"""
    n_train = int(n_samples * C.TRAIN_FRAC)
    n_oof = int(n_samples * C.OOF_FRAC)
    return (slice(0, n_train), slice(n_train, n_train + n_oof), slice(n_train + n_oof, n_samples))


def describe_split(n_samples: int) -> List[dict]:
    """返回各段的区间与根数, 供报告与断言使用。"""
    a, b, c = time_split(n_samples)
    return [
        {"name": "train", "start": a.start, "end": a.stop, "n": a.stop - a.start},
        {"name": "oof", "start": b.start, "end": b.stop, "n": b.stop - b.start},
        {"name": "ooc", "start": c.start, "end": c.stop, "n": c.stop - c.start},
    ]
