# -*- coding: utf-8 -*-
"""优化器 —— **嵌套 CV 选模 + 选执行参数**。

为什么不用「在 OOF 上网格搜索」:
  上一版在 OOF(2015 根)上比较了 350 个组合, 选出 OOF 夏普 +2.36, 但 OOC 转负 -2.29。
  在有限样本上做大规模搜索, 最优解几乎必然是噪声。
做法(嵌套 CV):
  外层: 在**训练段内部**做 Purged K-Fold, 得到若干折;
  内层: 每折用「其余折」训练模型, 在该折上评估 (模型候选 x 执行参数) 组合;
  聚合: 同一组合在各折上的得分取均值 -> 按均值排序选出最优组合;
  最后: 用**整个训练段**重训一次选定模型, 冻结参数。
  OOF 段只用于**一次性确认**, OOC 段只观察 —— 两者都不参与任何选择。
"""
from __future__ import annotations

from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config as C
from src import models
from src.cv import PurgedKFold
from src.execution import evaluate_with_params

MIN_TRADES_PER_FOLD = 8       # 单折最少成交笔数, 低于此该组合该折不计分


def _exec_grid() -> List[dict]:
    return [dict(thr_q=q, tp_mult=tp, sl_mult=sl)
            for q, tp, sl in product(C.EXEC_THR_GRID, C.TP_ATR_GRID, C.SL_ATR_GRID)]


def nested_select(F: pd.DataFrame, y: pd.Series, factors: List[str], side: str,
                  ohlc: dict, atr: np.ndarray, times: np.ndarray,
                  train_slice: slice, verbose: bool = True) -> Tuple[str, dict, pd.DataFrame]:
    """在训练段内部做嵌套 CV, 选出 (模型候选名, 执行参数)。"""
    ts, te = train_slice.start, train_slice.stop
    n_train = te - ts
    cands = models.MODEL_GRID
    grid = _exec_grid()
    rows: List[dict] = []

    folds = list(PurgedKFold(n_splits=C.NESTED_FOLDS, embargo=C.CV_EMBARGO_BARS,
                             horizon=C.HORIZON).split(n_train))
    for fi, (tr_l, va_l) in enumerate(folds, 1):
        tr_g, va_g = ts + tr_l, ts + va_l
        v0, v1 = int(va_g.min()), int(va_g.max()) + 1
        for mc in cands:
            booster = models.train_fold(F, y, factors, tr_g, va_g,
                                        params=models.resolve_params(mc))
            pred = models.predict_booster(booster, F, factors)
            for ec in grid:
                r = evaluate_with_params(pred, ohlc, atr, times, side, v0, v1,
                                         ec["thr_q"], ec["tp_mult"], ec["sl_mult"])
                m = r["metrics"]
                if m["n_trades"] < MIN_TRADES_PER_FOLD:
                    continue
                rows.append(dict(fold=fi, model=mc["name"], **ec,
                                 n=m["n_trades"], sharpe=m["sharpe"],
                                 total_return=m["total_return"], tp_rate=m["tp_rate"],
                                 win_rate=m["win_rate"], payoff=m["payoff_ratio"]))
        if verbose:
            print("    [%s] 外层折 %d/%d 完成 (验证 %d 根)"
                  % (side, fi, len(folds), v1 - v0))

    g = pd.DataFrame(rows)
    if g.empty:
        return cands[0]["name"], dict(thr_q=0.6, tp_mult=2.0, sl_mult=1.5), g

    # 退化参数淘汰: 止盈几乎不触发的组合直接作废
    g = g[g["tp_rate"] >= C.MIN_TP_RATE]
    if g.empty:
        g = pd.DataFrame(rows)

    agg = (g.groupby(["model", "thr_q", "tp_mult", "sl_mult"])
             .agg(n_folds=("sharpe", "size"), sharpe=("sharpe", "mean"),
                  total_return=("total_return", "mean"),
                  tp_rate=("tp_rate", "mean"), win_rate=("win_rate", "mean"),
                  payoff=("payoff", "mean"), n_trades=("n", "mean"))
             .reset_index()
             .sort_values(["sharpe", "total_return"], ascending=False)
             .reset_index(drop=True))
    best = agg.iloc[0]
    return best["model"], dict(thr_q=float(best["thr_q"]), tp_mult=float(best["tp_mult"]),
                               sl_mult=float(best["sl_mult"])), agg


def disclosure(n_combos: int, n_folds: int) -> str:
    return ("嵌套 CV: %d 个(模型×执行)组合 x %d 折 = %d 次评估, 选优依据为**各折均值**。"
            "聚合后比较的独立组合为 %d 个。OOF/OOC 均未参与选择。" %
            (n_combos, n_folds, n_combos * n_folds, n_combos))
