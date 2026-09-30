# -*- coding: utf-8 -*-
"""优化器 —— 模型层(超参候选) × 执行层(阈值/止盈/止损) 的选优。

本文件有两个入口:
  - `optimize_on_oof` (需求7/8/9 的**主路径**): 用 **train** 训练各模型候选,
    再在 **OOF** 上网格评估每个 (模型, 执行) 组合并选优; OOC 全程只观察。
    `optimize_full_on_oof` 是它的**扩展版**(主脚本使用): 外层再叠「去冗余阈值」,
    即把「因子集松紧」也交给 OOF 选优; 内层执行网格增加「最长持有期」。
  - `nested_select` (备用路径, 供旧流水线 run_pipeline.py 使用): 在 train 内部做
    嵌套 CV, 以各折均值选优, 不触碰 OOF/OOC。

为什么主路径把优化放在 OOF:
  用户规格明确要求「只在 OOF 上优化, OOC 只观察」。这时样本被切成
  train(拟合) / OOF(选择) / OOC(唯一纯净检验), 是标准的 train-valid-test 结构;
  OOC 因此是唯一未被任何选择过程污染的段, 其表现是对外可披露的诚实结果。
  代价: OOF 上做了 ~数百次比较, 最优解可能含噪声 -> 故设 OOF_MIN_TRADES 门槛,
  并要求 OOC 表现作为最终(而非可选)检验。design==runtime 见 DESIGN.md / tests。
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


# ================================================================ 主路径: OOF 优化
def _select_best(g: pd.DataFrame) -> pd.DataFrame:
    """选优规则(需求7/8/9): 成交数门槛 -> 退化参数淘汰 -> 主目标+次目标排序。"""
    usable = g[g["n"] >= C.OOF_MIN_TRADES]
    if usable.empty:                        # 门槛过严时退回全量, 保证总有解
        usable = g
    cand = usable[usable["tp_rate"] >= C.MIN_TP_RATE]
    if cand.empty:                          # 止盈几乎不触发的组合全部作废; 仍空则退回
        cand = usable
    return cand.sort_values([C.OBJECTIVE_PRIMARY, C.OBJECTIVE_SECONDARY],
                            ascending=False).reset_index(drop=True)


def optimize_on_oof(F: pd.DataFrame, y: pd.Series, factors: List[str], side: str,
                    ohlc: dict, atr: np.ndarray, times: np.ndarray,
                    train_slice: slice, oof_slice: slice,
                    verbose: bool = True) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """在 **OOF** 上优化 (模型候选 × 执行参数)。训练只用 train, 选优只用 OOF。

    返回 (best, agg, raw):
      best = dict(model_name, model, thr_q, thr_abs, tp_mult, sl_mult, oof_metrics)
      agg  = 按选优规则排序后的组合表(供落盘审阅)
      raw  = 所有 (模型×执行) 的原始 OOF 评估明细
    """
    ts, te = train_slice.start, train_slice.stop
    grid = _exec_grid()
    rows: List[dict] = []
    trained: Dict[str, models.TrainedModel] = {}
    for mc in models.MODEL_GRID:
        tm = models.train_side(F.iloc[ts:te], y.iloc[ts:te], factors, side,
                               params=models.resolve_params(mc), verbose=False)
        trained[mc["name"]] = tm
        pred = tm.predict(F)                       # 全序列预测, 仅取 OOF 段评估
        for ec in grid:
            r = evaluate_with_params(pred, ohlc, atr, times, side,
                                     oof_slice.start, oof_slice.stop,
                                     ec["thr_q"], ec["tp_mult"], ec["sl_mult"])
            m = r["metrics"]
            rows.append(dict(model=mc["name"], **ec, n=int(m["n_trades"]),
                             thr_abs=float(r["thr_abs"]), sharpe=float(m["sharpe"]),
                             total_return=float(m["total_return"]), calmar=float(m["calmar"]),
                             max_drawdown=float(m["max_drawdown"]), win_rate=float(m["win_rate"]),
                             payoff=float(m["payoff_ratio"]), tp_rate=float(m["tp_rate"])))
        if verbose:
            print("    [%s] 模型候选 %-9s 已评估 %d 组执行参数" % (side, mc["name"], len(grid)))

    raw = pd.DataFrame(rows)
    agg = _select_best(raw)
    best = agg.iloc[0]
    name = best["model"]
    out = dict(model_name=name, model=trained[name], thr_q=float(best["thr_q"]),
               thr_abs=float(best["thr_abs"]), tp_mult=float(best["tp_mult"]),
               sl_mult=float(best["sl_mult"]),
               oof_metrics={k: float(best[k]) for k in
                            ("n", "sharpe", "total_return", "calmar", "max_drawdown",
                             "win_rate", "payoff", "tp_rate")})
    return out, agg, raw


# ================================================ 主路径扩展: OOF 外层扫描(去冗余 × 持有期)
def _exec_grid_oof() -> List[dict]:
    """OOF 外层扫描用的执行网格: 阈值分位 × 止盈 × 止损 × **最长持有期**。"""
    return [dict(thr_q=q, tp_mult=tp, sl_mult=sl, max_hold=h)
            for q, tp, sl, h in product(C.EXEC_THR_GRID, C.TP_ATR_GRID,
                                        C.SL_ATR_GRID, C.OOF_HOLD_GRID)]


def optimize_full_on_oof(F: pd.DataFrame, y: pd.Series, factor_sets: Dict[float, List[str]],
                         side: str, ohlc: dict, atr: np.ndarray, times: np.ndarray,
                         train_slice: slice, oof_slice: slice,
                         verbose: bool = True) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """在 **OOF** 上做两阶段选优, 训练只用 train、选择只用 OOF。

    外层: 去冗余阈值(决定因子集) × 模型超参候选;
    内层: 阈值分位 × 止盈 × 止损 × 最长持有期。
    每个外层组合都用 train 段**重新筛因子并重新训练**, 避免"因子集"成为未优化的死参数。

    factor_sets: {去冗余阈值: 该方向的因子列表}(由 factor_select 在 train 上产出)
    返回 (best, agg, raw, trained); trained[(dedup, model_name)] 供冻结时取回模型。
    """
    ts, te = train_slice.start, train_slice.stop
    grid = _exec_grid_oof()
    rows: List[dict] = []
    trained: Dict[Tuple[float, str], models.TrainedModel] = {}

    for dedup, facs in factor_sets.items():
        if not facs:
            print("    [%s] 去冗余 |corr|<%.2f 无可用因子, 跳过" % (side, dedup))
            continue
        for mc in models.MODEL_GRID:
            tm = models.train_side(F.iloc[ts:te], y.iloc[ts:te], facs, side,
                                   params=models.resolve_params(mc), verbose=False)
            trained[(dedup, mc["name"])] = tm
            pred = tm.predict(F)                       # 全序列预测, 仅取 OOF 段评估
            for ec in grid:
                r = evaluate_with_params(pred, ohlc, atr, times, side,
                                         oof_slice.start, oof_slice.stop,
                                         ec["thr_q"], ec["tp_mult"], ec["sl_mult"],
                                         max_hold=ec["max_hold"])
                m = r["metrics"]
                rows.append(dict(dedup=float(dedup), n_factors=len(facs), model=mc["name"],
                                 thr_q=float(ec["thr_q"]), tp_mult=float(ec["tp_mult"]),
                                 sl_mult=float(ec["sl_mult"]), max_hold=int(ec["max_hold"]),
                                 thr_abs=float(r["thr_abs"]), n=int(m["n_trades"]),
                                 sharpe=float(m["sharpe"]), total_return=float(m["total_return"]),
                                 calmar=float(m["calmar"]),
                                 max_drawdown=float(m["max_drawdown"]),
                                 win_rate=float(m["win_rate"]), payoff=float(m["payoff_ratio"]),
                                 tp_rate=float(m["tp_rate"])))
        if verbose:
            print("    [%s] 去冗余 |corr|<%.2f -> 因子 %2d 个 | %d 个模型 × %d 组执行 = %d 组已评估"
                  % (side, dedup, len(facs), len(models.MODEL_GRID), len(grid),
                     len(models.MODEL_GRID) * len(grid)))

    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError("OOF 网格为空: %s" % side)
    agg = _select_best(raw)
    best = agg.iloc[0]
    name = best["model"]
    out = dict(dedup=float(best["dedup"]), model_name=name,
               model=trained[(float(best["dedup"]), name)],
               thr_q=float(best["thr_q"]), thr_abs=float(best["thr_abs"]),
               tp_mult=float(best["tp_mult"]), sl_mult=float(best["sl_mult"]),
               max_hold=int(best["max_hold"]),
               oof_metrics={k: float(best[k]) for k in
                            ("n", "sharpe", "total_return", "calmar", "max_drawdown",
                             "win_rate", "payoff", "tp_rate")})
    return out, agg, raw
