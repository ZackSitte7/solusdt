# -*- coding: utf-8 -*-
"""优化器 —— **只在 OOF 段上**优化模型层与执行层。

红线:
  - OOC 段绝不参与任何参数选择, 只在最后冻结配置后跑一次。
  - 记录并披露**实际比较过的组合数**, 因为多次比较必然产生"看起来很好"的假阳性。
目标:
  主目标 夏普(年化), 次目标 总收益; 并要求成交笔数 >= MIN_TRADES, 否则不足以判定。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config as C
from src.execution import simulate_signals, equity_from_trades
from src.metrics import summarize

MIN_TRADES = 20          # OOF 段最少成交笔数, 低于此不参与优选


def grid_execution(pred: np.ndarray, ohlc: Dict[str, np.ndarray], atr: np.ndarray,
                   times: np.ndarray, side: str, start: int, end: int,
                   notional: float = None) -> pd.DataFrame:
    """在 [start,end) 区间上网格搜索 阈值分位 x 止盈 x 止损。返回结果表。"""
    notional = C.TRADE_NOTIONAL if notional is None else notional
    seg_pred = pred[start:end]
    finite = np.isfinite(seg_pred)
    if finite.sum() < 100:
        return pd.DataFrame()

    thr_grid = (0.50, 0.60, 0.70, 0.80, 0.90)
    rows: List[dict] = []
    for q in thr_grid:
        thr = float(np.nanquantile(seg_pred[finite], 1.0 - q if side == "long" else q))
        sig_full = np.zeros(len(pred), dtype=bool)
        if side == "long":
            sig_full[start:end] = finite & (seg_pred >= thr)
        else:
            sig_full[start:end] = finite & (seg_pred <= thr)
        for tp_m in C.TP_ATR_GRID:
            for sl_m in C.SL_ATR_GRID:
                trades = simulate_signals(sig_full, ohlc["open"], ohlc["high"], ohlc["low"],
                                          ohlc["close"], atr, times, side, tp_m, sl_m,
                                          C.MAX_HOLD_BARS, notional)
                if len(trades) < 5:
                    continue
                eq = equity_from_trades(trades, len(pred))
                m = summarize(trades, eq[start:end] if eq[start:end].size else eq)
                rows.append(dict(side=side, thr_q=q, thr=thr, tp_mult=tp_m, sl_mult=sl_m,
                                 **{k: m[k] for k in
                                    ("n_trades", "win_rate", "payoff_ratio", "profit_factor",
                                     "sharpe", "calmar", "max_drawdown", "total_return",
                                     "final_equity")}))
    return pd.DataFrame(rows)


def pick_best(grid: pd.DataFrame) -> Optional[dict]:
    """按 (夏普 -> 总收益) 优选, 并强制最小笔数。"""
    if grid.empty:
        return None
    ok = grid[grid["n_trades"] >= MIN_TRADES].copy()
    if ok.empty:
        ok = grid.copy()                       # 样本不足时退化为全表(会在报告里标注)
    ok = ok.sort_values(["sharpe", "total_return"], ascending=False)
    return ok.iloc[0].to_dict()


def grid_models(cand_results: Dict[str, dict]) -> Optional[str]:
    """模型层优选: 各候选配置在 OOF 上的夏普, 取最优。"""
    best, best_key = None, None
    for k, v in cand_results.items():
        s = v.get("sharpe", float("-inf"))
        if best is None or s > best:
            best, best_key = s, k
    return best_key


def disclosure(grid_rows: int, model_rows: int) -> str:
    """多重比较披露。"""
    tot = grid_rows + model_rows
    return ("本次共比较 %d 个组合(执行层 %d + 模型层 %d)。"
            "未做多重检验校正时, 夏普最高者很可能是噪声; "
            "判定应同时看 OOC 段是否复现。" % (tot, grid_rows, model_rows))
