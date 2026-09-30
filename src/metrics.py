# -*- coding: utf-8 -*-
"""绩效指标 —— 胜率 / 盈亏比 / 夏普 / 卡玛 / 最大回撤 等。

约定:
  - 夏普用 **按 bar 的权益收益** 年化: 4h bar -> 每年 365*6 = 2190 根;
  - 无风险利率取 0(加密资产场景下不引入额外假设);
  - 最大回撤基于权益曲线峰值回撤; 卡玛 = 年化收益 / 最大回撤。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

import config as C

BARS_PER_YEAR = 365 * 6            # 4h bar: 6 根/天


def _ann_factor() -> float:
    return float(np.sqrt(BARS_PER_YEAR))


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(dd.min())


def sharpe(equity: np.ndarray) -> float:
    if len(equity) < 3:
        return 0.0
    r = np.diff(equity) / np.where(equity[:-1] == 0, np.nan, equity[:-1])
    r = r[np.isfinite(r)]
    if len(r) < 3 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * _ann_factor())


def calmar(equity: np.ndarray) -> float:
    if len(equity) < 2:
        return 0.0
    total = equity[-1] / equity[0] - 1.0
    years = len(equity) / BARS_PER_YEAR
    if years <= 0:
        return 0.0
    ann = (1.0 + total) ** (1.0 / years) - 1.0 if total > -1 else -1.0
    mdd = abs(max_drawdown(equity))
    return float(ann / mdd) if mdd > 1e-9 else 0.0


def trade_stats(trades: List) -> Dict[str, float]:
    """交易级统计。盈亏比 = 平均盈利 / 平均亏损(绝对值)。"""
    n = len(trades)
    if n == 0:
        return {k: 0.0 for k in
                ("n_trades", "win_rate", "avg_win", "avg_loss", "payoff_ratio",
                 "profit_factor", "avg_ret", "total_pnl", "avg_bars",
                 "tp_rate", "sl_rate", "timeout_rate")}
    pnl = np.array([t.pnl_usdt for t in trades], dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    reasons = [t.exit_reason for t in trades]
    return {
        "n_trades": float(n),
        "win_rate": float((pnl > 0).mean()),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": float(abs(avg_win / avg_loss)) if avg_loss < 0 else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.sum() < 0 else float("inf"),
        "avg_ret": float(np.mean([t.net_ret for t in trades])),
        "total_pnl": float(pnl.sum()),
        "avg_bars": float(np.mean([t.bars_held for t in trades])),
        "tp_rate": float(np.mean([r == "tp" for r in reasons])),
        "sl_rate": float(np.mean([r == "sl" for r in reasons])),
        "timeout_rate": float(np.mean([r == "timeout" for r in reasons])),
    }


def summarize(trades: List, equity: np.ndarray) -> Dict[str, float]:
    """交易级 + 权益级指标合并。"""
    out = trade_stats(trades)
    out.update({
        "final_equity": float(equity[-1]) if len(equity) else C.INIT_CAPITAL,
        "total_return": float(equity[-1] / equity[0] - 1.0) if len(equity) > 1 else 0.0,
        "sharpe": sharpe(equity),
        "calmar": calmar(equity),
        "max_drawdown": max_drawdown(equity),
    })
    return out
