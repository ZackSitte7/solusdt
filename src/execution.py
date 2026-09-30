# -*- coding: utf-8 -*-
"""执行层 —— 信号 -> 成交的模拟。

关键约定(防未来函数):
  - 信号在 bar t 收盘后产生, **在 bar t+1 的 open 成交**;
  - 止盈/止损用 bar t+1.. 的 high/low 判定, 同一根内若两者都触到, **按先止损**处理(保守);
  - 最长持有到期后, 以该根 close 平仓。
成本:
  - 单边 = 手续费 + 滑点, 开平各收一次。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
import pandas as pd

import config as C


@dataclass
class Trade:
    side: str            # long / short
    entry_idx: int
    exit_idx: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    tp: float
    sl: float
    exit_reason: str     # tp / sl / timeout
    bars_held: int
    gross_ret: float     # 价格收益率(未扣成本)
    net_ret: float       # 扣成本后的收益率
    pnl_usdt: float      # 该笔盈亏(USDT)


def _round_trip_cost() -> float:
    """一个来回的总成本率(开+平)。"""
    return 2.0 * (C.FEE_RATE + C.SLIP_RATE)


def simulate_signals(signals: np.ndarray, o: np.ndarray, h: np.ndarray, l: np.ndarray,
                     c: np.ndarray, atr: np.ndarray, times: np.ndarray,
                     side: str, tp_mult: float, sl_mult: float,
                     max_hold: int, notional: float) -> List[Trade]:
    """按信号数组模拟单腿交易(同一时刻最多一笔)。signals[t]=True 表示 t 收盘发信号。"""
    n = len(signals)
    trades: List[Trade] = []
    cost = _round_trip_cost()
    i = 1
    while i < n - 1:
        if not signals[i] or not np.isfinite(atr[i]) or atr[i] <= 0:
            i += 1
            continue
        e = i + 1                                   # 下一根 open 成交
        if e >= n or not np.isfinite(o[e]) or o[e] <= 0:
            i += 1
            continue
        entry = float(o[e])
        a = float(atr[i])
        if side == "long":
            tp, sl = entry + tp_mult * a, entry - sl_mult * a
        else:
            tp, sl = entry - tp_mult * a, entry + sl_mult * a

        exit_idx, exit_price, reason = None, None, "timeout"
        last = min(e + max_hold, n - 1)
        for j in range(e, last + 1):
            if side == "long":
                if l[j] <= sl:                      # 先判止损(保守)
                    exit_idx, exit_price, reason = j, sl, "sl"
                    break
                if h[j] >= tp:
                    exit_idx, exit_price, reason = j, tp, "tp"
                    break
            else:
                if h[j] >= sl:
                    exit_idx, exit_price, reason = j, sl, "sl"
                    break
                if l[j] <= tp:
                    exit_idx, exit_price, reason = j, tp, "tp"
                    break
        if exit_idx is None:
            exit_idx, exit_price, reason = last, float(c[last]), "timeout"

        gross = (exit_price - entry) / entry if side == "long" else (entry - exit_price) / entry
        net = gross - cost
        trades.append(Trade(
            side=side, entry_idx=e, exit_idx=exit_idx,
            entry_time=pd.Timestamp(times[e]), exit_time=pd.Timestamp(times[exit_idx]),
            entry_price=entry, exit_price=float(exit_price), tp=float(tp), sl=float(sl),
            exit_reason=reason, bars_held=int(exit_idx - e + 1),
            gross_ret=float(gross), net_ret=float(net),
            pnl_usdt=float(net * notional)))
        i = exit_idx + 1                            # 平仓后才允许下一笔
    return trades


def signal_from_pred(pred: np.ndarray, side: str, thr_q: float,
                     start: int, end: int) -> np.ndarray:
    """按预测值分位生成信号: 多头取 >= 上分位, 空头取 <= 下分位。仅作用于 [start,end)。"""
    sig = np.zeros(len(pred), dtype=bool)
    seg = pred[start:end]
    finite = np.isfinite(seg)
    if finite.sum() < 30:
        return sig
    q = 1.0 - thr_q if side == "long" else thr_q
    thr = float(np.nanquantile(seg[finite], q))
    if side == "long":
        sig[start:end] = finite & (seg >= thr)
    else:
        sig[start:end] = finite & (seg <= thr)
    return sig


def evaluate_with_params(pred: np.ndarray, ohlc: dict, atr: np.ndarray, times: np.ndarray,
                         side: str, start: int, end: int, thr_q: float,
                         tp_mult: float, sl_mult: float,
                         notional: float = None) -> dict:
    """统一评估入口: 信号 -> 成交 -> 指标 + 权益曲线。所有调用方共用, 保证口径一致。"""
    notional = C.TRADE_NOTIONAL if notional is None else notional
    sig = signal_from_pred(pred, side, thr_q, start, end)
    trades = simulate_signals(sig, ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"],
                              atr, times, side, tp_mult, sl_mult,
                              C.MAX_HOLD_BARS, notional)
    eq = equity_from_trades(trades, len(pred))
    from src.metrics import summarize
    m = summarize(trades, eq[start:end])
    return {"metrics": m, "trades": trades, "equity": eq[start:end]}


def equity_from_trades(trades: List[Trade], n_bars: int) -> np.ndarray:
    """按 bar 记权益曲线(含持仓浮盈), 用于算夏普/回撤。"""
    eq = np.full(n_bars, C.INIT_CAPITAL, dtype=float)
    realized = 0.0
    open_trades = []
    by_entry = {}
    for t in trades:
        by_entry.setdefault(t.entry_idx, []).append(t)
    for i in range(n_bars):
        for t in by_entry.get(i, []):
            open_trades.append(t)
        # 平掉已到期的
        still = []
        for t in open_trades:
            if t.exit_idx == i:
                realized += t.pnl_usdt
            else:
                still.append(t)
        open_trades = still
        unreal = 0.0
        for t in open_trades:
            # 极简浮盈: 用平仓价与入场价线性插值到当前 bar, 避免引入未定义价格
            span = max(1, t.exit_idx - t.entry_idx)
            frac = min(1.0, max(0.0, (i - t.entry_idx) / span))
            unreal += t.pnl_usdt * frac
        eq[i] = C.INIT_CAPITAL + realized + unreal
    return eq
