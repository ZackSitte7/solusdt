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

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

import config as C

MIN_SEG = 30                 # 段内最少有效预测数, 低于此不生成阈值(避免噪声分位)


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
                     max_hold: int, notional: float,
                     trail_mult: float = 0.0,
                     scan_lo: int = 1, scan_hi: int = None) -> List[Trade]:
    """按信号数组模拟单腿交易(同一时刻最多一笔)。signals[t]=True 表示 t 收盘发信号。

    scan_lo/scan_hi: 只在 [scan_lo, scan_hi) 内查找入场信号(出场仍可延伸到序列末端)。
    因信号在评估段之外恒为 False, 限定扫描区间不改变结果, 但可把每次评估的开销从
    全序列降到评估段长度(评估网格动辄上万组, 这是关键提速)。

    trail_mult > 0 时启用 **ATR 跟踪止损**: 止损随持仓期内的有利极值单向移动
        long : sl_dyn = max(sl_init, 最高价 - trail_mult * ATR)
        short: sl_dyn = min(sl_init, 最低价 + trail_mult * ATR)
    移动发生在**判定之后**, 即先判本根是否触及上一根收盘时的止损位(规避同根内
    "先看极值再回填止损"的未来函数)。跟踪止损用于替代"固定持有到期按收盘平仓":
    未走出方向的仓位被跟踪止损截断, 走出方向的仓位可以继续持有。
    """
    n = len(signals)
    trades: List[Trade] = []
    cost = _round_trip_cost()
    trail = trail_mult > 0
    hi = (n - 1) if scan_hi is None else int(scan_hi)
    i = max(1, int(scan_lo))
    while i < hi:
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
        sl_dyn = sl
        hh = ll = entry
        for j in range(e, last + 1):
            if side == "long":
                if l[j] <= sl_dyn:                  # 先判止损(保守)
                    exit_idx, exit_price, reason = j, sl_dyn, "sl"
                    break
                if h[j] >= tp:
                    exit_idx, exit_price, reason = j, tp, "tp"
                    break
                hh = max(hh, float(h[j]))
                if trail:
                    sl_dyn = max(sl_dyn, hh - trail_mult * a)
            else:
                if h[j] >= sl_dyn:
                    exit_idx, exit_price, reason = j, sl_dyn, "sl"
                    break
                if l[j] <= tp:
                    exit_idx, exit_price, reason = j, tp, "tp"
                    break
                ll = min(ll, float(l[j]))
                if trail:
                    sl_dyn = min(sl_dyn, ll + trail_mult * a)
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


def threshold_from_quantile(pred: np.ndarray, side: str, thr_q: float,
                            start: int, end: int) -> float:
    """从 [start,end) 段预测值取分位, 得到**绝对**阈值。
    多头取上分位(1-thr_q), 空头取下分位(thr_q)。"""
    seg = pred[start:end]
    finite = np.isfinite(seg)
    if finite.sum() < MIN_SEG:
        return float("nan")
    q = 1.0 - thr_q if side == "long" else thr_q
    return float(np.nanquantile(seg[finite], q))


def signal_from_threshold(pred: np.ndarray, side: str, thr_abs: float,
                          start: int, end: int, regime: np.ndarray = None) -> np.ndarray:
    """按**冻结的绝对阈值**生成信号。

    关键: OOC 评估必须传入由 OOF 冻结下来的 thr_abs, 而**不能**用 OOC 自身分布取分位,
    否则会用 OOC 的分布信息(等于偷看 OOC)。

    regime: 可选, 全序列 bool 数组(制度门控); 传入时信号需**同时**满足阈值与制度条件。
    """
    sig = np.zeros(len(pred), dtype=bool)
    if not np.isfinite(thr_abs):
        return sig
    seg = pred[start:end]
    finite = np.isfinite(seg)
    if side == "long":
        sig[start:end] = finite & (seg >= thr_abs)
    else:
        sig[start:end] = finite & (seg <= thr_abs)
    if regime is not None:
        sig &= np.asarray(regime, dtype=bool)
    return sig


def signal_from_pred(pred: np.ndarray, side: str, thr_q: float,
                     start: int, end: int) -> np.ndarray:
    """按预测值分位生成信号(便捷封装: 先算分位阈值, 再按绝对阈值比对)。"""
    return signal_from_threshold(
        pred, side, threshold_from_quantile(pred, side, thr_q, start, end), start, end)


def evaluate_with_threshold(pred: np.ndarray, ohlc: dict, atr: np.ndarray, times: np.ndarray,
                            side: str, start: int, end: int, thr_abs: float,
                            tp_mult: float, sl_mult: float,
                            notional: float = None, max_hold: int = None,
                            trail_mult: float = 0.0, regime: np.ndarray = None) -> dict:
    """统一评估入口(绝对阈值): 信号 -> 成交 -> 指标 + 权益曲线。所有调用方共用, 口径一致。

    regime: 可选制度门控(全序列 bool 数组), 与阈值信号取交集。
    """
    notional = C.TRADE_NOTIONAL if notional is None else notional
    max_hold = C.MAX_HOLD_BARS if max_hold is None else int(max_hold)
    sig = signal_from_threshold(pred, side, thr_abs, start, end, regime=regime)
    trades = simulate_signals(sig, ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"],
                              atr, times, side, tp_mult, sl_mult,
                              max_hold, notional, trail_mult=trail_mult,
                              scan_lo=start, scan_hi=end)
    eq = equity_from_trades(trades, len(pred), lo=start, hi=end)
    from src.metrics import summarize
    m = summarize(trades, eq)
    return {"metrics": m, "trades": trades, "equity": eq,
            "thr_abs": float(thr_abs), "n_signals": int(sig.sum())}


def evaluate_with_params(pred: np.ndarray, ohlc: dict, atr: np.ndarray, times: np.ndarray,
                         side: str, start: int, end: int, thr_q: float,
                         tp_mult: float, sl_mult: float,
                         notional: float = None, max_hold: int = None,
                         trail_mult: float = 0.0, regime: np.ndarray = None) -> dict:
    """分位阈值版入口(先由 [start,end) 段预测算分位, 再评估)。"""
    thr = threshold_from_quantile(pred, side, thr_q, start, end)
    return evaluate_with_threshold(pred, ohlc, atr, times, side, start, end, thr,
                                   tp_mult, sl_mult, notional, max_hold, trail_mult, regime)


def equity_from_trades(trades: List[Trade], n_bars: int, lo: int = 0, hi: int = None) -> np.ndarray:
    """按 bar 记权益曲线(含持仓浮盈), 用于算夏普/回撤。

    lo/hi: 只构造 [lo, hi) 区间的权益(评估段)。评估网格动辄上万组, 逐组算全序列
    权益会成为瓶颈; 因评估段内成交的建仓索引都 >= lo, 段内权益起点即 INIT_CAPITAL,
    限定区间不改变段内指标。默认 lo=0/hi=n_bars 即全序列。
    """
    hi = n_bars if hi is None else int(hi)
    lo = int(lo)
    eq = np.full(hi - lo, C.INIT_CAPITAL, dtype=float)
    realized = 0.0
    open_trades = []
    by_entry = {}
    for t in trades:
        by_entry.setdefault(t.entry_idx, []).append(t)
    for i in range(lo, hi):
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
        eq[i - lo] = C.INIT_CAPITAL + realized + unreal
    return eq
