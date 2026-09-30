# -*- coding: utf-8 -*-
"""4h 因子库 —— 只使用 t 及之前的 bar, 不含未来信息。

因子分组:
  收益/动量    return / momentum / ROC
  趋势(必备)  均线比值、均线斜率、MACD、趋势强度、线性回归斜率、区间位置
  波动        ATR 比、实现波动、布林带宽、Parkinson、振幅
  超买超卖    RSI / Stochastic / CCI
  量能       量 z-score、量比、OBV 斜率、VWAP 偏离
  订单流     taker 主动买占比(**仅当数据源提供** trades/taker_buy_* 时构建;
             欧易公开 K 线无此字段, 自动跳过)
  形态       收盘在 K 线内的位置、上下影线、实体比、跳空
  时间       小时 / 星期(加密市场存在日内与周内效应)
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

import config as C

# 订单流因子依赖的原始字段。欧易(OKX)公开 K 线缺少这些字段 -> 自动跳过相关因子。
ORDER_FLOW_COLS = ("trades", "taker_buy_base", "taker_buy_quote")


# ---------------------------------------------------------------- 基础工具
def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    """ADX(n) —— 经典趋势强度指标(需求14 趋势类因子)。"""
    up, dn = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=high.index)
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    plus_di = 100.0 * _safe_div(plus_dm.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean(), atr)
    minus_di = 100.0 * _safe_div(minus_dm.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean(), atr)
    dx = 100.0 * _safe_div((plus_di - minus_di).abs(), plus_di + minus_di)
    return dx.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


# ---------------------------------------------------------------- 因子构建
def build_factors(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """返回 (含全部因子列的 DataFrame, 因子名列表)。因子值只依赖 <= t 的信息。"""
    d = df.copy()
    o, h, l, c, v = d["open"], d["high"], d["low"], d["close"], d["volume"]
    qv = d["quote_volume"]
    lr = d["log_ret"]
    f: dict = {}

    # ---------------- 收益 / 动量
    for k in (1, 2, 3, 6, 12, 24, 48):
        f["ret_%d" % k] = np.log(c / c.shift(k))
    for k in (6, 12, 24):
        f["roc_%d" % k] = c / c.shift(k) - 1.0
    f["mom_accel"] = f["ret_6"] - f["ret_24"] / 4.0        # 短动量相对长动量的加速度

    # ---------------- 趋势(必备)
    for n in (20, 50, 100, 200):
        f["ma_ratio_%d" % n] = _safe_div(c, _sma(c, n)) - 1.0
    for n in (20, 50):
        f["ma_slope_%d" % n] = _safe_div(_sma(c, n), _sma(c, n).shift(5)) - 1.0
    f["ma_align"] = ((_sma(c, 20) > _sma(c, 50)).astype(float)
                     + (_sma(c, 50) > _sma(c, 200)).astype(float))   # 0~2 多头排列强度
    ema12, ema26 = _ema(c, 12), _ema(c, 26)
    f["macd"] = _safe_div(ema12 - ema26, c)
    f["macd_signal"] = _safe_div(_ema(ema12 - ema26, 9), c)
    f["macd_hist"] = f["macd"] - f["macd_signal"]
    atr14 = _atr(h, l, c, 14)
    f["adx_14"] = _adx(h, l, c, 14)                                 # 趋势强度(需求14)
    f["trend_strength_50"] = _safe_div(c - _sma(c, 50), atr14)      # 距 50 均线多少个 ATR
    # 线性回归斜率(标准化为每根 bar 的 %): 20/50 根窗口
    x = np.arange(20.0)
    xm = x.mean()
    denom20 = ((x - xm) ** 2).sum()
    f["lr_slope_20"] = c.rolling(20, min_periods=20).apply(
        lambda a: float(np.dot(a - a.mean(), x - xm) / denom20) / (a.mean() or np.nan), raw=True)
    x50 = np.arange(50.0)
    xm50, denom50 = x50.mean(), ((x50 - x50.mean()) ** 2).sum()
    f["lr_slope_50"] = c.rolling(50, min_periods=50).apply(
        lambda a: float(np.dot(a - a.mean(), x50 - xm50) / denom50) / (a.mean() or np.nan), raw=True)
    for n in (20, 50, 100):
        lo, hi = l.rolling(n, min_periods=n).min(), h.rolling(n, min_periods=n).max()
        f["pos_in_range_%d" % n] = _safe_div(c - lo, hi - lo)       # 0~1, 越接近 1 越在区间高位
    # 新高/新低计数(近 20 根内创多少根新高)
    f["hh_count_20"] = (c >= c.rolling(20, min_periods=20).max()).rolling(20, min_periods=5).sum() / 20.0
    f["ll_count_20"] = (c <= c.rolling(20, min_periods=20).min()).rolling(20, min_periods=5).sum() / 20.0

    # ---------------- 波动
    f["atr_ratio"] = _safe_div(atr14, c)
    f["atr_ratio_chg"] = _safe_div(atr14, atr14.shift(6)) - 1.0
    for n in (12, 20, 50):
        f["rv_%d" % n] = lr.rolling(n, min_periods=n).std()
    f["vol_ratio_5_20"] = _safe_div(lr.rolling(5, min_periods=5).std(),
                                    lr.rolling(20, min_periods=20).std())
    sd20 = c.rolling(20, min_periods=20).std()
    f["bb_width_20"] = _safe_div(4.0 * sd20, _sma(c, 20))
    f["bb_pos_20"] = _safe_div(c - _sma(c, 20), sd20)                # 布林带内位置(z 值)
    f["parkinson_20"] = np.sqrt((np.log(_safe_div(h, l)) ** 2).rolling(20, min_periods=20).mean()
                               / (4.0 * np.log(2.0)))
    f["range_ratio"] = _safe_div(h - l, c)

    # ---------------- 超买超卖
    f["rsi_14"] = _rsi(c, 14)
    f["rsi_28"] = _rsi(c, 28)
    f["rsi_div"] = f["rsi_14"] - f["rsi_28"]
    ll14, hh14 = l.rolling(14, min_periods=14).min(), h.rolling(14, min_periods=14).max()
    f["stoch_k_14"] = 100.0 * _safe_div(c - ll14, hh14 - ll14)
    tp = (h + l + c) / 3.0
    f["cci_20"] = _safe_div(tp - _sma(tp, 20), 0.015 * tp.rolling(20, min_periods=20).std())

    # ---------------- 量能
    vm, vs = _sma(v, 20), v.rolling(20, min_periods=20).std()
    f["vol_z_20"] = _safe_div(v - vm, vs)
    f["vol_ma_ratio"] = _safe_div(_sma(v, 5), vm)
    f["quote_vol_z_20"] = _safe_div(qv - _sma(qv, 20), qv.rolling(20, min_periods=20).std())
    if "trades" in d.columns:                                        # 订单流字段(欧易无)
        tr_n = d["trades"]
        f["trades_z_20"] = _safe_div(tr_n - _sma(tr_n, 20), tr_n.rolling(20, min_periods=20).std())
        f["avg_trade_size"] = _safe_div(v, tr_n)                     # 单笔均量(大单占比代理)
        f["avg_trade_size_z"] = _safe_div(f["avg_trade_size"] - _sma(f["avg_trade_size"], 20),
                                          f["avg_trade_size"].rolling(20, min_periods=20).std())
    # OBV 斜率
    obv = (np.sign(lr.fillna(0.0)) * v).cumsum()
    f["obv_slope_20"] = _safe_div(obv - obv.shift(20), v.rolling(20, min_periods=20).sum())
    # VWAP 偏离(20 根)
    f["vwap_dev_20"] = _safe_div(c, _safe_div((tp * v).rolling(20, min_periods=20).sum(),
                                             v.rolling(20, min_periods=20).sum())) - 1.0

    # ---------------- 订单流: taker 主动买占比(仅数据源提供 trades/taker_buy_* 时)
    if all(col in d.columns for col in ("taker_buy_base", "taker_buy_quote")):
        tbr = _safe_div(d["taker_buy_base"], v)
        f["taker_buy_ratio"] = tbr
        f["taker_buy_ratio_ma6"] = _sma(tbr, 6)
        f["taker_buy_ratio_dev"] = tbr - _sma(tbr, 20)
        f["taker_buy_z_20"] = _safe_div(tbr - _sma(tbr, 20), tbr.rolling(20, min_periods=20).std())
        qbr = _safe_div(d["taker_buy_quote"], qv)
        f["taker_buy_quote_ratio"] = qbr

    # ---------------- K线形态
    rng = (h - l).replace(0, np.nan)
    f["close_pos_in_bar"] = _safe_div(c - l, rng)
    f["upper_shadow"] = _safe_div(h - np.maximum(o, c), rng)
    f["lower_shadow"] = _safe_div(np.minimum(o, c) - l, rng)
    f["body_ratio"] = _safe_div(c - o, rng)
    f["gap"] = c.shift(1) / o - 1.0
    f["consec_up"] = (lr > 0).astype(int).groupby((lr <= 0).cumsum()).cumsum().astype(float)
    f["consec_dn"] = (-(lr < 0).astype(int)).groupby((lr >= 0).cumsum()).cumsum().astype(float)

    # ---------------- 时间
    dt = pd.to_datetime(d["datetime"], utc=True)
    f["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24.0)
    f["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24.0)
    f["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7.0)
    f["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7.0)

    F = pd.DataFrame(f, index=d.index)
    F = F.replace([np.inf, -np.inf], np.nan)
    return F, list(F.columns)


def add_label(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """标签。

    label_raw : 未来 horizon 根的对数收益(close_t -> close_{t+horizon})
    label     : **以 ATR 为单位**的未来涨跌 = label_raw / (ATR_t / close_t)。
                除以 ATR 是为了消除波动率/漂移的制度差异 —— 训练段是 +7941% 大牛市,
                验证段是熊市, 用原始收益会让模型学成「永远做多」。
    另给出**可成交的**前瞻收益(open_{t+1} -> close_{t+horizon}), 供回测/评估对照,
    避免用 close_t 建仓这种不可交易假设。
    """
    d = df.copy()
    atr = _atr(d["high"], d["low"], d["close"], 14)
    atr_rel = _safe_div(atr, d["close"])                 # 相对 ATR(占价格比例)
    d["atr"] = atr

    d["label_raw"] = np.log(d["close"].shift(-horizon) / d["close"])
    d["label_tradable_raw"] = np.log(d["close"].shift(-horizon) / d["open"].shift(-1))

    if getattr(C, "LABEL_MODE", "atr") == "atr":
        d["label"] = _safe_div(d["label_raw"], atr_rel)
        d["label_tradable"] = _safe_div(d["label_tradable_raw"], atr_rel)
    else:
        d["label"] = d["label_raw"]
        d["label_tradable"] = d["label_tradable_raw"]
    return d
