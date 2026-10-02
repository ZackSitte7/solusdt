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


def _bars_since(s: pd.Series, n: int, is_min: bool = False) -> pd.Series:
    """自窗口内极值(最高/最低)以来经过的根数(0 = 本根即极值)。只用 t 及之前的信息。"""
    fn = (lambda a: float(len(a) - 1 - int(np.argmin(a)))) if is_min \
        else (lambda a: float(len(a) - 1 - int(np.argmax(a))))
    return s.rolling(n, min_periods=n).apply(fn, raw=True)


def _rolling_corr(a: pd.Series, b: pd.Series, n: int) -> pd.Series:
    """滚动相关系数(向量化: 协方差 / 标准差之积)。"""
    return _safe_div(a.rolling(n, min_periods=n).cov(b),
                     a.rolling(n, min_periods=n).std() * b.rolling(n, min_periods=n).std())


def _rolling_r2(s: pd.Series, n: int) -> pd.Series:
    """滚动线性拟合决定系数 R² —— 价格对时间做一元回归的 R²(0~1)。

    衡量"这段走势有多像一条直线"(趋势质量), 与斜率大小无关。v14 趋势类因子。
    """
    x = np.arange(n, dtype=float)
    xm = x.mean()
    sxx = float(((x - xm) ** 2).sum())

    def _fn(a):
        ym = a.mean()
        sxy = float(np.dot(a - ym, x - xm))
        sst = float(((a - ym) ** 2).sum())
        return (sxy * sxy) / (sxx * sst) if sst > 0 else np.nan

    return s.rolling(n, min_periods=n).apply(_fn, raw=True)


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

    # ================================================================ v6: 扩充因子库
    # 目标: 在原有 61 个因子之外, 补足「波动率结构 / 高阶矩 / 自相关 / 趋势强度 /
    # 量能资金流 / 形态统计」六个方向, 给因子筛选更多可挑的**正交**候选。
    # 全部只用 rolling / shift / ewm(<=t), 无未来信息。
    atr_rel = _safe_div(atr14, c)

    # ---- 1) 波动率结构: 期限结构、波动率的波动率、更有效的波动率估计量
    for n in (6, 24, 48):
        f["rv_%d" % n] = lr.rolling(n, min_periods=n).std()
    f["rv_ratio_6_24"] = _safe_div(f["rv_6"], f["rv_24"])
    f["rv_ratio_12_48"] = _safe_div(lr.rolling(12, min_periods=12).std(), f["rv_48"])
    f["vol_of_vol_20"] = atr_rel.rolling(20, min_periods=20).std()
    f["atr_z_20"] = _safe_div(atr_rel - atr_rel.rolling(20, min_periods=20).mean(),
                              atr_rel.rolling(20, min_periods=20).std())
    f["bb_width_chg"] = _safe_div(f["bb_width_20"],
                                  f["bb_width_20"].rolling(20, min_periods=20).mean()) - 1.0
    hl_log, co_log = np.log(_safe_div(h, l)), np.log(_safe_div(c, o))
    f["gk_vol_20"] = np.sqrt((0.5 * hl_log ** 2
                              - (2.0 * np.log(2.0) - 1.0) * co_log ** 2)
                             .rolling(20, min_periods=20).mean())          # Garman-Klass
    hc, ho = np.log(_safe_div(h, c)), np.log(_safe_div(h, o))
    lc, lo = np.log(_safe_div(l, c)), np.log(_safe_div(l, o))
    f["rs_vol_20"] = np.sqrt((hc * ho + lc * lo).clip(lower=0)
                             .rolling(20, min_periods=20).mean())          # Rogers-Satchell

    # ---- 2) 高阶矩 / 涨跌波动不对称(恐慌与贪婪的形状差异)
    f["ret_skew_20"] = lr.rolling(20, min_periods=20).skew()
    f["ret_kurt_20"] = lr.rolling(20, min_periods=20).kurt()
    f["up_vol_20"] = lr.clip(lower=0.0).rolling(20, min_periods=20).std()
    f["dn_vol_20"] = (-lr).clip(lower=0.0).rolling(20, min_periods=20).std()
    f["vol_asym_20"] = _safe_div(f["up_vol_20"] - f["dn_vol_20"],
                                 f["up_vol_20"] + f["dn_vol_20"])

    # ---- 3) 自相关 / 均值回归(趋势延续 vs 反转的可预测性)
    f["autocorr_1_20"] = _rolling_corr(lr, lr.shift(1), 20)
    f["autocorr_5_50"] = _rolling_corr(lr, lr.shift(5), 50)
    f["ret1_z_20"] = _safe_div(lr, lr.rolling(20, min_periods=20).std())

    # ---- 4) 趋势强度(补充): EMA 交叉、Aroon、Vortex、区间距离、极值时间
    f["ema_ratio_9_21"] = _safe_div(_ema(c, 9) - _ema(c, 21), c)
    f["ma_slope_100"] = _safe_div(_sma(c, 100), _sma(c, 100).shift(12)) - 1.0
    f["trend_consistency_20"] = np.sign(lr).rolling(20, min_periods=20).mean()
    n_ar = 25
    # Aroon Up 用「距窗口最高价多少根」, Aroon Down 用「距窗口最低价多少根」
    f["aroon_up_25"] = 100.0 * (n_ar - _bars_since(h, n_ar + 1, is_min=False)) / n_ar
    f["aroon_dn_25"] = 100.0 * (n_ar - _bars_since(l, n_ar + 1, is_min=True)) / n_ar
    f["aroon_osc_25"] = f["aroon_up_25"] - f["aroon_dn_25"]
    tr_sum14 = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()],
                         axis=1).max(axis=1).rolling(14, min_periods=14).sum()
    f["vortex_pos_14"] = _safe_div((h - l.shift(1)).abs().rolling(14, min_periods=14).sum(), tr_sum14)
    f["vortex_neg_14"] = _safe_div((l - h.shift(1)).abs().rolling(14, min_periods=14).sum(), tr_sum14)
    f["vortex_diff_14"] = f["vortex_pos_14"] - f["vortex_neg_14"]
    kc_mid = _ema(c, 20)
    f["kc_pos_20"] = _safe_div(c - kc_mid, 2.0 * atr14)                  # Keltner 通道位置
    f["kc_width_20"] = _safe_div(4.0 * atr14, kc_mid)
    for n in (20, 50):
        f["donchian_width_%d" % n] = _safe_div(h.rolling(n, min_periods=n).max()
                                               - l.rolling(n, min_periods=n).min(), c)
    f["dist_high_50"] = _safe_div(h.rolling(50, min_periods=50).max(), c) - 1.0
    f["dist_low_50"] = _safe_div(l.rolling(50, min_periods=50).min(), c) - 1.0
    f["bars_since_high_50"] = _bars_since(h, 50, is_min=False) / 50.0
    f["bars_since_low_50"] = _bars_since(l, 50, is_min=True) / 50.0

    # ---- 5) 超买超卖(补充): RSI 斜率 / 随机 RSI / Williams %R / 加速度
    f["rsi_slope_14"] = f["rsi_14"] - f["rsi_14"].shift(5)
    f["williams_r_14"] = -100.0 * _safe_div(hh14 - c, hh14 - ll14)
    rmin = f["rsi_14"].rolling(14, min_periods=14).min()
    rmax = f["rsi_14"].rolling(14, min_periods=14).max()
    f["stoch_rsi_14"] = _safe_div(f["rsi_14"] - rmin, rmax - rmin)
    f["roc_48"] = c / c.shift(48) - 1.0
    f["price_accel"] = f["ret_3"] - f["ret_12"] / 4.0

    # ---- 6) 量能 / 资金流(补充): MFI / CMF / 量价相关 / 非流动性
    tp_ = (h + l + c) / 3.0
    mf_ = tp_ * v
    f["mfi_14"] = 100.0 - 100.0 / (1.0 + _safe_div(
        mf_.where(tp_.diff() > 0, 0.0).rolling(14, min_periods=14).sum(),
        mf_.where(tp_.diff() < 0, 0.0).rolling(14, min_periods=14).sum()))
    f["cmf_20"] = _safe_div((_safe_div((c - l) - (h - c), h - l) * v)
                            .rolling(20, min_periods=20).sum(), v.rolling(20, min_periods=20).sum())
    f["vol_price_corr_20"] = _rolling_corr(lr, v.pct_change(), 20)
    f["vratio_12_48"] = _safe_div(v.rolling(12, min_periods=12).mean(),
                                  v.rolling(48, min_periods=48).mean())
    f["amihud_20"] = _safe_div(lr.abs(), qv).rolling(20, min_periods=20).mean() * 1e6
    f["obv_slope_50"] = _safe_div(obv - obv.shift(50), v.rolling(50, min_periods=50).sum())
    f["quote_vol_ratio_5_20"] = _safe_div(_sma(qv, 5), _sma(qv, 20))

    # ---- 7) K线形态统计(近 5/20 根的平均形态)
    f["body_ratio_ma5"] = f["body_ratio"].rolling(5, min_periods=5).mean()
    f["upper_shadow_ma5"] = f["upper_shadow"].rolling(5, min_periods=5).mean()
    f["lower_shadow_ma5"] = f["lower_shadow"].rolling(5, min_periods=5).mean()
    f["doji_20"] = (f["body_ratio"].abs() < 0.10).rolling(20, min_periods=20).mean()
    f["hammer_20"] = ((f["lower_shadow"] > 0.5) & (f["body_ratio"] > -0.2)) \
        .rolling(20, min_periods=20).mean()
    f["shooting_20"] = ((f["upper_shadow"] > 0.5) & (f["body_ratio"] < 0.2)) \
        .rolling(20, min_periods=20).mean()

    # ---- 8) 时间(补充)
    f["is_weekend"] = (dt.dt.dayofweek >= 5).astype(float)

    # ================================================================ v7: 空头专用因子
    # v6 的归因显示: 空头在 OOC(下跌市)反而亏钱 —— 说明通用因子没能刻画"下跌的结构"。
    # 这里补一组只在空头逻辑里成立的因子: 下行动量的绝对强度、破位位置、高低点同步下移、
    # 相对自身历史的超卖、跳空低开。它们对多头无意义, 故单列一组, 由 OOF 决定是否采用。
    dn = (-lr).clip(lower=0.0)                    # 下跌幅度(正数)
    up = lr.clip(lower=0.0)                       # 上涨幅度(正数)
    f["dn_mom_12"] = dn.rolling(12, min_periods=12).sum()
    f["up_mom_12"] = up.rolling(12, min_periods=12).sum()
    f["mom_asym_12"] = _safe_div(f["dn_mom_12"] - f["up_mom_12"],
                                 f["dn_mom_12"] + f["up_mom_12"])
    f["ret_vol_adj_20"] = _safe_div(lr.rolling(20, min_periods=20).mean(),
                                    lr.rolling(20, min_periods=20).std())
    # 破位: 现价相对过去 20 根最低价/最高价的位置(负值 = 已跌破支撑)
    f["break_sup_20"] = _safe_div(c, l.rolling(20, min_periods=20).min()) - 1.0
    f["dist_res_20"] = _safe_div(h.rolling(20, min_periods=20).max(), c) - 1.0
    # 结构走弱: 近 20 根里「高点下移」/「低点下移」的比例
    f["lower_high_20"] = (h < h.shift(1)).rolling(20, min_periods=20).mean()
    f["lower_low_20"] = (l < l.shift(1)).rolling(20, min_periods=20).mean()
    f["bear_persist"] = f["lower_high_20"] * f["lower_low_20"]
    # 相对自身历史(50 根)的超卖: RSI 的 z 分数(负 = 比自身常态更超卖)
    f["rsi_bear_z_50"] = _safe_div(f["rsi_14"] - f["rsi_14"].rolling(50, min_periods=50).mean(),
                                   f["rsi_14"].rolling(50, min_periods=50).std())
    f["gap_down_20"] = ((o < c.shift(1)).astype(float)).rolling(20, min_periods=20).mean()

    # ================================================================ v13: 多头专用因子
    # v7 只给空头补了"下跌结构"专用因子, 多头一直没有对称的处理。v13 对 long 做同样的专属化:
    # 上涨的结构走强(高点/低点同时抬高)、突破阻力、相对自身历史的超买、跳空高开、上涨效率。
    # 这些对空头逻辑无意义, 故在 v13 中**仅进入 long 的候选池**(short 侧显式排除)。
    f["higher_high_20"] = (h > h.shift(1)).rolling(20, min_periods=20).mean()
    f["higher_low_20"] = (l > l.shift(1)).rolling(20, min_periods=20).mean()
    f["bull_persist"] = f["higher_high_20"] * f["higher_low_20"]     # 结构持续走强
    f["break_res_20"] = _safe_div(c, h.rolling(20, min_periods=20).max()) - 1.0  # 相对 20 根新高
    f["rsi_bull_z_50"] = _safe_div(f["rsi_14"] - f["rsi_14"].rolling(50, min_periods=50).mean(),
                                   f["rsi_14"].rolling(50, min_periods=50).std())
    f["gap_up_20"] = ((o > c.shift(1)).astype(float)).rolling(20, min_periods=20).mean()
    f["up_eff_20"] = _safe_div(up.rolling(20, min_periods=20).sum(),
                               _safe_div(h - l, c).rolling(20, min_periods=20).sum())

    # ================================================================ v14: 趋势类因子(long 专属)
    # 因子库里已有 adx/aroon/vortex/keltner/donchian/lr_slope/trend_consistency 等趋势因子, v14 只补
    # **尚未覆盖的 5 个趋势维度**, 且都做成"上涨为正"的有向形式, 便于 long 直接使用(short 侧显式排除):
    #   1) 趋势质量: 价格像不像一条直线(R²) × 斜率符号 —— 区分"真趋势"与"宽幅震荡";
    #   2) Kaufman 效率比: 净移动 / 路径长度 —— 同样区分趋势 vs 震荡, 但对噪声更稳健;
    #   3) 多周期同向度: 20/50/100 均线斜率方向是否一致 —— 捕捉大级别趋势共振;
    #   4) 趋势加速度: 短均线斜率的增量 —— 趋势是在走强还是走弱;
    #   5) ADX 斜率: 趋势强度指标的自身变化 —— ADX 向上=趋势正在增强。
    for n in (20, 50):
        r2 = _rolling_r2(c, n)
        f["trend_qual_%d" % n] = r2 * np.sign(f["lr_slope_%d" % n])   # 有向趋势质量
        net = (c - c.shift(n)).abs()
        path = c.diff().abs().rolling(n, min_periods=n).sum()
        f["eff_ratio_%d" % n] = _safe_div(net, path) * np.sign(c - c.shift(n))  # 有向效率比
    f["mtf_trend_align"] = (np.sign(f["ma_slope_20"]) + np.sign(f["ma_slope_50"])
                            + np.sign(f["ma_slope_100"])) / 3.0
    _sl20 = _safe_div(_sma(c, 20), _sma(c, 20).shift(5)) - 1.0
    f["ma_slope_accel"] = _sl20 - (_safe_div(_sma(c, 20).shift(5), _sma(c, 20).shift(10)) - 1.0)
    f["adx_slope_14"] = f["adx_14"] - f["adx_14"].shift(5)

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
