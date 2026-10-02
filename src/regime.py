# -*- coding: utf-8 -*-
"""制度门控(regime gate) —— 只在"顺势制度"里开仓。

动机(v4 结论): 样本外亏损的主因不是因子冗余, 而是**制度错配** —— train 段 SOL 是 +7916%
的超级牛市, OOC 段是 -27% 的熊市; long 模型在牛市里学"逢跌做多", 到熊市结构性失效。

做法: 用**只用过去信息**的均线制度做开关, 只在制度与方向一致时放行信号:
    long  仅当 close > MA           (上行制度)
    short 仅当 close < MA           (下行制度)
`sma200_slope` 再要求均线本身在该方向上行/下行, 过滤"价格刚穿均线但趋势未确立"的假信号。

门控发生在**信号 bar**(t 收盘), 与成交(t+1 开盘)一致, 不引入未来函数。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def regime_masks(df: pd.DataFrame, rule: str) -> dict:
    """返回 {"long": bool数组(全序列, True=允许开多), "short": ...}。

    任何规则的取值都只依赖 bar t 及之前的 close, 无未来信息; 早期 NaN 一律判定为
    False(不允许开仓)。
    """
    n = len(df)
    if rule == "none":
        return {"long": np.ones(n, dtype=bool), "short": np.ones(n, dtype=bool)}

    if rule in ("sma100", "sma200"):
        k = C.V5_REGIME_MA[rule]
        ma = _sma(df["close"], k)
        up = (df["close"] > ma).to_numpy(dtype=bool)
        dn = (df["close"] < ma).to_numpy(dtype=bool)
        return {"long": up, "short": dn}

    if rule == "sma200_slope":
        ma = _sma(df["close"], C.V5_REGIME_MA[rule])
        slope = ma.diff(C.V5_SLOPE_LOOKBACK)
        up = ((df["close"] > ma) & (slope > 0)).to_numpy(dtype=bool)
        dn = ((df["close"] < ma) & (slope < 0)).to_numpy(dtype=bool)
        return {"long": up, "short": dn}

    if rule == "atr_pct":
        # 波动率制度(v6): 只在 ATR% 处于过去 V6_VOL_WIN 根的低分位时放行(低波动=趋势更干净)。
        # 分位阈值用 rolling 计算, 只依赖 bar t 及之前 -> 无未来函数。
        atrp = df["atr"] / df["close"]
        thr = atrp.rolling(C.V6_VOL_WIN, min_periods=C.V6_VOL_MIN).quantile(C.V6_VOL_Q)
        ok = (atrp <= thr).to_numpy(dtype=bool)
        return {"long": ok, "short": ok}

    # ---------------------------------------------------------------- v7: 空头专用制度
    # 动机: v6 的空头在 OOC 亏钱 —— 通用的「close < MA」在下跌市里会**滞后**, 开空时下跌
    # 往往已近尾声。v7 给空头加三条更苛刻的结构条件 + 一条"空头排列 + 低波动"组合。
    if rule == "sma_align":
        # 空头排列: close < MA20 < MA50 < MA200(短中长均线依次向下)—— 趋势已确立才开空。
        n1, n2, n3 = C.V7_ALIGN_SMA
        m1, m2, m3 = _sma(df["close"], n1), _sma(df["close"], n2), _sma(df["close"], n3)
        dn = ((df["close"] < m1) & (m1 < m2) & (m2 < m3)).to_numpy(dtype=bool)
        up = ((df["close"] > m1) & (m1 > m2) & (m2 > m3)).to_numpy(dtype=bool)
        return {"long": up, "short": dn}

    if rule == "ema_bear":
        # 快慢 EMA 空头排列: close < EMA_fast < EMA_slow; 对均线反应更快, 更早确认下行。
        nf, ns = C.V7_EMA_PAIR
        ef, es = _ema(df["close"], nf), _ema(df["close"], ns)
        dn = ((df["close"] < ef) & (ef < es)).to_numpy(dtype=bool)
        up = ((df["close"] > ef) & (ef > es)).to_numpy(dtype=bool)
        return {"long": up, "short": dn}

    if rule == "breakdown_20":
        # 破位: 收盘跌破**过去 V7_BREAKDOWN_WIN 根出现过的最低价**(创新低)。只用 <= t 的信息
        # (shift(1) 后取滚动最小, 再与当根收盘比较), 属于最"事件驱动"的开空条件。
        k = C.V7_BREAKDOWN_WIN
        prior_low = df["low"].rolling(k, min_periods=k).min().shift(1)
        dn = (df["close"] < prior_low).to_numpy(dtype=bool)
        prior_high = df["high"].rolling(k, min_periods=k).max().shift(1)
        up = (df["close"] > prior_high).to_numpy(dtype=bool)
        return {"long": up, "short": dn}

    if rule == "bear_vol":
        # 空头排列 ∩ 低波动: 只在"趋势向下且波动收敛"时开空, 回避下跌末端的剧烈反抽。
        n1, n2, n3 = C.V7_ALIGN_SMA
        m1, m2, m3 = _sma(df["close"], n1), _sma(df["close"], n2), _sma(df["close"], n3)
        align_dn = (df["close"] < m1) & (m1 < m2) & (m2 < m3)
        atrp = df["atr"] / df["close"]
        thr = atrp.rolling(C.V6_VOL_WIN, min_periods=C.V6_VOL_MIN).quantile(C.V6_VOL_Q)
        dn = (align_dn & (atrp <= thr)).to_numpy(dtype=bool)
        align_up = (df["close"] > m1) & (m1 > m2) & (m2 > m3)
        up = (align_up & (atrp <= thr)).to_numpy(dtype=bool)
        return {"long": up, "short": dn}

    raise ValueError("未知制度规则: %s" % rule)
