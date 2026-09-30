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

    raise ValueError("未知制度规则: %s" % rule)
