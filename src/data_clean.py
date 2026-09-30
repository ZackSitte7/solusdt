# -*- coding: utf-8 -*-
"""数据清洗与完整性检查。

原则:
  - 只做**可复现、可解释**的清洗(排序去重、异常值剔除、缺口统计), 不做插值填充,
    以免用未来信息污染因子。
  - 所有剔除都在报告中留痕(剔除了哪几根、为什么), 保证设计与运行同一。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402

BAR_MS = {"4h": 4 * 3600 * 1000, "1d": 24 * 3600 * 1000}


def integrity_check(df: pd.DataFrame, interval: str = None) -> dict:
    """K线完整性体检: 重复、时序、缺口、OHLC 逻辑、零量。"""
    interval = interval or C.INTERVAL
    step = BAR_MS[interval]
    rep = {}
    rep["rows"] = len(df)
    rep["dup_open_time"] = int(df["open_time"].duplicated().sum())
    rep["monotonic"] = bool(df["open_time"].is_monotonic_increasing)

    # 缺口: open_time 间隔 > 1 步
    d = np.diff(df["open_time"].to_numpy())
    gaps = d[d > step]
    rep["gap_count"] = int(len(gaps))
    rep["gap_max_bars"] = int((gaps.max() // step) - 1) if len(gaps) else 0
    rep["gap_total_missing"] = int(((gaps // step) - 1).sum()) if len(gaps) else 0

    # OHLC 逻辑: high >= max(o,c), low <= min(o,c), high >= low
    o, h, l, c = (df["open"].to_numpy(), df["high"].to_numpy(),
                  df["low"].to_numpy(), df["close"].to_numpy())
    rep["bad_ohlc"] = int(((h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l)).sum())
    rep["nonpositive_price"] = int(((o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)).sum())
    rep["zero_volume"] = int((df["volume"] <= 0).sum())

    rng = h / np.where(l > 0, l, np.nan) - 1.0
    rep["range_p99"] = float(np.nanpercentile(rng, 99))
    rep["range_max"] = float(np.nanmax(rng))
    return rep


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """清洗主流程。返回干净 DataFrame(含 log_ret 辅助列)。"""
    n0 = len(df)
    df = df.copy()

    # 1) 排序 + 去重(同一 open_time 只保留最后一版)
    df = df.sort_values("open_time").drop_duplicates("open_time", keep="last")

    # 2) 价格必须为正
    mask_price = (df[["open", "high", "low", "close"]] > 0).all(axis=1)
    bad_price = int((~mask_price).sum())
    df = df[mask_price]

    # 3) 单根振幅异常(交易所故障/脏数据)剔除
    rng = df["high"] / df["low"] - 1.0
    bad_range = int((rng > C.MAX_BAR_RANGE).sum())
    df = df[rng <= C.MAX_BAR_RANGE]

    # 4) 单根收益异常剔除(基于 close-to-close)
    ret = df["close"].pct_change()
    bad_ret = int((ret.abs() > C.MAX_BAR_RET).sum())
    df = df[ret.abs().fillna(0) <= C.MAX_BAR_RET]

    # 5) OHLC 逻辑修正(high/low 与 o/c 取极值), 保证后续 ATR/振幅计算自洽
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    # 6) 零量剔除
    bad_vol = int((df["volume"] <= 0).sum())
    df = df[df["volume"] > 0]

    df = df.reset_index(drop=True)
    df["log_ret"] = np.log(df["close"]).diff()

    if verbose:
        print("清洗: %d -> %d 根  (剔除 价格异常%d / 振幅>%.0f%% %d / 收益>%.0f%% %d / 零量%d)"
              % (n0, len(df), bad_price, C.MAX_BAR_RANGE * 100, bad_range,
                 C.MAX_BAR_RET * 100, bad_ret, bad_vol))

    # 缺口告警(不填充, 只报告)
    step = BAR_MS[C.INTERVAL]
    g = np.diff(df["open_time"].to_numpy())
    big = g[g > step * (C.MAX_GAP_BARS + 1)]
    if len(big):
        print("  [告警] 存在 %d 处 >%d 根的数据缺口, 最大缺失 %d 根; 未做插值填充。"
              % (len(big), C.MAX_GAP_BARS, int((big.max() // step) - 1)))
    return df


def main() -> None:
    if not C.RAW_PARQUET.exists():
        raise SystemExit("缺少原始数据, 请先运行 src/data_fetch.py")
    raw = pd.read_parquet(C.RAW_PARQUET)
    rep0 = integrity_check(raw)
    print("=== 原始数据体检 ===")
    for k, v in rep0.items():
        print("  %-18s %s" % (k, v))

    df = clean(raw)
    rep1 = integrity_check(df)
    print("\n=== 清洗后体检 ===")
    for k, v in rep1.items():
        print("  %-18s %s" % (k, v))
    df.to_parquet(C.CLEAN_PARQUET, index=False)
    print("\n落盘: %s" % C.CLEAN_PARQUET)


if __name__ == "__main__":
    main()
