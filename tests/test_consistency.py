# -*- coding: utf-8 -*-
"""设计与运行一致性单元测试(需求15)。

覆盖: 切分完整性、因子无未来函数、订单流因子自适应、时序折叠无泄漏、
执行层(次根开盘成交/止盈止损/保守假设/成本/权益核算)、冻结阈值不偷看、
因子方向筛选、模型确定性、成本与资金口径。

运行: pytest tests/test_consistency.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import config as C                       # noqa: E402
from src import factors as F_lib         # noqa: E402
from src import execution, factor_select  # noqa: E402
from src import models                   # noqa: E402
from src.cv import PurgedKFold, time_split  # noqa: E402


# ================================================================ 工具
def _synth(n=300, seed=7, with_order_flow=False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.004, n)
    close = 1000 * np.exp(np.cumsum(ret))
    open_ = close * (1 + rng.normal(0, 0.001, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.002, n)))
    vol = rng.integers(500, 5000, n).astype(float)
    dt = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    df = pd.DataFrame({"open_time": (dt.view("int64") // 10 ** 6), "datetime": dt,
                       "open": open_, "high": high, "low": low, "close": close,
                       "volume": vol, "quote_volume": vol * close})
    if with_order_flow:
        tb = vol * rng.uniform(0.3, 0.7, n)
        df["trades"] = rng.integers(50, 500, n)
        df["taker_buy_base"] = tb
        df["taker_buy_quote"] = tb * close
    df["log_ret"] = np.log(df["close"]).diff()
    return df


# ================================================================ 切分(需求2)
def test_time_split_70_15_15():
    for n in (1000, 13447):
        tr, oof, ooc = time_split(n)
        assert tr.stop == int(n * 0.70)
        assert oof.stop - oof.start == int(n * 0.15)
        assert ooc.stop - ooc.start == n - tr.stop - (oof.stop - oof.start)
        assert tr.start == 0 and ooc.stop == n
        assert tr.stop == oof.start and oof.stop == ooc.start     # 严格时序、无缝、不重叠


# ================================================================ 因子(需求5/14)
def test_factors_no_lookahead():
    df = _synth(300)
    df2 = df.copy()
    df2.loc[200:, ["open", "high", "low", "close"]] *= 1.5        # 篡改未来
    f1, names1 = F_lib.build_factors(df)
    f2, _ = F_lib.build_factors(df2)
    pd.testing.assert_frame_equal(f1.iloc[:200], f2.iloc[:200])   # t<200 不得受影响


def test_factors_skip_order_flow_when_missing():
    """欧易数据无 trades/taker_* -> 订单流因子必须被跳过, 且不报错。"""
    _, names_ok = F_lib.build_factors(_synth(120))
    _, names_of = F_lib.build_factors(_synth(120, with_order_flow=True))
    for c in ("taker_buy_ratio", "taker_buy_quote_ratio", "trades_z_20"):
        assert c not in names_ok
        assert c in names_of


def test_trend_factors_present():
    """需求14: 必须包含趋势类因子。"""
    _, names = F_lib.build_factors(_synth(120))
    for c in ("adx_14", "ma_ratio_50", "ma_slope_20", "macd", "macd_hist",
              "lr_slope_20", "pos_in_range_50", "trend_strength_50"):
        assert c in names, "缺少趋势因子 %s" % c


def test_labels_atr_normalized():
    df = F_lib.add_label(_synth(60), C.HORIZON)
    assert {"atr", "label", "label_raw", "label_tradable"}.issubset(df.columns)
    # 标签仅依赖未来 -> 最后一根必须为 NaN; ATR 标准化因子为正
    assert np.isnan(df["label"].iloc[-1])
    rel = df["label_raw"] / (df["atr"] / df["close"])
    assert np.allclose(df["label"].dropna(), rel.dropna(), atol=1e-12)


# ================================================================ 时序折叠(需求6)
def test_purged_kfold_no_leak():
    """Purged K-Fold: 训练/验证不重叠, 且验证窗两侧(标签跨度+embargo)已被挖空。"""
    n = 600
    h, emb = C.HORIZON, C.CV_EMBARGO_BARS
    folds = list(PurgedKFold().split(n))
    assert len(folds) == C.CV_N_SPLITS
    seen_val = []
    for tr, va in folds:
        assert len(tr) and len(va)
        assert len(set(tr) & set(va)) == 0, "训练/验证重叠"
        lo, hi = int(va.min()), int(va.max()) + 1
        # purge: 训练样本的标签窗 [t, t+h) 不得落在验证区; embargo: 验证区后侧挖空
        assert all(t + h <= lo or t >= hi for t in tr), "标签窗泄漏"
        assert all(t >= hi + emb or t + h <= lo for t in tr), "embargo 未生效"
        seen_val.append((lo, hi))
    assert seen_val == sorted(seen_val)                       # 验证折按时间推进
    assert seen_val[0][0] == 0 and seen_val[-1][1] == n       # 覆盖全序列、无缝


# ================================================================ 执行层(需求11/12)
def _bar(t, o, h, l, c):
    return {"datetime": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=4 * t),
            "open": o, "high": h, "low": l, "close": c}


def _ohlc(rows):
    d = pd.DataFrame(rows)
    return ({k: d[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close")},
            d["datetime"].to_numpy())


def test_execution_long_tp_next_bar_open():
    # 信号在 bar1 收盘产生 -> bar2 开盘成交; bar3 触及止盈
    ohlc, times = _ohlc([_bar(0, 100, 101, 99, 100),
                         _bar(1, 100, 101, 99, 100),        # 信号(bar1)
                         _bar(2, 101, 102, 100, 101),       # 次根开盘 101 成交
                         _bar(3, 100, 110, 101, 109)])      # high>=106 -> 止盈
    sig = np.array([False, True, False, False])
    atr = np.array([1.0, 1.0, 1.0, 1.0])
    trades = execution.simulate_signals(sig, ohlc["open"], ohlc["high"], ohlc["low"],
                                        ohlc["close"], atr, times, "long", 5.0, 3.0,
                                        max_hold=12, notional=100.0)
    assert len(trades) == 1
    t = trades[0]
    assert t.entry_idx == 2 and t.entry_price == 101.0      # 次根开盘成交
    assert t.exit_reason == "tp" and t.exit_price == 106.0   # 101 + 5*1
    assert abs(t.net_ret - ((106 - 101) / 101 - 2 * C.FEE_RATE)) < 1e-12
    assert abs(t.pnl_usdt - t.net_ret * 100) < 1e-12


def test_execution_both_hit_conservative_sl():
    ohlc, times = _ohlc([_bar(0, 100, 101, 99, 100),
                         _bar(1, 100, 101, 99, 100),        # 信号
                         _bar(2, 101, 102, 100, 101),       # 成交 101
                         _bar(3, 100, 110, 95, 105)])       # 同根既触 tp(106) 又触 sl(98)
    sig = np.array([False, True, False, False])
    atr = np.array([1.0, 1.0, 1.0, 1.0])
    t = execution.simulate_signals(sig, ohlc["open"], ohlc["high"], ohlc["low"],
                                   ohlc["close"], atr, times, "long", 5.0, 3.0, 12, 100.0)[0]
    assert t.exit_reason == "sl" and t.exit_price == 98.0


def test_execution_short_tp():
    ohlc, times = _ohlc([_bar(0, 100, 101, 99, 100),
                         _bar(1, 100, 101, 99, 100),        # 信号
                         _bar(2, 101, 103, 100, 102),       # 成交 101
                         _bar(3, 102, 103, 94, 95)])        # low<=96 -> 空头止盈
    sig = np.array([False, True, False, False])
    atr = np.array([1.0, 1.0, 1.0, 1.0])
    t = execution.simulate_signals(sig, ohlc["open"], ohlc["high"], ohlc["low"],
                                   ohlc["close"], atr, times, "short", 5.0, 3.0, 12, 100.0)[0]
    assert t.exit_reason == "tp" and t.entry_price == 101.0 and t.exit_price == 96.0


def test_equity_equals_capital_plus_pnl():
    ohlc, times = _ohlc([_bar(0, 100, 101, 99, 100),
                         _bar(1, 100, 101, 99, 100),        # 信号
                         _bar(2, 101, 102, 100, 101),       # 成交 101
                         _bar(3, 102, 110, 101, 108),       # 止盈 106
                         _bar(4, 108, 109, 107, 108)])
    sig = np.array([False, True, False, False, False])
    atr = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    trades = execution.simulate_signals(sig, ohlc["open"], ohlc["high"], ohlc["low"],
                                        ohlc["close"], atr, times, "long", 5.0, 3.0, 12, 100.0)
    eq = execution.equity_from_trades(trades, 5)
    assert abs(eq[-1] - (C.INIT_CAPITAL + sum(t.pnl_usdt for t in trades))) < 1e-9


def test_cost_is_two_sided_fee():
    """需求11: 单边 5bp -> 往返成本 10bp。"""
    assert C.FEE_RATE == 0.0005
    assert abs(2 * (C.FEE_RATE + C.SLIP_RATE) - 0.001) < 1e-12


# ================================================================ 冻结阈值(需求9)
def test_frozen_threshold_no_ooc_peek():
    pred = np.linspace(-1, 1, 100)
    thr = execution.threshold_from_quantile(pred, "long", 0.7, 0, 100)
    assert abs(thr - np.quantile(pred, 0.3)) < 1e-12
    sig = execution.signal_from_threshold(pred, "long", 0.5, 0, 100)
    assert sig.sum() == int((pred >= 0.5).sum())            # 用绝对阈值, 不由分布决定
    # 空头: 取下分位
    thr_s = execution.threshold_from_quantile(pred, "short", 0.2, 0, 100)
    assert abs(thr_s - np.quantile(pred, 0.2)) < 1e-12
    # 段内有效样本不足 -> NaN(不生成信号)
    assert np.isnan(execution.threshold_from_quantile(pred, "long", 0.7, 0, 10))
    assert execution.signal_from_threshold(pred, "long", np.nan, 0, 100).sum() == 0


# ================================================================ 因子筛选(需求5/13)
def test_select_factors_long_short_direction():
    rng = np.random.default_rng(0)
    n = 400
    y = pd.Series(rng.normal(size=n))
    F = pd.DataFrame({"pos": y + rng.normal(0, 0.1, n),      # 正相关
                      "neg": -y + rng.normal(0, 0.1, n)})    # 负相关
    sel = factor_select.select_factors(F, y)
    assert "pos" in sel["selected"]["long"] and "neg" not in sel["selected"]["long"]
    assert "neg" in sel["selected"]["short"] and "pos" not in sel["selected"]["short"]


# ================================================================ 模型(需求3)
def test_model_deterministic():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(400, 5)), columns=["f%d" % i for i in range(5)])
    y = pd.Series(X["f0"] + 0.5 * X["f1"] + rng.normal(0, 0.1, 400))
    p = dict(C.LGBM_PARAMS_COMMON)
    p["n_jobs"] = 1
    m1 = models.train_fold(X, y, list(X.columns), np.arange(300), np.arange(300, 400), params=p)
    m2 = models.train_fold(X, y, list(X.columns), np.arange(300), np.arange(300, 400), params=p)
    p1 = models.predict_booster(m1, X, list(X.columns))
    p2 = models.predict_booster(m2, X, list(X.columns))
    np.testing.assert_array_equal(p1, p2)


# ================================================================ 资金口径(需求11)
def test_capital_and_notional():
    assert C.INIT_CAPITAL == 1000.0
    assert C.TRADE_NOTIONAL == 100.0
    assert C.MAX_HOLD_BARS == 12
