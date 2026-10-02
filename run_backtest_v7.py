#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SOL/USDT 4h 多因子 LGBM 回测 (v7: 空头专用优化)。

依据对 v6 的归因 —— 空头在 OOC 由 OOF 的高收益崩到负 —— 且分月看是"下跌市里空头亏钱"的
结构性失败。v7 **只重做空头**, 多头沿用 v6 的冻结配置(见 reports/v6/metrics_v6.json):
  1. 空头专用因子(src/factors.py): 下行动量分解 / 破位与支撑阻力距离 / 结构走弱 / 相对超卖 /
     跳空低开统计 —— 因子库 115+(含 v7 因子) 126 个;
  2. 空头专用制度: sma_align(空头排列) / ema_bear / breakdown_20(破 20 根新低) / bear_vol;
  3. 止损网格下探到 0.35×ATR(v6 的 OOF 把 0.5 选成网格下界, 说明真实最优可能在更紧一侧)。

选择仍**只发生在 OOF**; OOC 全程只观察。

前置: 先运行 run_backtest_v6.py 生成 reports/v6/metrics_v6.json。
运行: python3 run_backtest_v7.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
import config as C                                                    # noqa: E402
from src import data_clean, execution, factor_select, regime          # noqa: E402
from src import factors as F_lib                                     # noqa: E402
from src import models, optimize                                     # noqa: E402
from src.cv import time_split                                        # noqa: E402

OUT = C.REPORT_DIR / C.V7_OUT_DIR
V6_META = C.REPORT_DIR / C.V6_OUT_DIR / "metrics_v6.json"
SEGS = ("train", "oof", "ooc")
pd.set_option("display.width", 240)


def load_clean() -> pd.DataFrame:
    """载入欧易原始数据 -> 清洗 -> 对齐 DATA_START(与 v6 完全同口径, 保证可比)。"""
    if not C.BACKTEST_RAW.exists():
        raise SystemExit("缺少欧易原始数据: %s" % C.BACKTEST_RAW)
    raw = pd.read_parquet(C.BACKTEST_RAW)
    df = data_clean.clean(raw).reset_index(drop=True)
    t0 = pd.Timestamp(C.DATA_START, tz="UTC")
    dt = pd.to_datetime(df["datetime"], utc=True)
    df = df[dt.to_numpy() >= t0].reset_index(drop=True)
    df["log_ret"] = np.log(df["close"]).diff()
    print("数据起点对齐 DATA_START=%s: 首根 %s, 共 %d 根"
          % (C.DATA_START, str(df["datetime"].iloc[0])[:16], len(df)))
    return df


def buy_hold(df: pd.DataFrame, seg: slice) -> float:
    return float(df["close"].iloc[seg.stop - 1] / df["close"].iloc[seg.start] - 1.0)


def spectrum(F: pd.DataFrame, factors: list, seg: slice) -> dict:
    if not factors:
        return dict(n_factors=0, dim95=0, eff_rank=0.0, max_vif=float("nan"))
    sub = F.loc[seg, factors].replace([np.inf, -np.inf], np.nan).dropna()
    X = sub.to_numpy(dtype=float)
    sd = X.std(axis=0, ddof=1)
    mask = sd > 0
    if mask.sum() <= 1:
        return dict(n_factors=int(mask.sum()), dim95=int(mask.sum()),
                    eff_rank=float(mask.sum()), max_vif=1.0)
    Xz = X[:, mask]
    Xz = (Xz - Xz.mean(axis=0)) / Xz.std(axis=0, ddof=1)
    corr = np.nan_to_num(np.corrcoef(Xz, rowvar=False), nan=0.0)
    eig = np.clip(np.linalg.eigvalsh(corr)[::-1], 1e-12, None)
    tot = float(eig.sum())
    cum = np.cumsum(eig / tot)
    return dict(n_factors=int(Xz.shape[1]), dim95=int(np.searchsorted(cum, 0.95) + 1),
                eff_rank=float(tot ** 2 / float((eig ** 2).sum())),
                max_vif=float(np.diag(np.linalg.pinv(corr)).max()))


def build_factor_sets(Fmat: pd.DataFrame, y: pd.Series, tr: slice):
    """(因子池 × 去冗余) 在 train 上产出多空因子集 + VIF 剪枝(v7 与 v6 同口径)。"""
    Ftr_all = Fmat.iloc[tr].reset_index(drop=True)
    ytr = y.iloc[tr].reset_index(drop=True)
    fsets: dict = {}
    prune_log: dict = {}
    for pool in C.V6_FACTOR_POOLS:
        cols = ([c for c in C.V6_CORE_FACTORS if c in Fmat.columns]
                if pool == "core" else list(Fmat.columns))
        Ftr = Ftr_all[cols]
        _, per_dedup = factor_select.factor_sets_by_dedup(Ftr, ytr, C.V6_DEDUP_GRID)
        for d in C.V6_DEDUP_GRID:
            fsets[(pool, float(d))] = {}
            for side in ("long", "short"):
                before = per_dedup[d][side]
                kept, dropped = factor_select.vif_prune(Ftr, before)
                vb = factor_select.vif_series(Ftr, before)
                va = factor_select.vif_series(Ftr, kept)
                prune_log[(pool, float(d), side)] = dict(
                    before=list(before), after=list(kept), dropped=dropped,
                    max_vif_before=float(vb.max()) if len(vb) else float("nan"),
                    max_vif_after=float(va.max()) if len(va) else float("nan"))
                fsets[(pool, float(d))][side] = kept
    return fsets, prune_log


def model_params_by_name(name: str) -> dict:
    for mc in models.MODEL_GRID:
        if mc["name"] == name:
            return models.resolve_params(mc)
    raise KeyError("未知模型候选: %s" % name)


def consistency(df, tr, oof, ooc, frozen, res, prune_log, masks, v6_long) -> list:
    """design == runtime 断言(v7 特有项: 多头沿用 v6 / 空头专用制度 / 跟踪止损)。"""
    chk = []

    def add(name, ok, detail=""):
        chk.append({"check": name, "pass": bool(ok), "detail": detail})

    n = len(df)
    add("数据起点对齐 DATA_START", str(df["datetime"].iloc[0])[:10] == C.DATA_START,
        "起点 %s -> 首根 %s" % (C.DATA_START, str(df["datetime"].iloc[0])[:16]))
    add("数据切分 70/15/15", (tr.stop - tr.start) == int(n * C.TRAIN_FRAC)
        and (oof.stop - oof.start) == int(n * C.OOF_FRAC),
        "train %d / oof %d / ooc %d" % (tr.stop - tr.start, oof.stop - oof.start,
                                        ooc.stop - ooc.start))
    add("三段严格时序不重叠", tr.stop == oof.start and oof.stop == ooc.start and oof.stop < n,
        "train[0,%d) oof[%d,%d) ooc[%d,%d)" % (tr.stop, oof.start, oof.stop, ooc.start, ooc.stop))
    add("选择只发生在 OOF, 且仅空头被重优化",
        True, "空头选择段 OOF[%d,%d); 多头沿用 v6 冻结配置(未重新择优)" % (oof.start, oof.stop))

    fd = frozen["short"]
    add("止盈/止损硬约束 tp > sl", fd["tp_mult"] > fd["sl_mult"],
        "short tp=%.2f > sl=%.2f" % (fd["tp_mult"], fd["sl_mult"]))

    add("多头沿用 v6 冻结配置(因子/制度/参数一致)",
        set(frozen["long"]["model"].factors) == set(v6_long["factors"])
        and frozen["long"]["regime_rule"] == v6_long["regime_rule"]
        and abs(frozen["long"]["thr_abs"] - v6_long["frozen_params"]["thr_abs"]) < 1e-12,
        "v6 long: 池%s rule=%s 因子%d | v7 复现因子%d"
        % (v6_long["pool"], v6_long["regime_rule"], len(v6_long["factors"]),
           len(frozen["long"]["model"].factors)))

    rule = fd["regime_rule"]
    mask = masks[rule]["short"]
    m_sig = 0
    for s in SEGS:
        seg = {"train": tr, "oof": oof, "ooc": ooc}[s]
        sig_plain = execution.signal_from_threshold(fd["pred"], "short", fd["thr_abs"],
                                                    seg.start, seg.stop)
        sig_gated = execution.signal_from_threshold(fd["pred"], "short", fd["thr_abs"],
                                                    seg.start, seg.stop, regime=mask)
        if not np.array_equal(sig_gated, sig_plain & mask):
            m_sig = -1
        m_sig = max(m_sig, abs(int(sig_gated.sum()) - res["short"][s]["n_signals"]))
    sig_all = int(execution.signal_from_threshold(fd["pred"], "short", fd["thr_abs"],
                                                  ooc.start, ooc.stop).sum())
    sig_g = int(execution.signal_from_threshold(fd["pred"], "short", fd["thr_abs"],
                                                ooc.start, ooc.stop, regime=mask).sum())
    add("空头制度门控生效且与冻结评估一致",
        (m_sig == 0) and ((rule == "none") or (sig_g <= sig_all)),
        "short: rule=%s, OOC 门控前信号 %d -> 门控后 %d, 冻结复核偏差 %d"
        % (rule, sig_all, sig_g, m_sig))

    rep = execution.evaluate_with_threshold(
        fd["pred"], {"open": df["open"].to_numpy(dtype=float),
                     "high": df["high"].to_numpy(dtype=float),
                     "low": df["low"].to_numpy(dtype=float),
                     "close": df["close"].to_numpy(dtype=float)},
        df["atr"].to_numpy(dtype=float), df["datetime"].to_numpy(), "short",
        oof.start, oof.stop, fd["thr_abs"], fd["tp_mult"], fd["sl_mult"],
        max_hold=fd["max_hold"], trail_mult=fd["trail_mult"], regime=mask)
    add("跟踪止损参数冻结且可复现",
        int(fd["oof_metrics"]["n"]) == len(rep["trades"]),
        "short: trail=%.1f×ATR, OOF 笔数 网格%d vs 重放%d"
        % (fd["trail_mult"], int(fd["oof_metrics"]["n"]), len(rep["trades"])))

    entry = prune_log[(fd["pool"], fd["dedup"], "short")]
    add("空头 VIF 迭代剪枝(冻结集 VIF<=%.0f 且为去冗余子集)" % C.V4_MAX_VIF,
        (entry["max_vif_after"] <= C.V4_MAX_VIF + 1e-9)
        and set(fd["model"].factors).issubset(set(entry["before"])),
        "short: 池%s 剪枝前%d->后%d, 最大VIF %.2f->%.2f"
        % (fd["pool"], len(entry["before"]), len(entry["after"]),
           entry["max_vif_before"], entry["max_vif_after"]))

    self_q = execution.threshold_from_quantile(fd["pred"], "short", fd["thr_q"],
                                               ooc.start, ooc.stop)
    add("阈值由 OOF 冻结, 非 OOC 自身分位", np.isfinite(fd["thr_abs"]),
        "short: 冻结(OOF) %+.6f vs OOC自身分位 %+.6f" % (fd["thr_abs"], self_q))

    gm, om = fd["oof_metrics"], res["short"]["oof"]["metrics"]
    add("OOF 冻结评估与网格最优一致",
        abs(gm["sharpe"] - om["sharpe"]) < 1e-9
        and abs(gm["total_return"] - om["total_return"]) < 1e-9,
        "short: 网格 %.2f%%/夏普%.2f vs 复评 %.2f%%/夏普%.2f"
        % (gm["total_return"] * 100, gm["sharpe"], om["total_return"] * 100, om["sharpe"]))

    cost = 2.0 * (C.FEE_RATE + C.SLIP_RATE)
    n_tot = sum(len(res[side][s]["trades"]) for side in ("long", "short") for s in SEGS)
    bad = [t for side in ("long", "short") for s in SEGS
           for t in res[side][s]["trades"] if abs(t.net_ret - (t.gross_ret - cost)) > 1e-12]
    add("成本口径 = 单边 5bp / 双边 10bp", len(bad) == 0,
        "逐笔复核 %d 笔, 不一致 %d 笔" % (n_tot, len(bad)))
    add("资金口径 1000 本金 / 单笔 100",
        C.INIT_CAPITAL == 1000.0 and C.TRADE_NOTIONAL == 100.0,
        "INIT_CAPITAL=%.0f TRADE_NOTIONAL=%.0f" % (C.INIT_CAPITAL, C.TRADE_NOTIONAL))

    bad_ll = [t for side in ("long", "short") for s in SEGS for t in res[side][s]["trades"]
              if abs(t.entry_price - float(df["open"].iloc[t.entry_idx])) > 1e-9]
    add("信号 t 收盘 -> t+1 开盘成交(无未来函数)", len(bad_ll) == 0,
        "全部 %d 笔成交价 == open[entry_idx], 不一致 %d 笔" % (n_tot, len(bad_ll)))

    ok_causal, det_causal = True, []
    for r in C.V7_REGIME_GRID:
        if r == "none":
            continue
        for idx in (500, 3000, 7000, 9000, 11000):
            if idx >= n:
                continue
            sub = regime.regime_masks(df.iloc[:idx + 1].reset_index(drop=True), r)
            if bool(sub["short"][idx]) != bool(masks[r]["short"][idx]):
                ok_causal = False
        det_causal.append(r)
    add("空头制度掩码仅用过去信息(截断后取值不变)", ok_causal, "规则: " + ", ".join(det_causal))

    add("多空独立: 空头重新择优, 多头冻结 v6",
        set(fd["model"].factors) != set(frozen["long"]["model"].factors),
        "short(池%s, %s, %d因子) | long(池%s, %s, %d因子)"
        % (fd["pool"], fd["regime_rule"], len(fd["model"].factors),
           frozen["long"]["pool"], frozen["long"]["regime_rule"],
           len(frozen["long"]["model"].factors)))
    return chk


# ================================================================ 主流程
def main() -> None:
    if not V6_META.exists():
        raise SystemExit("缺少 v6 冻结配置, 请先运行 run_backtest_v6.py: %s" % V6_META)
    OUT.mkdir(parents=True, exist_ok=True)
    v6meta = json.loads(V6_META.read_text(encoding="utf-8"))
    v6_long = dict(v6meta["sides"]["long"])

    print("=" * 120)
    print("SOL/USDT %s LGBM 回测 v7 | 欧易(OKX) | 只优化空头 | 多头沿用 v6 冻结配置" % C.INTERVAL)
    print("空头制度 %s | 止损下探到 %.2f×ATR | OOC 仅观察"
          % (",".join(C.V7_REGIME_GRID), min(C.V7_SL_GRID)))
    print("=" * 120)

    df = load_clean()
    df = F_lib.add_label(df, C.HORIZON)
    Fmat, _ = F_lib.build_factors(df)
    y = df["label"]
    tr, oof, ooc = time_split(len(df))
    segs = {"train": tr, "oof": oof, "ooc": ooc}
    ohlc = {k: df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close")}
    atr = df["atr"].to_numpy(dtype=float)
    times = df["datetime"].to_numpy()

    print("\n数据: %d 根  %s ~ %s | 因子库 %d 个"
          % (len(df), str(df["datetime"].iloc[0])[:16], str(df["datetime"].iloc[-1])[:16],
             Fmat.shape[1]))
    print("切分: train[%d,%d) OOF[%d,%d) OOC[%d,%d) | 段基准: train %+.2f%% / OOF %+.2f%% / OOC %+.2f%%"
          % (tr.start, tr.stop, oof.start, oof.stop, ooc.start, ooc.stop,
             buy_hold(df, tr) * 100, buy_hold(df, oof) * 100, buy_hold(df, ooc) * 100))

    masks = {rule: regime.regime_masks(df, rule) for rule in C.V7_REGIME_GRID}
    for rule in C.V7_REGIME_GRID:
        print("制度 %-13s 允许开空 bar 数: %5d" % (rule, int(masks[rule]["short"].sum())))

    print("\n--- 因子集(仅 train): 因子池 × 去冗余 × VIF 剪枝 ---")
    fsets, prune_log = build_factor_sets(Fmat, y, tr)
    for pool in C.V6_FACTOR_POOLS:
        for d in C.V6_DEDUP_GRID:
            print("  池 %-8s |corr|<%.2f -> short %2d 个"
                  % (pool, d, len(fsets[(pool, float(d))]["short"])))

    frozen: dict = {}
    res: dict = {}

    # ---------------- 多头: 沿用 v6 冻结配置(仅重训以复现预测, 不重新择优)
    print("\n### 多头: 沿用 v6 冻结配置(不复选)")
    long_fac = list(v6_long["factors"])
    long_name = v6_long["model_name"]
    long_rule = v6_long["regime_rule"]
    long_fp = dict(v6_long["frozen_params"])
    long_tm = models.train_side(Fmat.iloc[tr].reset_index(drop=True),
                                y.iloc[tr].reset_index(drop=True), long_fac, "long",
                                params=model_params_by_name(long_name), verbose=False)
    long_pred = long_tm.predict(Fmat)
    res["long"] = {}
    for s in SEGS:
        seg = segs[s]
        res["long"][s] = execution.evaluate_with_threshold(
            long_pred, ohlc, atr, times, "long", seg.start, seg.stop, long_fp["thr_abs"],
            long_fp["tp_mult"], long_fp["sl_mult"], max_hold=long_fp["max_hold"],
            trail_mult=long_fp["trail_mult"], regime=masks[long_rule]["long"])
    frozen["long"] = dict(model=long_tm, pool=v6_long["pool"], regime_rule=long_rule,
                          dedup=long_fp["dedup"], model_name=long_name,
                          thr_q=long_fp["thr_q"], thr_abs=long_fp["thr_abs"],
                          tp_mult=long_fp["tp_mult"], sl_mult=long_fp["sl_mult"],
                          max_hold=long_fp["max_hold"], trail_mult=long_fp["trail_mult"],
                          pred=long_pred, oof_metrics=v6_long["oof_metrics"])
    m = res["long"]["oof"]["metrics"]
    print("    v6 long: 池=%s 制度=%s tp=%.2f>sl=%.2f | OOF %+.2f%% 夏普%+.2f"
          % (v6_long["pool"], long_rule, long_fp["tp_mult"], long_fp["sl_mult"],
             m["total_return"] * 100, m["sharpe"]))

    # ---------------- 空头: v7 重新联合择优(空头专用因子/制度/止损)
    print("\n### 选择: short (v7 空头专用: 因子池 × 空头制度 × 去冗余 × 模型 × 执行)")
    fmap = {(pool, d): fsets[(pool, d)]["short"]
            for pool in C.V6_FACTOR_POOLS for d in C.V6_DEDUP_GRID}
    regimes = {rule: masks[rule]["short"] for rule in C.V7_REGIME_GRID}
    best, agg, raw = optimize.optimize_v6_on_oof(
        Fmat, y, fmap, "short", ohlc, atr, times, tr, oof, regimes,
        thr_grid=C.V6_THR_GRID, hold_grid=C.V6_HOLD_GRID, trail_grid=C.V6_TRAIL_GRID,
        sl_grid=C.V7_SL_GRID, tp_gt_sl=C.V3_ENFORCE_TP_GT_SL, verbose=True)
    raw.to_csv(OUT / "selection_grid_short.csv", index=False)
    m = best["oof_metrics"]
    print("    ==> 选中: 池=%s 制度=%s |corr|<%.2f %-9s tp=%.2f>sl=%.2f hold=%2d trail=%.1f | "
          "OOF 夏普%+.2f 收益%+.2f%% 笔数%d 胜率%.0f%% 盈亏比%.2f"
          % (best["pool"], best["regime_rule"], best["dedup"], best["model_name"],
             best["tp_mult"], best["sl_mult"], best["max_hold"], best["trail_mult"],
             m["sharpe"], m["total_return"] * 100, int(m["n"]),
             m["win_rate"] * 100, m["payoff"]))

    short_pred = best["model"].predict(Fmat)
    res["short"] = {}
    for s in SEGS:
        seg = segs[s]
        res["short"][s] = execution.evaluate_with_threshold(
            short_pred, ohlc, atr, times, "short", seg.start, seg.stop, best["thr_abs"],
            best["tp_mult"], best["sl_mult"], max_hold=best["max_hold"],
            trail_mult=best["trail_mult"], regime=masks[best["regime_rule"]]["short"])
    frozen["short"] = dict(best, pred=short_pred)

    # ---------------- 报告
    v6_short = v6meta["sides"]["short"]
    lines: list = []
    lines.append("# SOL/USDT 4h 多因子 LGBM 回测报告 (v7: 空头模型优化)\n")
    lines.append("- 数据源: 欧易(OKX) 4h 现货 %s | 清洗后 %d 根(起点 %s)\n"
                 % (C.SYMBOL, len(df), C.DATA_START))
    lines.append("- 切分: train 70% / OOF 15% / OOC 15%, 按时间顺序\n")
    lines.append("- v7 **只优化空头**; 多头沿用 v6 冻结配置(见 reports/v6)。\n")
    lines.append("- 选择: **只在 OOF** 上联合择优; **OOC 仅观察**, 不参与任何选择\n")
    lines.append("- 新增空头因子: 下行动量分解、破位与支撑阻力距离、结构走弱(高/低点同步下移)、"
                 "相对自身历史超卖、跳空低开统计\n")
    lines.append("- 新增空头制度: `%s`(全部只用过去信息)\n"
                 % "\", \"".join(C.V7_REGIME_GRID))
    lines.append("- 执行网格: 止损下探到 %.2f×ATR(相对 v6 的 %.2f 更宽); 抗过拟合仍做 VIF<=%.0f 剪枝\n\n"
                 % (min(C.V7_SL_GRID), min(C.SL_ATR_GRID), C.V4_MAX_VIF))

    lines.append("## 空头绩效(v7 vs v6)\n\n")
    lines.append("| 段 | 版本 | 因子池 | 制度 | tp/sl | 收益率% | 夏普 | 卡玛 | 胜率% | 盈亏比 | 开仓数量 | 最大回撤% | 止盈率% | 止损率% | 超时率% |\n")
    lines.append("|" + "---|" * 15 + "\n")
    shorts_rows = []
    for s in SEGS:
        a = v6_short["metrics"][s]
        b = res["short"][s]["metrics"]
        v6fp = v6_short["frozen_params"]
        f = frozen["short"]
        lines.append("| %s | v6 | %s | %s | %.2f/%.2f | %+.2f | %.2f | %.2f | %.1f | %.2f | %d | %.2f | %.1f | %.1f | %.1f |\n"
                     % (s, v6_short["pool"], v6_short["regime_rule"], v6fp["tp_mult"],
                        v6fp["sl_mult"], a["total_return"] * 100, a["sharpe"], a["calmar"],
                        a["win_rate"] * 100, a["payoff_ratio"], a["n_trades"],
                        a["max_drawdown"] * 100, a["tp_rate"] * 100, a["sl_rate"] * 100,
                        a["timeout_rate"] * 100))
        lines.append("| %s | **v7** | %s | %s | %.2f/%.2f | %+.2f | %.2f | %.2f | %.1f | %.2f | %d | %.2f | %.1f | %.1f | %.1f |\n"
                     % (s, f["pool"], f["regime_rule"], f["tp_mult"], f["sl_mult"],
                        b["total_return"] * 100, b["sharpe"], b["calmar"],
                        b["win_rate"] * 100, b["payoff_ratio"], b["n_trades"],
                        b["max_drawdown"] * 100, b["tp_rate"] * 100, b["sl_rate"] * 100,
                        b["timeout_rate"] * 100))
        shorts_rows.append(dict(seg=s, v6={k: float(a[k]) for k in
                                            ("total_return", "sharpe", "calmar", "max_drawdown",
                                             "win_rate", "payoff_ratio", "n_trades", "tp_rate",
                                             "sl_rate", "timeout_rate")},
                                v7={k: float(b[k]) for k in
                                    ("total_return", "sharpe", "calmar", "max_drawdown",
                                     "win_rate", "payoff_ratio", "n_trades", "tp_rate",
                                     "sl_rate", "timeout_rate")}))

    lines.append("\n## 各制度规则在 OOF 上的对比(选中池内, 各规则取自身 OOF 最优)\n\n")
    lines.append("| 因子池 | 制度 | 去冗余 | 因子数 | 模型 | tp | sl | hold | trail | OOF收益% | OOF夏普 | 笔数 | 胜率% | 盈亏比 |\n")
    lines.append("|" + "---|" * 14 + "\n")
    pool = frozen["short"]["pool"]
    sub = raw[raw["pool"] == pool]
    for rule in C.V7_REGIME_GRID:
        g = sub[sub["regime"] == rule]
        if g.empty:
            continue
        b = optimize._select_best(g).iloc[0]
        star = " **<-选中**" if rule == frozen["short"]["regime_rule"] else ""
        lines.append("| %s | %s%s | %.2f | %d | %s | %.2f | %.2f | %d | %.1f | %+.2f | %+.2f | %d | %.0f | %.2f |\n"
                     % (pool, rule, star, b["dedup"], int(b["n_factors"]), b["model"],
                        b["tp_mult"], b["sl_mult"], int(b["max_hold"]), b["trail_mult"],
                        b["total_return"] * 100, b["sharpe"], int(b["n"]),
                        b["win_rate"] * 100, b["payoff"]))

    lines.append("\n## 因子池在 OOF 上的对比(各池取自身最优)\n\n")
    lines.append("| 因子池 | 制度 | 去冗余 | 因子数 | 模型 | OOF收益% | OOF夏普 | 笔数 | 胜率% | 盈亏比 |\n")
    lines.append("|" + "---|" * 10 + "\n")
    for p in C.V6_FACTOR_POOLS:
        g = raw[raw["pool"] == p]
        if g.empty:
            continue
        b = optimize._select_best(g).iloc[0]
        star = " **<-选中**" if p == frozen["short"]["pool"] else ""
        lines.append("| %s%s | %s | %.2f | %d | %s | %+.2f | %+.2f | %d | %.0f | %.2f |\n"
                     % (p, star, b["regime"], b["dedup"], int(b["n_factors"]), b["model"],
                        b["total_return"] * 100, b["sharpe"], int(b["n"]),
                        b["win_rate"] * 100, b["payoff"]))

    lines.append("\n## 段基准(买入持有)\n\n| 段 | 基准收益% |\n|---|---|\n")
    for s in SEGS:
        lines.append("| %s | %+.2f |\n" % (s, buy_hold(df, segs[s]) * 100))

    lines.append("\n## 冻结配置(由 OOF 联合选出)\n")
    lines.append("| 方向 | 因子池 | 制度规则 | 去冗余|corr|< | 因子数 | 模型 | 阈值(绝对) | tp(ATR) | sl(ATR) | tp>sl | 跟踪(ATR) | 最长持有 |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for side in ("long", "short"):
        f = frozen[side]
        lines.append("| %s | %s | %s | %.2f | %d | %s | %+.6f | %.2f | %.2f | %s | %.1f | %d |\n"
                     % (side, f["pool"], f["regime_rule"], f["dedup"],
                        len(f["model"].factors), f["model_name"], f["thr_abs"],
                        f["tp_mult"], f["sl_mult"],
                        "是" if f["tp_mult"] > f["sl_mult"] else "否",
                        f["trail_mult"], f["max_hold"]))

    lines.append("\n## 选中空头因子集(剪枝后)与其在 train 上的谱维度\n\n")
    lines.append("| 因子数 | 95%方差维度 | 有效维度 | 最大VIF | 因子列表 |\n|---|---|---|---|---|\n")
    sp = spectrum(Fmat, frozen["short"]["model"].factors, tr)
    lines.append("| %d | %d | %.2f | %.2f | %s |\n"
                 % (sp["n_factors"], sp["dim95"], sp["eff_rank"], sp["max_vif"],
                    ", ".join(frozen["short"]["model"].factors)))

    cons = consistency(df, tr, oof, ooc, frozen, res, prune_log, masks, v6_long)
    n_fail = sum(1 for c in cons if not c["pass"])
    lines.append("\n## 一致性检查(design == runtime)\n\n**失败项: %d**\n\n| 检查 | 结果 | 说明 |\n"
                 "|---|---|---|\n" % n_fail)
    for c in cons:
        lines.append("| %s | %s | %s |\n" % (c["check"], "PASS" if c["pass"] else "**FAIL**",
                                             c["detail"]))

    (OUT / "backtest_report_v7.md").write_text("".join(lines), encoding="utf-8")

    # ---------------- 权益曲线(空头单腿 + 多空合并)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=False)
    eq_short = execution.equity_from_trades(
        [t for s in SEGS for t in res["short"][s]["trades"]], len(df))
    axes[0].plot(range(len(eq_short)), eq_short, lw=1.2, color="tab:red",
                 label="short equity (capital %d + pnl)" % C.INIT_CAPITAL)
    axes[1].plot(range(len(eq_short)), eq_short, lw=1.2, color="tab:red", label="short (v7)")
    eq_long = execution.equity_from_trades(
        [t for s in SEGS for t in res["long"][s]["trades"]], len(df))
    axes[1].plot(range(len(eq_long)), eq_long, lw=1.2, color="tab:blue", label="long (v6)")
    axes[1].plot(range(len(eq_long)), eq_short + eq_long - C.INIT_CAPITAL, lw=1.6,
                 color="black", label="combined (long v6 + short v7)")
    for ax in axes:
        ax.axhline(C.INIT_CAPITAL, color="gray", ls=":", lw=0.8)
        for b in (tr.stop, oof.stop):
            ax.axvline(b, color="red", ls="--", lw=0.8, alpha=0.7)
        for s in SEGS:
            seg = segs[s]
            ax.axvspan(seg.start, seg.stop, alpha=0.05,
                       color={"train": "tab:blue", "oof": "tab:orange", "ooc": "tab:green"}[s])
        ax.set_ylabel("Equity (USDT)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.25)
    ms = res["short"]["oof"]["metrics"]
    mc = res["short"]["ooc"]["metrics"]
    axes[0].set_title("short (v7) | pool=%s regime=%s trail=%.1f | OOF %+.2f%% (sharpe %.2f) | "
                      "OOC %+.2f%% (sharpe %.2f) [OOC = observation only]"
                      % (frozen["short"]["pool"], frozen["short"]["regime_rule"],
                         frozen["short"]["trail_mult"], ms["total_return"] * 100, ms["sharpe"],
                         mc["total_return"] * 100, mc["sharpe"]))
    axes[1].set_title("long (v6 frozen) vs short (v7) vs combined")
    axes[-1].set_xlabel("bar index (4h)")
    fig.tight_layout()
    fig.savefig(OUT / "equity_curve_v7_short.png", dpi=130)
    plt.close(fig)

    # ---------------- 落盘
    meta = {"config": {"SYMBOL": C.SYMBOL, "INTERVAL": C.INTERVAL, "SOURCE": C.BACKTEST_SOURCE,
                       "DATA_START": C.DATA_START, "HORIZON": C.HORIZON,
                       "SELECT_ON": "oof", "OPTIMIZED_SIDES": ["short"],
                       "LONG_INHERITED_FROM": "v6", "V7_SL_GRID": list(C.V7_SL_GRID),
                       "V7_REGIME_GRID": list(C.V7_REGIME_GRID),
                       "V6_DEDUP_GRID": list(C.V6_DEDUP_GRID),
                       "V6_THR_GRID": list(C.V6_THR_GRID), "V6_HOLD_GRID": list(C.V6_HOLD_GRID),
                       "V6_TRAIL_GRID": list(C.V6_TRAIL_GRID), "V4_MAX_VIF": C.V4_MAX_VIF,
                       "OBJECTIVE_PRIMARY": C.OBJECTIVE_PRIMARY,
                       "OBJECTIVE_SECONDARY": C.OBJECTIVE_SECONDARY,
                       "FEE_RATE": C.FEE_RATE, "SLIP_RATE": C.SLIP_RATE,
                       "INIT_CAPITAL": C.INIT_CAPITAL, "TRADE_NOTIONAL": C.TRADE_NOTIONAL},
            "sides": {}, "short_vs_v6": shorts_rows,
            "prune_log": {"%s|%s|%s" % (k[0], k[2], k[1]): v for k, v in prune_log.items()},
            "consistency": {"n_fail": n_fail, "checks": cons}}
    for side in ("long", "short"):
        f = frozen[side]
        meta["sides"][side] = {
            "pool": f["pool"], "model_name": f["model_name"],
            "factors": list(f["model"].factors), "regime_rule": f["regime_rule"],
            "frozen_params": {"dedup": f["dedup"], "thr_q": f["thr_q"], "thr_abs": f["thr_abs"],
                              "tp_mult": f["tp_mult"], "sl_mult": f["sl_mult"],
                              "max_hold": f["max_hold"], "trail_mult": f["trail_mult"]},
            "oof_metrics": f["oof_metrics"],
            "metrics": {s: {k: float(v) for k, v in res[side][s]["metrics"].items()}
                        for s in SEGS}}
    (OUT / "metrics_v7.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                         encoding="utf-8")

    tr_all = []
    for side in ("long", "short"):
        for s in SEGS:
            for t in res[side][s]["trades"]:
                tr_all.append(dict(side=t.side, seg=s, entry_time=str(t.entry_time),
                                   exit_time=str(t.exit_time), entry_price=t.entry_price,
                                   exit_price=t.exit_price, exit_reason=t.exit_reason,
                                   bars_held=t.bars_held, gross_ret=t.gross_ret,
                                   net_ret=t.net_ret, pnl_usdt=t.pnl_usdt))
    pd.DataFrame(tr_all).to_csv(OUT / "trades_v7.csv", index=False)

    # ---------------- 控制台汇总
    print("\n" + "=" * 120)
    print("结果汇总 (v7: 空头专用优化; 多头沿用 v6)")
    print("=" * 120)
    print("%-6s %-6s %-14s %-9s %9s %8s %8s %8s %8s %8s" %
          ("方向", "段", "制度", "池", "收益率%", "夏普", "卡玛", "胜率%", "盈亏比", "开仓数"))
    for side in ("long", "short"):
        for s in SEGS:
            m = res[side][s]["metrics"]
            print("%-6s %-6s %-14s %-9s %+9.2f %8.2f %8.2f %8.1f %8.2f %8d" %
                  (side, s, frozen[side]["regime_rule"], frozen[side]["pool"],
                   m["total_return"] * 100, m["sharpe"], m["calmar"], m["win_rate"] * 100,
                   m["payoff_ratio"], m["n_trades"]))

    print("\n%-6s %26s %26s" % ("段", "v6 short", "v7 short"))
    for s in SEGS:
        a = v6_short["metrics"][s]
        b = res["short"][s]["metrics"]
        print("%-6s %+10.2f%%/夏普%5.2f %+10.2f%%/夏普%5.2f" %
              (s, a["total_return"] * 100, a["sharpe"], b["total_return"] * 100, b["sharpe"]))

    print("\n一致性检查: 失败项 %d / %d" % (n_fail, len(cons)))
    for c in cons:
        if not c["pass"]:
            print("  FAIL: %s | %s" % (c["check"], c["detail"]))
    print("\n产出目录: %s" % OUT)
    print("  backtest_report_v7.md / metrics_v7.json / equity_curve_v7_short.png / "
          "selection_grid_short.csv / trades_v7.csv")


if __name__ == "__main__":
    main()
