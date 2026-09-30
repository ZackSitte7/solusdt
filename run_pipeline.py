#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SOL/USDT 4h 多因子收益预测与回测 —— 主流程。

流程(严格按使用者规格):
  1. 载入清洗后的 4h 数据, 构建因子与标签
  2. 按时间切分: train 70% / OOF 15% / OOC 15%
  3. **仅在训练段**做因子 IC 筛选 + 相关性聚类去冗余; 多空分别筛
  4. **仅在训练段**用 Purged K-Fold + Embargo 训练 LightGBM; 多空分别训练
  5. **仅在 OOF 段**优化模型层(候选配置)与执行层(阈值/止盈/止损)
  6. 冻结配置, 在 OOC 段**只跑一次、只观察, 不参与任何选择**
  7. 输出: 收益曲线、胜率、盈亏比、夏普、卡玛、最大回撤
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C                       # noqa: E402
from src import data_clean, factors as F_lib, factor_select, models, optimize  # noqa: E402
from src.cv import describe_split, time_split                                   # noqa: E402
from src.execution import simulate_signals, equity_from_trades                  # noqa: E402
from src.metrics import summarize                                               # noqa: E402

pd.set_option("display.width", 200)


def load_data() -> pd.DataFrame:
    if not C.CLEAN_PARQUET.exists():
        print("清洗数据不存在, 先跑清洗...")
        raw = pd.read_parquet(C.RAW_PARQUET)
        df = data_clean.clean(raw).reset_index(drop=True)
    else:
        df = pd.read_parquet(C.CLEAN_PARQUET).reset_index(drop=True)
    return df


def evaluate_side(seg_pred: np.ndarray, ohlc: dict, atr: np.ndarray, times: np.ndarray,
                  side: str, start: int, end: int, params: dict) -> dict:
    """按给定执行参数, 在 [start,end) 段上评估一侧。返回指标与交易明细。"""
    q = params["thr_q"]
    finite = np.isfinite(seg_pred)
    thr = float(np.nanquantile(seg_pred[finite], 1.0 - q if side == "long" else q))
    sig = np.zeros(len(seg_pred), dtype=bool)
    if side == "long":
        sig[start:end] = finite[start:end] & (seg_pred[start:end] >= thr)
    else:
        sig[start:end] = finite[start:end] & (seg_pred[start:end] <= thr)
    trades = simulate_signals(sig, ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"],
                              atr, times, side, params["tp_mult"], params["sl_mult"],
                              C.MAX_HOLD_BARS, C.TRADE_NOTIONAL)
    eq = equity_from_trades(trades, len(seg_pred))
    m = summarize(trades, eq[start:end])
    m["thr"] = thr
    return {"metrics": m, "trades": trades, "equity": eq[start:end]}


def main() -> None:
    print("=" * 100)
    print("SOL/USDT 4h 多因子模型  |  目标: 未来 %d 根(=%d 小时)对数收益(回归)"
          % (C.HORIZON, C.HORIZON * 4))
    print("=" * 100)

    # ---------------- 1. 数据 + 因子
    df = load_data()
    df = F_lib.add_label(df, C.HORIZON)
    Fmat, fnames = F_lib.build_factors(df)
    print("数据: %d 根 4h K线  %s ~ %s" %
          (len(df), df["datetime"].iloc[0], df["datetime"].iloc[-1]))
    print("原始因子: %d 个" % len(fnames))

    ohlc = {k: df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close")}
    atr = F_lib._atr(df["high"], df["low"], df["close"], C.ATR_WINDOW).to_numpy(dtype=float)
    times = df["datetime"].to_numpy()
    y = df["label"]

    # ---------------- 2. 切分
    sl = describe_split(len(df))
    print("\n=== 数据切分(按时间顺序) ===")
    for s in sl:
        print("  %-6s 行 %5d~%5d  n=%5d  %s ~ %s" %
              (s["name"], s["start"], s["end"] - 1, s["n"],
               df["datetime"].iloc[s["start"]], df["datetime"].iloc[s["end"] - 1]))
    (tr, oof, ooc) = time_split(len(df))

    # ---------------- 3. 因子筛选(仅训练段)
    print("\n=== 因子筛选(仅用训练段, IC=Spearman) ===")
    Ftr = Fmat.iloc[tr]
    ytr = y.iloc[tr]
    sel = factor_select.select_factors(Ftr.reset_index(drop=True), ytr.reset_index(drop=True))
    ic: pd.Series = sel["ic"]
    print("  全因子 |IC| 分位: P50=%.4f P75=%.4f P90=%.4f 最大=%.4f" %
          (ic.abs().median(), ic.abs().quantile(.75), ic.abs().quantile(.90), ic.abs().max()))
    print("  |IC| >= %.3f 的因子: %d / %d" % (C.IC_MIN_ABS, int((ic.abs() >= C.IC_MIN_ABS).sum()), len(ic)))
    for side in ("long", "short"):
        print("  [%s] 聚类去冗余后保留 %d 个: %s" %
              (side, len(sel["selected"][side]), ", ".join(sel["selected"][side][:12])
               + (" ..." if len(sel["selected"][side]) > 12 else "")))

    # ---------------- 4. 模型层优化(仅在 OOF 上比较, 训练只用训练段)
    print("\n=== 4. 模型层: 训练段训练 -> OOF 段比较候选配置 ===")
    eval_params = {"thr_q": 0.70, "tp_mult": 2.5, "sl_mult": 2.0}   # 仅用于模型候选比较
    cand_best = {}
    for side in ("long", "short"):
        facs = sel["selected"][side]
        if not facs:
            print("  [%s] 无可用因子, 跳过" % side)
            continue
        print("  [%s] 候选配置 %d 个" % (side, len(models.MODEL_GRID)))
        best_key, best_res, best_tm = None, None, None
        for cand in models.MODEL_GRID:
            tm = models.train_side(Ftr, ytr, facs, side,
                                   params=models.resolve_params(cand), verbose=False)
            pred = tm.predict(Fmat)
            r = evaluate_side(pred, ohlc, atr, times, side, oof.start, oof.stop, eval_params)
            sh = r["metrics"]["sharpe"]
            print("      %-10s OOF 夏普=%+.3f 笔数=%d" % (cand["name"], sh, r["metrics"]["n_trades"]))
            if best_res is None or sh > best_res["metrics"]["sharpe"]:
                best_key, best_res, best_tm = cand["name"], r, tm
        cand_best[side] = dict(name=best_key, model=best_tm, res=best_res)
        print("    -> [%s] 选定模型配置: %s" % (side, best_key))

    # ---------------- 5. 执行层优化(仅在 OOF)
    print("\n=== 5. 执行层: 仅在 OOF 段网格搜索(阈值分位 x 止盈 x 止损) ===")
    frozen = {}
    grid_rows_total = 0
    for side in ("long", "short"):
        if side not in cand_best:
            continue
        tm = cand_best[side]["model"]
        pred = tm.predict(Fmat)
        g = optimize.grid_execution(pred, ohlc, atr, times, side, oof.start, oof.stop)
        grid_rows_total += len(g)
        if g.empty:
            print("  [%s] 无有效组合" % side)
            continue
        best = optimize.pick_best(g)
        print("  [%s] 组合 %d 个; 最优: 阈值分位=%.2f 止盈=%.1fATR 止损=%.1fATR "
              "-> 夏普=%+.2f 净利=%.2f%% 笔数=%d 胜率=%.1f%% 盈亏比=%.2f"
              % (side, len(g), best["thr_q"], best["tp_mult"], best["sl_mult"],
                 best["sharpe"], best["total_return"] * 100, best["n_trades"],
                 best["win_rate"] * 100, best["payoff_ratio"]))
        frozen[side] = dict(thr_q=best["thr_q"], tp_mult=best["tp_mult"],
                            sl_mult=best["sl_mult"], model=tm)
        g.to_csv(C.REPORT_DIR / ("oof_grid_%s.csv" % side), index=False)

    # ---------------- 6. 冻结后在 OOF 与 OOC 各跑一次
    print("\n=== 6. 冻结配置 -> OOF(用于确认) 与 OOC(仅观察) ===")
    report = {"config": {k: getattr(C, k) for k in
                         ("SYMBOL", "INTERVAL", "HORIZON", "TRAIN_FRAC", "OOF_FRAC",
                          "OOC_FRAC", "CV_N_SPLITS", "CV_EMBARGO_BARS", "IC_MIN_ABS",
                          "INIT_CAPITAL", "TRADE_NOTIONAL", "MAX_HOLD_BARS",
                          "FEE_RATE", "SLIP_RATE")},
              "split": sl, "sides": {}}
    curves = {}
    for side, cfg in frozen.items():
        tm = cfg["model"]
        pred = tm.predict(Fmat)
        entry = {}
        for seg_name, seg in (("oof", oof), ("ooc", ooc)):
            r = evaluate_side(pred, ohlc, atr, times, side, seg.start, seg.stop, cfg)
            entry[seg_name] = r["metrics"]
            curves["%s_%s" % (side, seg_name)] = (r["equity"], df["datetime"].iloc[seg])
        report["sides"][side] = {
            "factors": tm.factors,
            "frozen_params": {k: cfg[k] for k in ("thr_q", "tp_mult", "sl_mult")},
            "cv_rmse": tm.cv_rmse, "best_iters": tm.best_iters,
            "oof": entry["oof"], "ooc": entry["ooc"],
        }
        print("\n  --- %s ---" % side)
        print("  %-6s %6s %8s %8s %8s %9s %9s %9s %9s" %
              ("段", "笔数", "胜率%", "盈亏比", "总收益%", "夏普", "卡玛", "最大回撤%", "盈亏(USDT)"))
        for seg_name in ("oof", "ooc"):
            m = entry[seg_name]
            print("  %-6s %6d %8.1f %8.2f %8.2f %9.2f %9.2f %9.2f %9.2f" %
                  (seg_name, m["n_trades"], m["win_rate"] * 100, m["payoff_ratio"],
                   m["total_return"] * 100, m["sharpe"], m["calmar"],
                   m["max_drawdown"] * 100, m["total_pnl"]))

    # 合并多空权益曲线
    if curves:
        fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=False)
        for side in ("long", "short"):
            for seg_name in ("oof", "ooc"):
                k = "%s_%s" % (side, seg_name)
                if k in curves:
                    eq, ts = curves[k]
                    axes[0 if seg_name == "oof" else 1].plot(ts, eq, label="%s-%s" % (side, seg_name))
        axes[0].set_title("OOF equity (optimization segment)")
        axes[1].set_title("OOC equity (held-out, observed only)")
        for ax in axes:
            ax.axhline(C.INIT_CAPITAL, color="grey", ls="--", lw=1)
            ax.legend(); ax.grid(alpha=.3); ax.set_ylabel("USDT")
        plt.tight_layout()
        plt.savefig(C.REPORT_DIR / "equity_curve.png", dpi=130)
        print("\n曲线已保存: %s" % (C.REPORT_DIR / "equity_curve.png"))

    # ---------------- 7. 落盘
    report["disclosure"] = optimize.disclosure(grid_rows_total, 3 * 2)
    (C.REPORT_DIR / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    ic.to_frame("ic").sort_values("ic", key=lambda s: s.abs(), ascending=False).to_csv(
        C.REPORT_DIR / "factor_ic.csv")
    json.dump({s: sel["selected"][s] for s in sel["selected"]},
              open(C.REPORT_DIR / "selected_factors.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("报告已落盘: %s" % C.REPORT_DIR)
    print("\n%s" % report["disclosure"])


if __name__ == "__main__":
    main()
