#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SOL/USDT 4h 多因子收益预测与回测 —— 主流程。

流程(严格按使用者规格):
  1. 载入清洗后的 4h 数据, 构建因子与标签
  2. 按时间切分: train 70% / OOF 15% / OOC 15%
  3. **仅在训练段**做因子 IC 筛选 + 相关性阈值去冗余; 多空分别筛
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
import pandas as pd                      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C                       # noqa: E402
from src import data_clean, factors as F_lib, factor_select, models, optimize, execution  # noqa: E402
from src.cv import describe_split, time_split                                   # noqa: E402

pd.set_option("display.width", 200)


def load_data() -> pd.DataFrame:
    if not C.CLEAN_PARQUET.exists():
        print("清洗数据不存在, 先跑清洗...")
        raw = pd.read_parquet(C.RAW_PARQUET)
        df = data_clean.clean(raw).reset_index(drop=True)
    else:
        df = pd.read_parquet(C.CLEAN_PARQUET).reset_index(drop=True)
    return df


def buy_hold(df: pd.DataFrame, seg: slice) -> float:
    """该段的买入持有收益(基准), 用于判断模型是否真的提供了超额。"""
    a = float(df["close"].iloc[seg.start])
    b = float(df["close"].iloc[seg.stop - 1])
    return b / a - 1.0


def main() -> None:
    print("=" * 104)
    print("SOL/USDT %s 多因子模型  |  标签: 未来 %d 根(=%d 小时)收益 / ATR  [制度中性]"
          % (C.INTERVAL, C.HORIZON, C.HORIZON * 4))
    print("=" * 104)

    # ---------------- 1. 数据 + 因子
    df = load_data()
    df = F_lib.add_label(df, C.HORIZON)
    Fmat, fnames = F_lib.build_factors(df)
    print("数据: %d 根 %s K线  %s ~ %s" %
          (len(df), C.INTERVAL, df["datetime"].iloc[0], df["datetime"].iloc[-1]))
    print("原始因子: %d 个" % len(fnames))

    ohlc = {k: df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close")}
    atr = df["atr"].to_numpy(dtype=float)
    times = df["datetime"].to_numpy()
    y = df["label"]

    # ---------------- 2. 切分
    sl = describe_split(len(df))
    (tr, oof, ooc) = time_split(len(df))
    print("\n=== 数据切分(按时间顺序) + 各段买入持有基准 ===")
    for s in sl:
        seg = slice(s["start"], s["end"])
        bh = buy_hold(df, seg)
        print("  %-6s 行 %5d~%5d  n=%5d  %s ~ %s   买入持有 %+8.1f%%" %
              (s["name"], s["start"], s["end"] - 1, s["n"],
               str(df["datetime"].iloc[s["start"]])[:16],
               str(df["datetime"].iloc[s["end"] - 1])[:16], bh * 100))

    # ---------------- 3. 因子筛选(仅训练段)
    print("\n=== 因子筛选(仅用训练段, IC=Spearman, 标签已 ATR 标准化) ===")
    sel = factor_select.select_factors(Fmat.iloc[tr].reset_index(drop=True),
                                       y.iloc[tr].reset_index(drop=True))
    ic: pd.Series = sel["ic"]
    print("  全因子 |IC| 分位: P50=%.4f P75=%.4f P90=%.4f 最大=%.4f" %
          (ic.abs().median(), ic.abs().quantile(.75), ic.abs().quantile(.90), ic.abs().max()))
    print("  |IC| >= %.3f 的因子: %d / %d" % (C.IC_MIN_ABS,
                                             int((ic.abs() >= C.IC_MIN_ABS).sum()), len(ic)))
    for side in ("long", "short"):
        print("  [%s] 去冗余后保留 %d 个: %s" %
              (side, len(sel["selected"][side]), ", ".join(sel["selected"][side])))

    # ---------------- 4+5. 嵌套 CV: 在训练段内部选模型与执行参数
    print("\n=== 嵌套 CV(训练段内部 %d 折) 选模型 x 执行参数 ===" % C.NESTED_FOLDS)
    frozen, aggs = {}, {}
    n_combo = len(models.MODEL_GRID) * len(C.EXEC_THR_GRID) * len(C.TP_ATR_GRID) * len(C.SL_ATR_GRID)
    print("  候选组合: %d(模型%d x 阈值%d x 止盈%d x 止损%d) x %d 折"
          % (n_combo, len(models.MODEL_GRID), len(C.EXEC_THR_GRID), len(C.TP_ATR_GRID),
             len(C.SL_ATR_GRID), C.NESTED_FOLDS))
    for side in ("long", "short"):
        facs = sel["selected"][side]
        if not facs:
            print("  [%s] 无可用因子, 跳过" % side)
            continue
        best_model, best_exec, agg = optimize.nested_select(
            Fmat, y, facs, side, ohlc, atr, times, tr)
        aggs[side] = agg
        print("  [%s] 选定: 模型=%s  阈值分位=%.2f  止盈=%.2fATR  止损=%.2fATR"
              % (side, best_model, best_exec["thr_q"], best_exec["tp_mult"], best_exec["sl_mult"]))
        if not agg.empty:
            print("      各折均值 Top3:")
            for _, r in agg.head(3).iterrows():
                print("        %-10s q=%.1f tp=%.1f sl=%.1f -> 夏普%+.2f 收益%+.2f%% "
                      "笔数%.0f 止盈率%.0f%% 胜率%.0f%%"
                      % (r["model"], r["thr_q"], r["tp_mult"], r["sl_mult"], r["sharpe"],
                         r["total_return"] * 100, r["n_trades"], r["tp_rate"] * 100,
                         r["win_rate"] * 100))
        # 用整个训练段重训选定模型(集成多折)
        mc = next(m for m in models.MODEL_GRID if m["name"] == best_model)
        tm = models.train_side(Fmat.iloc[tr], y.iloc[tr], facs, side,
                               params=models.resolve_params(mc), verbose=False)
        frozen[side] = dict(**best_exec, model=tm, model_name=best_model, factors=facs)
        if agg is not None and not agg.empty:
            agg.to_csv(C.REPORT_DIR / ("nested_cv_%s.csv" % side), index=False)

    # ---------------- 6. 冻结后在 OOF(确认) 与 OOC(仅观察) 各跑一次
    print("\n=== 冻结配置 -> OOF(一次性确认) 与 OOC(仅观察, 不参与选择) ===")
    report = {"config": {k: getattr(C, k) for k in
                         ("SYMBOL", "INTERVAL", "HORIZON", "LABEL_MODE", "TRAIN_FRAC",
                          "OOF_FRAC", "OOC_FRAC", "NESTED_FOLDS", "CV_EMBARGO_BARS",
                          "IC_MIN_ABS", "INIT_CAPITAL", "TRADE_NOTIONAL",
                          "MAX_HOLD_BARS", "FEE_RATE", "SLIP_RATE", "MIN_TP_RATE")},
              "split": sl, "sides": {}}
    report["buy_hold"] = {s["name"]: buy_hold(df, slice(s["start"], s["end"])) for s in sl}
    curves = {}
    for side, cfg in frozen.items():
        pred = cfg["model"].predict(Fmat)
        entry = {}
        for seg_name, seg in (("oof", oof), ("ooc", ooc)):
            r = execution.evaluate_with_params(pred, ohlc, atr, times, side, seg.start, seg.stop,
                                               cfg["thr_q"], cfg["tp_mult"], cfg["sl_mult"])
            entry[seg_name] = r["metrics"]
            curves["%s_%s" % (side, seg_name)] = (r["equity"], df["datetime"].iloc[seg])
        report["sides"][side] = {
            "model": cfg["model_name"],
            "factors": cfg["factors"],
            "frozen_params": {k: cfg[k] for k in ("thr_q", "tp_mult", "sl_mult")},
            "cv_rmse": cfg["model"].cv_rmse,
            "oof": entry["oof"], "ooc": entry["ooc"],
        }
        print("\n  --- %s  (模型 %s, 因子 %d 个) ---" % (side, cfg["model_name"], len(cfg["factors"])))
        print("  %-6s %6s %7s %7s %9s %8s %8s %9s %10s %9s" %
              ("段", "笔数", "胜率%", "盈亏比", "总收益%", "买入持有%", "夏普", "卡玛",
               "最大回撤%", "盈亏USDT"))
        for seg_name in ("oof", "ooc"):
            m = entry[seg_name]
            bh = report["buy_hold"][seg_name] * 100
            print("  %-6s %6d %7.1f %7.2f %9.2f %10.2f %8.2f %9.2f %10.2f %9.2f" %
                  (seg_name, m["n_trades"], m["win_rate"] * 100, m["payoff_ratio"],
                   m["total_return"] * 100, bh, m["sharpe"], m["calmar"],
                   m["max_drawdown"] * 100, m["total_pnl"]))
            print("         出场构成: 止盈%.0f%% 止损%.0f%% 到期%.0f%%  平均持有%.1f根" %
                  (m["tp_rate"] * 100, m["sl_rate"] * 100, m["timeout_rate"] * 100, m["avg_bars"]))

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
    report["disclosure"] = optimize.disclosure(n_combo, C.NESTED_FOLDS)
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
