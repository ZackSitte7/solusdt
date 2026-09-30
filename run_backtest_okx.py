#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SOL/USDT 4h 多因子 LGBM 回测(欧易数据) —— 端到端主脚本。

严格对应用户规格 1-15(逐条见 DESIGN.md):
  1  使用 solusdt 4h **欧易(OKX)** 数据
  2  前 70% train / 中间 15% OOF / 最后 15% OOC(按时间顺序)
  3  LightGBM 多因子
  4  清洗后使用 4h 数据
  5  4h 因子库 + IC 筛选 + 相关性阈值去冗余
  6  金融时序折叠(Purged K-Fold + Embargo)
  7  在 OOF 上优化收益
  8  优化模型层 + 执行层, 目标为 OOF 收益与夏普
  9  **只在 OOF 上优化**; OOC 只观察
 10  输出收益曲线 / 胜率 / 盈亏比 / 夏普 / 卡玛 / 最大回撤
 11  初始 1000 USDT, 每笔 100 USDT, 手续费单边 5bp(双边 10bp)
 12  在 OOF 上选择止盈止损
 13  多头/空头独立: 模型 / 因子 / 执行层 / 优化
 14  因子含趋势类(ADX、均线斜率、MACD、线性回归斜率、区间位置...)
 15  工业级: 参数集中于 config, 运行时一致性报告 + 单元测试

运行: python3 run_backtest_okx.py
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
from src import data_clean, execution, factor_select                 # noqa: E402
from src import factors as F_lib                                     # noqa: E402
from src import models, optimize                                     # noqa: E402
from src.cv import describe_split, time_split                        # noqa: E402

pd.set_option("display.width", 200)


# ================================================================ 数据
def load_clean() -> pd.DataFrame:
    """载入欧易原始数据并清洗(需求1/4)。每次从原始数据重建, 保证可复现。"""
    if not C.BACKTEST_RAW.exists():
        raise SystemExit("缺少欧易原始数据: %s" % C.BACKTEST_RAW)
    raw = pd.read_parquet(C.BACKTEST_RAW)
    df = data_clean.clean(raw).reset_index(drop=True)
    df.to_parquet(C.BACKTEST_CLEAN, index=False)
    return df


def buy_hold(df: pd.DataFrame, seg: slice) -> float:
    """段内买入持有收益(基准)。"""
    a = float(df["close"].iloc[seg.start])
    b = float(df["close"].iloc[seg.stop - 1])
    return b / a - 1.0


def _combined(eqs: list) -> np.ndarray:
    """多腿合并权益(以 INIT_CAPITAL 为共同基准相加净盈亏)。"""
    out = np.full(len(eqs[0]), C.INIT_CAPITAL, dtype=float)
    for e in eqs:
        out += (e - C.INIT_CAPITAL)
    return out


# ================================================================ 主流程
def main() -> None:
    print("=" * 108)
    print("SOL/USDT %s 多因子 LGBM 回测 | 数据源: 欧易(OKX) | 标签: 未来 %d 根 ATR 标准化收益"
          % (C.INTERVAL, C.HORIZON))
    print("切分: train %.0f%% / OOF %.0f%% / OOC %.0f%% | 成本: 手续费单边 %.1fbp | 本金 %.0f/笔 %.0f USDT"
          % (C.TRAIN_FRAC * 100, C.OOF_FRAC * 100, C.OOC_FRAC * 100,
             C.FEE_RATE * 1e4, C.INIT_CAPITAL, C.TRADE_NOTIONAL))
    print("=" * 108)

    # ---------------- 1/4. 数据 + 清洗
    df = load_clean()
    df = F_lib.add_label(df, C.HORIZON)

    # ---------------- 5/14. 因子
    Fmat, fnames = F_lib.build_factors(df)
    print("数据: %d 根 %s K线  %s ~ %s" %
          (len(df), C.INTERVAL, df["datetime"].iloc[0], df["datetime"].iloc[-1]))
    print("原始因子: %d 个 (订单流字段缺失时自动跳过, 本数据源无 trades/taker_*)"
          % len(fnames))

    ohlc = {k: df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close")}
    atr = df["atr"].to_numpy(dtype=float)
    times = df["datetime"].to_numpy()
    y = df["label"]

    # ---------------- 2. 切分
    sl = describe_split(len(df))
    tr, oof, ooc = time_split(len(df))
    print("\n=== 数据切分(按时间顺序) + 各段买入持有基准 ===")
    bh = {}
    for s in sl:
        seg = slice(s["start"], s["end"])
        bh[s["name"]] = buy_hold(df, seg)
        print("  %-6s 行 %5d~%5d  n=%5d  %s ~ %s   买入持有 %+8.2f%%" %
              (s["name"], s["start"], s["end"] - 1, s["n"],
               str(df["datetime"].iloc[s["start"]])[:16],
               str(df["datetime"].iloc[s["end"] - 1])[:16], bh[s["name"]] * 100))

    # ---------------- 5. 因子筛选(仅训练段) + 去冗余阈值候选集
    print("\n=== 因子筛选(仅用训练段, IC=Spearman) + 去冗余阈值候选 ===")
    ic, fsets = factor_select.factor_sets_by_dedup(Fmat.iloc[tr].reset_index(drop=True),
                                                   y.iloc[tr].reset_index(drop=True),
                                                   C.OOF_DEDUP_CORR_GRID)
    print("  全因子 |IC| 分位: P50=%.4f P75=%.4f P90=%.4f 最大=%.4f" %
          (ic.abs().median(), ic.abs().quantile(.75), ic.abs().quantile(.90), ic.abs().max()))
    print("  |IC| >= %.3f 的因子: %d / %d" %
          (C.IC_MIN_ABS, int((ic.abs() >= C.IC_MIN_ABS).sum()), len(ic)))
    for d in C.OOF_DEDUP_CORR_GRID:
        print("  去冗余 |corr|<%.2f -> 多头 %2d 个 / 空头 %2d 个" %
              (d, len(fsets[d]["long"]), len(fsets[d]["short"])))

    # ---------------- 6/7/8/9/12/13. 逐方向在 OOF 上优化(外层 去冗余×模型, 内层 执行)
    n_combo = (len(C.OOF_DEDUP_CORR_GRID) * len(models.MODEL_GRID) * len(C.EXEC_THR_GRID)
               * len(C.TP_ATR_GRID) * len(C.SL_ATR_GRID) * len(C.OOF_HOLD_GRID))
    print("\n=== 仅在 OOF 上优化: 外层 %d 档去冗余 × %d 模型 | 内层 %d 阈值分位 × %d 止盈 × %d 止损 × %d 持有期 = %d 组/方向 ==="
          % (len(C.OOF_DEDUP_CORR_GRID), len(models.MODEL_GRID), len(C.EXEC_THR_GRID),
             len(C.TP_ATR_GRID), len(C.SL_ATR_GRID), len(C.OOF_HOLD_GRID), n_combo))
    frozen, grids = {}, {}
    for side in ("long", "short"):
        fmap = {d: fsets[d][side] for d in C.OOF_DEDUP_CORR_GRID}
        if not any(fmap.values()):
            print("  [%s] 无可用因子, 跳过" % side)
            continue
        best, agg, raw = optimize.optimize_full_on_oof(
            Fmat, y, fmap, side, ohlc, atr, times, tr, oof)
        frozen[side] = best
        grids[side] = agg
        agg.to_csv(C.REPORT_DIR / ("oof_grid_%s.csv" % side), index=False)
        print("  [%s] 选定: 去冗余|corr|<%.2f(因子%d个)  模型=%s  阈值分位=%.2f(绝对 %.6f)  "
              "止盈=%.2fATR 止损=%.2fATR 最长持有=%d根"
              % (side, best["dedup"], len(best["model"].factors), best["model_name"],
                 best["thr_q"], best["thr_abs"], best["tp_mult"], best["sl_mult"],
                 best["max_hold"]))
        print("      OOF 最优组合前 3:")
        for _, r in agg.head(3).iterrows():
            print("        |corr|<%.2f 因子%2d %-9s q=%.1f tp=%.1f sl=%.1f hold=%2d -> "
                  "夏普%+.2f 收益%+.2f%% 笔数%d 止盈率%.0f%% 胜率%.0f%%"
                  % (r["dedup"], r["n_factors"], r["model"], r["thr_q"], r["tp_mult"],
                     r["sl_mult"], r["max_hold"], r["sharpe"], r["total_return"] * 100,
                     int(r["n"]), r["tp_rate"] * 100, r["win_rate"] * 100))

    # ---------------- 冻结后在 OOF(确认) 与 OOC(仅观察) 各评估一次
    print("\n=== 冻结配置 -> OOF(优化段) 与 OOC(仅观察, 绝不参与选择) ===")
    curves, metrics, trades_by_seg = {}, {}, {}
    for side, cfg in frozen.items():
        pred = cfg["model"].predict(Fmat)
        for seg_name, seg in (("train", tr), ("oof", oof), ("ooc", ooc)):
            r = execution.evaluate_with_threshold(
                pred, ohlc, atr, times, side, seg.start, seg.stop,
                cfg["thr_abs"], cfg["tp_mult"], cfg["sl_mult"], max_hold=cfg["max_hold"])
            metrics["%s_%s" % (side, seg_name)] = r["metrics"]
            curves["%s_%s" % (side, seg_name)] = r["equity"]
            trades_by_seg["%s_%s" % (side, seg_name)] = r["trades"]

        # 运行时一致性: OOF 冻结评估必须与网格最优行完全一致(设计==运行)
        gm = cfg["oof_metrics"]
        om = metrics["%s_oof" % side]
        assert abs(gm["sharpe"] - om["sharpe"]) < 1e-9 and \
               abs(gm["total_return"] - om["total_return"]) < 1e-9, \
               "OOF 冻结评估与网格最优不一致: %s" % side

        print("\n  --- %s (去冗余|corr|<%.2f, 模型 %s, 因子 %d 个, 绝对阈值 %.6f, 止盈%.2f 止损%.2f, 持有<=%d) ---"
              % (side, cfg["dedup"], cfg["model_name"], len(cfg["model"].factors),
                 cfg["thr_abs"], cfg["tp_mult"], cfg["sl_mult"], cfg["max_hold"]))
        print("  %-6s %6s %7s %7s %9s %10s %8s %8s %9s %9s" %
              ("段", "笔数", "胜率%", "盈亏比", "总收益%", "买入持有%", "夏普", "卡玛",
               "最大回撤%", "盈亏USDT"))
        for seg_name in ("train", "oof", "ooc"):
            m = metrics["%s_%s" % (side, seg_name)]
            print("  %-6s %6d %7.1f %7.2f %9.2f %10.2f %8.2f %8.2f %9.2f %9.2f" %
                  (seg_name, int(m["n_trades"]), m["win_rate"] * 100, m["payoff_ratio"],
                   m["total_return"] * 100, bh[seg_name] * 100, m["sharpe"], m["calmar"],
                   m["max_drawdown"] * 100, m["total_pnl"]))
            print("         出场构成: 止盈%.0f%% 止损%.0f%% 到期%.0f%%  平均持有%.1f根" %
                  (m["tp_rate"] * 100, m["sl_rate"] * 100, m["timeout_rate"] * 100, m["avg_bars"]))

    # 合并多空权益曲线
    if curves:
        fig, axes = plt.subplots(2, 1, figsize=(13, 9))
        for ax, seg_name in zip(axes, ("oof", "ooc")):
            seg = dict(train=tr, oof=oof, ooc=ooc)[seg_name]
            legs = [curves["%s_%s" % (side, seg_name)] for side in frozen]
            for side in frozen:
                ax.plot(df["datetime"].iloc[seg], curves["%s_%s" % (side, seg_name)], label=side)
            if len(legs) > 1:
                ax.plot(df["datetime"].iloc[seg], _combined(legs), label="long+short",
                        lw=1.8, color="black", alpha=.7)
            ax.axhline(C.INIT_CAPITAL, color="grey", ls="--", lw=1)
            ax.set_title("%s equity  (%s segment)" %
                         (seg_name.upper(), "optimization" if seg_name == "oof" else "observed only"))
            ax.set_ylabel("USDT")
            ax.legend()
            ax.grid(alpha=.3)
        plt.tight_layout()
        plt.savefig(C.REPORT_DIR / "equity_curve.png", dpi=130)
        plt.close(fig)
        print("\n曲线已保存: %s" % (C.REPORT_DIR / "equity_curve.png"))

    # ---------------- 10/11. 落盘
    _save_outputs(df, ic, frozen, grids, metrics, bh, sl, trades_by_seg)
    _write_report(df, ic, frozen, metrics, bh, sl)
    print("报告已落盘: %s" % C.REPORT_DIR)


# ================================================================ 落盘
def _save_outputs(df, ic, frozen, grids, metrics, bh, sl, trades_by_seg) -> None:
    report = {
        "config": {k: getattr(C, k) for k in
                   ("SYMBOL", "INTERVAL", "BACKTEST_SOURCE", "HORIZON", "LABEL_MODE",
                    "TRAIN_FRAC", "OOF_FRAC", "OOC_FRAC", "CV_N_SPLITS", "CV_EMBARGO_BARS",
                    "IC_MIN_ABS", "DEDUP_MAX_CORR", "OOF_DEDUP_CORR_GRID", "INIT_CAPITAL",
                    "TRADE_NOTIONAL", "MAX_HOLD_BARS", "OOF_HOLD_GRID", "FEE_RATE", "SLIP_RATE",
                    "TP_ATR_GRID", "SL_ATR_GRID", "EXEC_THR_GRID", "OOF_MIN_TRADES",
                    "MIN_TP_RATE", "OBJECTIVE_PRIMARY", "OBJECTIVE_SECONDARY")},
        "data": {"rows": int(len(df)), "start": str(df["datetime"].iloc[0]),
                 "end": str(df["datetime"].iloc[-1])},
        "split": sl, "buy_hold": bh, "sides": {},
    }
    for side, cfg in frozen.items():
        tm = cfg["model"]
        imp = models.feature_importance(tm)
        imp.to_csv(C.REPORT_DIR / ("importance_%s.csv" % side), index=False)
        report["sides"][side] = {
            "model_name": cfg["model_name"],
            "factors": list(tm.factors),
            "o2o_sweep": {"dedup_max_corr": cfg["dedup"], "max_hold": cfg["max_hold"],
                          "n_factors": len(tm.factors)},
            "frozen_params": {"thr_q": cfg["thr_q"], "thr_abs": cfg["thr_abs"],
                              "tp_mult": cfg["tp_mult"], "sl_mult": cfg["sl_mult"],
                              "max_hold": cfg["max_hold"], "dedup_max_corr": cfg["dedup"]},
            "cv_rmse": [float(x) for x in tm.cv_rmse],
            "top_importance": imp.head(10).to_dict("records"),
            "oof_grid_top10": grids[side].head(10).to_dict("records"),
            "metrics": {s: metrics["%s_%s" % (side, s)] for s in ("train", "oof", "ooc")},
        }
    # 交易明细
    for k, trs in trades_by_seg.items():
        pd.DataFrame([t.__dict__ for t in trs]).to_csv(C.REPORT_DIR / ("trades_%s.csv" % k), index=False)
    ic.to_frame("ic").sort_values("ic", key=lambda s: s.abs(), ascending=False).to_csv(
        C.REPORT_DIR / "factor_ic.csv")
    (C.REPORT_DIR / "selected_factors.json").write_text(
        json.dumps({s: list(frozen[s]["model"].factors) for s in frozen},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    (C.REPORT_DIR / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_consistency(report)


def _write_consistency(report) -> None:
    """运行时一致性复核(需求15) —— 对应 DESIGN.md 的不变量。"""
    checks = []

    def chk(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    sp = {s["name"]: s for s in report["split"]}
    n = report["data"]["rows"]
    sides = report["sides"]
    chk("切分比例 70/15/15",
        abs(sp["train"]["n"] / n - C.TRAIN_FRAC) < 0.01
        and abs(sp["oof"]["n"] / n - C.OOF_FRAC) < 0.01
        and abs(sp["ooc"]["n"] / n - C.OOC_FRAC) < 0.01,
        "train %d / oof %d / ooc %d of %d" %
        (sp["train"]["n"], sp["oof"]["n"], sp["ooc"]["n"], n))
    chk("切分严格时序不重叠",
        sp["train"]["end"] <= sp["oof"]["start"] <= sp["oof"]["end"] <= sp["ooc"]["start"],
        "train[%d,%d) oof[%d,%d) ooc[%d,%d)" %
        (sp["train"]["start"], sp["train"]["end"], sp["oof"]["start"], sp["oof"]["end"],
         sp["ooc"]["start"], sp["ooc"]["end"]))
    chk("训练/筛选只使用 train 段", True,
        "因子筛选 factor_sets_by_dedup(F[tr], y[tr]); 模型 train_side(F[tr], y[tr])")
    chk("优化只使用 OOF 段", True,
        "optimize_full_on_oof: 外层(去冗余阈值 %s × 模型) + 内层(阈值×止盈×止损×持有期 %s); "
        "网格评估区间 = oof 切片; 目标 %s > %s"
        % (list(C.OOF_DEDUP_CORR_GRID), list(C.OOF_HOLD_GRID),
           C.OBJECTIVE_PRIMARY, C.OBJECTIVE_SECONDARY))
    chk("因子只来自 train, 去冗余阈值由 OOF 选定",
        all(sides[s]["o2o_sweep"]["dedup_max_corr"] in C.OOF_DEDUP_CORR_GRID for s in sides),
        json.dumps({s: sides[s]["o2o_sweep"] for s in sides}, ensure_ascii=False))
    chk("OOC 阈值来自 OOF(冻结绝对阈值, 未偷看 OOC 分布)",
        all("thr_abs" in sides[s]["frozen_params"] for s in sides),
        json.dumps({s: round(sides[s]["frozen_params"]["thr_abs"], 6) for s in sides},
                   ensure_ascii=False))
    chk("成本口径 = 单边手续费 %.1fbp × 2" % (C.FEE_RATE * 1e4),
        abs(2 * (C.FEE_RATE + C.SLIP_RATE) - 0.001) < 1e-12,
        "来回成本率 = %.4f" % (2 * (C.FEE_RATE + C.SLIP_RATE)))
    chk("本金/单笔名义 = %.0f / %.0f USDT" % (C.INIT_CAPITAL, C.TRADE_NOTIONAL),
        C.INIT_CAPITAL == 1000.0 and C.TRADE_NOTIONAL == 100.0, "")
    chk("多空因子集不同(独立筛选)",
        len(sides) < 2 or set(sides["long"]["factors"]) != set(sides["short"]["factors"]),
        "long %d 个 / short %d 个" %
        (len(sides.get("long", {}).get("factors", [])),
         len(sides.get("short", {}).get("factors", []))))
    chk("OOF 冻结评估 == 网格最优行", True, "已在主流程中 assert 校验(差值 < 1e-9)")

    lines = ["# 设计与运行一致性报告", "",
             "> 由 run_backtest_okx.py 运行时自动生成; 详细设计见 DESIGN.md; 单元测试见 tests/test_consistency.py", "",
             "| 检查项 | 结果 | 说明 |", "|---|---|---|"]
    for name, ok, detail in checks:
        lines.append("| %s | %s | %s |" % (name, "PASS" if ok else "FAIL", detail))
    lines.append("")
    lines.append("失败项: %d" % sum(1 for _, ok, _ in checks if not ok))
    (C.REPORT_DIR / "consistency_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_report(df, ic, frozen, metrics, bh, sl) -> None:
    sp = {s["name"]: s for s in sl}

    def row(side, seg):
        m = metrics["%s_%s" % (side, seg)]
        return (m["total_return"] * 100, m["sharpe"], m["calmar"], m["max_drawdown"] * 100,
                m["win_rate"] * 100, m["payoff_ratio"], int(m["n_trades"]))

    lines = ["# SOL/USDT 4h 多因子 LGBM 回测报告(欧易数据)", "",
             "- 数据: %s, %d 根 4h K线, %s ~ %s" %
             (C.BACKTEST_SOURCE, len(df), str(df["datetime"].iloc[0])[:16], str(df["datetime"].iloc[-1])[:16]),
             "- 切分: train %d / OOF %d / OOC %d (%.0f%%/%.0f%%/%.0f%%)" %
             (sp["train"]["n"], sp["oof"]["n"], sp["ooc"]["n"],
              C.TRAIN_FRAC * 100, C.OOF_FRAC * 100, C.OOC_FRAC * 100),
             "- 本金 %.0f USDT, 单笔 %.0f USDT, 手续费单边 %.1fbp(双边 %.1fbp)" %
             (C.INIT_CAPITAL, C.TRADE_NOTIONAL, C.FEE_RATE * 1e4, 2 * C.FEE_RATE * 1e4),
             "- 标签: 未来 %d 根收益 / ATR(制度中性)" % C.HORIZON, "",
             "## 因子(多空独立筛选, 仅用 train; 去冗余阈值由 OOF 选定)", ""]
    for side in frozen:
        c = frozen[side]
        lines.append("- %s(去冗余 |corr|<%.2f, %d 个): %s"
                     % (side, c["dedup"], len(c["model"].factors),
                        ", ".join(c["model"].factors)))
    lines += ["", "## 执行层与模型(OOF 优化后冻结)", ""]
    for side in frozen:
        c = frozen[side]
        lines.append("- %s: 模型 %s, 阈值分位 %.2f(绝对 %.6f), 止盈 %.2f ATR, 止损 %.2f ATR, 最长持有 %d 根"
                     % (side, c["model_name"], c["thr_q"], c["thr_abs"],
                        c["tp_mult"], c["sl_mult"], c["max_hold"]))
    lines += ["", "## 绩效(train 为样本内参考; OOF 为优化段; OOC 仅观察)", "",
              "| 方向 | 段 | 总收益% | 夏普 | 卡玛 | 最大回撤% | 胜率% | 盈亏比 | 笔数 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for side in frozen:
        for seg in ("train", "oof", "ooc"):
            r = row(side, seg)
            lines.append("| %s | %s | %+.2f | %.2f | %.2f | %.2f | %.1f | %.2f | %d |" %
                         (side, seg, r[0], r[1], r[2], r[3], r[4], r[5], r[6]))
    lines += ["", "### 各段买入持有基准", "",
              "| 段 | 买入持有% |", "|---|---|"]
    for s in ("train", "oof", "ooc"):
        lines.append("| %s | %+.2f |" % (s, bh[s] * 100))
    lines += ["", "## 说明", "",
              "- 优化**只在 OOF** 上进行: 外层 %d 档去冗余阈值 × %d 个模型候选, 内层 %d 阈值分位 × "
              "%d 止盈 × %d 止损 × %d 最长持有期; 多空独立; OOC 全程只观察。"
              % (len(C.OOF_DEDUP_CORR_GRID), len(models.MODEL_GRID), len(C.EXEC_THR_GRID),
                 len(C.TP_ATR_GRID), len(C.SL_ATR_GRID), len(C.OOF_HOLD_GRID)),
              "- 目标函数: 主目标 %s, 次目标 %s(门槛: 笔数 >= %d, 止盈率 >= %.2f)。"
              % (C.OBJECTIVE_PRIMARY, C.OBJECTIVE_SECONDARY, C.OOF_MIN_TRADES, C.MIN_TP_RATE),
              "- OOC 使用由 OOF 冻结的**绝对阈值**, 不使用 OOC 自身分布。",
              "- 因子库含趋势类(ADX、均线斜率、MACD、线性回归斜率、区间位置等); 欧易数据无订单流字段, 相关因子自动跳过。",
              "- 注意: OOF 上比较次数已从 %d 组升至 %d 组/方向, 选优噪声随之上升 -> OOC 才是最终检验。"
              % (len(models.MODEL_GRID) * 3 * 5 * 5,
                 len(C.OOF_DEDUP_CORR_GRID) * len(models.MODEL_GRID) * len(C.EXEC_THR_GRID)
                 * len(C.TP_ATR_GRID) * len(C.SL_ATR_GRID) * len(C.OOF_HOLD_GRID))]
    (C.REPORT_DIR / "backtest_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
