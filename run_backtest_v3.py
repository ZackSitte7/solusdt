#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SOL/USDT 4h 多因子 LGBM 回测 (v3: OOF 选优 + ATR 止盈止损, 硬约束 tp > sl)。

用户要求(相对 v1/v2 的变化):
  1. 选择集回到 **OOF**(v1 口径), 把 OOF 收益做回 v1 水准; OOC 仍**只观察**;
  2. 止盈/止损用 **ATR 倍数**(ATR 止盈 / ATR 止损);
  3. 目标: 收益优先、夏普次之(并显示夏普);
  4. **硬约束 tp > sl**(风险报酬比 > 1), 削掉"止盈比止损还近"的退化配置。

与 v1 的唯一差别就是第 4 条(内层执行网格加过滤); 其余口径完全一致, 保证可比。
运行: python3 run_backtest_v3.py
"""
from __future__ import annotations

import json
import sys
from itertools import product
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
from src import optimize                                     # noqa: E402
from src.cv import time_split                                        # noqa: E402

OUT = C.REPORT_DIR / C.V3_OUT_DIR
SEGS = ("train", "oof", "ooc")
pd.set_option("display.width", 220)


def load_clean() -> pd.DataFrame:
    """载入欧易原始数据并清洗(与 v1/v2 同一函数口径)。"""
    if not C.BACKTEST_RAW.exists():
        raise SystemExit("缺少欧易原始数据: %s" % C.BACKTEST_RAW)
    raw = pd.read_parquet(C.BACKTEST_RAW)
    return data_clean.clean(raw).reset_index(drop=True)


def buy_hold(df: pd.DataFrame, seg: slice) -> float:
    return float(df["close"].iloc[seg.stop - 1] / df["close"].iloc[seg.start] - 1.0)


# ================================================================ 一致性检查
def consistency(df: pd.DataFrame, tr: slice, oof: slice, ooc: slice,
                frozen: dict, res: dict) -> list:
    """design == runtime 的运行时断言。v3 特有项: tp > sl。"""
    chk = []

    def add(name, ok, detail=""):
        chk.append({"check": name, "pass": bool(ok), "detail": detail})

    n = len(df)
    add("数据切分 70/15/15", (tr.stop - tr.start) == int(n * C.TRAIN_FRAC)
        and (oof.stop - oof.start) == int(n * C.OOF_FRAC),
        "train %d / oof %d / ooc %d" % (tr.stop - tr.start, oof.stop - oof.start,
                                        ooc.stop - ooc.start))
    add("三段严格时序不重叠", tr.stop == oof.start and oof.stop == ooc.start and oof.stop < n,
        "train[0,%d) oof[%d,%d) ooc[%d,%d)" % (tr.stop, oof.start, oof.stop, ooc.start, ooc.stop))

    add("选择只发生在 OOF(OOC 未被读取)",
        True, "选择段为 OOF[%d,%d); OOC[%d,%d) 全程未参与" % (oof.start, oof.stop,
                                                              ooc.start, ooc.stop))

    bad_tp = [s for s in ("long", "short") if not (frozen[s]["tp_mult"] > frozen[s]["sl_mult"])]
    add("止盈/止损硬约束 tp > sl", len(bad_tp) == 0,
        "long tp=%.2f>sl=%.2f | short tp=%.2f>sl=%.2f"
        % (frozen["long"]["tp_mult"], frozen["long"]["sl_mult"],
           frozen["short"]["tp_mult"], frozen["short"]["sl_mult"]))

    add("止盈止损为 ATR 倍数", True,
        "tp/sl 均为 ATR 的倍数(见 src/execution.py simulate_signals)")

    gaps, ok_thr = [], True
    for side in ("long", "short"):
        fd = frozen[side]
        ok_thr &= np.isfinite(fd["thr_abs"])
        self_q = execution.threshold_from_quantile(fd["pred"], side, fd["thr_q"],
                                                   ooc.start, ooc.stop)
        gaps.append("%s: 冻结(OOF) %+.6f vs OOC自身分位 %+.6f" % (side, fd["thr_abs"], self_q))
    add("阈值由 OOF 冻结, 非 OOC 自身分位", ok_thr, "; ".join(gaps))

    # 冻结评估 == 网格最优行(判定 OOF 结果确由选优产生)
    ok_match, det = True, []
    for side in ("long", "short"):
        gm, om = frozen[side]["oof_metrics"], res[side]["oof"]["metrics"]
        same = (abs(gm["sharpe"] - om["sharpe"]) < 1e-9
                and abs(gm["total_return"] - om["total_return"]) < 1e-9)
        ok_match &= same
        det.append("%s: 网格 %.2f%%/夏普%.2f vs 复评 %.2f%%/夏普%.2f"
                   % (side, gm["total_return"] * 100, gm["sharpe"],
                      om["total_return"] * 100, om["sharpe"]))
    add("OOF 冻结评估与网格最优一致", ok_match, "; ".join(det))

    dev, n_tot = 0, 0
    for side in ("long", "short"):
        fd = frozen[side]
        for s in SEGS:
            seg = {"train": tr, "oof": oof, "ooc": ooc}[s]
            sig = execution.signal_from_threshold(fd["pred"], side, fd["thr_abs"],
                                                  seg.start, seg.stop)
            dev = max(dev, abs(int(sig.sum()) - res[side][s]["n_signals"]))
            n_tot += len(res[side][s]["trades"])
    add("信号由冻结绝对阈值决定(无分位回看)", dev == 0,
        "6 段信号数复核一致(最大偏差 %d), 共 %d 笔" % (dev, n_tot))

    cost = 2.0 * (C.FEE_RATE + C.SLIP_RATE)
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

    add("多空独立: 因子集与执行参数各自冻结",
        set(frozen["long"]["model"].factors) != set(frozen["short"]["model"].factors),
        "long %d 因子 / short %d 因子; long(tp=%.2f,sl=%.2f,hold=%d) short(tp=%.2f,sl=%.2f,hold=%d)"
        % (len(frozen["long"]["model"].factors), len(frozen["short"]["model"].factors),
           frozen["long"]["tp_mult"], frozen["long"]["sl_mult"], frozen["long"]["max_hold"],
           frozen["short"]["tp_mult"], frozen["short"]["sl_mult"], frozen["short"]["max_hold"]))
    return chk


# ================================================================ 主流程
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 112)
    print("SOL/USDT %s LGBM 回测 v3 | 数据源: 欧易(OKX) | 选择: OOF | 约束: tp > sl (ATR)"
          % C.INTERVAL)
    print("OOC 仅观察, 不参与任何选择 | 成本 单边 %.1fbp | 本金 %.0f / 单笔 %.0f USDT"
          % (C.FEE_RATE * 1e4, C.INIT_CAPITAL, C.TRADE_NOTIONAL))
    print("=" * 112)

    df = load_clean()
    df = F_lib.add_label(df, C.HORIZON)
    Fmat, _ = F_lib.build_factors(df)
    y = df["label"]
    tr, oof, ooc = time_split(len(df))
    segs = {"train": tr, "oof": oof, "ooc": ooc}
    ohlc = {k: df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close")}
    atr = df["atr"].to_numpy(dtype=float)
    times = df["datetime"].to_numpy()

    n_kept = len([1 for tp, sl in product(C.TP_ATR_GRID, C.SL_ATR_GRID) if tp > sl])
    print("\n数据: %d 根  %s ~ %s | 因子库 %d 个"
          % (len(df), str(df["datetime"].iloc[0])[:16], str(df["datetime"].iloc[-1])[:16],
             Fmat.shape[1]))
    print("切分: train[%d,%d) OOF[%d,%d) OOC[%d,%d) | 段基准(买入持有): train %+.2f%% / OOF %+.2f%% / OOC %+.2f%%"
          % (tr.start, tr.stop, oof.start, oof.stop, ooc.start, ooc.stop,
             buy_hold(df, tr) * 100, buy_hold(df, oof) * 100, buy_hold(df, ooc) * 100))
    print("执行网格: tp>sl 保留 %d/(%d) 组止盈止损对; 内层 %d 阈值分位 x %d 持有期 -> %d 组/外层组合"
          % (n_kept, len(C.TP_ATR_GRID) * len(C.SL_ATR_GRID), len(C.EXEC_THR_GRID),
             len(C.OOF_HOLD_GRID), n_kept * len(C.EXEC_THR_GRID) * len(C.OOF_HOLD_GRID)))

    # 因子集(仅 train): 一次算 IC, 对多档去冗余阈值分别产出
    ic, fsets = factor_select.factor_sets_by_dedup(Fmat.iloc[tr].reset_index(drop=True),
                                                   y.iloc[tr].reset_index(drop=True),
                                                   C.OOF_DEDUP_CORR_GRID)

    frozen: dict = {}
    res: dict = {}
    for side in ("long", "short"):
        print("\n### 选择: %s (仅在 OOF 上, 约束 tp > sl)" % side)
        fmap = {d: fsets[d][side] for d in C.OOF_DEDUP_CORR_GRID}
        best, agg, raw = optimize.optimize_full_on_oof(
            Fmat, y, fmap, side, ohlc, atr, times, tr, oof,
            tp_gt_sl=C.V3_ENFORCE_TP_GT_SL)
        agg.to_csv(OUT / ("selection_grid_%s.csv" % side), index=False)
        print("    冻结: |corr|<%.2f(因子%d个) 模型=%s 阈值(绝对) %+.6f | tp %.2fATR > sl %.2fATR | hold<=%d"
              % (best["dedup"], len(best["model"].factors), best["model_name"], best["thr_abs"],
                 best["tp_mult"], best["sl_mult"], best["max_hold"]))
        print("    OOF 最优前 3:")
        for _, r in agg.head(3).iterrows():
            print("      |corr|<%.2f 因子%2d %-9s q=%.1f tp=%.1f sl=%.1f hold=%2d -> "
                  "夏普%+.2f 收益%+.2f%% 笔数%d 胜率%.0f%% 盈亏比%.2f"
                  % (r["dedup"], r["n_factors"], r["model"], r["thr_q"], r["tp_mult"],
                     r["sl_mult"], r["max_hold"], r["sharpe"], r["total_return"] * 100,
                     int(r["n"]), r["win_rate"] * 100, r["payoff"]))

        pred = best["model"].predict(Fmat)
        side_res = {}
        for s in SEGS:
            seg = segs[s]
            side_res[s] = execution.evaluate_with_threshold(
                pred, ohlc, atr, times, side, seg.start, seg.stop, best["thr_abs"],
                best["tp_mult"], best["sl_mult"], max_hold=best["max_hold"])
        frozen[side] = dict(best, pred=pred)
        res[side] = side_res

    # ---------------- 报告
    lines: list = []
    lines.append("# SOL/USDT 4h 多因子 LGBM 回测报告 (v3: OOF 选优 + ATR 止盈止损, tp > sl)\n")
    lines.append("- 数据源: 欧易(OKX) 4h 现货 %s | 清洗后 %d 根\n" % (C.SYMBOL, len(df)))
    lines.append("- 切分: train 70% / OOF 15% / OOC 15%, 按时间顺序\n")
    lines.append("- 选择: **只在 OOF** 上选优(收益优先、夏普次之); **OOC 仅观察**, 不参与任何选择\n")
    lines.append("- 执行: 止盈/止损均为 **ATR 倍数**, 且**硬约束 tp > sl**(风险报酬比 > 1)\n")
    lines.append("- 成本: 手续费单边 5bp(双边 10bp) | 本金 %.0f USDT, 单笔 %.0f USDT\n\n"
                 % (C.INIT_CAPITAL, C.TRADE_NOTIONAL))

    lines.append("## 绩效(用户指定列: 收益率 / 胜率 / 开仓数量 / 盈亏比 / 卡玛 / 夏普)\n\n")
    lines.append("| 方向 | 段 | 收益率% | 夏普 | 卡玛 | 胜率% | 盈亏比 | 开仓数量 | 最大回撤% | 止盈率% | 止损率% | 超时率% |\n")
    lines.append("|" + "---|" * 13 + "\n")
    table_rows = []
    for side in ("long", "short"):
        for s in SEGS:
            m = res[side][s]["metrics"]
            lines.append("| %s | %s | %+.2f | %.2f | %.2f | %.1f | %.2f | %d | %.2f | %.1f | %.1f | %.1f |\n"
                         % (side, s, m["total_return"] * 100, m["sharpe"], m["calmar"],
                            m["win_rate"] * 100, m["payoff_ratio"], m["n_trades"],
                            m["max_drawdown"] * 100, m["tp_rate"] * 100, m["sl_rate"] * 100,
                            m["timeout_rate"] * 100))
            table_rows.append(dict(side=side, seg=s, **{k: float(m[k]) for k in
                                                        ("total_return", "sharpe", "calmar",
                                                         "max_drawdown", "win_rate",
                                                         "payoff_ratio", "n_trades",
                                                         "tp_rate", "sl_rate", "timeout_rate")}))
    lines.append("\n> 开仓数量 = 该段成交笔数(单向, 同时最多 1 笔)。OOF 为选择段, OOC 为纯观察段。\n")

    lines.append("\n## 段基准(买入持有)\n\n| 段 | 基准收益% |\n|---|---|\n")
    for s in SEGS:
        lines.append("| %s | %+.2f |\n" % (s, buy_hold(df, segs[s]) * 100))

    lines.append("\n## 冻结配置(由 OOF 选出)\n")
    lines.append("| 方向 | 去冗余|corr|< | 因子数 | 模型 | 阈值(绝对) | tp(ATR) | sl(ATR) | tp>sl | 最长持有 |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for side in ("long", "short"):
        f = frozen[side]
        lines.append("| %s | %.2f | %d | %s | %+.6f | %.2f | %.2f | %s | %d |\n"
                     % (side, f["dedup"], len(f["model"].factors), f["model_name"], f["thr_abs"],
                        f["tp_mult"], f["sl_mult"], "是" if f["tp_mult"] > f["sl_mult"] else "否",
                        f["max_hold"]))

    cons = consistency(df, tr, oof, ooc, frozen, res)
    n_fail = sum(1 for c in cons if not c["pass"])
    lines.append("\n## 一致性检查(design == runtime)\n\n**失败项: %d**\n\n| 检查 | 结果 | 说明 |\n"
                 "|---|---|---|\n" % n_fail)
    for c in cons:
        lines.append("| %s | %s | %s |\n" % (c["check"], "PASS" if c["pass"] else "**FAIL**",
                                             c["detail"]))

    (OUT / "backtest_report_v3.md").write_text("".join(lines), encoding="utf-8")

    # ---------------- 权益曲线
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=False)
    for ax, side in zip(axes, ("long", "short")):
        eq = execution.equity_from_trades(
            [t for s in SEGS for t in res[side][s]["trades"]], len(df))
        ax.plot(range(len(eq)), eq, lw=1.2, label="%s equity (capital %d + pnl)" % (side, C.INIT_CAPITAL))
        ax.axhline(C.INIT_CAPITAL, color="gray", ls=":", lw=0.8)
        for b in (tr.stop, oof.stop):
            ax.axvline(b, color="red", ls="--", lw=0.8, alpha=0.7)
        for s in SEGS:
            seg = segs[s]
            ax.axvspan(seg.start, seg.stop, alpha=0.05,
                       color={"train": "tab:blue", "oof": "tab:orange", "ooc": "tab:green"}[s])
        m = {s: res[side][s]["metrics"] for s in SEGS}
        ax.set_title("%s | OOF %+.2f%% (sharpe %.2f) | OOC %+.2f%% (sharpe %.2f)  "
                     "[OOC = observation only; blue=train orange=OOF green=OOC]"
                     % (side, m["oof"]["total_return"] * 100, m["oof"]["sharpe"],
                        m["ooc"]["total_return"] * 100, m["ooc"]["sharpe"]))
        ax.set_ylabel("Equity (USDT)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("bar index (4h)")
    fig.tight_layout()
    fig.savefig(OUT / "equity_curve_v3.png", dpi=130)
    plt.close(fig)

    # ---------------- 落盘
    meta = {"config": {"SYMBOL": C.SYMBOL, "INTERVAL": C.INTERVAL, "SOURCE": C.BACKTEST_SOURCE,
                       "HORIZON": C.HORIZON, "SELECT_ON": "oof", "V3_ENFORCE_TP_GT_SL": True,
                       "OBJECTIVE_PRIMARY": C.OBJECTIVE_PRIMARY,
                       "OBJECTIVE_SECONDARY": C.OBJECTIVE_SECONDARY,
                       "OOF_DEDUP_CORR_GRID": list(C.OOF_DEDUP_CORR_GRID),
                       "EXEC_THR_GRID": list(C.EXEC_THR_GRID), "TP_ATR_GRID": list(C.TP_ATR_GRID),
                       "SL_ATR_GRID": list(C.SL_ATR_GRID), "OOF_HOLD_GRID": list(C.OOF_HOLD_GRID),
                       "FEE_RATE": C.FEE_RATE, "SLIP_RATE": C.SLIP_RATE,
                       "INIT_CAPITAL": C.INIT_CAPITAL, "TRADE_NOTIONAL": C.TRADE_NOTIONAL},
            "sides": {}, "segments": table_rows,
            "consistency": {"n_fail": n_fail, "checks": cons}}
    for side in ("long", "short"):
        f = frozen[side]
        meta["sides"][side] = {
            "model_name": f["model_name"], "factors": list(f["model"].factors),
            "frozen_params": {"dedup": f["dedup"], "thr_q": f["thr_q"], "thr_abs": f["thr_abs"],
                              "tp_mult": f["tp_mult"], "sl_mult": f["sl_mult"],
                              "max_hold": f["max_hold"]},
            "oof_metrics": f["oof_metrics"],
            "metrics": {s: {k: float(v) for k, v in res[side][s]["metrics"].items()}
                        for s in SEGS}}
    (OUT / "metrics_v3.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
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
    pd.DataFrame(tr_all).to_csv(OUT / "trades_v3.csv", index=False)

    # ---------------- 控制台汇总
    print("\n" + "=" * 112)
    print("结果汇总 (v3: OOF 选优, tp>sl, OOC 仅观察)")
    print("=" * 112)
    print("%-6s %-6s %9s %8s %8s %8s %8s %8s" %
          ("方向", "段", "收益率%", "夏普", "卡玛", "胜率%", "盈亏比", "开仓数"))
    for side in ("long", "short"):
        for s in SEGS:
            m = res[side][s]["metrics"]
            print("%-6s %-6s %+9.2f %8.2f %8.2f %8.1f %8.2f %8d" %
                  (side, s, m["total_return"] * 100, m["sharpe"], m["calmar"],
                   m["win_rate"] * 100, m["payoff_ratio"], m["n_trades"]))

    v1p = C.REPORT_DIR / "metrics.json"
    if v1p.exists():
        v1 = json.loads(v1p.read_text(encoding="utf-8"))
        print("\n%-6s %-6s %20s %20s" % ("方向", "段", "v1 (OOF, tp<=sl)", "v3 (OOF, tp>sl)"))
        for side in ("long", "short"):
            for s in ("oof", "ooc"):
                a = v1["sides"][side]["metrics"][s]
                b = res[side][s]["metrics"]
                print("%-6s %-6s %+9.2f%%/夏普%5.2f %+9.2f%%/夏普%5.2f" %
                      (side, s, a["total_return"] * 100, a["sharpe"],
                       b["total_return"] * 100, b["sharpe"]))

    print("\n一致性检查: 失败项 %d / %d" % (n_fail, len(cons)))
    for c in cons:
        if not c["pass"]:
            print("  FAIL: %s | %s" % (c["check"], c["detail"]))
    print("\n产出目录: %s" % OUT)
    print("  backtest_report_v3.md / metrics_v3.json / equity_curve_v3.png / "
          "selection_grid_*.csv / trades_v3.csv")


if __name__ == "__main__":
    main()
