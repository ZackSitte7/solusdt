#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SOL/USDT 4h 多因子 LGBM 回测 (v2: 抗过拟合选择协议) —— 端到端主脚本。

与 v1 (run_backtest_okx.py) 的唯一区别是**选择协议**, 其余口径(数据/清洗/因子/标签/
成本/资金/指标)完全一致, 保证可比:

  v1: 在 15% 的 OOF 上比较 6480 组 -> OOF 既是选择集又是报告段(选择偏差)
  v2: 在 train 内部的前推折上比较 64 组 -> OOF 与 OOC **都只被读一次**

设计逐条依据见 config.py「抗过拟合选择协议」与 DESIGN.md 第 4 节。运行: python3 run_backtest_v2.py
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
from src import data_clean, execution, protocol                        # noqa: E402
from src import factors as F_lib                                       # noqa: E402
from src.cv import time_split                                          # noqa: E402

OUT = C.REPORT_DIR / "v2"
SEGS = ("train", "oof", "ooc")
pd.set_option("display.width", 220)


def load_clean() -> pd.DataFrame:
    """载入欧易原始数据并清洗(与 v1 同一函数口径)。"""
    if not C.BACKTEST_RAW.exists():
        raise SystemExit("缺少欧易原始数据: %s" % C.BACKTEST_RAW)
    raw = pd.read_parquet(C.BACKTEST_RAW)
    return data_clean.clean(raw).reset_index(drop=True)


def buy_hold(df: pd.DataFrame, seg: slice) -> float:
    return float(df["close"].iloc[seg.stop - 1] / df["close"].iloc[seg.start] - 1.0)


# ================================================================ 一致性检查
def consistency(df: pd.DataFrame, tr: slice, oof: slice, ooc: slice,
                frozen: dict, res: dict, folds_by_side: dict) -> list:
    """design == runtime 的运行时断言(逐条对应 DESIGN.md 第 4 节)。"""
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

    gap = C.HORIZON + C.WF_EMBARGO
    ok_seq, det = True, []
    for side, fl in folds_by_side.items():
        prev_end = None
        for f in fl:
            ok_seq &= (f["train_end"] <= f["val_start"] - gap)
            ok_seq &= (prev_end is None or f["val_start"] >= prev_end)
            det.append("%s折%d train<%d val[%d,%d)" % (side, f["fold"], f["train_end"],
                                                       f["val_start"], f["val_end"]))
            prev_end = f["val_end"]
    add("前推折时序 + purge/embargo(>=%d 根)" % gap, ok_seq, " | ".join(det))

    sel_max = max(f["val_end"] for fl in folds_by_side.values() for f in fl)
    add("选择只发生在 train 内部(OOF/OOC 未被读取)",
        sel_max <= tr.stop and tr.stop <= oof.start,
        "选择段最远到 %d, train 止于 %d, OOF 起于 %d -> OOF/OOC 从未进入选择" %
        (sel_max, tr.stop, oof.start))

    gaps = []
    ok_thr = True
    for side in ("long", "short"):
        fd = frozen[side]
        ok_thr &= np.isfinite(fd["thr_abs"])
        self_q = execution.threshold_from_quantile(fd["pred"], side, fd["thr_q"],
                                                   oof.start, oof.stop)
        gaps.append("%s: 冻结 %+.6f vs OOF自身分位 %+.6f (差 %+.6f)"
                    % (side, fd["thr_abs"], self_q, fd["thr_abs"] - self_q))
    add("阈值来自选择阶段, 非评估段自身分位", ok_thr,
        "; ".join(gaps) + " —— 评估一律使用左侧冻结值")

    dev, n_tot = 0, 0
    for side in ("long", "short"):
        fd = frozen[side]
        for s in SEGS:
            seg = {"train": tr, "oof": oof, "ooc": ooc}[s]
            sig = execution.signal_from_threshold(fd["pred"], side, fd["thr_abs"],
                                                  seg.start, seg.stop)
            dev = max(dev, abs(int(sig.sum()) - res[side][s]["n_signals"]))
            n_tot += len(res[side][s]["trades"])
    add("信号由绝对阈值决定(无分位回看)", dev == 0,
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
        set(frozen["long"]["factors"]) != set(frozen["short"]["factors"]),
        "long %d 因子 / short %d 因子; long(tp=%.1f,sl=%.1f,hold=%d,trail=%.1f) "
        "short(tp=%.1f,sl=%.1f,hold=%d,trail=%.1f)" %
        (len(frozen["long"]["factors"]), len(frozen["short"]["factors"]),
         frozen["long"]["tp_mult"], frozen["long"]["sl_mult"], frozen["long"]["max_hold"],
         frozen["long"]["trail_mult"], frozen["short"]["tp_mult"], frozen["short"]["sl_mult"],
         frozen["short"]["max_hold"], frozen["short"]["trail_mult"]))

    add("因子数受硬上限约束",
        all(len(frozen[s]["factors"]) <= C.MAX_FACTORS_PER_SIDE for s in ("long", "short")),
        "上限 %d; 实际 long=%d short=%d" % (C.MAX_FACTORS_PER_SIDE,
                                            len(frozen["long"]["factors"]),
                                            len(frozen["short"]["factors"])))

    # 跟踪止损只在有利方向上单向移动: 任何一笔止损成交价都不得比初始止损更松
    # (long: exit >= entry - sl*ATR; short: exit <= entry + sl*ATR)。放宽即为未来函数。
    ok_trail, det = True, []
    for side in ("long", "short"):
        fd = frozen[side]
        n_sl, bad = 0, 0
        for s in SEGS:
            for t in res[side][s]["trades"]:
                if t.exit_reason != "sl":
                    continue
                n_sl += 1
                a = float(df["atr"].iloc[t.entry_idx - 1])      # 信号根(entry_idx-1)的 ATR
                init = (t.entry_price - fd["sl_mult"] * a if side == "long"
                        else t.entry_price + fd["sl_mult"] * a)
                if side == "long":
                    bad += int(t.exit_price < init - 1e-9)
                else:
                    bad += int(t.exit_price > init + 1e-9)
        ok_trail &= (bad == 0)
        det.append("%s: 止损成交 %d 笔, 比初始止损更松的 %d 笔%s"
                   % (side, n_sl, bad, "(已启用跟踪)" if fd["trail_mult"] > 0 else "(未启用)"))
    add("跟踪止损单向移动(从不反向放宽)", ok_trail, " | ".join(det))
    return chk


# ================================================================ 主流程
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 112)
    print("SOL/USDT %s LGBM 回测 v2 | 数据源: 欧易(OKX) | 选择协议: train 内部前推折(选择准则 %s)"
          % (C.INTERVAL, C.SELECT_RULE))
    print("OOF 与 OOC 均为纯检验段(各只读一次) | 成本 单边 %.1fbp | 本金 %.0f / 单笔 %.0f USDT"
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

    print("\n数据: %d 根  %s ~ %s | 因子库 %d 个"
          % (len(df), str(df["datetime"].iloc[0])[:16], str(df["datetime"].iloc[-1])[:16],
             Fmat.shape[1]))
    print("切分: train[%d,%d) OOF[%d,%d) OOC[%d,%d) | 段基准(买入持有): train %+.2f%% / OOF %+.2f%% / OOC %+.2f%%"
          % (tr.start, tr.stop, oof.start, oof.stop, ooc.start, ooc.stop,
             buy_hold(df, tr) * 100, buy_hold(df, oof) * 100, buy_hold(df, ooc) * 100))

    frozen: dict = {}
    res: dict = {}
    for side in ("long", "short"):
        print("\n### 选择: %s (仅用 train 内前推折)" % side)
        info = protocol.select_on_wf(Fmat, y, side, ohlc, atr, times, tr)
        print("    " + protocol.disclosure(info))
        print("    冻结: 模型 %s | 因子 %d 个 | 阈值(绝对) %+.6f | tp %.1fATR sl %.1fATR hold %d trail %.1f"
              % (info["model_name"], len(info["factors"]), info["thr_abs"], info["tp_mult"],
                 info["sl_mult"], info["max_hold"], info["trail_mult"]))
        print("    前推折表现(选择依据): 最差折 %+.2f%% | 平均折 %+.2f%% | 平均夏普 %.2f | 折均笔数 %.0f"
              % (info["cv_metrics"]["worst_return"] * 100, info["cv_metrics"]["mean_return"] * 100,
                 info["cv_metrics"]["mean_sharpe"], info["cv_metrics"]["mean_n"]))
        print("    因子筛选: 候选 %d -> 去冗余后 %d -> 上限截断至 %d"
              % (info["factor_report"]["n_candidates"], info["factor_report"]["n_after_dedup"],
                 len(info["factors"])))

        pred = info["model"].predict(Fmat)
        side_res = {}
        for s in SEGS:
            seg = segs[s]
            side_res[s] = execution.evaluate_with_threshold(
                pred, ohlc, atr, times, side, seg.start, seg.stop, info["thr_abs"],
                info["tp_mult"], info["sl_mult"], max_hold=info["max_hold"],
                trail_mult=info["trail_mult"])
        frozen[side] = dict(info, pred=pred)
        res[side] = side_res
        info["raw"].to_csv(OUT / ("selection_grid_%s.csv" % side), index=False)

    # ---------------- 报告
    lines: list = []
    lines.append("# SOL/USDT 4h 多因子 LGBM 回测报告 (v2: 抗过拟合选择协议)\n")
    lines.append("- 数据源: 欧易(OKX) 4h 现货 %s | 清洗后 %d 根\n" % (C.SYMBOL, len(df)))
    lines.append("- 切分: train 70% / OOF 15% / OOC 15%, 按时间顺序\n")
    lines.append("- 选择: **train 内部 %d 个前推折**上比较 %d 个预注册组合/方向, 准则 `%s`\n"
                 % (C.WF_FOLDS, frozen["long"]["n_combos"], C.SELECT_RULE))
    lines.append("- OOF/OOC 未参与任何选择, 各只评估一次; 阈值由选择阶段样本外预测冻结\n")
    lines.append("- 成本: 手续费单边 5bp(双边 10bp) | 本金 %.0f USDT, 单笔 %.0f USDT\n\n"
                 % (C.INIT_CAPITAL, C.TRADE_NOTIONAL))

    head = ("| 方向 | 段 | 收益% | 夏普 | 卡玛 | 最大回撤% | 胜率% | 盈亏比 | 笔数 |"
            " 止盈率% | 止损率% | 超时率% |\n")
    sep = "|" + "---|" * 12 + "\n"
    lines.append(head + sep)
    table_rows = []
    for side in ("long", "short"):
        for s in SEGS:
            m = res[side][s]["metrics"]
            lines.append("| %s | %s | %+.2f | %.2f | %.2f | %.2f | %.1f | %.2f | %d | %.1f | %.1f | %.1f |\n"
                         % (side, s, m["total_return"] * 100, m["sharpe"], m["calmar"],
                            m["max_drawdown"] * 100, m["win_rate"] * 100, m["payoff_ratio"],
                            m["n_trades"], m["tp_rate"] * 100, m["sl_rate"] * 100,
                            m["timeout_rate"] * 100))
            table_rows.append(dict(side=side, seg=s, **{k: float(m[k]) for k in
                                                        ("total_return", "sharpe", "calmar",
                                                         "max_drawdown", "win_rate",
                                                         "payoff_ratio", "n_trades",
                                                         "tp_rate", "sl_rate", "timeout_rate")}))
    lines.append("\n## 段基准(买入持有)\n\n| 段 | 基准收益% |\n|---|---|\n")
    for s in SEGS:
        lines.append("| %s | %+.2f |\n" % (s, buy_hold(df, segs[s]) * 100))

    lines.append("\n## 冻结配置(由 train 前推折选出)\n")
    lines.append("| 方向 | 模型 | 因子数 | 阈值(绝对) | tp(ATR) | sl(ATR) | 最长持有 | 跟踪止损 |\n")
    lines.append("|---|---|---|---|---|---|---|---|\n")
    for side in ("long", "short"):
        f = frozen[side]
        lines.append("| %s | %s | %d | %+.6f | %.1f | %.1f | %d | %s |\n"
                     % (side, f["model_name"], len(f["factors"]), f["thr_abs"], f["tp_mult"],
                        f["sl_mult"], f["max_hold"],
                        ("%.1fATR" % f["trail_mult"]) if f["trail_mult"] > 0 else "关闭"))

    lines.append("\n## 选择质量诊断(判断冻结配置是否值得信任)\n")
    lines.append("选择段 = train 内的前推折。若某方向**最差折仍为负**, 说明该方向在选择段里"
                 "找不到稳健盈利的组合, 其冻结配置只是「最不坏」的一个, OOC 结果不得归因于"
                 "被验证过的边际。\n\n")
    lines.append("| 方向 | 最差折收益% | 平均折收益% | 平均夏普 | 折均笔数 | 诊断 |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for side in ("long", "short"):
        cm = frozen[side]["cv_metrics"]
        ok = cm["worst_return"] > 0
        lines.append("| %s | %+.2f | %+.2f | %.2f | %.0f | %s |\n"
                     % (side, cm["worst_return"] * 100, cm["mean_return"] * 100,
                        cm["mean_sharpe"], cm["mean_n"],
                        "选择段每折为正 -> 边际经前推验证" if ok
                        else "**选择段最优组合仍为负 -> 无被验证的边际**"))
    lines.append("\n> 制度提示: train 段基准为 %+.0f%%(单边牛市), 空头在该制度下天然不利; "
                 "选择段与 OOC 段的制度不同, 这本身就是 v2 尚未解决的问题。\n"
                 % (buy_hold(df, tr) * 100))

    lines.append("\n## OOC 稳健性(仅观察; 分块自助 %d 次, 块长 %d 笔)\n" % (C.BOOT_N, C.BOOT_BLOCK))
    lines.append("| 方向 | OOC 净盈亏 | 95% 区间 | 区间含 0? | P(总盈亏>0) |\n|---|---|---|---|---|\n")
    boot = {}
    for side in ("long", "short"):
        pnl = np.array([t.pnl_usdt for t in res[side]["ooc"]["trades"]], dtype=float)
        b = protocol.block_bootstrap_ci(pnl)
        boot[side] = b
        lines.append("| %s | %+.2f | [%+.2f, %+.2f] | %s | %.1f%% |\n"
                     % (side, b["total"], b["lo"], b["hi"],
                        "是" if (b["lo"] <= 0 <= b["hi"]) else "否", b["p_pos"] * 100))

    lines.append("\n## 前推折明细(选择依据)\n")
    for side in ("long", "short"):
        lines.append("\n**%s**\n\n| 折 | 训练止 | 验证区间 | 训练根数 | 验证根数 | 因子数 |\n"
                     "|---|---|---|---|---|---|\n" % side)
        for f in frozen[side]["folds"]:
            lines.append("| %d | %d | [%d, %d) | %d | %d | %d |\n"
                         % (f["fold"], f["train_end"], f["val_start"], f["val_end"],
                            f["n_train"], f["n_val"], f["n_factors"]))
        lines.append("\n各折因子集(体现因子也在前推):\n")
        for f in frozen[side]["folds"]:
            lines.append("- 折%d: %s\n" % (f["fold"], f["factors"]))

    cons = consistency(df, tr, oof, ooc, frozen, res,
                       {s: frozen[s]["folds"] for s in ("long", "short")})
    n_fail = sum(1 for c in cons if not c["pass"])
    lines.append("\n## 一致性检查(design == runtime)\n\n**失败项: %d**\n\n| 检查 | 结果 | 说明 |\n"
                 "|---|---|---|\n" % n_fail)
    for c in cons:
        lines.append("| %s | %s | %s |\n" % (c["check"], "PASS" if c["pass"] else "**FAIL**",
                                             c["detail"]))

    (OUT / "backtest_report_v2.md").write_text("".join(lines), encoding="utf-8")

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
    fig.savefig(OUT / "equity_curve_v2.png", dpi=130)
    plt.close(fig)

    # ---------------- 落盘
    meta = {"config": {"SYMBOL": C.SYMBOL, "INTERVAL": C.INTERVAL, "SOURCE": C.BACKTEST_SOURCE,
                       "HORIZON": C.HORIZON, "SELECT_RULE": C.SELECT_RULE, "WF_FOLDS": C.WF_FOLDS,
                       "MAX_FACTORS_PER_SIDE": C.MAX_FACTORS_PER_SIDE, "FACTOR_RANK": C.FACTOR_RANK,
                       "DEDUP_MAX_CORR_V2": C.DEDUP_MAX_CORR_V2,
                       "PROTO_MODELS": list(C.PROTO_MODELS), "PROTO_THR_GRID": list(C.PROTO_THR_GRID),
                       "PROTO_TP_GRID": list(C.PROTO_TP_GRID), "PROTO_SL_GRID": list(C.PROTO_SL_GRID),
                       "PROTO_HOLD_GRID": list(C.PROTO_HOLD_GRID),
                       "PROTO_TRAIL_GRID": list(C.PROTO_TRAIL_GRID),
                       "FEE_RATE": C.FEE_RATE, "SLIP_RATE": C.SLIP_RATE,
                       "INIT_CAPITAL": C.INIT_CAPITAL, "TRADE_NOTIONAL": C.TRADE_NOTIONAL},
            "sides": {}, "segments": table_rows, "bootstrap_ooc": boot,
            "consistency": {"n_fail": n_fail, "checks": cons}}
    for side in ("long", "short"):
        f = frozen[side]
        meta["sides"][side] = {
            "model_name": f["model_name"], "factors": f["factors"],
            "frozen_params": {"thr_q": f["thr_q"], "thr_abs": f["thr_abs"], "tp_mult": f["tp_mult"],
                              "sl_mult": f["sl_mult"], "max_hold": f["max_hold"],
                              "trail_mult": f["trail_mult"]},
            "cv_metrics": f["cv_metrics"], "folds": f["folds"],
            "factor_report": {"n_candidates": f["factor_report"]["n_candidates"],
                              "n_after_dedup": f["factor_report"]["n_after_dedup"],
                              "dropped_by_ic": f["factor_report"]["dropped_by_ic"],
                              "dropped_by_icir": f["factor_report"]["dropped_by_icir"],
                              "dropped_by_cap": f["factor_report"]["dropped_by_cap"]},
            "metrics": {s: {k: float(v) for k, v in res[side][s]["metrics"].items()}
                        for s in SEGS}}
    (OUT / "metrics_v2.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
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
    pd.DataFrame(tr_all).to_csv(OUT / "trades_v2.csv", index=False)

    # ---------------- v1 对照
    print("\n" + "=" * 112)
    print("结果汇总 (OOC 仅观察, 不参与任何选择)")
    print("=" * 112)
    print("%-6s %-6s %9s %8s %8s %9s %8s %7s %7s" %
          ("方向", "段", "收益%", "夏普", "卡玛", "回撤%", "胜率%", "盈亏比", "笔数"))
    for side in ("long", "short"):
        for s in SEGS:
            m = res[side][s]["metrics"]
            print("%-6s %-6s %+9.2f %8.2f %8.2f %9.2f %8.1f %7.2f %7d" %
                  (side, s, m["total_return"] * 100, m["sharpe"], m["calmar"],
                   m["max_drawdown"] * 100, m["win_rate"] * 100, m["payoff_ratio"],
                   m["n_trades"]))

    v1p = C.REPORT_DIR / "metrics.json"
    if v1p.exists():
        v1 = json.loads(v1p.read_text(encoding="utf-8"))
        print("\n%-6s %-6s %18s %18s" % ("方向", "段", "v1 (OOF 选优)", "v2 (train 前推折选优)"))
        for side in ("long", "short"):
            for s in SEGS:
                a = v1["sides"][side]["metrics"][s]
                b = res[side][s]["metrics"]
                print("%-6s %-6s %+8.2f%%/夏普%5.2f %+8.2f%%/夏普%5.2f" %
                      (side, s, a["total_return"] * 100, a["sharpe"],
                       b["total_return"] * 100, b["sharpe"]))

    print("\n一致性检查: 失败项 %d / %d" % (n_fail, len(cons)))
    for c in cons:
        if not c["pass"]:
            print("  FAIL: %s | %s" % (c["check"], c["detail"]))
    print("\n产出目录: %s" % OUT)
    print("  backtest_report_v2.md / metrics_v2.json / equity_curve_v2.png / "
          "selection_grid_*.csv / trades_v2.csv")


if __name__ == "__main__":
    main()
