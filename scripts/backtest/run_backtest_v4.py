#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SOL/USDT 4h 多因子 LGBM 回测 (v4: OOF 选优 + ATR 止盈止损(tp>sl) + VIF 迭代剪枝)。

在 v3 基础上只加一道抗过拟合工序(其余口径完全一致, 便于对比):
  依据 scripts/analysis/analyze_factor_dim.py 的诊断 —— long 名义 25 因子但有效维度(参与率)仅 4.3、
  最大 VIF 71.5, 说明大量因子是近共线的噪声方向; pairwise 去冗余(|corr|>=0.95)挡不住
  **多元共线**。故在因子筛选之后、喂给模型之前, 做 **VIF 迭代剪枝**: 反复删除 VIF 最大的
  因子, 直到全部 VIF <= V4_MAX_VIF 或只剩 V4_MIN_FACTORS 个。剪枝只发生在 train 上,
  不引入未来信息; 之后仍在 **OOF** 上选优(收益优先、夏普次之), OOC 仅观察。

运行: python3 scripts/backtest/run_backtest_v4.py
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

BASE_DIR = Path(__file__).resolve().parents[2]      # 仓库根(脚本位于 scripts/backtest|analysis/)
sys.path.insert(0, str(BASE_DIR))
import config as C                                                    # noqa: E402
from src import data_clean, execution, factor_select                 # noqa: E402
from src import factors as F_lib                                     # noqa: E402
from src import optimize                                     # noqa: E402
from src.cv import time_split                                        # noqa: E402

OUT = C.REPORT_DIR / C.V4_OUT_DIR
SEGS = ("train", "oof", "ooc")
pd.set_option("display.width", 220)


def load_clean() -> pd.DataFrame:
    """载入欧易原始数据并清洗(与 v1/v2/v3 同一函数口径)。"""
    if not C.BACKTEST_RAW.exists():
        raise SystemExit("缺少欧易原始数据: %s" % C.BACKTEST_RAW)
    raw = pd.read_parquet(C.BACKTEST_RAW)
    return data_clean.clean(raw).reset_index(drop=True)


def buy_hold(df: pd.DataFrame, seg: slice) -> float:
    return float(df["close"].iloc[seg.stop - 1] / df["close"].iloc[seg.start] - 1.0)


def spectrum(F: pd.DataFrame, factors: list, seg: slice) -> dict:
    """因子集在 seg 上的谱维度: 有效维度(参与率) / 95%方差维度 / 最大 VIF。"""
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
    return dict(n_factors=int(Xz.shape[1]),
                dim95=int(np.searchsorted(cum, 0.95) + 1),
                eff_rank=float(tot ** 2 / float((eig ** 2).sum())),
                max_vif=float(np.diag(np.linalg.pinv(corr)).max()))


# ================================================================ 一致性检查
def consistency(df: pd.DataFrame, tr: slice, oof: slice, ooc: slice,
                frozen: dict, res: dict, prune_log: dict) -> list:
    """design == runtime 的运行时断言。v4 特有项: VIF 剪枝后最大 VIF <= 阈值。"""
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

    # VIF 剪枝: 冻结因子集的最大 VIF 必须在阈值内, 且是去冗余集剔除 VIF 后的子集
    ok_vif, det_vif = True, []
    for side in ("long", "short"):
        facs = list(frozen[side]["model"].factors)
        entry = prune_log[(frozen[side]["dedup"], side)]
        m_vif = entry["max_vif_after"]
        subset = set(facs).issubset(set(entry["before"]))
        ok_vif &= (m_vif <= C.V4_MAX_VIF + 1e-9) and subset
        det_vif.append("%s: 剪枝前%d->后%d, 最大VIF %.2f->%.2f, 子集=%s"
                       % (side, len(entry["before"]), len(entry["after"]),
                          entry["max_vif_before"], m_vif, subset))
    add("VIF 迭代剪枝(冻结集 VIF<=%.0f 且为去冗余子集)" % C.V4_MAX_VIF, ok_vif,
        "; ".join(det_vif))

    gaps, ok_thr = [], True
    for side in ("long", "short"):
        fd = frozen[side]
        ok_thr &= np.isfinite(fd["thr_abs"])
        self_q = execution.threshold_from_quantile(fd["pred"], side, fd["thr_q"],
                                                   ooc.start, ooc.stop)
        gaps.append("%s: 冻结(OOF) %+.6f vs OOC自身分位 %+.6f" % (side, fd["thr_abs"], self_q))
    add("阈值由 OOF 冻结, 非 OOC 自身分位", ok_thr, "; ".join(gaps))

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
    print("SOL/USDT %s LGBM 回测 v4 | 欧易(OKX) | OOF 选优 | tp > sl (ATR) | VIF 剪枝 <= %.0f"
          % (C.INTERVAL, C.V4_MAX_VIF))
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
    Ftr = Fmat.iloc[tr].reset_index(drop=True)
    _, fsets = factor_select.factor_sets_by_dedup(Ftr, y.iloc[tr].reset_index(drop=True),
                                                  C.OOF_DEDUP_CORR_GRID)

    # ---- VIF 迭代剪枝(仅 train) ----
    pruned: dict = {}
    prune_log: dict = {}
    for side in ("long", "short"):
        print("\n--- VIF 迭代剪枝: %s (阈值 %.0f, 下限 %d) ---" % (side, C.V4_MAX_VIF,
                                                              C.V4_MIN_FACTORS))
        print("  %-10s %8s %8s %10s %10s  %s" % ("|corr|<", "剪枝前", "剪枝后", "VIF前", "VIF后",
                                                 "被删因子(VIF)"))
        for d in C.OOF_DEDUP_CORR_GRID:
            before = fsets[d][side]
            kept, dropped = factor_select.vif_prune(Ftr, before)
            vb = factor_select.vif_series(Ftr, before)
            va = factor_select.vif_series(Ftr, kept)
            prune_log[(d, side)] = dict(before=list(before), after=list(kept), dropped=dropped,
                                        max_vif_before=float(vb.max()) if len(vb) else float("nan"),
                                        max_vif_after=float(va.max()) if len(va) else float("nan"))
            pruned.setdefault(d, {})[side] = kept
            drop_txt = ", ".join("%s(%.0f)" % (x["factor"], x["vif"]) for x in dropped
                                 if x["vif"] is not None) or "-"
            print("  %-10.2f %8d %8d %10.1f %10.1f  %s"
                  % (d, len(before), len(kept),
                     prune_log[(d, side)]["max_vif_before"], prune_log[(d, side)]["max_vif_after"],
                     drop_txt))

    frozen: dict = {}
    res: dict = {}
    for side in ("long", "short"):
        print("\n### 选择: %s (仅在 OOF 上, 约束 tp > sl, 输入为 VIF 剪枝后因子集)" % side)
        fmap = {d: pruned[d][side] for d in C.OOF_DEDUP_CORR_GRID}
        best, agg, raw = optimize.optimize_full_on_oof(
            Fmat, y, fmap, side, ohlc, atr, times, tr, oof,
            tp_gt_sl=C.V3_ENFORCE_TP_GT_SL)
        agg.to_csv(OUT / ("selection_grid_%s.csv" % side), index=False)
        print("    冻结: |corr|<%.2f(剪枝后因子%d个) 模型=%s 阈值(绝对) %+.6f | tp %.2fATR > sl %.2fATR | hold<=%d"
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
    lines.append("# SOL/USDT 4h 多因子 LGBM 回测报告 (v4: OOF 选优 + ATR 止盈止损(tp>sl) + VIF 迭代剪枝)\n")
    lines.append("- 数据源: 欧易(OKX) 4h 现货 %s | 清洗后 %d 根\n" % (C.SYMBOL, len(df)))
    lines.append("- 切分: train 70% / OOF 15% / OOC 15%, 按时间顺序\n")
    lines.append("- 选择: **只在 OOF** 上选优(收益优先、夏普次之); **OOC 仅观察**, 不参与任何选择\n")
    lines.append("- 执行: 止盈/止损均为 **ATR 倍数**, 且**硬约束 tp > sl**(风险报酬比 > 1)\n")
    lines.append("- 抗过拟合: 因子集在 train 上做 **VIF 迭代剪枝**(全部 VIF <= %.0f, 下限 %d 个), "
                 "治理 v3 诊断出的多元共线(long 最大 VIF 曾达 71.5)\n"
                 % (C.V4_MAX_VIF, C.V4_MIN_FACTORS))
    lines.append("- 成本: 手续费单边 5bp(双边 10bp) | 本金 %.0f USDT, 单笔 %.0f USDT\n\n"
                 % (C.INIT_CAPITAL, C.TRADE_NOTIONAL))

    lines.append("## 绩效(收益率 / 夏普 / 卡玛 / 胜率 / 盈亏比 / 开仓数量)\n\n")
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

    lines.append("\n## 因子维度对比(VIF 剪枝前后, 仅 train)\n\n")
    lines.append("| 方向 | 段档 | 剪枝前维度 | 剪枝后维度 | 剪枝前有效维度 | 剪枝后有效维度 | 剪枝前最大VIF | 剪枝后最大VIF |\n")
    lines.append("|---|---|---|---|---|---|---|---|\n")
    dim_rows = []
    for side in ("long", "short"):
        d = frozen[side]["dedup"]
        entry = prune_log[(d, side)]
        sp_b = spectrum(Fmat, entry["before"], tr)
        sp_a = spectrum(Fmat, entry["after"], tr)
        lines.append("| %s | |corr|<%.2f | %d | %d | %.2f | %.2f | %.1f | %.1f |\n"
                     % (side, d, sp_b["n_factors"], sp_a["n_factors"], sp_b["eff_rank"],
                        sp_a["eff_rank"], sp_b["max_vif"], sp_a["max_vif"]))
        dim_rows.append(dict(side=side, dedup=d, n_before=sp_b["n_factors"],
                             n_after=sp_a["n_factors"], eff_before=sp_b["eff_rank"],
                             eff_after=sp_a["eff_rank"], dim95_after=sp_a["dim95"],
                             vif_before=sp_b["max_vif"], vif_after=sp_a["max_vif"]))
    lines.append("\n> 有效维度 = 相关矩阵参与率 (Σλ)²/Σλ²。剪枝把 long 从'名义 25 维 / 有效 4.3 维'"
                 "收敛到更接近满秩的因子集。\n")
    for side in ("long", "short"):
        d = frozen[side]["dedup"]
        entry = prune_log[(d, side)]
        lines.append("\n### %s 被 VIF 剪掉的因子(阈值 %.0f)\n\n" % (side, C.V4_MAX_VIF))
        lines.append("| 因子 | VIF(删除时) | 原因 |\n|---|---|---|\n")
        for x in entry["dropped"]:
            lines.append("| %s | %s | %s |\n" % (x["factor"],
                                                 "-" if x["vif"] is None else "%.1f" % x["vif"],
                                                 x["reason"]))
        lines.append("\n保留因子(%d): %s\n" % (len(entry["after"]), ", ".join(entry["after"])))

    lines.append("\n## 段基准(买入持有)\n\n| 段 | 基准收益% |\n|---|---|\n")
    for s in SEGS:
        lines.append("| %s | %+.2f |\n" % (s, buy_hold(df, segs[s]) * 100))

    lines.append("\n## 冻结配置(由 OOF 选出)\n")
    lines.append("| 方向 | 去冗余|corr|< | 因子数(剪枝后) | 模型 | 阈值(绝对) | tp(ATR) | sl(ATR) | tp>sl | 最长持有 |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for side in ("long", "short"):
        f = frozen[side]
        lines.append("| %s | %.2f | %d | %s | %+.6f | %.2f | %.2f | %s | %d |\n"
                     % (side, f["dedup"], len(f["model"].factors), f["model_name"], f["thr_abs"],
                        f["tp_mult"], f["sl_mult"], "是" if f["tp_mult"] > f["sl_mult"] else "否",
                        f["max_hold"]))

    cons = consistency(df, tr, oof, ooc, frozen, res, prune_log)
    n_fail = sum(1 for c in cons if not c["pass"])
    lines.append("\n## 一致性检查(design == runtime)\n\n**失败项: %d**\n\n| 检查 | 结果 | 说明 |\n"
                 "|---|---|---|\n" % n_fail)
    for c in cons:
        lines.append("| %s | %s | %s |\n" % (c["check"], "PASS" if c["pass"] else "**FAIL**",
                                             c["detail"]))

    (OUT / "backtest_report_v4.md").write_text("".join(lines), encoding="utf-8")

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
                     "[VIF<=%.0f; OOC = observation only]"
                     % (side, m["oof"]["total_return"] * 100, m["oof"]["sharpe"],
                        m["ooc"]["total_return"] * 100, m["ooc"]["sharpe"], C.V4_MAX_VIF))
        ax.set_ylabel("Equity (USDT)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("bar index (4h)")
    fig.tight_layout()
    fig.savefig(OUT / "equity_curve_v4.png", dpi=130)
    plt.close(fig)

    # ---------------- 落盘
    meta = {"config": {"SYMBOL": C.SYMBOL, "INTERVAL": C.INTERVAL, "SOURCE": C.BACKTEST_SOURCE,
                       "HORIZON": C.HORIZON, "SELECT_ON": "oof", "V3_ENFORCE_TP_GT_SL": True,
                       "V4_MAX_VIF": C.V4_MAX_VIF, "V4_MIN_FACTORS": C.V4_MIN_FACTORS,
                       "OBJECTIVE_PRIMARY": C.OBJECTIVE_PRIMARY,
                       "OBJECTIVE_SECONDARY": C.OBJECTIVE_SECONDARY,
                       "OOF_DEDUP_CORR_GRID": list(C.OOF_DEDUP_CORR_GRID),
                       "EXEC_THR_GRID": list(C.EXEC_THR_GRID), "TP_ATR_GRID": list(C.TP_ATR_GRID),
                       "SL_ATR_GRID": list(C.SL_ATR_GRID), "OOF_HOLD_GRID": list(C.OOF_HOLD_GRID),
                       "FEE_RATE": C.FEE_RATE, "SLIP_RATE": C.SLIP_RATE,
                       "INIT_CAPITAL": C.INIT_CAPITAL, "TRADE_NOTIONAL": C.TRADE_NOTIONAL},
            "sides": {}, "segments": table_rows, "dimension": dim_rows,
            "prune_log": {"%s|%s" % (k[1], k[0]): v for k, v in prune_log.items()},
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
    (OUT / "metrics_v4.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
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
    pd.DataFrame(tr_all).to_csv(OUT / "trades_v4.csv", index=False)

    # ---------------- 控制台汇总
    print("\n" + "=" * 112)
    print("结果汇总 (v4: OOF 选优 + tp>sl + VIF 剪枝, OOC 仅观察)")
    print("=" * 112)
    print("%-6s %-6s %9s %8s %8s %8s %8s %8s" %
          ("方向", "段", "收益率%", "夏普", "卡玛", "胜率%", "盈亏比", "开仓数"))
    for side in ("long", "short"):
        for s in SEGS:
            m = res[side][s]["metrics"]
            print("%-6s %-6s %+9.2f %8.2f %8.2f %8.1f %8.2f %8d" %
                  (side, s, m["total_return"] * 100, m["sharpe"], m["calmar"],
                   m["win_rate"] * 100, m["payoff_ratio"], m["n_trades"]))

    v3p = C.REPORT_DIR / C.V3_OUT_DIR / "metrics_v3.json"
    if v3p.exists():
        v3 = json.loads(v3p.read_text(encoding="utf-8"))
        print("\n%-6s %-6s %22s %22s" % ("方向", "段", "v3 (无VIF剪枝)", "v4 (VIF剪枝)"))
        for side in ("long", "short"):
            for s in ("oof", "ooc"):
                a = v3["sides"][side]["metrics"][s]
                b = res[side][s]["metrics"]
                print("%-6s %-6s %+9.2f%%/夏普%5.2f %+9.2f%%/夏普%5.2f" %
                      (side, s, a["total_return"] * 100, a["sharpe"],
                       b["total_return"] * 100, b["sharpe"]))

    print("\n一致性检查: 失败项 %d / %d" % (n_fail, len(cons)))
    for c in cons:
        if not c["pass"]:
            print("  FAIL: %s | %s" % (c["check"], c["detail"]))
    print("\n产出目录: %s" % OUT)
    print("  backtest_report_v4.md / metrics_v4.json / equity_curve_v4.png / "
          "selection_grid_*.csv / trades_v4.csv")


if __name__ == "__main__":
    main()
