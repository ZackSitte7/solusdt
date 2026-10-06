#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SOL/USDT 4h 多因子 LGBM 回测 (v15: 放开 long 止盈上限 + 冻结 long 门控)。

基线 = **v14**(数据 >= 2021-10-01 + long 趋势因子 + 真实 OOF 段「收益锚 + 夏普择优」)。
依据 v14 的 OOF 归因与两组 OOF 实验(tools/exp_v15_trendgate.py / tools/exp_v15_tp_ext.py),
v15 只改 long 两处:
  1. **门控冻结**: 把 long 的制度门控从 v14 的 4 种扩到 9 种(ADX/效率比/趋势质量等)后,
     没有一种门控能在 OOF 上超过「不门控(none)」-> long 门控**冻结为 none**, 不再当旋钮;
  2. **止盈放开**: v14 的 tp 网格上界 6.0 被 4 个制度一致顶到(空间在决定解, 而非数据),
     v15 把 long 的 tp 网格外扩到 12.0(同时在 tp>sl 约束下放开 sl 下界到 1.0)。
     放开后 long 的 OOF 收益 +9.18% -> +9.48%、夏普 8.59 -> 9.64, 选中 tp=10/sl=2/hold=72。
**short 逐位不变**(仍用 v13/v14 口径与网格, 并排除 long 专用因子); OOC 全程只观察。

运行: python3 scripts/backtest/run_backtest_v15.py
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
from src import data_clean, execution, factor_select, regime          # noqa: E402
from src import factors as F_lib                                     # noqa: E402
from src import optimize                                     # noqa: E402
from src.cv import time_split                                        # noqa: E402

OUT = C.REPORT_DIR / C.V15_OUT_DIR
SEGS = ("train", "oof", "ooc")
pd.set_option("display.width", 240)


def load_clean() -> pd.DataFrame:
    """载入欧易原始数据 -> 清洗 -> 对齐 DATA_START(与 v6~v14 同口径)。"""
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
    return dict(n_factors=int(Xz.shape[1]), dim95=int(np.searchsorted(cum, 0.95) + 1),
                eff_rank=float(tot ** 2 / float((eig ** 2).sum())),
                max_vif=float(np.diag(np.linalg.pinv(corr)).max()))


# ================================================================ 一致性检查
def consistency(df: pd.DataFrame, tr: slice, oof: slice, ooc: slice,
                frozen: dict, res: dict, prune_log: dict, masks: dict) -> list:
    """design == runtime 的运行时断言。v15 特有项: long 门控冻结 + tp 上界放开。"""
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
    add("选择只发生在 OOF(OOC 未被读取)",
        True, "选择段为 OOF[%d,%d); OOC[%d,%d) 全程未参与" % (oof.start, oof.stop,
                                                              ooc.start, ooc.stop))

    bad_tp = [s for s in ("long", "short") if not (frozen[s]["tp_mult"] > frozen[s]["sl_mult"])]
    add("止盈/止损硬约束 tp > sl", len(bad_tp) == 0,
        "long tp=%.2f>sl=%.2f | short tp=%.2f>sl=%.2f"
        % (frozen["long"]["tp_mult"], frozen["long"]["sl_mult"],
           frozen["short"]["tp_mult"], frozen["short"]["sl_mult"]))

    # v15 设计=运行: long 门控冻结为 none, 且 tp 网格上界已放开到 12
    ok_v15, det_v15 = True, []
    ok_v15 &= (frozen["long"]["regime_rule"] == "none")
    ok_v15 &= (C.V15_LONG_REGIME_GRID == ("none",))
    ok_v15 &= (C.V15_LONG_TP_GRID[-1] == 12.0 and frozen["long"]["tp_mult"] in C.V15_LONG_TP_GRID)
    ok_v15 &= (frozen["short"]["regime_rule"] in C.V5_REGIME_GRID)
    det_v15.append("long 门控=%s(冻结网格 %s), long tp=%.1f(网格上界 %.1f), short 门控=%s(未改动)"
                   % (frozen["long"]["regime_rule"], list(C.V15_LONG_REGIME_GRID),
                      frozen["long"]["tp_mult"], C.V15_LONG_TP_GRID[-1],
                      frozen["short"]["regime_rule"]))
    add("v15 long 门控冻结=none 且 tp 上界放开到 12(short 不变)", ok_v15, "; ".join(det_v15))

    # 制度门控一致性: 冻结信号 == 阈值信号 ∩ 制度掩码, 且门控确实筛掉了信号
    ok_reg, det_reg = True, []
    for side in ("long", "short"):
        fd = frozen[side]
        rule = fd["regime_rule"]
        mask = masks[rule][side]
        m_sig = 0
        for s in SEGS:
            seg = {"train": tr, "oof": oof, "ooc": ooc}[s]
            sig_plain = execution.signal_from_threshold(fd["pred"], side, fd["thr_abs"],
                                                        seg.start, seg.stop)
            sig_gated = execution.signal_from_threshold(fd["pred"], side, fd["thr_abs"],
                                                        seg.start, seg.stop, regime=mask)
            if not np.array_equal(sig_gated, sig_plain & mask):
                ok_reg = False
            m_sig = max(m_sig, abs(int(sig_gated.sum()) - res[side][s]["n_signals"]))
        sig_all = int(execution.signal_from_threshold(fd["pred"], side, fd["thr_abs"],
                                                      ooc.start, ooc.stop).sum())
        sig_g = int(execution.signal_from_threshold(fd["pred"], side, fd["thr_abs"],
                                                    ooc.start, ooc.stop, regime=mask).sum())
        ok_reg &= (m_sig == 0) and ((rule == "none") or (sig_g <= sig_all))
        det_reg.append("%s: rule=%s, OOC 门控前信号 %d -> 门控后 %d, 冻结复核偏差 %d"
                       % (side, rule, sig_all, sig_g, m_sig))
    add("制度门控生效且与冻结评估一致", ok_reg, "; ".join(det_reg))

    ok_vif, det_vif = True, []
    for side in ("long", "short"):
        entry = prune_log[(frozen[side]["dedup"], side)]
        m_vif = entry["max_vif_after"]
        subset = set(frozen[side]["model"].factors).issubset(set(entry["before"]))
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

    # 制度掩码因果性: 用截断到 [0,idx] 的数据重算, 掩码在 idx 处的取值必须不变
    ok_causal, det_causal = True, []
    for rule in C.V5_REGIME_GRID:
        if rule == "none":
            continue
        for idx in (500, 3000, 7000, 9000, 11000, 12500):
            if idx >= n:
                continue
            sub = regime.regime_masks(df.iloc[:idx + 1].reset_index(drop=True), rule)
            for side in ("long", "short"):
                if bool(sub[side][idx]) != bool(masks[rule][side][idx]):
                    ok_causal = False
        det_causal.append(rule)
    add("制度掩码仅用过去信息(截断后取值不变)", ok_causal, "规则: " + ", ".join(det_causal))

    add("多空独立: 因子集/执行参数/制度规则各自冻结",
        set(frozen["long"]["model"].factors) != set(frozen["short"]["model"].factors),
        "long(%s, %d因子, tp=%.2f,sl=%.2f) | short(%s, %d因子, tp=%.2f,sl=%.2f)"
        % (frozen["long"]["regime_rule"], len(frozen["long"]["model"].factors),
           frozen["long"]["tp_mult"], frozen["long"]["sl_mult"],
           frozen["short"]["regime_rule"], len(frozen["short"]["model"].factors),
           frozen["short"]["tp_mult"], frozen["short"]["sl_mult"]))

    # v15 选优规则可复现: 在"各(long)门控自身最优"的集合上复核 收益锚 + 夏普择优
    ok_anchor, det_anchor = True, []
    for side in ("long", "short"):
        rules = C.V15_LONG_REGIME_GRID if side == "long" else C.V5_REGIME_GRID
        cs = pd.read_csv(OUT / ("selection_grid_%s.csv" % side))
        wins = pd.DataFrame([cs[cs["regime"] == r].iloc[0] for r in rules
                             if not cs[cs["regime"] == r].empty])
        anchor = float(wins["total_return"].max())
        band = wins[wins["total_return"] >= anchor - C.V15_RET_DROP - 1e-12]
        band_best = band.sort_values(["sharpe", "total_return"], ascending=False).iloc[0]
        chosen = frozen[side]["oof_metrics"]
        same = (abs(float(band_best["total_return"]) - chosen["total_return"]) < 1e-9
                and abs(float(band_best["sharpe"]) - chosen["sharpe"]) < 1e-9)
        ok_anchor &= same and (chosen["total_return"] >= anchor - C.V15_RET_DROP - 1e-12)
        det_anchor.append("%s: 收益锚 %+.4f, 让步带(%+.4f)内最高夏普 %+.3f, 选中 收益 %+.4f/夏普 %+.3f"
                          % (side, anchor, anchor - C.V15_RET_DROP, float(band["sharpe"].max()),
                             chosen["total_return"], chosen["sharpe"]))
    add("v15 收益锚+夏普择优可复现(收益让步 <= %.4f 内取夏普最高)" % C.V15_RET_DROP,
        ok_anchor, "; ".join(det_anchor))
    return chk


# ================================================================ 主流程
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 116)
    print("SOL/USDT %s LGBM 回测 v15 | 欧易(OKX) | 数据 >= %s | long: 门控冻结 none + tp 上界 6->12 | "
          "short: v14 口径不变" % (C.INTERVAL, C.DATA_START))
    print("选择: 真实 OOF 段收益锚 + 让步<=%.2fpp 内取夏普最高 | OOC 仅观察, 不参与任何选择 | "
          "成本 单边 %.1fbp | 本金 %.0f / 单笔 %.0f USDT"
          % (C.V15_RET_DROP * 100, C.FEE_RATE * 1e4, C.INIT_CAPITAL, C.TRADE_NOTIONAL))
    print("=" * 116)

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

    masks = {rule: regime.regime_masks(df, rule) for rule in C.V5_REGIME_GRID}
    for rule in C.V5_REGIME_GRID:
        lu, su = int(masks[rule]["long"].sum()), int(masks[rule]["short"].sum())
        print("制度 %-13s 允许开仓 bar 数: long %5d / short %5d | long 是否采用: %s"
              % (rule, lu, su, "是(冻结)" if rule in C.V15_LONG_REGIME_GRID else "否"))

    # 因子集(仅 train) + VIF 迭代剪枝(仅 train, 同 v4)
    # long 用再外扩的去冗余网格; v13/v14 的 long 专用因子(含趋势因子)显式挡在 short 之外,
    # 保证空头口径与 v13/v14 基线逐位不变。
    Ftr = Fmat.iloc[tr].reset_index(drop=True)
    y_tr = y.iloc[tr].reset_index(drop=True)
    long_only = list(C.V15_LONG_FACTORS)
    dedup_by_side = {"long": C.V15_LONG_DEDUP_GRID, "short": C.OOF_DEDUP_CORR_GRID}
    _, fs_long = factor_select.factor_sets_by_dedup(Ftr, y_tr, dedup_by_side["long"])
    _, fs_short = factor_select.factor_sets_by_dedup(Ftr, y_tr, dedup_by_side["short"],
                                                     exclude=long_only)
    fsets = {"long": fs_long, "short": fs_short}
    print("\nv15 long 沿用 v14 趋势因子(仅进 long): %s" % ", ".join(C.V14_LONG_TREND_FACTORS))
    print("long 门控冻结为: %s(其余 %s 不参与 long 选优)"
          % (list(C.V15_LONG_REGIME_GRID),
             [r for r in C.V5_REGIME_GRID if r not in C.V15_LONG_REGIME_GRID]))
    pruned: dict = {}
    prune_log: dict = {}
    for side in ("long", "short"):
        print("\n--- VIF 迭代剪枝: %s (阈值 %.0f, 下限 %d) ---" % (side, C.V4_MAX_VIF,
                                                              C.V4_MIN_FACTORS))
        for d in dedup_by_side[side]:
            before = fsets[side][d][side]
            kept, dropped = factor_select.vif_prune(Ftr, before)
            vb = factor_select.vif_series(Ftr, before)
            va = factor_select.vif_series(Ftr, kept)
            prune_log[(d, side)] = dict(before=list(before), after=list(kept), dropped=dropped,
                                        max_vif_before=float(vb.max()) if len(vb) else float("nan"),
                                        max_vif_after=float(va.max()) if len(va) else float("nan"))
            pruned.setdefault(side, {})[d] = kept
            print("  |corr|<%.2f 因子 %2d -> %2d | 最大VIF %.1f -> %.1f" %
                  (d, len(before), len(kept), prune_log[(d, side)]["max_vif_before"],
                   prune_log[(d, side)]["max_vif_after"]))

    frozen: dict = {}
    res: dict = {}
    for side in ("long", "short"):
        rules = C.V15_LONG_REGIME_GRID if side == "long" else C.V5_REGIME_GRID
        print("\n### 选择: %s (OOF 上联合择优: %s × 去冗余 × 模型 × 执行, OOC 不参与)"
              % (side, "制度规则" if side == "short" else "制度规则已冻结=none"))
        fmap = {d: pruned[side][d] for d in dedup_by_side[side]}
        per_rule_best = {}
        aggs = []
        # v15: long 用再外扩的执行网格(tp 上界 12) + v14 的模型候选; short 沿用全局默认(逐位不变)
        opt_kw = {}
        if side == "long":
            opt_kw = dict(thr_grid=C.V15_LONG_THR_GRID, tp_grid=C.V15_LONG_TP_GRID,
                          sl_grid=C.V15_LONG_SL_GRID, hold_grid=C.V15_LONG_HOLD_GRID,
                          model_grid=list(C.V15_LONG_MODEL_GRID))
        for rule in rules:
            best, agg, _ = optimize.optimize_full_on_oof(
                Fmat, y, fmap, side, ohlc, atr, times, tr, oof,
                tp_gt_sl=C.V3_ENFORCE_TP_GT_SL, regime=masks[rule][side],
                ret_drop=C.V15_RET_DROP, verbose=False, **opt_kw)
            agg = agg.copy()
            agg.insert(0, "regime", rule)
            aggs.append(agg)
            per_rule_best[rule] = best
            m = best["oof_metrics"]
            print("    制度 %-13s -> |corr|<%.2f 因子%2d %-9s tp=%.1f>sl=%.1f hold=%2d | "
                  "OOF 夏普%+.2f 收益%+.2f%% 笔数%d 胜率%.0f%% 盈亏比%.2f"
                  % (rule, best["dedup"], len(best["model"].factors), best["model_name"],
                     best["tp_mult"], best["sl_mult"], best["max_hold"],
                     m["sharpe"], m["total_return"] * 100, int(m["n"]),
                     m["win_rate"] * 100, m["payoff"]))
        combo = pd.concat(aggs, ignore_index=True)
        combo.to_csv(OUT / ("selection_grid_%s.csv" % side), index=False)
        # 制度规则之间也用同一规则: 收益锚(各规则自身最优) + 让步带内夏普择优
        wins = pd.DataFrame([dict(regime=r, **per_rule_best[r]["oof_metrics"])
                             for r in rules])
        anchor = float(wins["total_return"].max())
        wsel = optimize.select_return_anchor_sharpe(wins, C.V15_RET_DROP)
        best_rule = wsel.iloc[0]["regime"]
        best = per_rule_best[best_rule]
        print("    ==> 收益锚(制度最优收益) %+.2f%% | 让步带 >= %+.2f%% | 选定制度规则: %s "
              "(收益 %+.2f%% / 夏普 %+.2f)"
              % (anchor * 100, (anchor - C.V15_RET_DROP) * 100, best_rule,
                 best["oof_metrics"]["total_return"] * 100, best["oof_metrics"]["sharpe"]))

        pred = best["model"].predict(Fmat)
        side_res = {}
        for s in SEGS:
            seg = segs[s]
            side_res[s] = execution.evaluate_with_threshold(
                pred, ohlc, atr, times, side, seg.start, seg.stop, best["thr_abs"],
                best["tp_mult"], best["sl_mult"], max_hold=best["max_hold"],
                regime=masks[best_rule][side])
        frozen[side] = dict(best, pred=pred, regime_rule=best_rule)
        res[side] = side_res

    # ---------------- 报告
    lines: list = []
    lines.append("# SOL/USDT 4h 多因子 LGBM 回测报告 (v15: 放开 long 止盈上限 + 冻结 long 门控)\n")
    lines.append("- 数据源: 欧易(OKX) 4h 现货 %s | 清洗后 %d 根 | **起点对齐 DATA_START=%s**(首根 %s)\n"
                 % (C.SYMBOL, len(df), C.DATA_START, str(df["datetime"].iloc[0])[:16]))
    lines.append("- 切分: train 70% / OOF 15% / OOC 15%, 按时间顺序\n")
    lines.append("- 选择规则(同 v14): 真实 OOF 段上 **收益锚 + 夏普择优** —— 取 OOF 收益最高者为锚, "
                 "在「OOF 收益 >= 锚收益 - %.4f(让出上限 %.2f 个百分点)」的有界让步带内取 OOF 夏普最高者; "
                 "让步带用**绝对值**(收益可能为负, 比例门槛会方向错误)。\n"
                 % (C.V15_RET_DROP, C.V15_RET_DROP * 100))
    lines.append("- 选择**只在 OOF**; **OOC 仅观察**, 不参与任何选择\n")
    lines.append("- 执行: 止盈/止损均为 **ATR 倍数**, 且**硬约束 tp > sl**(风险报酬比 > 1)\n")
    lines.append("- 抗过拟合: 因子集做 **VIF 迭代剪枝**(全部 VIF <= %.0f);\n" % C.V4_MAX_VIF)
    lines.append("- 与 v14 的关系: 成本 / 资金 / 切分 / 标签 / VIF 纪律 / 选择规则**完全一致**; "
                 "v15 只改 long 两处 —— (1) **门控冻结为 none**(实验: 9 种趋势/制度门控在 OOF 上都不如不门控); "
                 "(2) **tp 网格上界 6.0 -> 12.0**(v14 中 tp=6 被 4 个制度一致顶到, 是空间在决定解)、"
                 "并把 tp>sl 下的 sl 网格收敛为 %s; long 专用因子沿用 v14 的趋势因子; "
                 "**short 逐位不变**(仍用 v13/v14 原网格 + 排除 long 专用因子)\n\n"
                 % (list(C.V15_LONG_SL_GRID),))

    lines.append("## 绩效(收益率 / 夏普 / 卡玛 / 胜率 / 盈亏比 / 开仓数量)\n\n")
    lines.append("| 方向 | 段 | 制度 | 收益率% | 夏普 | 卡玛 | 胜率% | 盈亏比 | 开仓数量 | 最大回撤% | 止盈率% | 止损率% | 超时率% |\n")
    lines.append("|" + "---|" * 14 + "\n")
    table_rows = []
    for side in ("long", "short"):
        for s in SEGS:
            m = res[side][s]["metrics"]
            lines.append("| %s | %s | %s | %+.2f | %.2f | %.2f | %.1f | %.2f | %d | %.2f | %.1f | %.1f | %.1f |\n"
                         % (side, s, frozen[side]["regime_rule"], m["total_return"] * 100,
                            m["sharpe"], m["calmar"], m["win_rate"] * 100, m["payoff_ratio"],
                            m["n_trades"], m["max_drawdown"] * 100, m["tp_rate"] * 100,
                            m["sl_rate"] * 100, m["timeout_rate"] * 100))
            table_rows.append(dict(side=side, seg=s, regime=frozen[side]["regime_rule"],
                                   **{k: float(m[k]) for k in
                                      ("total_return", "sharpe", "calmar", "max_drawdown",
                                       "win_rate", "payoff_ratio", "n_trades",
                                       "tp_rate", "sl_rate", "timeout_rate")}))
    lines.append("\n> 开仓数量 = 该段成交笔数(单向, 同时最多 1 笔)。OOF 为选择段, OOC 为纯观察段。\n")
    lines.append("> v15: long 门控冻结为 **none**, short 门控沿用 v14 口径(sma200_slope 等)。\n")

    lines.append("\n## 制度规则在 OOF 上的对比(各规则取自身 OOF 最优)\n\n")
    lines.append("| 方向 | 制度 | 去冗余 | 模型 | tp | sl | hold | OOF收益% | OOF夏普 | 笔数 | 胜率% | 盈亏比 |\n")
    lines.append("|" + "---|" * 12 + "\n")
    for side in ("long", "short"):
        rules = C.V15_LONG_REGIME_GRID if side == "long" else C.V5_REGIME_GRID
        cs = pd.read_csv(OUT / ("selection_grid_%s.csv" % side))
        for rule in rules:
            sub = cs[cs["regime"] == rule]
            if sub.empty:
                continue
            r = sub.iloc[0]
            star = " **<-选中**" if rule == frozen[side]["regime_rule"] else ""
            lines.append("| %s | %s%s | %.2f | %s | %.1f | %.1f | %d | %+.2f | %+.2f | %d | %.0f | %.2f |\n"
                         % (side, rule, star, r["dedup"], r["model"], r["tp_mult"], r["sl_mult"],
                            int(r["max_hold"]), r["total_return"] * 100, r["sharpe"],
                            int(r["n"]), r["win_rate"] * 100, r["payoff"]))
    lines.append("\n> long 的门控已在 v15 冻结为 none(实验结论), 故该表 long 仅一行; "
                 "short 仍对 v14 的全部制度规则择优。\n")

    lines.append("\n## v15 选择过程(收益锚 → 让步带内夏普择优)\n\n")
    lines.append("| 方向 | 收益锚(各制度最优中的最高收益)% | 让步带上限% | 选中制度 | 选中收益% | 选中夏普 | 收益让步% |\n")
    lines.append("|" + "---|" * 7 + "\n")
    for side in ("long", "short"):
        rules = C.V15_LONG_REGIME_GRID if side == "long" else C.V5_REGIME_GRID
        cs = pd.read_csv(OUT / ("selection_grid_%s.csv" % side))
        wins = pd.DataFrame([cs[cs["regime"] == r].iloc[0] for r in rules
                             if not cs[cs["regime"] == r].empty])
        anchor = float(wins["total_return"].max())
        ch = frozen[side]["oof_metrics"]
        lines.append("| %s | %+.2f | %+.2f | %s | %+.2f | %+.2f | %.2f |\n"
                     % (side, anchor * 100, (anchor - C.V15_RET_DROP) * 100,
                        frozen[side]["regime_rule"], ch["total_return"] * 100, ch["sharpe"],
                        (anchor - ch["total_return"]) * 100))

    lines.append("\n## 段基准(买入持有)\n\n| 段 | 基准收益% |\n|---|---|\n")
    for s in SEGS:
        lines.append("| %s | %+.2f |\n" % (s, buy_hold(df, segs[s]) * 100))

    lines.append("\n## 冻结配置(由 OOF 联合选出)\n")
    lines.append("| 方向 | 制度规则 | 去冗余|corr|< | 因子数(剪枝后) | 模型 | 阈值(绝对) | tp(ATR) | sl(ATR) | tp>sl | 最长持有 |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for side in ("long", "short"):
        f = frozen[side]
        lines.append("| %s | %s | %.2f | %d | %s | %+.6f | %.2f | %.2f | %s | %d |\n"
                     % (side, f["regime_rule"], f["dedup"], len(f["model"].factors),
                        f["model_name"], f["thr_abs"], f["tp_mult"], f["sl_mult"],
                        "是" if f["tp_mult"] > f["sl_mult"] else "否", f["max_hold"]))

    cons = consistency(df, tr, oof, ooc, frozen, res, prune_log, masks)
    n_fail = sum(1 for c in cons if not c["pass"])
    lines.append("\n## 一致性检查(design == runtime)\n\n**失败项: %d**\n\n| 检查 | 结果 | 说明 |\n"
                 "|---|---|---|\n" % n_fail)
    for c in cons:
        lines.append("| %s | %s | %s |\n" % (c["check"], "PASS" if c["pass"] else "**FAIL**",
                                             c["detail"]))

    (OUT / "backtest_report_v15.md").write_text("".join(lines), encoding="utf-8")

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
        ax.set_title("%s | regime=%s | OOF %+.2f%% (sharpe %.2f) | OOC %+.2f%% (sharpe %.2f) "
                     "[OOC = observation only]"
                     % (side, frozen[side]["regime_rule"], m["oof"]["total_return"] * 100,
                        m["oof"]["sharpe"], m["ooc"]["total_return"] * 100, m["ooc"]["sharpe"]))
        ax.set_ylabel("Equity (USDT)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("bar index (4h)")
    fig.tight_layout()
    fig.savefig(OUT / "equity_curve_v15.png", dpi=130)
    plt.close(fig)

    # ---------------- 落盘
    meta = {"config": {"SYMBOL": C.SYMBOL, "INTERVAL": C.INTERVAL, "SOURCE": C.BACKTEST_SOURCE,
                       "DATA_START": C.DATA_START,
                       "HORIZON": C.HORIZON, "SELECT_ON": "oof", "V3_ENFORCE_TP_GT_SL": True,
                       "V15_SELECT_RULE": "return_anchor_then_sharpe",
                       "V15_RET_DROP": C.V15_RET_DROP,
                       "V15_LONG_REGIME_GRID": list(C.V15_LONG_REGIME_GRID),
                       "V15_LONG_DEDUP_GRID": list(C.V15_LONG_DEDUP_GRID),
                       "V15_LONG_THR_GRID": list(C.V15_LONG_THR_GRID),
                       "V15_LONG_TP_GRID": list(C.V15_LONG_TP_GRID),
                       "V15_LONG_SL_GRID": list(C.V15_LONG_SL_GRID),
                       "V15_LONG_HOLD_GRID": list(C.V15_LONG_HOLD_GRID),
                       "V15_LONG_MODEL_GRID": [m["name"] for m in C.V15_LONG_MODEL_GRID],
                       "V14_LONG_TREND_FACTORS": list(C.V14_LONG_TREND_FACTORS),
                       "V4_MAX_VIF": C.V4_MAX_VIF, "V4_MIN_FACTORS": C.V4_MIN_FACTORS,
                       "V5_REGIME_GRID": list(C.V5_REGIME_GRID), "V5_REGIME_MA": C.V5_REGIME_MA,
                       "OBJECTIVE_PRIMARY": C.OBJECTIVE_PRIMARY,
                       "OBJECTIVE_SECONDARY": C.OBJECTIVE_SECONDARY,
                       "OOF_DEDUP_CORR_GRID": list(C.OOF_DEDUP_CORR_GRID),
                       "EXEC_THR_GRID": list(C.EXEC_THR_GRID), "TP_ATR_GRID": list(C.TP_ATR_GRID),
                       "SL_ATR_GRID": list(C.SL_ATR_GRID), "OOF_HOLD_GRID": list(C.OOF_HOLD_GRID),
                       "FEE_RATE": C.FEE_RATE, "SLIP_RATE": C.SLIP_RATE,
                       "INIT_CAPITAL": C.INIT_CAPITAL, "TRADE_NOTIONAL": C.TRADE_NOTIONAL},
            "sides": {}, "segments": table_rows,
            "prune_log": {"%s|%s" % (k[1], k[0]): v for k, v in prune_log.items()},
            "consistency": {"n_fail": n_fail, "checks": cons}}
    for side in ("long", "short"):
        f = frozen[side]
        meta["sides"][side] = {
            "model_name": f["model_name"], "factors": list(f["model"].factors),
            "regime_rule": f["regime_rule"],
            "frozen_params": {"dedup": f["dedup"], "thr_q": f["thr_q"], "thr_abs": f["thr_abs"],
                              "tp_mult": f["tp_mult"], "sl_mult": f["sl_mult"],
                              "max_hold": f["max_hold"]},
            "oof_metrics": f["oof_metrics"],
            "metrics": {s: {k: float(v) for k, v in res[side][s]["metrics"].items()}
                        for s in SEGS}}
    (OUT / "metrics_v15.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
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
    pd.DataFrame(tr_all).to_csv(OUT / "trades_v15.csv", index=False)

    # ---------------- 控制台汇总
    print("\n" + "=" * 116)
    print("结果汇总 (v15: 数据 >= %s | long 门控冻结 none + tp 上界 12 | OOF 收益锚+夏普择优, OOC 仅观察)"
          % C.DATA_START)
    print("=" * 116)
    print("%-6s %-6s %-14s %9s %8s %8s %8s %8s %8s" %
          ("方向", "段", "制度", "收益率%", "夏普", "卡玛", "胜率%", "盈亏比", "开仓数"))
    for side in ("long", "short"):
        for s in SEGS:
            m = res[side][s]["metrics"]
            print("%-6s %-6s %-14s %+9.2f %8.2f %8.2f %8.1f %8.2f %8d" %
                  (side, s, frozen[side]["regime_rule"], m["total_return"] * 100, m["sharpe"],
                   m["calmar"], m["win_rate"] * 100, m["payoff_ratio"], m["n_trades"]))

    p14 = C.REPORT_DIR / C.V14_OUT_DIR / "metrics_v14.json"
    if p14.exists():
        m14 = json.loads(p14.read_text(encoding="utf-8"))
        print("\n%-6s %-6s %28s %28s" % ("方向", "段", "v14 (基线)", "v15 (long tp 上界 6->12)"))
        for side in ("long", "short"):
            for s in ("oof", "ooc"):
                a = m14["sides"][side]["metrics"][s]
                b = res[side][s]["metrics"]
                print("%-6s %-6s %+13.2f%%/夏普%5.2f %+16.2f%%/夏普%5.2f" %
                      (side, s, a["total_return"] * 100, a["sharpe"],
                       b["total_return"] * 100, b["sharpe"]))
    else:
        print("\n(未找到 %s, 跳过与 v14 的对照)" % p14)

    print("\n一致性检查: 失败项 %d / %d" % (n_fail, len(cons)))
    for c in cons:
        if not c["pass"]:
            print("  FAIL: %s | %s" % (c["check"], c["detail"]))
    print("\n产出目录: %s" % OUT)
    print("  backtest_report_v15.md / metrics_v15.json / equity_curve_v15.png / "
          "selection_grid_*.csv / trades_v15.csv")


if __name__ == "__main__":
    main()
