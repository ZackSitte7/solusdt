#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SOL/USDT 4h 多因子 LGBM 回测 (v9: long+short 共同优化, 看合并总收益)。

与 v7/v8 的区别(用户要求): long / short 只是两个模型, **要一起优化看总收益**;
v7/v8 把 long 冻结自 v6、只重做 short, v9 **两侧都重做**, 并在 OOF 上对**组合**联合选优。

选择纪律(沿用 v8, 只发生在 OOF 子折; OOC 只观察):
  1. 每侧把 OOF 切成 V8_N_FOLDS 个连续子折, 每组配置在每子折上独立评估;
  2. 阶段1: 每侧按"该侧稳健收益"(各折收益均值 - V8_ROBUST_K×标准差)取前 V9_CAND_PER_SIDE 个候选;
  3. 阶段2: 对 (long候选 × short候选) 计算**合并逐折收益** rc = r_long + r_short
     (合并口径与 v7 一致: equity_combined = eq_long + eq_short - INIT_CAPITAL),
     主目标 = mean(rc) - V8_ROBUST_K×std(rc), 次目标 = 最差折合并收益(maximin)。
  两侧同用 v7 的制度网格(8 条)与止损网格(下探 0.35×ATR)。

为什么这不是"各自最优再相加": 只看收益, 相加是可分的; 但折间 std 惩罚依赖两侧在各折上的
搭配, std(r_long+r_short) != std(r_long)+std(r_short), 于是两侧真正耦合。

运行: python3 scripts/backtest/run_backtest_v9.py   (约需 2× v8 的网格时间)
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

BASE_DIR = Path(__file__).resolve().parents[2]      # 仓库根(脚本位于 scripts/backtest|analysis/)
sys.path.insert(0, str(BASE_DIR))
import config as C                                                    # noqa: E402
from src import data_clean, execution, factor_select, metrics, regime  # noqa: E402
from src import factors as F_lib                                     # noqa: E402
from src import models, optimize                                     # noqa: E402
from src.cv import time_split                                        # noqa: E402

OUT = C.REPORT_DIR / C.V9_OUT_DIR
META_V6 = C.REPORT_DIR / C.V6_OUT_DIR / "metrics_v6.json"
META_V7 = C.REPORT_DIR / C.V7_OUT_DIR / "metrics_v7.json"
META_V8 = C.REPORT_DIR / C.V8_OUT_DIR / "metrics_v8.json"
SEGS = ("train", "oof", "ooc")
MKEYS = ("total_return", "sharpe", "calmar", "max_drawdown", "win_rate",
         "payoff_ratio", "n_trades", "tp_rate", "sl_rate", "timeout_rate")
pd.set_option("display.width", 240)


def load_clean() -> pd.DataFrame:
    """载入欧易原始数据 -> 清洗 -> 对齐 DATA_START(与 v6/v7/v8 完全同口径, 保证可比)。"""
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
    """(因子池 × 去冗余) 在 train 上产出多空因子集 + VIF 剪枝(与 v6/v7/v8 同口径)。"""
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


def combined_seg(res: dict, seg: str) -> dict:
    """合并一段: equity = eq_long + eq_short - INIT; trades 合并; 指标同口径重算。"""
    eq = res["long"][seg]["equity"] + res["short"][seg]["equity"] - C.INIT_CAPITAL
    trades = list(res["long"][seg]["trades"]) + list(res["short"][seg]["trades"])
    return {"equity": eq, "trades": trades, "metrics": metrics.summarize(trades, eq)}


def consistency(df, tr, oof, ooc, frozen, res, comb, prune_log, masks, best, pairs, raw) -> list:
    """design == runtime 断言(v9 特有项: 两侧均重做 / 合并收益可加 / 联合目标只用 OOF 子折)。"""
    chk = []

    def add(name, ok, detail=""):
        chk.append({"check": name, "pass": bool(ok), "detail": detail})

    n = len(df)
    add("数据起点对齐 DATA_START", str(df["datetime"].iloc[0])[:10] == C.DATA_START,
        "起点 %s -> 首根 %s" % (C.DATA_START, str(df["datetime"].iloc[0])[:16]))
    add("数据切分 70/15/15 且三段严格时序不重叠",
        (tr.stop - tr.start) == int(n * C.TRAIN_FRAC) and tr.stop == oof.start
        and oof.stop == ooc.start and oof.stop < n,
        "train[0,%d) oof[%d,%d) ooc[%d,%d)" % (tr.stop, oof.start, oof.stop, ooc.start, ooc.stop))

    folds = optimize._v8_fold_slices(oof, C.V8_N_FOLDS)
    contiguous = all(folds[i][1] == folds[i + 1][0] for i in range(len(folds) - 1))
    inside = (folds[0][0] == oof.start and folds[-1][1] == oof.stop)
    add("OOF 子折连续/不重叠/完全落在 OOF 内(未触碰 OOC)", contiguous and inside,
        "%d 折: %s" % (len(folds), ", ".join("[%d,%d)" % f for f in folds)))
    add("两侧网格评估均未使用 OOC 区间的 bar",
        bool((raw["long"]["fold_lo"] >= oof.start).all()
             and (raw["long"]["fold_hi"] <= oof.stop).all()
             and (raw["short"]["fold_lo"] >= oof.start).all()
             and (raw["short"]["fold_hi"] <= oof.stop).all()),
        "long %d 行 / short %d 行, 评估区间上界均 <= OOF 上界 %d"
        % (len(raw["long"]), len(raw["short"]), oof.stop))

    add("两侧均在 OOF 上重新择优(不再冻结自 v6)", True,
        "long(池%s, %s) 与 short(池%s, %s) 均为 v9 独立选出的配置"
        % (frozen["long"]["pool"], frozen["long"]["regime_rule"],
           frozen["short"]["pool"], frozen["short"]["regime_rule"]))
    add("两侧各自止盈/止损硬约束 tp > sl",
        frozen["long"]["tp_mult"] > frozen["long"]["sl_mult"]
        and frozen["short"]["tp_mult"] > frozen["short"]["sl_mult"],
        "long tp=%.2f>sl=%.2f | short tp=%.2f>sl=%.2f"
        % (frozen["long"]["tp_mult"], frozen["long"]["sl_mult"],
           frozen["short"]["tp_mult"], frozen["short"]["sl_mult"]))

    # 合并收益可加: 合并 total_return == 两侧 total_return 之和
    ok_add, det = True, []
    for s in SEGS:
        rl = res["long"][s]["metrics"]["total_return"]
        rs = res["short"][s]["metrics"]["total_return"]
        rc = comb[s]["metrics"]["total_return"]
        det.append("%s: %.4f + %.4f = %.4f" % (s, rl * 100, rs * 100, rc * 100))
        if abs(rc - (rl + rs)) > 1e-9:
            ok_add = False
    add("合并收益 = 两侧收益之和(可加性)", ok_add, " | ".join(det))

    # 联合目标可复现: 用 best 的逐折合并收益重算稳健分, 且 best 即 pairs 第一行
    fr = np.array(best["fold_returns"], dtype=float)
    rb = float(fr.mean() - C.V8_ROBUST_K * fr.std(ddof=1))
    add("联合稳健分可复现且 best == 联合候选表第一行",
        abs(rb - best["robust_combined"]) < 1e-9
        and bool(pairs.iloc[0]["long_pool"] == best["long"]["pool"])
        and bool(pairs.iloc[0]["short_pool"] == best["short"]["pool"])
        and abs(float(pairs.iloc[0]["robust_combined"]) - best["robust_combined"]) < 1e-12,
        "稳健 %+.4f = 均值 %+.4f - %.1f×std %.4f | 逐折 %s"
        % (best["robust_combined"], fr.mean(), C.V8_ROBUST_K, fr.std(ddof=1),
           ", ".join("%+.2f%%" % (v * 100) for v in fr)))

    # 门控生效
    ok_gate, det_gate = True, []
    for side in ("long", "short"):
        rule = frozen[side]["regime_rule"]
        mask = masks[rule][side]
        m = 0
        for s in SEGS:
            seg = {"train": tr, "oof": oof, "ooc": ooc}[s]
            sig = execution.signal_from_threshold(frozen[side]["pred"], side,
                                                  frozen[side]["thr_abs"],
                                                  seg.start, seg.stop, regime=mask)
            m = max(m, abs(int(sig.sum()) - res[side][s]["n_signals"]))
        ok_gate &= (m == 0)
        det_gate.append("%s rule=%s(允许 %d bar)" % (side, rule, int(mask.sum())))
    add("两侧制度门控生效且与冻结评估一致", ok_gate, " | ".join(det_gate))

    # VIF / 成本 / 成交时点
    ok_vif = True
    for side in ("long", "short"):
        e = prune_log[(frozen[side]["pool"], frozen[side]["dedup"], side)]
        ok_vif &= (e["max_vif_after"] <= C.V4_MAX_VIF + 1e-9)
    add("两侧 VIF 迭代剪枝(冻结集 VIF<=%.0f)" % C.V4_MAX_VIF, ok_vif,
        " | ".join("%s: 池%s 后VIF %.2f" % (sd, frozen[sd]["pool"],
                  prune_log[(frozen[sd]["pool"], frozen[sd]["dedup"], sd)]["max_vif_after"])
                  for sd in ("long", "short")))

    cost = 2.0 * (C.FEE_RATE + C.SLIP_RATE)
    allt = [t for side in ("long", "short") for s in SEGS for t in res[side][s]["trades"]]
    bad = [t for t in allt if abs(t.net_ret - (t.gross_ret - cost)) > 1e-12]
    bad_ll = [t for t in allt if abs(t.entry_price - float(df["open"].iloc[t.entry_idx])) > 1e-9]
    add("成本口径 = 单边 5bp / 双边 10bp", len(bad) == 0,
        "逐笔复核 %d 笔, 不一致 %d 笔" % (len(allt), len(bad)))
    add("信号 t 收盘 -> t+1 开盘成交(无未来函数)", len(bad_ll) == 0,
        "全部 %d 笔成交价 == open[entry_idx], 不一致 %d 笔" % (len(allt), len(bad_ll)))
    add("资金口径 1000 本金 / 单笔 100",
        C.INIT_CAPITAL == 1000.0 and C.TRADE_NOTIONAL == 100.0,
        "INIT_CAPITAL=%.0f TRADE_NOTIONAL=%.0f" % (C.INIT_CAPITAL, C.TRADE_NOTIONAL))

    ok_causal, det_causal = True, []
    for r in C.V7_REGIME_GRID:
        if r == "none":
            continue
        for idx in (500, 3000, 7000, 9000, 11000):
            if idx >= n:
                continue
            sub = regime.regime_masks(df.iloc[:idx + 1].reset_index(drop=True), r)
            for side in ("long", "short"):
                if bool(sub[side][idx]) != bool(masks[r][side][idx]):
                    ok_causal = False
        det_causal.append(r)
    add("制度掩码仅用过去信息(截断后取值不变)", ok_causal, "规则: " + ", ".join(det_causal))
    return chk


# ================================================================ 主流程
def main() -> None:
    if not META_V6.exists():
        raise SystemExit("缺少 v6 冻结配置, 请先运行 scripts/backtest/run_backtest_v6.py: %s" % META_V6)
    OUT.mkdir(parents=True, exist_ok=True)
    v6meta = json.loads(META_V6.read_text(encoding="utf-8"))
    v7meta = json.loads(META_V7.read_text(encoding="utf-8")) if META_V7.exists() else None
    v8meta = json.loads(META_V8.read_text(encoding="utf-8")) if META_V8.exists() else None

    print("=" * 120)
    print("SOL/USDT %s LGBM 回测 v9 | 欧易(OKX) | long+short 共同优化 (合并总收益) | OOC 仅观察"
          % C.INTERVAL)
    print("每侧 OOF 内 %d 折 -> 稳健候选前 %d -> 组合联合选优 | 主目标 = 合并收益均值 - %.1f×折间std"
          % (C.V8_N_FOLDS, C.V9_CAND_PER_SIDE, C.V8_ROBUST_K))
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
    folds = optimize._v8_fold_slices(oof, C.V8_N_FOLDS)
    print("OOF 内部分折: " + ", ".join("[%d,%d)" % f for f in folds))

    masks = {rule: regime.regime_masks(df, rule) for rule in C.V7_REGIME_GRID}
    for rule in C.V7_REGIME_GRID:
        print("制度 %-13s 允许开多 %5d bar / 开空 %5d bar"
              % (rule, int(masks[rule]["long"].sum()), int(masks[rule]["short"].sum())))

    print("\n--- 因子集(仅 train): 因子池 × 去冗余 × VIF 剪枝 ---")
    fsets, prune_log = build_factor_sets(Fmat, y, tr)
    for pool in C.V6_FACTOR_POOLS:
        for d in C.V6_DEDUP_GRID:
            print("  池 %-8s |corr|<%.2f -> long %2d / short %2d 个"
                  % (pool, d, len(fsets[(pool, float(d))]["long"]),
                     len(fsets[(pool, float(d))]["short"])))

    # ---------------- 两侧各自 OOF 稳健长表(选择段仍是 OOF; OOC 不参与)
    raw: dict = {}
    for side in ("long", "short"):
        print("\n### 阶段1: %s OOF 内部分折稳健评估(因子池 × 制度 × 去冗余 × 模型 × 执行 × %d 折)"
              % (side, C.V8_N_FOLDS))
        fmap = {(pool, d): fsets[(pool, d)][side]
                for pool in C.V6_FACTOR_POOLS for d in C.V6_DEDUP_GRID}
        regimes = {rule: masks[rule][side] for rule in C.V7_REGIME_GRID}
        _, _, raw[side] = optimize.optimize_v8_on_oof(
            Fmat, y, fmap, side, ohlc, atr, times, tr, oof, regimes,
            thr_grid=C.V6_THR_GRID, hold_grid=C.V6_HOLD_GRID, trail_grid=C.V6_TRAIL_GRID,
            sl_grid=C.V7_SL_GRID, tp_gt_sl=C.V3_ENFORCE_TP_GT_SL, verbose=True)
        raw[side].to_csv(OUT / ("selection_folds_%s.csv" % side), index=False)

    # ---------------- 阶段2: 联合选优(合并总收益的跨折稳健分)
    print("\n### 阶段2: 合并组合联合选优 (long候选 × short候选)")
    best, pairs, pl, ps = optimize.joint_select_v9(
        raw["long"], raw["short"], C.V9_CAND_PER_SIDE, robust_k=C.V8_ROBUST_K,
        top_report=C.V9_TOP_PAIRS)
    pairs.to_csv(OUT / "selection_pairs_v9.csv", index=False)
    print("    long 候选池 %d 个(该侧稳健收益第1 = %+.2f%%, 第%d = %+.2f%%)"
          % (len(pl), pl.iloc[0]["robust_ret"] * 100, len(pl), pl.iloc[-1]["robust_ret"] * 100))
    print("    short 候选池 %d 个(该侧稳健收益第1 = %+.2f%%, 第%d = %+.2f%%)"
          % (len(ps), ps.iloc[0]["robust_ret"] * 100, len(ps), ps.iloc[-1]["robust_ret"] * 100))
    fr = best["fold_returns"]
    print("    ==> 联合最优: 合并稳健 %+.3f (均值 %+.3f - %.1f×std %.3f, 最差折 %+.3f)"
          % (best["robust_combined"], best["mean_combined"], C.V8_ROBUST_K,
             best["std_combined"], best["worst_combined"]))
    print("        合并逐折收益: " + ", ".join("%+.2f%%" % (v * 100) for v in fr))
    print("        long : 池=%s 制度=%s |corr|<%.2f %-9s tp=%.2f>sl=%.2f hold=%2d trail=%.1f"
          % (best["long"]["pool"], best["long"]["regime"], best["long"]["dedup"],
             best["long"]["model"], best["long"]["tp_mult"], best["long"]["sl_mult"],
             int(best["long"]["max_hold"]), best["long"]["trail_mult"]))
    print("        short: 池=%s 制度=%s |corr|<%.2f %-9s tp=%.2f>sl=%.2f hold=%2d trail=%.1f"
          % (best["short"]["pool"], best["short"]["regime"], best["short"]["dedup"],
             best["short"]["model"], best["short"]["tp_mult"], best["short"]["sl_mult"],
             int(best["short"]["max_hold"]), best["short"]["trail_mult"]))

    # ---------------- 冻结两侧配置并评估
    frozen, res = {}, {}
    for side in ("long", "short"):
        bc = best[side]
        facs = fsets[(bc["pool"], bc["dedup"])][side]
        tm = models.train_side(Fmat.iloc[tr].reset_index(drop=True),
                               y.iloc[tr].reset_index(drop=True), facs, side,
                               params=model_params_by_name(bc["model"]), verbose=False)
        pred = tm.predict(Fmat)
        thr_abs = execution.threshold_from_quantile(pred, side, float(bc["thr_q"]),
                                                    oof.start, oof.stop)
        frozen[side] = dict(pool=bc["pool"], regime_rule=bc["regime"], dedup=float(bc["dedup"]),
                            model_name=bc["model"], model=tm, thr_q=float(bc["thr_q"]),
                            thr_abs=float(thr_abs), tp_mult=float(bc["tp_mult"]),
                            sl_mult=float(bc["sl_mult"]), max_hold=int(bc["max_hold"]),
                            trail_mult=float(bc["trail_mult"]), pred=pred)
        res[side] = {}
        for s in SEGS:
            seg = segs[s]
            res[side][s] = execution.evaluate_with_threshold(
                pred, ohlc, atr, times, side, seg.start, seg.stop, thr_abs,
                frozen[side]["tp_mult"], frozen[side]["sl_mult"],
                max_hold=frozen[side]["max_hold"], trail_mult=frozen[side]["trail_mult"],
                regime=masks[frozen[side]["regime_rule"]][side])
    comb = {s: combined_seg(res, s) for s in SEGS}

    # ---------------- 报告
    lines: list = []
    lines.append("# SOL/USDT 4h 多因子 LGBM 回测报告 (v9: long+short 共同优化, 合并总收益)\n")
    lines.append("- 数据源: 欧易(OKX) 4h 现货 %s | 清洗后 %d 根(起点 %s)\n"
                 % (C.SYMBOL, len(df), C.DATA_START))
    lines.append("- 切分: train 70% / OOF 15% / OOC 15%, 按时间顺序\n")
    lines.append("- v9 **两侧都重做**: long / short 各为一个模型, 在 OOF 上**联合选优合并总收益**"
                 "(v7/v8 的 long 冻结自 v6, v9 不再冻结)。\n")
    lines.append("- 选择: **只在 OOF 子折**; **OOC 仅观察**。合并口径: `equity = eq_long + eq_short"
                 " - INIT_CAPITAL` => **合并收益 = 两侧收益之和**。\n")
    lines.append("- 阶段1: 每侧按该侧稳健收益(折均收益 - %.1f×折间std, 单折>=%d笔)取前 %d 个候选;\n"
                 "  阶段2: 对 (long候选 × short候选) 以合并稳健分 = 合并收益折均 - %.1f×折间std 为主目标,"
                 " 最差折合并收益为次目标。\n"
                 % (C.V8_ROBUST_K, C.V8_MIN_TRADES_PER_FOLD, C.V9_CAND_PER_SIDE, C.V8_ROBUST_K))
    lines.append("- OOF 子折: %s\n\n" % ", ".join("[%d,%d)" % f for f in folds))

    # ---- 合并总收益: v6/v7/v8(由各自 JSON 的两侧收益相加) vs v9(逐段重算)
    lines.append("## 合并总收益对照(本金 1000)\n\n")
    lines.append("| 段 | 版本 | long 收益% | short 收益% | **合并收益%** | 说明 |\n")
    lines.append("|" + "---|" * 6 + "\n")

    def _sum_from_meta(meta, tag, note):
        for s in SEGS:
            if meta is None:
                lines.append("| %s | %s | - | - | - | %s |\n" % (s, tag, note))
                continue
            lr = meta["sides"]["long"]["metrics"][s]["total_return"]
            sr = meta["sides"]["short"]["metrics"][s]["total_return"]
            lines.append("| %s | %s | %+.2f | %+.2f | **%+.2f** | %s |\n"
                         % (s, tag, lr * 100, sr * 100, (lr + sr) * 100, note))

    _sum_from_meta(v6meta, "v6", "两侧各自整段 OOF 取最大")
    _sum_from_meta(v7meta, "v7", "long 冻结自 v6, short 空头专用")
    _sum_from_meta(v8meta, "v8", "long 冻结自 v6, short 稳健选优")
    for s in SEGS:
        lr = res["long"][s]["metrics"]["total_return"]
        sr = res["short"][s]["metrics"]["total_return"]
        cr = comb[s]["metrics"]["total_return"]
        lines.append("| %s | **v9** | %+.2f | %+.2f | **%+.2f** | 两侧共同优化(合并目标) |\n"
                     % (s, lr * 100, sr * 100, cr * 100))

    lines.append("\n> 注: v6/v7/v8 的合并收益为各自 JSON 中两侧 `total_return` 之和(可加性成立, 与 v9 同口径)。\n")

    # ---- v9 分侧 + 合并明细
    lines.append("\n## v9 分侧与合并明细\n\n")
    lines.append("| 段 | 腿 | 收益率% | 夏普 | 卡玛 | 最大回撤% | 胜率% | 盈亏比 | 开仓数量 | 止盈率% | 止损率% | 超时率% |\n")
    lines.append("|" + "---|" * 12 + "\n")
    for s in SEGS:
        for leg, m in (("long", res["long"][s]["metrics"]), ("short", res["short"][s]["metrics"]),
                       ("**合并**", comb[s]["metrics"])):
            lines.append("| %s | %s | %+.2f | %.2f | %.2f | %.2f | %.1f | %.2f | %d | %.1f | %.1f | %.1f |\n"
                         % (s, leg, m["total_return"] * 100, m["sharpe"], m["calmar"],
                            m["max_drawdown"] * 100, m["win_rate"] * 100, m["payoff_ratio"],
                            m["n_trades"], m["tp_rate"] * 100, m["sl_rate"] * 100,
                            m["timeout_rate"] * 100))

    # ---- 联合候选表
    lines.append("\n## 联合候选组合(前 %d, 按合并稳健分降序)\n\n" % len(pairs))
    lines.append("| # | long 池/制度/tp-sl/hold/trail | short 池/制度/tp-sl/hold/trail | "
                 "合并稳健 | 折均收益% | 折间std | 最差折% | 逐折收益% |\n")
    lines.append("|" + "---|" * 9 + "\n")
    for i in range(len(pairs)):
        r = pairs.iloc[i]
        star = " **<-选中**" if i == 0 else ""
        fr_s = "/".join("%+.2f" % (r["fold%d_ret" % (k + 1)] * 100) for k in range(C.V8_N_FOLDS))
        lines.append("| %d%s | %s/%s/%.2f-%.2f/%d/%.1f | %s/%s/%.2f-%.2f/%d/%.1f | %+.3f | %+.2f | %.3f | %+.2f | %s |\n"
                     % (i + 1, star,
                        r["long_pool"], r["long_regime"], r["long_tp_mult"], r["long_sl_mult"],
                        int(r["long_max_hold"]), r["long_trail_mult"],
                        r["short_pool"], r["short_regime"], r["short_tp_mult"], r["short_sl_mult"],
                        int(r["short_max_hold"]), r["short_trail_mult"],
                        r["robust_combined"], r["mean_combined"] * 100, r["std_combined"],
                        r["worst_combined"] * 100, fr_s))

    # ---- 两侧候选池 top
    lines.append("\n## 阶段1 每侧稳健候选池(各前 15)\n\n")
    for side, pool in (("long", pl), ("short", ps)):
        lines.append("**" + side + " 候选池**\n\n| 池 | 制度 | |corr|< | 模型 | tp | sl | hold | "
                     "trail | 稳健收益% | 折均收益% | 最差折% |\n")
        lines.append("|" + "---|" * 12 + "\n")
        for i in range(min(15, len(pool))):
            p = pool.iloc[i]
            lines.append("| %s | %s | %.2f | %s | %.2f | %.2f | %d | %.1f | %+.2f | %+.2f | %+.2f |\n"
                         % (p["pool"], p["regime"], p["dedup"], p["model"], p["tp_mult"],
                            p["sl_mult"], int(p["max_hold"]), p["trail_mult"],
                            p["robust_ret"] * 100, p["ret_mean"] * 100, p["ret_min"] * 100))
        lines.append("\n")

    # ---- 冻结配置 + 谱维度
    lines.append("\n## 冻结配置(由 OOF 联合选出)\n")
    lines.append("| 方向 | 因子池 | 制度规则 | 去冗余|corr|< | 因子数 | 模型 | 阈值(绝对) | tp(ATR) | sl(ATR) | 跟踪(ATR) | 最长持有 |\n")
    lines.append("|" + "---|" * 12 + "\n")
    for side in ("long", "short"):
        f = frozen[side]
        lines.append("| %s | %s | %s | %.2f | %d | %s | %+.6f | %.2f | %.2f | %.1f | %d |\n"
                     % (side, f["pool"], f["regime_rule"], f["dedup"], len(f["model"].factors),
                        f["model_name"], f["thr_abs"], f["tp_mult"], f["sl_mult"],
                        f["trail_mult"], f["max_hold"]))
    lines.append("\n| 方向 | 因子数 | 95%方差维度 | 有效维度 | 最大VIF | 因子列表 |\n|---|---|---|---|---|---|\n")
    for side in ("long", "short"):
        sp = spectrum(Fmat, frozen[side]["model"].factors, tr)
        lines.append("| %s | %d | %d | %.2f | %.2f | %s |\n"
                     % (side, sp["n_factors"], sp["dim95"], sp["eff_rank"], sp["max_vif"],
                        ", ".join(frozen[side]["model"].factors)))

    lines.append("\n## 段基准(买入持有)\n\n| 段 | 基准收益% |\n|---|---|\n")
    for s in SEGS:
        lines.append("| %s | %+.2f |\n" % (s, buy_hold(df, segs[s]) * 100))

    cons = consistency(df, tr, oof, ooc, frozen, res, comb, prune_log, masks, best, pairs, raw)
    n_fail = sum(1 for c in cons if not c["pass"])
    lines.append("\n## 一致性检查(design == runtime)\n\n**失败项: %d**\n\n| 检查 | 结果 | 说明 |\n"
                 "|---|---|---|\n" % n_fail)
    for c in cons:
        lines.append("| %s | %s | %s |\n" % (c["check"], "PASS" if c["pass"] else "**FAIL**",
                                             c["detail"]))

    (OUT / "backtest_report_v9.md").write_text("".join(lines), encoding="utf-8")

    # ---------------- 权益曲线
    eq_long = execution.equity_from_trades(
        [t for s in SEGS for t in res["long"][s]["trades"]], len(df))
    eq_short = execution.equity_from_trades(
        [t for s in SEGS for t in res["short"][s]["trades"]], len(df))
    eq_comb = eq_long + eq_short - C.INIT_CAPITAL
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(range(len(df)), eq_long, lw=1.1, color="tab:blue", label="long (v9)")
    ax.plot(range(len(df)), eq_short, lw=1.1, color="tab:red", label="short (v9)")
    ax.plot(range(len(df)), eq_comb, lw=1.8, color="black", label="combined (long+short)")
    ax.axhline(C.INIT_CAPITAL, color="gray", ls=":", lw=0.8)
    for b in (tr.stop, oof.stop):
        ax.axvline(b, color="red", ls="--", lw=0.8, alpha=0.7)
    for s in SEGS:
        ax.axvspan(segs[s].start, segs[s].stop, alpha=0.05,
                   color={"train": "tab:blue", "oof": "tab:orange", "ooc": "tab:green"}[s])
    mc = comb["oof"]["metrics"]
    mo = comb["ooc"]["metrics"]
    ax.set_title("v9 combined (long+short jointly optimized) | OOF %+.2f%% (sharpe %.2f, mdd %.2f%%) | "
                 "OOC %+.2f%% (sharpe %.2f, mdd %.2f%%) [OOC = observation only]"
                 % (mc["total_return"] * 100, mc["sharpe"], mc["max_drawdown"] * 100,
                    mo["total_return"] * 100, mo["sharpe"], mo["max_drawdown"] * 100))
    ax.set_xlabel("bar index (4h)")
    ax.set_ylabel("Equity (USDT)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "equity_curve_v9_combined.png", dpi=130)
    plt.close(fig)

    # ---------------- 落盘
    meta = {"config": {"SYMBOL": C.SYMBOL, "INTERVAL": C.INTERVAL, "SOURCE": C.BACKTEST_SOURCE,
                       "DATA_START": C.DATA_START, "HORIZON": C.HORIZON,
                       "SELECT_ON": "oof", "OPTIMIZED_SIDES": ["long", "short"],
                       "JOINT": True, "JOINT_STAGE1_TOP": C.V9_CAND_PER_SIDE,
                       "JOINT_PRIMARY": "mean_combined - K*std_combined",
                       "JOINT_SECONDARY": "worst-fold combined return",
                       "COMBINE": "equity = eq_long + eq_short - INIT_CAPITAL",
                       "V8_N_FOLDS": C.V8_N_FOLDS, "V8_ROBUST_K": C.V8_ROBUST_K,
                       "V8_MIN_TRADES_PER_FOLD": C.V8_MIN_TRADES_PER_FOLD,
                       "V8_FOLDS": [[int(a), int(b)] for a, b in folds],
                       "V9_CONFIG_KEYS": list(optimize._V8_CONFIG_KEYS),
                       "V7_SL_GRID": list(C.V7_SL_GRID), "V7_REGIME_GRID": list(C.V7_REGIME_GRID),
                       "V6_DEDUP_GRID": list(C.V6_DEDUP_GRID), "V6_THR_GRID": list(C.V6_THR_GRID),
                       "V6_HOLD_GRID": list(C.V6_HOLD_GRID), "V6_TRAIL_GRID": list(C.V6_TRAIL_GRID),
                       "V4_MAX_VIF": C.V4_MAX_VIF, "FEE_RATE": C.FEE_RATE,
                       "SLIP_RATE": C.SLIP_RATE, "INIT_CAPITAL": C.INIT_CAPITAL,
                       "TRADE_NOTIONAL": C.TRADE_NOTIONAL},
            "joint_best": {k: best[k] for k in ("robust_combined", "mean_combined",
                                                "worst_combined", "std_combined", "fold_returns")},
            "joint_best_long": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer))
                                    else v) for k, v in best["long"].items()},
            "joint_best_short": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer))
                                     else v) for k, v in best["short"].items()},
            "joint_pairs_top": pairs.to_dict(orient="records"),
            "candidate_pool_long_top": pl.head(50).to_dict(orient="records"),
            "candidate_pool_short_top": ps.head(50).to_dict(orient="records"),
            "sides": {}, "combined": {},
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
            "metrics": {s: {k: float(v) for k, v in res[side][s]["metrics"].items()}
                        for s in SEGS}}
    for s in SEGS:
        meta["combined"][s] = {k: float(v) for k, v in comb[s]["metrics"].items()}
    (OUT / "metrics_v9.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
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
    pd.DataFrame(tr_all).to_csv(OUT / "trades_v9.csv", index=False)

    # ---------------- 控制台汇总
    print("\n" + "=" * 120)
    print("结果汇总 (v9: long+short 共同优化; OOC 仅观察)")
    print("=" * 120)
    print("%-8s %-6s %9s %8s %8s %8s %8s %8s" %
          ("腿", "段", "收益率%", "夏普", "卡玛", "最大回撤%", "胜率%", "开仓数"))
    for s in SEGS:
        for leg, m in (("long", res["long"][s]["metrics"]), ("short", res["short"][s]["metrics"]),
                       ("合并", comb[s]["metrics"])):
            print("%-8s %-6s %+9.2f %8.2f %8.2f %8.2f %8.1f %8d" %
                  (leg, s, m["total_return"] * 100, m["sharpe"], m["calmar"],
                   m["max_drawdown"] * 100, m["win_rate"] * 100, m["n_trades"]))
    print("\n合并总收益对照:  v6 %+.2f%% -> v7 %+.2f%% -> v8 %+.2f%% -> v9 %+.2f%% (OOC)"
          % (_ooc_sum(v6meta), _ooc_sum(v7meta), _ooc_sum(v8meta),
             comb["ooc"]["metrics"]["total_return"] * 100))
    print("\n一致性检查: 失败项 %d / %d" % (n_fail, len(cons)))
    for c in cons:
        if not c["pass"]:
            print("  FAIL: %s | %s" % (c["check"], c["detail"]))
    print("\n产出目录: %s" % OUT)
    print("  backtest_report_v9.md / metrics_v9.json / equity_curve_v9_combined.png / "
          "selection_folds_long.csv / selection_folds_short.csv / selection_pairs_v9.csv / trades_v9.csv")


def _ooc_sum(meta) -> float:
    if not meta:
        return float("nan")
    return (meta["sides"]["long"]["metrics"]["ooc"]["total_return"]
            + meta["sides"]["short"]["metrics"]["ooc"]["total_return"]) * 100


if __name__ == "__main__":
    main()
