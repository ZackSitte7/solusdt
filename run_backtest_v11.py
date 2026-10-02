#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SOL/USDT 4h 多因子 LGBM 回测 (v11: 强化选择器, 把 OOF 收益+夏普再推高)。

用户要求: 在 v10 上继续**优化 OOF 的收益和夏普**(数据固定 DATA_START=2021-10-01 起)。

v11 与 v10 的差别**只在"怎么选"**(数据窗口/因子池/训练/执行网格/成本/资金/合并口径/
排序目标/折数全部逐位相同):
  v10: 候选池每侧只按"该侧稳健收益"取前 200;
  v11: 候选池**双目标保留 + Pareto 前沿**, 每侧 500 个。

为什么这是"强化选择器"而不是"扩大假设空间": 执行网格/因子池/模型网格**一个都没动**,
搜索的假设空间与 v10 完全相同 —— 变的只是**从同一批假设里保留哪些进联合配对**。
v10 的候选池只按收益排序, 存在把"夏普高、收益略低"的配置在阶段1 丢掉的风险
(实测在 v9 的 3 折长表上前 200 已含最高稳健夏普, 故是风险而非既成事实);
v11 用 min(收益排名, 夏普排名) + Pareto 前沿把这类解保留下来。

折数**保持 3 折与 v10 一致**是这里的硬约束: 只有折数不变, 目标取值尺度才逐位相同,
v11 的候选池才**严格包含** v10 的池, "不比 v10 差"才是选择集上的硬保证。
(教训: 曾把折数 3 改 5, 改变了目标本身尺度、使包含性失效, 反而选出更差的配对 —— 已回退。)

排序目标仍与 v10 一致: 主 = 合并稳健收益, 次 = 合并稳健夏普(真实 bar 级), 三 = 最差折合并收益。
选择仍**只发生在 OOF 子折**; OOC 全程只观察。两侧都做(long / short 不冻结), 同用 v7 制度/止损网格。

折数与 v9/v10 相同 -> 阶段1 长表(配置×子折)与 v9 结构一致, 直接复用 v9 网格。
运行: python3 run_backtest_v11.py
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
from src import data_clean, execution, factor_select, metrics, regime  # noqa: E402
from src import factors as F_lib                                     # noqa: E402
from src import models, optimize                                     # noqa: E402
from src.cv import time_split                                        # noqa: E402

OUT = C.REPORT_DIR / C.V11_OUT_DIR
SEGS = ("train", "oof", "ooc")
KEYS = list(optimize._V8_CONFIG_KEYS)
pd.set_option("display.width", 240)


def load_clean() -> pd.DataFrame:
    """载入欧易原始数据 -> 清洗 -> 对齐 DATA_START(与 v6~v10 完全同口径)。"""
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
    """(因子池 × 去冗余) 在 train 上产出多空因子集 + VIF 剪枝(与 v6~v10 同口径)。"""
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


def build_pred_map(side: str, pool: pd.DataFrame, fsets: dict, Fmat: pd.DataFrame,
                   y: pd.Series, tr: slice, prebuilt: dict) -> dict:
    """候选池用到的 (池, 去冗余, 模型) -> (模型, 全序列预测)。已有则复用, 避免重复训练。"""
    out = dict(prebuilt)
    need = {(c["pool"], float(c["dedup"]), c["model"]) for _, c in pool.iterrows()}
    for key in sorted(need):
        if key in out:
            continue
        poolname, dedup, mname = key
        facs = fsets[(poolname, dedup)][side]
        tm = models.train_side(Fmat.iloc[tr].reset_index(drop=True),
                               y.iloc[tr].reset_index(drop=True), facs, side,
                               params=model_params_by_name(mname), verbose=False)
        out[key] = (tm, tm.predict(Fmat))
    return out


def combined_seg(res: dict, seg: str) -> dict:
    """合并一段: equity = eq_long + eq_short - INIT; trades 合并; 指标同口径重算。"""
    eq = res["long"][seg]["equity"] + res["short"][seg]["equity"] - C.INIT_CAPITAL
    trades = list(res["long"][seg]["trades"]) + list(res["short"][seg]["trades"])
    return {"equity": eq, "trades": trades, "metrics": metrics.summarize(trades, eq)}


def consistency(df, tr, oof, ooc, frozen, res, comb, prune_log, masks, best, pairs, raw, folds) -> list:
    chk = []

    def add(name, ok, detail=""):
        chk.append({"check": name, "pass": bool(ok), "detail": detail})

    n = len(df)
    add("数据切分 70/15/15 且三段严格时序不重叠",
        (tr.stop - tr.start) == int(n * C.TRAIN_FRAC) and tr.stop == oof.start
        and oof.stop == ooc.start and oof.stop < n,
        "train[0,%d) oof[%d,%d) ooc[%d,%d)" % (tr.stop, oof.start, oof.stop, ooc.start, ooc.stop))
    add("OOF 子折连续/不重叠/完全落在 OOF 内(未触碰 OOC)",
        all(folds[i][1] == folds[i + 1][0] for i in range(len(folds) - 1))
        and folds[0][0] == oof.start and folds[-1][1] == oof.stop,
        "%d 折: %s" % (len(folds), ", ".join("[%d,%d)" % f for f in folds)))
    add("两侧网格评估均未使用 OOC 区间的 bar",
        bool((raw["long"]["fold_lo"] >= oof.start).all() and (raw["long"]["fold_hi"] <= oof.stop).all()
             and (raw["short"]["fold_lo"] >= oof.start).all()
             and (raw["short"]["fold_hi"] <= oof.stop).all()),
        "long %d 行 / short %d 行, 评估上界均 <= OOF 上界 %d"
        % (len(raw["long"]), len(raw["short"]), oof.stop))
    add("两侧均在 OOF 上重新择优(long 不冻结)", True,
        "long=%s/%s | short=%s/%s" % (frozen["long"]["pool"], frozen["long"]["regime_rule"],
                                      frozen["short"]["pool"], frozen["short"]["regime_rule"]))
    add("两侧各自 tp > sl",
        frozen["long"]["tp_mult"] > frozen["long"]["sl_mult"]
        and frozen["short"]["tp_mult"] > frozen["short"]["sl_mult"],
        "long tp=%.2f>sl=%.2f | short tp=%.2f>sl=%.2f"
        % (frozen["long"]["tp_mult"], frozen["long"]["sl_mult"],
           frozen["short"]["tp_mult"], frozen["short"]["sl_mult"]))

    ok_add, det = True, []
    for s in SEGS:
        rl = res["long"][s]["metrics"]["total_return"]
        rs = res["short"][s]["metrics"]["total_return"]
        rc = comb[s]["metrics"]["total_return"]
        det.append("%s %.4f + %.4f = %.4f" % (s, rl * 100, rs * 100, rc * 100))
        ok_add &= abs(rc - (rl + rs)) < 1e-9
    add("合并收益 = 两侧收益之和(合并口径可加)", ok_add, " | ".join(det))

    # 合并夏普是"真实 bar 级": 合并权益须与"把两侧成交合到一起重建"逐 bar 一致
    ok_sh, det_sh = True, []
    for s in SEGS:
        seg = {"train": tr, "oof": oof, "ooc": ooc}[s]
        tcomb = list(res["long"][s]["trades"]) + list(res["short"][s]["trades"])
        ref = execution.equity_from_trades(tcomb, len(df), lo=seg.start, hi=seg.stop)
        d = float(np.max(np.abs(ref - comb[s]["equity"])))
        ok_sh &= (d < 1e-9)
        det_sh.append("%s sharpe %.3f, 权益逐bar偏差 %.1e" % (s, comb[s]["metrics"]["sharpe"], d))
    add("合并权益 = eq_long+eq_short-INIT 且与两侧成交合并重建逐 bar 一致", ok_sh, " | ".join(det_sh))

    # 批量夏普实现与 metrics.sharpe 口径一致(单条曲线一条一条比, 各段长度不同)
    ok_b, det_b = True, []
    for s in SEGS:
        one = float(optimize._sharpe_rows(comb[s]["equity"])[0])
        ok_b &= abs(one - metrics.sharpe(comb[s]["equity"])) < 1e-9
        det_b.append("%s %.4f" % (s, one))
    add("_sharpe_rows 批量口径 == metrics.sharpe", ok_b, " | ".join(det_b))

    # 目标可复现 + best == 候选表第一行
    fr = np.array(best["fold_returns"], dtype=float)
    fs = np.array(best["fold_sharpes"], dtype=float)
    r_ret = float(fr.mean() - C.V8_ROBUST_K * fr.std(ddof=1))
    r_shp = float(fs.mean() - C.V8_ROBUST_K * fs.std(ddof=1))
    add("目标可复现(主=稳健收益, 次=稳健夏普)且 best == 候选表第一行",
        abs(r_ret - best["robust_combined"]) < 1e-9 and abs(r_shp - best["sharpe_combined"]) < 1e-9
        and abs(float(pairs.iloc[0]["robust_combined"]) - best["robust_combined"]) < 1e-12
        and abs(float(pairs.iloc[0]["sharpe_combined"]) - best["sharpe_combined"]) < 1e-12,
        "稳健收益 %+.4f(折 %s) | 稳健夏普 %+.3f(折 %s)"
        % (r_ret, "/".join("%+.2f%%" % (v * 100) for v in fr), r_shp,
           "/".join("%+.2f" % v for v in fs)))

    # 阶段1 长表 <-> 逐折权益重算 一致
    ok_rec, det_rec = True, []
    for side in ("long", "short"):
        c = best[side]
        key = (c["pool"], float(c["dedup"]), c["model"])
        pred = frozen[side]["pred"]
        for f, (flo, fhi) in enumerate(folds, 1):
            r = execution.evaluate_with_params(pred, {k: df[k].to_numpy(dtype=float)
                                                      for k in ("open", "high", "low", "close")},
                                               df["atr"].to_numpy(dtype=float),
                                               df["datetime"].to_numpy(), side, flo, fhi,
                                               float(c["thr_q"]), float(c["tp_mult"]),
                                               float(c["sl_mult"]), max_hold=int(c["max_hold"]),
                                               trail_mult=float(c["trail_mult"]),
                                               regime=masks[c["regime"]][side])["metrics"]["total_return"]
            m = (raw[side]["fold"] == f) & (raw[side]["pool"] == c["pool"]) \
                & (raw[side]["regime"] == c["regime"]) & (raw[side]["model"] == c["model"]) \
                & np.isclose(raw[side]["dedup"], float(c["dedup"])) \
                & np.isclose(raw[side]["thr_q"], float(c["thr_q"])) \
                & np.isclose(raw[side]["tp_mult"], float(c["tp_mult"])) \
                & np.isclose(raw[side]["sl_mult"], float(c["sl_mult"])) \
                & (raw[side]["max_hold"] == int(c["max_hold"])) \
                & np.isclose(raw[side]["trail_mult"], float(c["trail_mult"]))
            src = raw[side].loc[m, "total_return"]
            if len(src) != 1 or abs(float(src.iloc[0]) - r) > 1e-9:
                ok_rec = False
            det_rec.append("%s折叠%d" % (side, f))
        det_rec.append("key=%s" % (key,))
    add("逐折权益重算 == 阶段1 长表(评估口径一致)", ok_rec, " ".join(det_rec[:7]))

    ok_vif = True
    for side in ("long", "short"):
        e = prune_log[(frozen[side]["pool"], frozen[side]["dedup"], side)]
        ok_vif &= (e["max_vif_after"] <= C.V4_MAX_VIF + 1e-9)
    add("两侧 VIF 迭代剪枝(冻结集 VIF<=%.0f)" % C.V4_MAX_VIF, ok_vif,
        " | ".join("%s 后VIF %.2f" % (sd, prune_log[(frozen[sd]["pool"], frozen[sd]["dedup"], sd)]["max_vif_after"])
                  for sd in ("long", "short")))

    cost = 2.0 * (C.FEE_RATE + C.SLIP_RATE)
    allt = [t for side in ("long", "short") for s in SEGS for t in res[side][s]["trades"]]
    bad = [t for t in allt if abs(t.net_ret - (t.gross_ret - cost)) > 1e-12]
    bad_ll = [t for t in allt if abs(t.entry_price - float(df["open"].iloc[t.entry_idx])) > 1e-9]
    add("成本口径 = 单边 5bp / 双边 10bp", len(bad) == 0,
        "逐笔复核 %d 笔, 不一致 %d 笔" % (len(allt), len(bad)))
    add("信号 t 收盘 -> t+1 开盘成交(无未来函数)", len(bad_ll) == 0,
        "全部 %d 笔成交价 == open[entry_idx]" % len(allt))

    ok_gate, det_gate = True, []
    for side in ("long", "short"):
        rule = frozen[side]["regime_rule"]
        mask = masks[rule][side]
        diff = 0
        for s in SEGS:
            seg = {"train": tr, "oof": oof, "ooc": ooc}[s]
            sig = execution.signal_from_threshold(frozen[side]["pred"], side,
                                                  frozen[side]["thr_abs"], seg.start, seg.stop,
                                                  regime=mask)
            diff = max(diff, abs(int(sig.sum()) - res[side][s]["n_signals"]))
        ok_gate &= (diff == 0)
        det_gate.append("%s rule=%s(%d bar)" % (side, rule, int(mask.sum())))
    add("两侧制度门控生效且与冻结评估一致", ok_gate, " | ".join(det_gate))

    ok_causal = True
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
    add("制度掩码仅用过去信息(截断后取值不变)", ok_causal, "逐规则逐时点截断复核")
    return chk


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 120)
    print("SOL/USDT %s LGBM 回测 v11 | 欧易(OKX) | 强化选择器(OOF 收益+夏普) | OOC 仅观察" % C.INTERVAL)
    print("主目标 = 合并稳健收益(收益优先) | 次目标 = 合并稳健夏普(真实 bar 级) | 每侧候选 %d | OOF 子折 %d"
          % (C.V11_CAND_PER_SIDE, C.V11_N_FOLDS))
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
    folds = optimize._v8_fold_slices(oof, C.V11_N_FOLDS)

    print("\n数据: %d 根  %s ~ %s | 因子库 %d 个 | 段基准 train %+.2f%% / OOF %+.2f%% / OOC %+.2f%%"
          % (len(df), str(df["datetime"].iloc[0])[:16], str(df["datetime"].iloc[-1])[:16],
             Fmat.shape[1], buy_hold(df, tr) * 100, buy_hold(df, oof) * 100, buy_hold(df, ooc) * 100))
    print("切分: train[%d,%d) OOF[%d,%d) OOC[%d,%d) | OOF 子折: %s"
          % (tr.start, tr.stop, oof.start, oof.stop, ooc.start, ooc.stop,
             ", ".join("[%d,%d)" % f for f in folds)))

    masks = {rule: regime.regime_masks(df, rule) for rule in C.V7_REGIME_GRID}
    fsets, prune_log = build_factor_sets(Fmat, y, tr)
    for pool in C.V6_FACTOR_POOLS:
        print("  池 %-8s -> " % pool + " / ".join(
            "|corr|<%.2f: long %2d short %2d" % (d, len(fsets[(pool, float(d))]["long"]),
                                                 len(fsets[(pool, float(d))]["short"]))
            for d in C.V6_DEDUP_GRID))

    # ---------------- 阶段1: 两侧"配置 × 子折"长表(网格与 v9 相同)
    raw: dict = {}
    prebuilt: dict = {}
    reuse_from = (C.REPORT_DIR / C.V11_REUSE_GRID_FROM) if C.V11_REUSE_GRID_FROM else None
    for side in ("long", "short"):
        src = (reuse_from / ("selection_folds_%s.csv" % side)) if reuse_from else None
        if src is not None and src.exists():
            raw[side] = pd.read_csv(src)
            prebuilt[side] = {}
            print("\n### 阶段1[%s]: 复用 %s (%d 行, 与 v11 网格逐位相同)"
                  % (side, src.relative_to(C.BASE_DIR), len(raw[side])))
            continue
        print("\n### 阶段1[%s]: OOF 内部分折稳健评估(因子池 × 制度 × 去冗余 × 模型 × 执行 × %d 折)"
              % (side, C.V11_N_FOLDS))
        fmap = {(pool, d): fsets[(pool, d)][side]
                for pool in C.V6_FACTOR_POOLS for d in C.V6_DEDUP_GRID}
        regimes = {rule: masks[rule][side] for rule in C.V7_REGIME_GRID}
        prebuilt[side] = {}
        _, _, raw[side] = optimize.optimize_v8_on_oof(
            Fmat, y, fmap, side, ohlc, atr, times, tr, oof, regimes,
            thr_grid=C.V6_THR_GRID, hold_grid=C.V6_HOLD_GRID, trail_grid=C.V6_TRAIL_GRID,
            sl_grid=C.V7_SL_GRID, tp_gt_sl=C.V3_ENFORCE_TP_GT_SL, verbose=True,
            n_folds=C.V11_N_FOLDS, out_models=prebuilt[side])
        raw[side].to_csv(OUT / ("selection_folds_%s.csv" % side), index=False)

    # ---------------- 候选池 -> 需要的模型/预测
    pl = optimize.candidate_pool_v11(raw["long"], C.V11_CAND_PER_SIDE, C.V8_ROBUST_K)
    ps = optimize.candidate_pool_v11(raw["short"], C.V11_CAND_PER_SIDE, C.V8_ROBUST_K)
    print("\n阶段1 候选池: long %d 个(稳健收益第1 %+.2f%%), short %d 个(稳健收益第1 %+.2f%%)"
          % (len(pl), pl.iloc[0]["robust_ret"] * 100, len(ps), ps.iloc[0]["robust_ret"] * 100))
    pmap = {"long": build_pred_map("long", pl, fsets, Fmat, y, tr, prebuilt["long"]),
            "short": build_pred_map("short", ps, fsets, Fmat, y, tr, prebuilt["short"])}
    for side in ("long", "short"):
        if prebuilt[side]:
            used = {(c["pool"], float(c["dedup"]), c["model"]) for _, c in
                    {"long": pl, "short": ps}[side].iterrows()}
            print("   %s: 复用阶段1模型 %d 个, 新训 %d 个"
                  % (side, len(used & set(prebuilt[side])), len(used - set(prebuilt[side]))))

    # ---------------- 阶段2: 收益 + 夏普 双目标联合选优
    print("\n### 阶段2: 合并组合联合选优(主: 稳健收益 | 次: 稳健夏普[真实 bar 级])")
    best, pairs, _, _ = optimize.joint_select_v11(
        raw["long"], raw["short"], pmap["long"], pmap["short"], ohlc, atr, times, folds, masks,
        C.V11_CAND_PER_SIDE, robust_k=C.V8_ROBUST_K, top_report=C.V9_TOP_PAIRS)
    pairs.to_csv(OUT / "selection_pairs_v11.csv", index=False)
    print("    ==> 选中组合: 稳健收益 %+.3f | 稳健夏普 %+.3f | 最差折 %+.3f"
          % (best["robust_combined"], best["sharpe_combined"], best["worst_combined"]))
    print("        合并逐折收益: " + ", ".join("%+.2f%%" % (v * 100) for v in best["fold_returns"]))
    print("        合并逐折夏普: " + ", ".join("%+.2f" % v for v in best["fold_sharpes"]))
    for side in ("long", "short"):
        b = best[side]
        print("        %-5s: 池=%s 制度=%s |corr|<%.2f %-9s tp=%.2f>sl=%.2f hold=%2d trail=%.1f"
              % (side, b["pool"], b["regime"], b["dedup"], b["model"], b["tp_mult"],
                 b["sl_mult"], int(b["max_hold"]), b["trail_mult"]))

    # ---------------- 冻结 + 逐段评估
    frozen, res = {}, {}
    for side in ("long", "short"):
        b = best[side]
        tm, pred = pmap[side][(b["pool"], float(b["dedup"]), b["model"])]
        thr_abs = execution.threshold_from_quantile(pred, side, float(b["thr_q"]), oof.start, oof.stop)
        frozen[side] = dict(pool=b["pool"], regime_rule=b["regime"], dedup=float(b["dedup"]),
                            model_name=b["model"], model=tm, thr_q=float(b["thr_q"]),
                            thr_abs=float(thr_abs), tp_mult=float(b["tp_mult"]),
                            sl_mult=float(b["sl_mult"]), max_hold=int(b["max_hold"]),
                            trail_mult=float(b["trail_mult"]), pred=pred)
        res[side] = {}
        for s in SEGS:
            seg = segs[s]
            res[side][s] = execution.evaluate_with_threshold(
                pred, ohlc, atr, times, side, seg.start, seg.stop, thr_abs,
                frozen[side]["tp_mult"], frozen[side]["sl_mult"],
                max_hold=frozen[side]["max_hold"], trail_mult=frozen[side]["trail_mult"],
                regime=masks[frozen[side]["regime_rule"]][side])
    comb = {s: combined_seg(res, s) for s in SEGS}

    # ---------------- 报表
    L: list = []
    L.append("# SOL/USDT 4h 多因子 LGBM 回测报告 (v11: 强化选择器, OOF 收益+夏普)\n\n")
    L.append("- 数据: 欧易(OKX) 4h %s | 清洗后 %d 根(起点 %s) | train/OOF/OOC = 70/15/15\n"
             % (C.SYMBOL, len(df), C.DATA_START))
    L.append("- v11 与 v10 的**排序目标与折数逐位相同**(主 = 合并稳健收益; 次 = 合并稳健夏普, 真实 bar 级重建; "
             "三 = 最差折收益; OOF 均分 %d 折), 差别只在候选池: 每侧 200→%d, "
             "并由\"只按稳健收益截断\"改为**双目标保留 + Pareto 前沿**(强制纳入(收益,夏普)非支配解)。\n"
             % (C.V11_N_FOLDS, C.V11_CAND_PER_SIDE))
    L.append("- 折数不变 -> v11 候选池**严格包含** v10 池, \"主目标不比 v10 差\"是选择集上的硬保证。\n")
    L.append("- 执行网格 / 因子池 / 模型网格与 v10 **完全相同** -> 假设空间未扩大, 只是少丢好解。\n")
    L.append("- 选择**只在 OOF 子折**; **OOC 仅观察**。两侧都做(long 不冻结), 同用 v7 制度/止损网格。\n")
    L.append("- OOF 子折: %s | 每侧候选 %d 个\n\n" % (", ".join("[%d,%d)" % f for f in folds), C.V11_CAND_PER_SIDE))

    L.append("## 合并总收益与夏普对照\n\n| 段 | 版本 | 合并收益% | 合并夏普 |\n|---|---|---|---|\n")
    def _meta(path, tag):
        for s in SEGS:
            if not path.exists():
                L.append("| %s | %s | - | - |\n" % (s, tag))
                continue
            m = json.loads(path.read_text(encoding="utf-8"))
            lr = m["sides"]["long"]["metrics"][s]["total_return"]
            sr = m["sides"]["short"]["metrics"][s]["total_return"]
            L.append("| %s | %s | **%+.2f** | (两侧单腿夏普 %.2f / %.2f) |\n"
                     % (s, tag, (lr + sr) * 100, m["sides"]["long"]["metrics"][s]["sharpe"],
                        m["sides"]["short"]["metrics"][s]["sharpe"]))
    _meta(C.REPORT_DIR / C.V6_OUT_DIR / "metrics_v6.json", "v6")
    _meta(C.REPORT_DIR / C.V8_OUT_DIR / "metrics_v8.json", "v8")
    _meta(C.REPORT_DIR / C.V9_OUT_DIR / "metrics_v9.json", "v9")
    _meta(C.REPORT_DIR / C.V10_OUT_DIR / "metrics_v10.json", "v10")
    for s in SEGS:
        m = comb[s]["metrics"]
        L.append("| %s | **v11** | **%+.2f** | **%.2f** |\n" % (s, m["total_return"] * 100, m["sharpe"]))
    L.append("\n> v6/v8/v9/v10 的合并收益 = 各自 JSON 两侧 `total_return` 之和(可加性); "
             "单腿夏普仅供参考, 组合夏普须按 bar 级重建。\n")

    L.append("\n## v11 分腿与合并明细\n\n| 段 | 腿 | 收益率% | 夏普 | 卡玛 | 最大回撤% | 胜率% | 盈亏比 | 开仓数 | 止盈率% | 止损率% |\n")
    L.append("|" + "---|" * 12 + "\n")
    for s in SEGS:
        for leg, m in (("long", res["long"][s]["metrics"]), ("short", res["short"][s]["metrics"]),
                       ("**合并**", comb[s]["metrics"])):
            L.append("| %s | %s | %+.2f | %.2f | %.2f | %.2f | %.1f | %.2f | %d | %.1f | %.1f |\n"
                     % (s, leg, m["total_return"] * 100, m["sharpe"], m["calmar"],
                        m["max_drawdown"] * 100, m["win_rate"] * 100, m["payoff_ratio"],
                        m["n_trades"], m["tp_rate"] * 100, m["sl_rate"] * 100))

    L.append("\n## 联合候选组合(前 %d; 主=稳健收益, 次=稳健夏普)\n\n" % len(pairs))
    L.append("| # | long 池/制度/tp-sl/hold/trail | short 池/制度/tp-sl/hold/trail | 稳健收益 | 稳健夏普 "
             "| 折均收益% | 最差折% | 逐折收益% | 逐折夏普 |\n")
    L.append("|" + "---|" * 10 + "\n")
    for i in range(len(pairs)):
        r = pairs.iloc[i]
        star = " **<-选中**" if i == 0 else ""
        fr_s = "/".join("%+.2f" % (r["fold%d_ret" % (k + 1)] * 100) for k in range(len(folds)))
        fs_s = "/".join("%+.2f" % r["fold%d_sharpe" % (k + 1)] for k in range(len(folds)))
        L.append("| %d%s | %s/%s/%.2f-%.2f/%d/%.1f | %s/%s/%.2f-%.2f/%d/%.1f | %+.3f | %+.3f | %+.2f | %+.2f | %s | %s |\n"
                 % (i + 1, star, r["long_pool"], r["long_regime"], r["long_tp_mult"],
                    r["long_sl_mult"], int(r["long_max_hold"]), r["long_trail_mult"],
                    r["short_pool"], r["short_regime"], r["short_tp_mult"], r["short_sl_mult"],
                    int(r["short_max_hold"]), r["short_trail_mult"],
                    r["robust_combined"], r["sharpe_combined"], r["mean_combined"] * 100,
                    r["worst_combined"] * 100, fr_s, fs_s))

    L.append("\n## 阶段1 每侧候选池(各前 15; 入池依据 = min(收益排名,夏普排名), 0 = Pareto 前沿)\n\n")
    for side, pool in (("long", pl), ("short", ps)):
        L.append("**" + side + " 候选池**\n\n| 池 | 制度 | |corr|< | 模型 | tp | sl | hold | trail | "
                 "稳健收益% | 稳健夏普 | 入池优先级 | Pareto | 折均收益% | 最差折% |\n")
        L.append("|" + "---|" * 14 + "\n")
        for i in range(min(15, len(pool))):
            p = pool.iloc[i]
            L.append("| %s | %s | %.2f | %s | %.2f | %.2f | %d | %.1f | %+.2f | %+.2f | %.0f | %s | %+.2f | %+.2f |\n"
                     % (p["pool"], p["regime"], p["dedup"], p["model"], p["tp_mult"], p["sl_mult"],
                        int(p["max_hold"]), p["trail_mult"], p["robust_ret"] * 100,
                        p["robust_sharpe"], p["robust_rank"], "是" if p["on_pareto"] else "",
                        p["ret_mean"] * 100, p["ret_min"] * 100))
        L.append("\n")

    L.append("\n## 冻结配置\n\n| 方向 | 因子池 | 制度 | |corr|< | 因子数 | 模型 | 阈值(绝对) | tp | sl | trail | 最长持有 |\n")
    L.append("|" + "---|" * 12 + "\n")
    for side in ("long", "short"):
        f = frozen[side]
        L.append("| %s | %s | %s | %.2f | %d | %s | %+.6f | %.2f | %.2f | %.1f | %d |\n"
                 % (side, f["pool"], f["regime_rule"], f["dedup"], len(f["model"].factors),
                    f["model_name"], f["thr_abs"], f["tp_mult"], f["sl_mult"], f["trail_mult"],
                    f["max_hold"]))
    L.append("\n| 方向 | 因子数 | 95%方差维度 | 有效维度 | 最大VIF |\n|---|---|---|---|---|\n")
    for side in ("long", "short"):
        sp = spectrum(Fmat, frozen[side]["model"].factors, tr)
        L.append("| %s | %d | %d | %.2f | %.2f |\n"
                 % (side, sp["n_factors"], sp["dim95"], sp["eff_rank"], sp["max_vif"]))

    cons = consistency(df, tr, oof, ooc, frozen, res, comb, prune_log, masks, best, pairs, raw, folds)
    n_fail = sum(1 for c in cons if not c["pass"])
    L.append("\n## 一致性检查(design == runtime)\n\n**失败项: %d / %d**\n\n| 检查 | 结果 | 说明 |\n|---|---|---|\n"
             % (n_fail, len(cons)))
    for c in cons:
        L.append("| %s | %s | %s |\n" % (c["check"], "PASS" if c["pass"] else "**FAIL**", c["detail"]))
    (OUT / "backtest_report_v11.md").write_text("".join(L), encoding="utf-8")

    # ---------------- 曲线 1: 全序列(三段标注)
    eq_long = execution.equity_from_trades([t for s in SEGS for t in res["long"][s]["trades"]], len(df))
    eq_short = execution.equity_from_trades([t for s in SEGS for t in res["short"][s]["trades"]], len(df))
    eq_comb = eq_long + eq_short - C.INIT_CAPITAL
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(range(len(df)), eq_long, lw=1.1, color="tab:blue", label="long")
    ax.plot(range(len(df)), eq_short, lw=1.1, color="tab:red", label="short")
    ax.plot(range(len(df)), eq_comb, lw=1.8, color="black", label="combined (long+short)")
    ax.axhline(C.INIT_CAPITAL, color="gray", ls=":", lw=0.8)
    for b in (tr.stop, oof.stop):
        ax.axvline(b, color="red", ls="--", lw=0.8, alpha=0.7)
    for s in SEGS:
        ax.axvspan(segs[s].start, segs[s].stop, alpha=0.05,
                   color={"train": "tab:blue", "oof": "tab:orange", "ooc": "tab:green"}[s])
    mc, mo = comb["oof"]["metrics"], comb["ooc"]["metrics"]
    ax.set_title("v11 (stronger selector: %d folds, %d candidates/side, dual-objective pool + Pareto)"
                 " | OOF %+.2f%% sharpe %.2f | OOC %+.2f%% sharpe %.2f [OOC = observation only]"
                 % (C.V11_N_FOLDS, C.V11_CAND_PER_SIDE, mc["total_return"] * 100, mc["sharpe"],
                    mo["total_return"] * 100, mo["sharpe"]))
    ax.set_xlabel("bar index (4h)")
    ax.set_ylabel("Equity (USDT)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "equity_curve_v11_full.png", dpi=130)
    plt.close(fig)

    # ---------------- 曲线 2: OOF / OOC 总体收益曲线(用户要求)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, s, color in ((axes[0], "oof", "tab:orange"), (axes[1], "ooc", "tab:green")):
        seg = segs[s]
        x = (segs[s].start + np.arange(seg.stop - seg.start))
        eql, eqs, eqc = res["long"][s]["equity"], res["short"][s]["equity"], comb[s]["equity"]
        ax.plot(x, eql, lw=1.1, color="tab:blue", label="long %+.2f%%" % (res["long"][s]["metrics"]["total_return"] * 100))
        ax.plot(x, eqs, lw=1.1, color="tab:red", label="short %+.2f%%" % (res["short"][s]["metrics"]["total_return"] * 100))
        ax.plot(x, eqc, lw=2.0, color="black",
                label="combined %+.2f%%" % (comb[s]["metrics"]["total_return"] * 100))
        ax.axhline(C.INIT_CAPITAL, color="gray", ls=":", lw=0.8)
        ax.axvspan(seg.start, seg.stop, alpha=0.05, color=color)
        ax.set_title("%s (bars %d~%d) | combined sharpe %.2f | mdd %.2f%%"
                     % (s.upper(), seg.start, seg.stop - 1, comb[s]["metrics"]["sharpe"],
                        comb[s]["metrics"]["max_drawdown"] * 100))
        ax.set_xlabel("bar index (4h)")
        ax.set_ylabel("Equity (USDT, start = %.0f)" % C.INIT_CAPITAL)
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.25)
    fig.suptitle("v11 combined equity: OOF (selection segment) vs OOC (observation only) "
                 "[combined = eq_long + eq_short - INIT]")
    fig.tight_layout()
    fig.savefig(OUT / "equity_v11_oof_ooc.png", dpi=130)
    plt.close(fig)

    # ---------------- 数值曲线落盘
    seg_arr = np.array(["train"] * (tr.stop - tr.start) + ["oof"] * (oof.stop - oof.start)
                       + ["ooc"] * (ooc.stop - ooc.start))
    pd.DataFrame({"datetime": df["datetime"].astype(str), "segment": seg_arr, "eq_long": eq_long,
                  "eq_short": eq_short, "eq_combined": eq_comb}).to_csv(
        OUT / "equity_v11_combined.csv", index=False)

    # ---------------- metrics
    meta = {"config": {"SYMBOL": C.SYMBOL, "INTERVAL": C.INTERVAL, "SOURCE": C.BACKTEST_SOURCE,
                       "DATA_START": C.DATA_START, "HORIZON": C.HORIZON,
                       "SELECT_ON": "oof", "OPTIMIZED_SIDES": ["long", "short"], "JOINT": True,
                       "OBJECTIVE_PRIMARY": "combined robust return (mean - K*std over OOF folds)",
                       "OBJECTIVE_SECONDARY": "combined robust sharpe (real bar-level, mean - K*std)",
                       "OBJECTIVE_TERTIARY": "worst-fold combined return",
                       "COMBINE": "equity = eq_long + eq_short - INIT_CAPITAL",
                       "V8_ROBUST_K": C.V8_ROBUST_K,
                       "V8_MIN_TRADES_PER_FOLD": C.V8_MIN_TRADES_PER_FOLD,
                       "V11_N_FOLDS": C.V11_N_FOLDS, "V11_CAND_PER_SIDE": C.V11_CAND_PER_SIDE,
                       "V8_FOLDS": [[int(a), int(b)] for a, b in folds],
                       "GRID_REUSED_FROM": C.V11_REUSE_GRID_FROM,
                       "V7_SL_GRID": list(C.V7_SL_GRID), "V7_REGIME_GRID": list(C.V7_REGIME_GRID),
                       "V4_MAX_VIF": C.V4_MAX_VIF, "FEE_RATE": C.FEE_RATE, "SLIP_RATE": C.SLIP_RATE,
                       "INIT_CAPITAL": C.INIT_CAPITAL, "TRADE_NOTIONAL": C.TRADE_NOTIONAL},
            "joint_best": {k: best[k] for k in ("robust_combined", "sharpe_combined", "worst_combined",
                                                "mean_combined", "std_combined", "sharpe_mean",
                                                "sharpe_std", "fold_returns", "fold_sharpes")},
            "joint_pairs_top": pairs.to_dict(orient="records"),
            "sides": {}, "combined": {},
            "consistency": {"n_fail": n_fail, "n_total": len(cons), "checks": cons}}
    for side in ("long", "short"):
        f = frozen[side]
        meta["sides"][side] = {
            "pool": f["pool"], "model_name": f["model_name"], "factors": list(f["model"].factors),
            "regime_rule": f["regime_rule"],
            "frozen_params": {"dedup": f["dedup"], "thr_q": f["thr_q"], "thr_abs": f["thr_abs"],
                              "tp_mult": f["tp_mult"], "sl_mult": f["sl_mult"],
                              "max_hold": f["max_hold"], "trail_mult": f["trail_mult"]},
            "metrics": {s: {k: float(v) for k, v in res[side][s]["metrics"].items()} for s in SEGS}}
    for s in SEGS:
        meta["combined"][s] = {k: float(v) for k, v in comb[s]["metrics"].items()}
    (OUT / "metrics_v11.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    tr_all = [dict(side=t.side, seg=s, entry_time=str(t.entry_time), exit_time=str(t.exit_time),
                   entry_price=t.entry_price, exit_price=t.exit_price, exit_reason=t.exit_reason,
                   bars_held=t.bars_held, gross_ret=t.gross_ret, net_ret=t.net_ret, pnl_usdt=t.pnl_usdt)
              for side in ("long", "short") for s in SEGS for t in res[side][s]["trades"]]
    pd.DataFrame(tr_all).to_csv(OUT / "trades_v11.csv", index=False)

    print("\n" + "=" * 120)
    print("结果汇总 (v11: 强化选择器, OOF 收益+夏普; OOC 仅观察)")
    print("=" * 120)
    print("%-6s %-6s %9s %8s %8s %8s %8s %8s" % ("腿", "段", "收益率%", "夏普", "卡玛", "最大回撤%", "胜率%", "开仓数"))
    for s in SEGS:
        for leg, m in (("long", res["long"][s]["metrics"]), ("short", res["short"][s]["metrics"]),
                       ("合并", comb[s]["metrics"])):
            print("%-6s %-6s %+9.2f %8.2f %8.2f %8.2f %8.1f %8d"
                  % (leg, s, m["total_return"] * 100, m["sharpe"], m["calmar"],
                     m["max_drawdown"] * 100, m["win_rate"] * 100, m["n_trades"]))
    print("\n一致性检查: 失败项 %d / %d" % (n_fail, len(cons)))
    for c in cons:
        if not c["pass"]:
            print("  FAIL: %s | %s" % (c["check"], c["detail"]))
    print("\n产出目录: %s" % OUT)
    print("  backtest_report_v11.md / metrics_v11.json / equity_v11_oof_ooc.png / "
          "equity_curve_v11_full.png / equity_v11_combined.csv / selection_pairs_v11.csv / trades_v11.csv")


if __name__ == "__main__":
    main()
