#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SOL/USDT 4h 多因子 LGBM 回测 (v8: OOF 内部分折 + 稳健选优)。

依据对 v7 的归因: v7 空头 OOF +6.31% 而 OOC 仅 +1.11%, 且 `selection_grid_short.csv`
在**整段 OOF** 上比较了 193,536 组 —— "最优"只是 19 万个含噪估计的**最大值**
(winner's curse, 天然高估); v7 选中的 5 个执行参数里更有 3 个落在网格边界。

v8 **不改选择段**(仍是 OOF, OOC 仍只观察), 只改"怎么选"这条纪律:
  1. 把 OOF 段再切成 V8_N_FOLDS 个**连续子折**, 每组配置在**每个子折**上独立评估
     (阈值分位按子折自身预测计算; 子折仍在 OOF 内, 不触碰 OOC);
  2. 稳健得分 = 各折夏普**均值 - V8_ROBUST_K × 标准差**, 次目标为**最差折夏普**,
     替代原来的"整段 OOF 取最大 total_return" —— 用折间一致性压低 max 效应;
  3. 单折成交数门槛 V8_MIN_TRADES_PER_FOLD, 淘汰"只在个别子折蒙对"的低频组合。

与 v7 一样, v8 **只重做空头**; 多头沿用 v6 冻结配置。除选择纪律外, 因子/训练/网格/
成本/资金/选择段与 v7 完全一致, 便于横向对比。

前置: 先运行 run_backtest_v6.py 生成 reports/v6/metrics_v6.json。
运行: python3 run_backtest_v8.py
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

OUT = C.REPORT_DIR / C.V8_OUT_DIR
V6_META = C.REPORT_DIR / C.V6_OUT_DIR / "metrics_v6.json"
V7_META = C.REPORT_DIR / C.V7_OUT_DIR / "metrics_v7.json"
SEGS = ("train", "oof", "ooc")
MKEYS = ("total_return", "sharpe", "calmar", "max_drawdown", "win_rate",
         "payoff_ratio", "n_trades", "tp_rate", "sl_rate", "timeout_rate")
pd.set_option("display.width", 240)


def load_clean() -> pd.DataFrame:
    """载入欧易原始数据 -> 清洗 -> 对齐 DATA_START(与 v6/v7 完全同口径, 保证可比)。"""
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
    """(因子池 × 去冗余) 在 train 上产出多空因子集 + VIF 剪枝(v8 与 v6/v7 同口径)。"""
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


def _selected_folds(raw: pd.DataFrame, sel: pd.Series) -> pd.DataFrame:
    """从"配置×子折"长表里取出**选中配置**的各子折明细。

    sel 取 `agg.iloc[0]`(稳健表第一行) —— 它含 _V8_CONFIG_KEYS 的全部维度。
    """
    m = pd.Series(True, index=raw.index)
    for k in optimize._V8_CONFIG_KEYS:
        v = sel[k]
        m &= (raw[k] == (float(v) if k in ("dedup", "thr_q", "tp_mult", "sl_mult",
                                           "trail_mult") else v))
    return raw[m].sort_values("fold").reset_index(drop=True)


def consistency(df, tr, oof, ooc, frozen, res, prune_log, masks, v6_long, best, agg, raw) -> list:
    """design == runtime 断言(v8 特有项: OOF 子折稳健选优 / OOC 未参与选择)。"""
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
    add("选择只发生在 OOF, 且仅空头被重优化", True,
        "空头选择段 OOF[%d,%d); 多头沿用 v6 冻结配置(未重新择优)" % (oof.start, oof.stop))

    # ---- v8 特有: 子折全部落在 OOF 内, 连续且不重叠, OOC 未参与
    folds = optimize._v8_fold_slices(oof, C.V8_N_FOLDS)
    contiguous = all(folds[i][1] == folds[i + 1][0] for i in range(len(folds) - 1))
    inside = (folds[0][0] == oof.start and folds[-1][1] == oof.stop
              and all(flo >= oof.start and fhi <= oof.stop for flo, fhi in folds))
    add("OOF 子折连续/不重叠/完全落在 OOF 内(未触碰 OOC)", contiguous and inside,
        "%d 折: %s" % (len(folds), ", ".join("[%d,%d)" % f for f in folds)))
    add("网格评估未使用 OOC 区间的 bar",
        bool((raw["fold_lo"] >= oof.start).all() and (raw["fold_hi"] <= oof.stop).all()),
        "全部 %d 行评估区间落在 [%d,%d] 内, 上界 = OOF 上界 %d"
        % (len(raw), int(raw["fold_lo"].min()), int(raw["fold_hi"].max()), oof.stop))

    fd = frozen["short"]
    add("止盈/止损硬约束 tp > sl", fd["tp_mult"] > fd["sl_mult"],
        "short tp=%.2f > sl=%.2f" % (fd["tp_mult"], fd["sl_mult"]))

    add("多头沿用 v6 冻结配置(因子/制度/参数一致)",
        set(frozen["long"]["model"].factors) == set(v6_long["factors"])
        and frozen["long"]["regime_rule"] == v6_long["regime_rule"]
        and abs(frozen["long"]["thr_abs"] - v6_long["frozen_params"]["thr_abs"]) < 1e-12,
        "v6 long: 池%s rule=%s 因子%d | v8 复现因子%d"
        % (v6_long["pool"], v6_long["regime_rule"], len(v6_long["factors"]),
           len(frozen["long"]["model"].factors)))

    # ---- v8 特有: 稳健得分 = 均值 - K×标准差, 且选中项确为稳健表第一
    sel = agg.iloc[0]
    robust_ok = abs(float(sel["robust"])
                    - (float(sel["sharpe_mean"]) - C.V8_ROBUST_K * float(sel["sharpe_std"]))) < 1e-9
    add("稳健得分 = 各折夏普均值 - %.1f×标准差 且为稳健表第一" % C.V8_ROBUST_K,
        robust_ok and (str(sel["pool"]) == fd["pool"])
        and abs(float(sel["dedup"]) - fd["dedup"]) < 1e-12,
        "选中 稳健=%.3f (均值%.3f - %.1f×std%.3f) | 正收益折 %d/%d"
        % (sel["robust"], sel["sharpe_mean"], C.V8_ROBUST_K, sel["sharpe_std"],
           int(sel["n_pos_folds"]), int(sel["n_folds"])))
    add("单折成交数门槛 V8_MIN_TRADES_PER_FOLD=%d" % C.V8_MIN_TRADES_PER_FOLD,
        int(sel["n_min"]) >= C.V8_MIN_TRADES_PER_FOLD,
        "选中配置单折最少笔数 %d, 合计 %d" % (int(sel["n_min"]), int(sel["n_total"])))

    rule = fd["regime_rule"]
    mask = masks[rule]["short"]
    m_sig = 0
    for s in SEGS:
        seg = {"train": tr, "oof": oof, "ooc": ooc}[s]
        sig_gated = execution.signal_from_threshold(fd["pred"], "short", fd["thr_abs"],
                                                    seg.start, seg.stop, regime=mask)
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
        "short: trail=%.1f×ATR, OOF 笔数 冻结%d vs 重放%d"
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
    add("选定配置整段 OOF 复评可复现(与冻结指标一致)",
        abs(gm["sharpe"] - om["sharpe"]) < 1e-9
        and abs(gm["total_return"] - om["total_return"]) < 1e-9,
        "short: 冻结 %.2f%%/夏普%.2f vs 复评 %.2f%%/夏普%.2f"
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
    v7meta = json.loads(V7_META.read_text(encoding="utf-8")) if V7_META.exists() else None

    print("=" * 120)
    print("SOL/USDT %s LGBM 回测 v8 | 欧易(OKX) | OOF 内部 %d 折 + 稳健选优 | 只优化空头"
          % (C.INTERVAL, C.V8_N_FOLDS))
    print("稳健得分 = 各折夏普均值 - %.1f×标准差 | 单折>=%d 笔 | OOC 仅观察"
          % (C.V8_ROBUST_K, C.V8_MIN_TRADES_PER_FOLD))
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
    v8_folds = optimize._v8_fold_slices(oof, C.V8_N_FOLDS)
    print("OOF 内部分折: " + ", ".join("[%d,%d)" % f for f in v8_folds))

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

    # ---------------- 空头: v8 稳健择优(OOF 内部分折; 网格/训练与 v7 完全一致)
    print("\n### 选择: short (v8 稳健选优: 因子池 × 空头制度 × 去冗余 × 模型 × 执行 × %d 子折)"
          % C.V8_N_FOLDS)
    fmap = {(pool, d): fsets[(pool, d)]["short"]
            for pool in C.V6_FACTOR_POOLS for d in C.V6_DEDUP_GRID}
    regimes = {rule: masks[rule]["short"] for rule in C.V7_REGIME_GRID}
    best, agg, raw = optimize.optimize_v8_on_oof(
        Fmat, y, fmap, "short", ohlc, atr, times, tr, oof, regimes,
        thr_grid=C.V6_THR_GRID, hold_grid=C.V6_HOLD_GRID, trail_grid=C.V6_TRAIL_GRID,
        sl_grid=C.V7_SL_GRID, tp_gt_sl=C.V3_ENFORCE_TP_GT_SL, verbose=True)
    raw.to_csv(OUT / "selection_folds_short.csv", index=False)
    agg.to_csv(OUT / "selection_robust_short.csv", index=False)
    m = best["oof_metrics"]
    print("    ==> 选中: 池=%s 制度=%s |corr|<%.2f %-9s tp=%.2f>sl=%.2f hold=%2d trail=%.1f | "
          "稳健%+.3f(均值%+.3f-%.1f×std%.3f, 最差折%+.3f) 正折%d/%d"
          % (best["pool"], best["regime_rule"], best["dedup"], best["model_name"],
             best["tp_mult"], best["sl_mult"], best["max_hold"], best["trail_mult"],
             best["robust"], best["sharpe_mean"], C.V8_ROBUST_K, best["sharpe_std"],
             best["sharpe_min"], best["n_pos_folds"], best["n_folds"]))
    print("        OOF 复评: 夏普%+.2f 收益%+.2f%% 笔数%d 胜率%.0f%% 盈亏比%.2f"
          % (m["sharpe"], m["total_return"] * 100, int(m["n"]),
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
    v7_short = v7meta["sides"]["short"] if v7meta else None
    lines: list = []
    lines.append("# SOL/USDT 4h 多因子 LGBM 回测报告 (v8: OOF 内部分折 + 稳健选优)\n")
    lines.append("- 数据源: 欧易(OKX) 4h 现货 %s | 清洗后 %d 根(起点 %s)\n"
                 % (C.SYMBOL, len(df), C.DATA_START))
    lines.append("- 切分: train 70% / OOF 15% / OOC 15%, 按时间顺序\n")
    lines.append("- v8 **只优化空头**; 多头沿用 v6 冻结配置(见 reports/v6)。\n")
    lines.append("- 选择: **只在 OOF** 上稳健择优; **OOC 仅观察**, 不参与任何选择\n")
    lines.append("- v8 唯一改动 = **选择纪律**: OOF 内切 %d 个连续子折, 稳健得分 = 各折夏普"
                 "均值 - %.1f×标准差, 次目标为最差折夏普; 单折成交数 >= %d 笔。\n"
                 "  网格/训练/因子/成本/资金/选择段与 v7 **完全一致**, 便于横向对比。\n"
                 % (C.V8_N_FOLDS, C.V8_ROBUST_K, C.V8_MIN_TRADES_PER_FOLD))
    lines.append("- OOF 子折: %s\n\n" % ", ".join("[%d,%d)" % f for f in v8_folds))

    # ---- 空头绩效 v6 / v7 / v8
    lines.append("## 空头绩效(v6 整段取最大 / v7 空头专用 / v8 稳健选优)\n\n")
    lines.append("| 段 | 版本 | 因子池 | 制度 | tp/sl | 收益率% | 夏普 | 卡玛 | 胜率% | 盈亏比 | 开仓数量 | 最大回撤% | 止盈率% | 止损率% | 超时率% |\n")
    lines.append("|" + "---|" * 15 + "\n")

    def _row(s, tag, f, mm):
        return ("| %s | %s | %s | %s | %.2f/%.2f | %+.2f | %.2f | %.2f | %.1f | %.2f | %d | %.2f | %.1f | %.1f | %.1f |\n"
                % (s, tag, f["pool"], f["regime_rule"], f["tp_mult"], f["sl_mult"],
                   mm["total_return"] * 100, mm["sharpe"], mm["calmar"], mm["win_rate"] * 100,
                   mm["payoff_ratio"], mm["n_trades"], mm["max_drawdown"] * 100,
                   mm["tp_rate"] * 100, mm["sl_rate"] * 100, mm["timeout_rate"] * 100))

    shorts_rows = []
    for s in SEGS:
        v6fp = v6_short["frozen_params"]
        a = v6_short["metrics"][s]
        lines.append(_row(s, "v6", dict(pool=v6_short["pool"], regime_rule=v6_short["regime_rule"],
                                        tp_mult=v6fp["tp_mult"], sl_mult=v6fp["sl_mult"]), a))
        if v7_short:
            v7fp = v7_short["frozen_params"]
            b = v7_short["metrics"][s]
            lines.append(_row(s, "v7", dict(pool=v7_short["pool"], regime_rule=v7_short["regime_rule"],
                                            tp_mult=v7fp["tp_mult"], sl_mult=v7fp["sl_mult"]), b))
        f = frozen["short"]
        c = res["short"][s]["metrics"]
        lines.append(_row(s, "**v8**", f, c))
        row = dict(seg=s, v6={k: float(a[k]) for k in MKEYS},
                   v8={k: float(c[k]) for k in MKEYS})
        if v7_short:
            row["v7"] = {k: float(v7_short["metrics"][s][k]) for k in MKEYS}
        shorts_rows.append(row)

    # ---- 稳健选优明细(前 15)
    lines.append("\n## 稳健得分榜(前 15 组, 按稳健得分降序)\n\n")
    lines.append("| # | 因子池 | 制度 | |corr|< | 因子数 | 模型 | tp | sl | hold | trail | "
                 "稳健 | 夏普均值 | 折间std | 最差折 | 最差折收益% | 平均收益% | 正折 | 总笔数 |\n")
    lines.append("|" + "---|" * 18 + "\n")
    for i in range(min(15, len(agg))):
        r = agg.iloc[i]
        star = " **<-选中**" if i == 0 else ""
        lines.append("| %d%s | %s | %s | %.2f | %d | %s | %.2f | %.2f | %d | %.1f | %+.3f | %+.3f | %.3f | %+.3f | %+.2f | %+.2f | %d/%d | %d |\n"
                     % (i + 1, star, r["pool"], r["regime"], r["dedup"], int(r["n_factors"]),
                        r["model"], r["tp_mult"], r["sl_mult"], int(r["max_hold"]),
                        r["trail_mult"], r["robust"], r["sharpe_mean"], r["sharpe_std"],
                        r["sharpe_min"], r["ret_min"] * 100, r["ret_mean"] * 100,
                        int(r["n_pos_folds"]), int(r["n_folds"]), int(r["n_total"])))

    # ---- 选中配置的各子折明细
    lines.append("\n## 选中空头配置在各 OOF 子折上的表现(稳健选优的直接依据)\n\n")
    lines.append("| 子折 | 区间 | 收益率% | 夏普 | 卡玛 | 最大回撤% | 胜率% | 盈亏比 | 笔数 | 止盈率% |\n")
    lines.append("|" + "---|" * 10 + "\n")
    sf = _selected_folds(raw, agg.iloc[0])
    for _, r in sf.iterrows():
        lines.append("| %d | [%d,%d) | %+.2f | %+.2f | %.2f | %.2f | %.1f | %.2f | %d | %.1f |\n"
                     % (int(r["fold"]), int(r["fold_lo"]), int(r["fold_hi"]),
                        r["total_return"] * 100, r["sharpe"], r["calmar"],
                        r["max_drawdown"] * 100, r["win_rate"] * 100, r["payoff"],
                        int(r["n"]), r["tp_rate"] * 100))
    lines.append("\n> 稳健得分 %.3f = 均值 %+.3f - %.1f×标准差 %.3f; 最差折夏普 %+.3f; 正收益折 %d/%d。\n"
                 % (best["robust"], best["sharpe_mean"], C.V8_ROBUST_K, best["sharpe_std"],
                    best["sharpe_min"], best["n_pos_folds"], best["n_folds"]))

    # ---- 各制度规则 / 因子池(稳健口径)
    lines.append("\n## 各制度规则在 OOF 上的对比(各规则取自身稳健最优)\n\n")
    lines.append("| 制度 | 因子池 | 去冗余 | 因子数 | 模型 | tp | sl | hold | trail | 稳健 | 夏普均值 | 最差折 | 平均收益% | 总笔数 |\n")
    lines.append("|" + "---|" * 15 + "\n")
    for rule in C.V7_REGIME_GRID:
        g = agg[agg["regime"] == rule]
        if g.empty:
            continue
        b = g.iloc[0]
        star = " **<-选中**" if rule == frozen["short"]["regime_rule"] else ""
        lines.append("| %s%s | %s | %.2f | %d | %s | %.2f | %.2f | %d | %.1f | %+.3f | %+.3f | %+.3f | %+.2f | %d |\n"
                     % (rule, star, b["pool"], b["dedup"], int(b["n_factors"]), b["model"],
                        b["tp_mult"], b["sl_mult"], int(b["max_hold"]), b["trail_mult"],
                        b["robust"], b["sharpe_mean"], b["sharpe_min"], b["ret_mean"] * 100,
                        int(b["n_total"])))

    lines.append("\n## 因子池在 OOF 上的对比(各池取自身稳健最优)\n\n")
    lines.append("| 因子池 | 制度 | 去冗余 | 因子数 | 模型 | 稳健 | 夏普均值 | 最差折 | 平均收益% | 总笔数 |\n")
    lines.append("|" + "---|" * 11 + "\n")
    for p in C.V6_FACTOR_POOLS:
        g = agg[agg["pool"] == p]
        if g.empty:
            continue
        b = g.iloc[0]
        star = " **<-选中**" if p == frozen["short"]["pool"] else ""
        lines.append("| %s%s | %s | %.2f | %d | %s | %+.3f | %+.3f | %+.3f | %+.2f | %d |\n"
                     % (p, star, b["regime"], b["dedup"], int(b["n_factors"]), b["model"],
                        b["robust"], b["sharpe_mean"], b["sharpe_min"], b["ret_mean"] * 100,
                        int(b["n_total"])))

    lines.append("\n## 段基准(买入持有)\n\n| 段 | 基准收益% |\n|---|---|\n")
    for s in SEGS:
        lines.append("| %s | %+.2f |\n" % (s, buy_hold(df, segs[s]) * 100))

    lines.append("\n## 冻结配置(由 OOF 稳健选出)\n")
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

    cons = consistency(df, tr, oof, ooc, frozen, res, prune_log, masks, v6_long, best, agg, raw)
    n_fail = sum(1 for c in cons if not c["pass"])
    lines.append("\n## 一致性检查(design == runtime)\n\n**失败项: %d**\n\n| 检查 | 结果 | 说明 |\n"
                 "|---|---|---|\n" % n_fail)
    for c in cons:
        lines.append("| %s | %s | %s |\n" % (c["check"], "PASS" if c["pass"] else "**FAIL**",
                                             c["detail"]))

    (OUT / "backtest_report_v8.md").write_text("".join(lines), encoding="utf-8")

    # ---------------- 权益曲线(空头单腿 + 多空合并)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=False)
    eq_short = execution.equity_from_trades(
        [t for s in SEGS for t in res["short"][s]["trades"]], len(df))
    axes[0].plot(range(len(eq_short)), eq_short, lw=1.2, color="tab:red",
                 label="short equity (capital %d + pnl)" % C.INIT_CAPITAL)
    axes[1].plot(range(len(eq_short)), eq_short, lw=1.2, color="tab:red", label="short (v8)")
    eq_long = execution.equity_from_trades(
        [t for s in SEGS for t in res["long"][s]["trades"]], len(df))
    axes[1].plot(range(len(eq_long)), eq_long, lw=1.2, color="tab:blue", label="long (v6)")
    axes[1].plot(range(len(eq_long)), eq_short + eq_long - C.INIT_CAPITAL, lw=1.6,
                 color="black", label="combined (long v6 + short v8)")
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
    axes[0].set_title("short (v8 robust) | pool=%s regime=%s trail=%.1f | OOF %+.2f%% (sharpe %.2f) | "
                      "OOC %+.2f%% (sharpe %.2f) [OOC = observation only]"
                      % (frozen["short"]["pool"], frozen["short"]["regime_rule"],
                         frozen["short"]["trail_mult"], ms["total_return"] * 100, ms["sharpe"],
                         mc["total_return"] * 100, mc["sharpe"]))
    axes[1].set_title("long (v6 frozen) vs short (v8) vs combined")
    axes[-1].set_xlabel("bar index (4h)")
    fig.tight_layout()
    fig.savefig(OUT / "equity_curve_v8_short.png", dpi=130)
    plt.close(fig)

    # ---------------- 落盘
    meta = {"config": {"SYMBOL": C.SYMBOL, "INTERVAL": C.INTERVAL, "SOURCE": C.BACKTEST_SOURCE,
                       "DATA_START": C.DATA_START, "HORIZON": C.HORIZON,
                       "SELECT_ON": "oof", "OPTIMIZED_SIDES": ["short"],
                       "LONG_INHERITED_FROM": "v6",
                       "V8_N_FOLDS": C.V8_N_FOLDS, "V8_ROBUST_K": C.V8_ROBUST_K,
                       "V8_MIN_TRADES_PER_FOLD": C.V8_MIN_TRADES_PER_FOLD,
                       "V8_FOLDS": [[int(a), int(b)] for a, b in v8_folds],
                       "V8_CONFIG_KEYS": list(optimize._V8_CONFIG_KEYS),
                       "V8_SELECT_RULE": "robust = sharpe_mean - K*sharpe_std",
                       "V7_SL_GRID": list(C.V7_SL_GRID),
                       "V7_REGIME_GRID": list(C.V7_REGIME_GRID),
                       "V6_DEDUP_GRID": list(C.V6_DEDUP_GRID),
                       "V6_THR_GRID": list(C.V6_THR_GRID), "V6_HOLD_GRID": list(C.V6_HOLD_GRID),
                       "V6_TRAIL_GRID": list(C.V6_TRAIL_GRID), "V4_MAX_VIF": C.V4_MAX_VIF,
                       "FEE_RATE": C.FEE_RATE, "SLIP_RATE": C.SLIP_RATE,
                       "INIT_CAPITAL": C.INIT_CAPITAL, "TRADE_NOTIONAL": C.TRADE_NOTIONAL},
            "sides": {}, "short_vs_prev": shorts_rows,
            "robust_top": agg.head(50).to_dict(orient="records"),
            "selected_folds": sf.to_dict(orient="records"),
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
    meta["sides"]["short"]["robust"] = {"robust": best["robust"], "sharpe_mean": best["sharpe_mean"],
                                        "sharpe_std": best["sharpe_std"], "sharpe_min": best["sharpe_min"],
                                        "ret_mean": best["ret_mean"], "n_pos_folds": best["n_pos_folds"],
                                        "n_folds": best["n_folds"]}
    (OUT / "metrics_v8.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
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
    pd.DataFrame(tr_all).to_csv(OUT / "trades_v8.csv", index=False)

    # ---------------- 控制台汇总
    print("\n" + "=" * 120)
    print("结果汇总 (v8: OOF 内部分折 + 稳健选优; 多头沿用 v6)")
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

    print("\n%-6s %22s %22s %22s" % ("段", "v6 short", "v7 short", "v8 short"))
    for s in SEGS:
        a = v6_short["metrics"][s]
        b = v7_short["metrics"][s] if v7_short else {}
        c = res["short"][s]["metrics"]
        bstr = ("%+.2f%%/夏普%5.2f" % (b["total_return"] * 100, b["sharpe"])) if v7_short else "n/a"
        print("%-6s %+9.2f%%/夏普%5.2f %22s %+9.2f%%/夏普%5.2f" %
              (s, a["total_return"] * 100, a["sharpe"], bstr,
               c["total_return"] * 100, c["sharpe"]))

    print("\n一致性检查: 失败项 %d / %d" % (n_fail, len(cons)))
    for c in cons:
        if not c["pass"]:
            print("  FAIL: %s | %s" % (c["check"], c["detail"]))
    print("\n产出目录: %s" % OUT)
    print("  backtest_report_v8.md / metrics_v8.json / equity_curve_v8_short.png / "
          "selection_folds_short.csv / selection_robust_short.csv / trades_v8.csv")


if __name__ == "__main__":
    main()
