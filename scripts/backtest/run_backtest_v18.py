#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SOL/USDT 4h 多因子 LGBM 回测 (v18: 滚动 walk-forward)。

基线 = **v15**(long 门控冻结 none + tp 上界 12 + 真实 OOF 段「收益锚 + 夏普择优」)。
v15/v16/v17 的诊断一致指向**选择层**才是过拟合主因: 单次 train/OOF/OOC 切分下,
「在 OOF 网格里取最大值」是噪声选择(long 网格两万余组里前 8 名相差 <1pp、冠亚军仅差
0.09pp); v17 改用 train 内部嵌套 CV 虽消除了选择污染, 却牺牲了**制度适配**。

v18 因此改为**滚动 walk-forward**(只改切分与选择时机, 因子/标签/成本/网格/训练口径
与选优规则(收益锚 + 夏普择优)全部沿用 v15; 另修正一处边界口径: 取成交上界
`entry_hi = 窗口 stop`, 窗内最后一根信号不得把仓位开进下一折/下一段):
  - 训练窗 = 固定长度**滚动**窗(约 2 年), 只含最近制度;
  - 选参窗 = 紧邻前测窗之前的 6 个月(样本外), 每步在其上重选一轮全部配置;
  - 前测窗 = 选参窗之后的 6 个月, 只用冻结配置前测, **从不参与任何选择**;
  - V18_N_STEPS 步首尾相接 -> 前测窗拼成**连续 OOS 段**(约 50% 数据), 取代脆弱的 15% OOC;
  - 每步阈值 thr_abs 由**该步选参窗**的预测分位冻结, 再应用到前测窗(纯前视)。

运行:  python3 scripts/backtest/run_backtest_v18.py                 # 完整(约 10 分钟)
       python3 scripts/backtest/run_backtest_v18.py --from-state    # 复用已缓存的逐折结果, 仅重出报告
"""
from __future__ import annotations

import json
import pickle
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
from src import data_clean, execution, factor_select, regime          # noqa: E402
from src import factors as F_lib                                     # noqa: E402
from src import optimize                                             # noqa: E402

OUT = C.REPORT_DIR / C.V18_OUT_DIR
STATE = OUT / "state_v18.pkl"
pd.set_option("display.width", 240)


# ================================================================ 数据
def load_clean() -> pd.DataFrame:
    """载入欧易原始数据 -> 清洗 -> 对齐 DATA_START(与 v6~v15 同口径)。"""
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


def buy_hold(df: pd.DataFrame, lo: int, hi: int) -> float:
    return float(df["close"].iloc[hi - 1] / df["close"].iloc[lo] - 1.0)


# ================================================================ 滚动窗口
def walk_forward_windows(n: int) -> list:
    """生成滚动 walk-forward 窗口, 全部按时间升序、且 train<val<test 严格不重叠。

    对齐方式: 先让**最后一次前测窗恰好落在数据末端**, 再向前推 V18_N_STEPS 步。
      test_k  = [oos_start + k*T, oos_start + (k+1)*T)
      val_k   = [test_k.start - V, test_k.start)     # 紧邻前测窗之前的样本外窗口
      train_k = [val_k.start - Tr, val_k.start)      # 固定长度滚动窗(首步可短, 有下限保护)
    """
    T, V, Tr = C.V18_TEST_BARS, C.V18_VAL_BARS, C.V18_TRAIN_BARS
    oos_start = n - C.V18_N_STEPS * T
    if oos_start - V < C.V18_MIN_TRAIN_BARS:
        raise SystemExit("滚动窗配置超出数据长度: oos_start=%d, val=%d" % (oos_start, V))
    out = []
    for k in range(C.V18_N_STEPS):
        ts = oos_start + k * T
        te = ts + T
        vs, ve = ts - V, ts
        trs = max(0, vs - Tr)
        out.append(dict(step=k, train=slice(trs, vs), val=slice(vs, ve), test=slice(ts, te)))
    return out


def side_grids(side: str):
    """与 v15 逐位一致的制度网格与执行/模型网格(long 专属, short 用全局默认)。"""
    if side == "long":
        return C.V15_LONG_REGIME_GRID, dict(
            thr_grid=C.V15_LONG_THR_GRID, tp_grid=C.V15_LONG_TP_GRID,
            sl_grid=C.V15_LONG_SL_GRID, hold_grid=C.V15_LONG_HOLD_GRID,
            model_grid=list(C.V15_LONG_MODEL_GRID))
    return C.V5_REGIME_GRID, {}


def _summarize(trades, equity) -> dict:
    from src.metrics import summarize
    return summarize(trades, equity)


# ================================================================ 一致性检查
def consistency(df: pd.DataFrame, wins: list, masks: dict, steps: list) -> list:
    """design == runtime 的运行时断言(v18 关注: 无前视 / 选参在前测之前 / 配置在网格内)。"""
    chk = []
    n = len(df)

    def add(name, ok, detail=""):
        chk.append({"check": name, "pass": bool(ok), "detail": detail})

    add("数据起点对齐 DATA_START", str(df["datetime"].iloc[0])[:10] == C.DATA_START,
        "起点 %s -> 首根 %s" % (C.DATA_START, str(df["datetime"].iloc[0])[:16]))

    ok_ord, det = True, []
    for w in wins:
        a, b, c = w["train"], w["val"], w["test"]
        ok_ord &= (a.stop == b.start) and (b.stop == c.start) and a.start >= 0
        ok_ord &= (a.stop - a.start) >= C.V18_MIN_TRAIN_BARS
        det.append("step%d train[%d,%d) val[%d,%d) test[%d,%d)"
                   % (w["step"], a.start, a.stop, b.start, b.stop, c.start, c.stop))
    add("三步严格时序: train < val < test 且首尾相接", ok_ord, "; ".join(det))

    ok_tile = all(wins[k]["test"].stop == wins[k + 1]["test"].start for k in range(len(wins) - 1))
    ok_tile &= wins[-1]["test"].stop == n
    add("前测窗首尾相接铺满连续 OOS 段", ok_tile,
        "OOS[%d,%d) 共 %d 根 / %d 折" % (wins[0]["test"].start, wins[-1]["test"].stop,
                                       wins[-1]["test"].stop - wins[0]["test"].start, len(wins)))

    long_regimes = [s["frozen"]["regime_rule"] for s in steps if s["side"] == "long"]
    ok_v15 = all(r == "none" for r in long_regimes) and C.V15_LONG_REGIME_GRID == ("none",)
    add("long 门控沿用 v15 冻结=none", ok_v15, "各步 long 制度: " + ", ".join(long_regimes))

    ok_grid, det_g = True, []
    for s in steps:
        f = s["frozen"]
        rules = C.V15_LONG_REGIME_GRID if s["side"] == "long" else C.V5_REGIME_GRID
        tg = C.V15_LONG_THR_GRID if s["side"] == "long" else C.EXEC_THR_GRID
        pg = C.V15_LONG_TP_GRID if s["side"] == "long" else C.TP_ATR_GRID
        sg = C.V15_LONG_SL_GRID if s["side"] == "long" else C.SL_ATR_GRID
        hg = C.V15_LONG_HOLD_GRID if s["side"] == "long" else C.OOF_HOLD_GRID
        ok_one = (f["regime_rule"] in rules and f["thr_q"] in tg and f["tp_mult"] in pg
                  and f["sl_mult"] in sg and f["max_hold"] in hg and f["tp_mult"] > f["sl_mult"])
        ok_grid &= ok_one
        if not ok_one:
            det_g.append("step%d/%s 越界:%s" % (s["step"], s["side"], f))
    add("每步冻结配置 ∈ v15 网格 且 tp > sl", ok_grid,
        "; ".join(det_g) if det_g else "全部 %d 个(步×方向)配置均在网格内" % len(steps))

    ok_thr, det_t = True, []
    for s in steps:
        f = s["frozen"]
        w = wins[s["step"]]
        q = execution.threshold_from_quantile(f["pred"], s["side"], f["thr_q"],
                                              w["val"].start, w["val"].stop)
        ok_thr &= abs(q - f["thr_abs"]) < 1e-12
        det_t.append("step%d/%s 选参窗分位 %.6f vs 冻结 %.6f"
                     % (s["step"], s["side"], q, f["thr_abs"]))
    add("阈值 thr_abs 由**选参窗**预测分位冻结(前测窗未参与)", ok_thr, "; ".join(det_t))

    bad_ll = [t for s in steps for t in s["trades"]
              if abs(t.entry_price - float(df["open"].iloc[t.entry_idx])) > 1e-9]
    n_tot = sum(len(s["trades"]) for s in steps)
    add("信号 t 收盘 -> t+1 开盘成交(无未来函数)", len(bad_ll) == 0,
        "全部 %d 笔成交价 == open[entry_idx], 不一致 %d 笔" % (n_tot, len(bad_ll)))

    cost = 2.0 * (C.FEE_RATE + C.SLIP_RATE)
    bad_c = [t for s in steps for t in s["trades"]
             if abs(t.net_ret - (t.gross_ret - cost)) > 1e-12]
    add("成本口径 = 单边 5bp / 双边 10bp", len(bad_c) == 0,
        "逐笔复核 %d 笔, 不一致 %d 笔" % (n_tot, len(bad_c)))

    ok_in = all(wins[s["step"]]["test"].start <= t.entry_idx < wins[s["step"]]["test"].stop
                for s in steps for t in s["trades"])
    add("前测成交均落在本步前测窗内(entry_hi=test.stop, 不穿越折边界)", ok_in,
        "全部 %d 笔入场索引 ∈ 本步前测窗" % n_tot)

    ok_reg, det_r = True, []
    for rule in C.V5_REGIME_GRID:
        if rule == "none":
            continue
        for idx in (3000, 6000, 9000, 10800):
            if idx >= len(df):
                continue
            sub = regime.regime_masks(df.iloc[:idx + 1].reset_index(drop=True), rule)
            for side in ("long", "short"):
                if bool(sub[side][idx]) != bool(masks[rule][side][idx]):
                    ok_reg = False
        det_r.append(rule)
    add("制度掩码仅用过去信息(截断后取值不变)", ok_reg, "规则: " + ", ".join(det_r))

    add("资金口径 1000 本金 / 单笔 100",
        C.INIT_CAPITAL == 1000.0 and C.TRADE_NOTIONAL == 100.0,
        "INIT_CAPITAL=%.0f TRADE_NOTIONAL=%.0f" % (C.INIT_CAPITAL, C.TRADE_NOTIONAL))
    return chk


# ================================================================ 计算层(昂贵)
def compute(df: pd.DataFrame, Fmat: pd.DataFrame, y: pd.Series, ohlc: dict,
            atr: np.ndarray, times: np.ndarray, wins: list, masks: dict) -> list:
    """逐步: 滚动重训 + 选参窗重选 + 前测窗前测。返回逐折结果列表。"""
    n = len(df)
    long_only = list(C.V15_LONG_FACTORS)
    dedup_by_side = {"long": C.V15_LONG_DEDUP_GRID, "short": C.OOF_DEDUP_CORR_GRID}
    steps: list = []
    for w in wins:
        tr, va, te = w["train"], w["val"], w["test"]
        print("\n" + "=" * 116)
        print("STEP %d | train[%d,%d) val[%d,%d) test[%d,%d) | test %s ~ %s"
              % (w["step"], tr.start, tr.stop, va.start, va.stop, te.start, te.stop,
                 str(df["datetime"].iloc[te.start])[:10], str(df["datetime"].iloc[te.stop - 1])[:10]))
        print("=" * 116)

        Ftr = Fmat.iloc[tr].reset_index(drop=True)
        y_tr = y.iloc[tr].reset_index(drop=True)
        _, fs_long = factor_select.factor_sets_by_dedup(Ftr, y_tr, dedup_by_side["long"])
        _, fs_short = factor_select.factor_sets_by_dedup(Ftr, y_tr, dedup_by_side["short"],
                                                        exclude=long_only)
        fsets = {"long": fs_long, "short": fs_short}

        pruned: dict = {}
        for side in ("long", "short"):
            for d in dedup_by_side[side]:
                before = fsets[side][d][side]
                pruned.setdefault(side, {})[d] = \
                    factor_select.vif_prune(Ftr, before)[0] if before else []

        for side in ("long", "short"):
            rules, opt_kw = side_grids(side)
            fmap = pruned[side]
            per_rule_best = {}
            aggs = []
            for rule in rules:
                best, agg, _ = optimize.optimize_full_on_oof(
                    Fmat, y, fmap, side, ohlc, atr, times, tr, va,
                    tp_gt_sl=C.V3_ENFORCE_TP_GT_SL, regime=masks[rule][side],
                    ret_drop=C.V18_RET_DROP, verbose=False, entry_hi=va.stop, **opt_kw)
                agg = agg.copy()
                agg.insert(0, "regime", rule)
                aggs.append(agg)
                per_rule_best[rule] = best
            pd.concat(aggs, ignore_index=True).to_csv(
                OUT / ("selection_grid_step%d_%s.csv" % (w["step"], side)), index=False)

            wins_r = pd.DataFrame([dict(regime=r, **per_rule_best[r]["oof_metrics"])
                                   for r in rules])
            anchor = float(wins_r["total_return"].max())
            wsel = optimize.select_return_anchor_sharpe(wins_r, C.V18_RET_DROP)
            best_rule = wsel.iloc[0]["regime"]
            best = per_rule_best[best_rule]

            pred = best["model"].predict(Fmat)
            r_val = execution.evaluate_with_threshold(
                pred, ohlc, atr, times, side, va.start, va.stop, best["thr_abs"],
                best["tp_mult"], best["sl_mult"], max_hold=best["max_hold"],
                regime=masks[best_rule][side], entry_hi=va.stop)
            r_test = execution.evaluate_with_threshold(
                pred, ohlc, atr, times, side, te.start, te.stop, best["thr_abs"],
                best["tp_mult"], best["sl_mult"], max_hold=best["max_hold"],
                regime=masks[best_rule][side], entry_hi=te.stop)

            steps.append(dict(step=w["step"], side=side, train=tr, val=va, test=te,
                              anchor=anchor,
                              frozen=dict(best, pred=pred, regime_rule=best_rule),
                              val_metrics=r_val["metrics"], test_metrics=r_test["metrics"],
                              trades=r_test["trades"]))
            mv, mt = r_val["metrics"], r_test["metrics"]
            print("  %-5s | %-13s | |corr|<%.2f 因子%2d %-9s thr_q=%.2f tp=%4.1f>sl=%.1f hold=%2d | "
                  "选参窗 收益%+.2f%%/夏普%+.2f/%d笔  ->  前测窗 收益%+.2f%%/夏普%+.2f/%d笔 胜率%.0f%%"
                  % (side, best_rule, best["dedup"], len(best["model"].factors),
                     best["model_name"], best["thr_q"], best["tp_mult"], best["sl_mult"],
                     best["max_hold"], mv["total_return"] * 100, mv["sharpe"],
                     int(mv["n_trades"]), mt["total_return"] * 100, mt["sharpe"],
                     int(mt["n_trades"]), mt["win_rate"] * 100))
    return steps


def steps_frame(steps: list, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in steps:
        f, mv, mt, w = s["frozen"], s["val_metrics"], s["test_metrics"], s["test"]
        rows.append(dict(step=s["step"], side=s["side"], test_lo=w.start, test_hi=w.stop,
                         test_start=str(df["datetime"].iloc[w.start])[:10],
                         test_end=str(df["datetime"].iloc[w.stop - 1])[:10],
                         regime=f["regime_rule"], dedup=f["dedup"],
                         n_factors=len(f["model"].factors), model=f["model_name"],
                         thr_q=f["thr_q"], thr_abs=f["thr_abs"], tp_mult=f["tp_mult"],
                         sl_mult=f["sl_mult"], max_hold=f["max_hold"],
                         val_return=mv["total_return"], val_sharpe=mv["sharpe"],
                         val_n=int(mv["n_trades"]), val_anchor=s.get("anchor", float("nan")),
                         test_return=mt["total_return"], test_sharpe=mt["sharpe"],
                         test_n=int(mt["n_trades"]), test_win=mt["win_rate"],
                         test_payoff=mt["payoff_ratio"], test_mdd=mt["max_drawdown"],
                         bench=buy_hold(df, w.start, w.stop)))
    return pd.DataFrame(rows)


# ================================================================ 报告层(廉价)
def report(df: pd.DataFrame, wins: list, masks: dict, steps: list) -> None:
    n = len(df)
    oos_lo, oos_hi = wins[0]["test"].start, wins[-1]["test"].stop
    steps_df = steps_frame(steps, df)
    steps_df.to_csv(OUT / "walk_forward_steps.csv", index=False)

    # 聚合: 连续 OOS 段的合并权益(前测窗首尾相接, 逐笔只算一次)
    agg = {}
    for side in ("long", "short"):
        trs = [t for s in steps if s["side"] == side for t in s["trades"]]
        eq = execution.equity_from_trades(trs, n, lo=oos_lo, hi=oos_hi)
        agg[side] = {"trades": trs, "equity": eq, "metrics": _summarize(trs, eq)}
    tr_all = agg["long"]["trades"] + agg["short"]["trades"]
    eq_combo = agg["long"]["equity"] + agg["short"]["equity"] - C.INIT_CAPITAL
    agg["combined"] = {"trades": tr_all, "equity": eq_combo,
                       "metrics": _summarize(tr_all, eq_combo)}

    # 逐折汇总(一致性诊断: 选参窗 -> 前测窗 的落差)
    per_step_gap = {}
    for side in ("long", "short"):
        vr = np.array([s["val_metrics"]["total_return"] for s in steps if s["side"] == side])
        tr = np.array([s["test_metrics"]["total_return"] for s in steps if s["side"] == side])
        vs = np.array([s["val_metrics"]["sharpe"] for s in steps if s["side"] == side])
        ts = np.array([s["test_metrics"]["sharpe"] for s in steps if s["side"] == side])
        per_step_gap[side] = dict(val_ret_mean=float(vr.mean()), test_ret_mean=float(tr.mean()),
                                  ret_gap=float(vr.mean() - tr.mean()),
                                  val_sharpe_mean=float(vs.mean()),
                                  test_sharpe_mean=float(ts.mean()),
                                  sharpe_gap=float(vs.mean() - ts.mean()),
                                  n_pos_test=int((tr > 0).sum()), n_steps=len(tr))

    cons = consistency(df, wins, masks, steps)
    n_fail = sum(1 for c in cons if not c["pass"])

    # ---------------- Markdown 报告
    lines: list = []
    lines.append("# SOL/USDT 4h 多因子 LGBM 回测报告 (v18: 滚动 walk-forward)\n")
    lines.append("- 数据源: 欧易(OKX) 4h 现货 %s | 清洗后 %d 根 | **起点对齐 DATA_START=%s**(首根 %s)\n"
                 % (C.SYMBOL, n, C.DATA_START, str(df["datetime"].iloc[0])[:16]))
    lines.append("- **v18 只改切分与选择时机**: 由 v15 的单次 train/OOF/OOC 切分, 改为滚动 "
                 "walk-forward —— 滚动训练窗 %d 根(约 2 年) + 选参窗 %d 根(约 6 个月, 样本外) + "
                 "前测窗 %d 根(约 6 个月), 共 %d 步; 每步在**选参窗**上重选一轮全部配置, "
                 "再在紧随其后的**前测窗**上只用冻结配置前测。\n"
                 % (C.V18_TRAIN_BARS, C.V18_VAL_BARS, C.V18_TEST_BARS, C.V18_N_STEPS))
    lines.append("- 前测窗首尾相接 -> 连续 OOS 段 [%d,%d) 共 %d 根(约占全样本 %.0f%%); "
                 "**前测窗从不参与任何选择**, 取代 v15 中脆弱的 15%% 单窗 OOC。\n"
                 % (oos_lo, oos_hi, oos_hi - oos_lo, (oos_hi - oos_lo) / n * 100))
    lines.append("- 因子/标签/成本/执行网格/模型候选/训练口径与 v15 **一致**; 唯一差异是**成交索引上界** "
                 "`entry_hi = 窗口 stop`(不含), 使窗内最后一根信号不再把仓位开进下一折/下一段, "
                 "保证每折自洽、选参窗不渗入前测窗。选优规则仍是 "
                 "v15 的**收益锚 + 夏普择优**(让步带 %.2f 个百分点); long 门控沿用 v15 冻结 "
                 "**none**, short 每步在 %s 中重选。\n"
                 % (C.V18_RET_DROP * 100, list(C.V5_REGIME_GRID)))
    lines.append("- 每步阈值 `thr_abs` 由**该步选参窗**的预测分位冻结, 再应用到前测窗(纯前视)。\n\n")

    lines.append("## 滚动窗口\n\n")
    lines.append("| 步 | train | val | test | test 区间 | 基准收益% |\n")
    lines.append("|" + "---|" * 6 + "\n")
    for w in wins:
        a, b, c = w["train"], w["val"], w["test"]
        lines.append("| %d | [%d,%d) %d根 | [%d,%d) %d根 | [%d,%d) %d根 | %s ~ %s | %+.2f |\n"
                     % (w["step"], a.start, a.stop, a.stop - a.start, b.start, b.stop,
                        b.stop - b.start, c.start, c.stop, c.stop - c.start,
                        str(df["datetime"].iloc[c.start])[:10],
                        str(df["datetime"].iloc[c.stop - 1])[:10],
                        buy_hold(df, c.start, c.stop) * 100))

    lines.append("\n## 逐折选择与前后对照(选参窗选优 -> 前测窗检验)\n\n")
    lines.append("| 步 | 方向 | 制度 | 去冗余 | 模型 | 因子数 | thr_q | tp | sl | hold | "
                 "选参窗收益% | 选参窗夏普 | 选参笔数 | 前测收益% | 前测夏普 | 前测笔数 | 前测胜率% |\n")
    lines.append("|" + "---|" * 18 + "\n")
    for _, r in steps_df.iterrows():
        lines.append("| %d | %s | %s | %.2f | %s | %d | %.2f | %.1f | %.1f | %d | %+.2f | %+.2f | %d | "
                     "%+.2f | %+.2f | %d | %.0f |\n"
                     % (r["step"], r["side"], r["regime"], r["dedup"], r["model"], r["n_factors"],
                        r["thr_q"], r["tp_mult"], r["sl_mult"], r["max_hold"],
                        r["val_return"] * 100, r["val_sharpe"], r["val_n"],
                        r["test_return"] * 100, r["test_sharpe"], r["test_n"], r["test_win"] * 100))

    lines.append("\n## 连续 OOS 段聚合绩效(前测窗拼接)\n\n")
    lines.append("| 方向 | OOS收益% | 夏普 | 卡玛 | 胜率% | 盈亏比 | 开仓数量 | 最大回撤% | "
                 "止盈率% | 止损率% | 超时率% |\n")
    lines.append("|" + "---|" * 11 + "\n")
    for side in ("long", "short", "combined"):
        m = agg[side]["metrics"]
        lines.append("| %s | %+.2f | %.2f | %.2f | %.1f | %.2f | %d | %.2f | %.1f | %.1f | %.1f |\n"
                     % (side, m["total_return"] * 100, m["sharpe"], m["calmar"],
                        m["win_rate"] * 100, m["payoff_ratio"], int(m["n_trades"]),
                        m["max_drawdown"] * 100, m["tp_rate"] * 100, m["sl_rate"] * 100,
                        m["timeout_rate"] * 100))
    lines.append("\n> combined = long + short 两腿权益相加(equity = eq_long + eq_short - INIT_CAPITAL)。\n")

    lines.append("\n## 前后一致性(选参窗 vs 前测窗, 取代 v15 的 OOF/OOC 落差)\n\n")
    lines.append("| 方向 | 选参窗平均收益% | 前测窗平均收益% | 收益落差(pp) | 选参窗平均夏普 | "
                 "前测窗平均夏普 | 夏普落差 | 前测正收益折数 |\n")
    lines.append("|" + "---|" * 8 + "\n")
    for side in ("long", "short"):
        g = per_step_gap[side]
        lines.append("| %s | %+.2f | %+.2f | %.2f | %+.2f | %+.2f | %.2f | %d/%d |\n"
                     % (side, g["val_ret_mean"] * 100, g["test_ret_mean"] * 100,
                        g["ret_gap"] * 100, g["val_sharpe_mean"], g["test_sharpe_mean"],
                        g["sharpe_gap"], g["n_pos_test"], g["n_steps"]))
    lines.append("\n> 落差越小、越稳定, 说明选择层没有靠「选参窗过拟合」虚增数字; "
                 "落差越大、且前测普遍低于选参, 说明**在选参窗上取最优**本身仍是噪声选择。\n")

    p15 = C.V18_REF_METRICS
    if p15.exists():
        m15 = json.loads(p15.read_text(encoding="utf-8"))
        lines.append("\n## 与基线 v15 对照(v15: 单次切分, OOF 选择 / OOC 观察)\n\n")
        lines.append("| 方向 | v15 OOF 收益% | v15 OOF 夏普 | v15 OOC 收益% | v15 OOC 夏普 | "
                     "v18 OOS 收益% | v18 OOS 夏普 |\n")
        lines.append("|" + "---|" * 7 + "\n")
        for side in ("long", "short"):
            a = m15["sides"][side]["metrics"]
            b = agg[side]["metrics"]
            lines.append("| %s | %+.2f | %+.2f | %+.2f | %+.2f | %+.2f | %+.2f |\n"
                         % (side, a["oof"]["total_return"] * 100, a["oof"]["sharpe"],
                            a["ooc"]["total_return"] * 100, a["ooc"]["sharpe"],
                            b["total_return"] * 100, b["sharpe"]))
        lines.append("\n> v15 的 OOF 是**被选择污染**的段, OOC 是单根 15%% 脆样本; v18 的 OOS 是"
                     "**从未参与选择**的连续约 50%% 段, 口径更诚实、样本更长。\n")

    lines.append("\n## 段基准(买入持有)\n\n| 段 | 基准收益% |\n|---|---|\n")
    lines.append("| 连续 OOS | %+.2f |\n" % (buy_hold(df, oos_lo, oos_hi) * 100))

    lines.append("\n## 一致性检查(design == runtime)\n\n**失败项: %d / %d**\n\n| 检查 | 结果 | 说明 |\n"
                 "|---|---|---|\n" % (n_fail, len(cons)))
    for c in cons:
        lines.append("| %s | %s | %s |\n" % (c["check"], "PASS" if c["pass"] else "**FAIL**",
                                             c["detail"]))

    (OUT / "backtest_report_v18.md").write_text("".join(lines), encoding="utf-8")

    # ---------------- 权益曲线
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    for ax, side in zip(axes, ("long", "short", "combined")):
        eq = agg[side]["equity"]
        m = agg[side]["metrics"]
        ax.plot(np.arange(oos_lo, oos_hi), eq, lw=1.3,
                label="%s OOS equity (capital %d + pnl)" % (side, C.INIT_CAPITAL))
        ax.axhline(C.INIT_CAPITAL, color="gray", ls=":", lw=0.8)
        for w in wins:
            ax.axvline(w["test"].start, color="red", ls="--", lw=0.8, alpha=0.6)
            ax.axvspan(w["test"].start, w["test"].stop, alpha=0.04, color="tab:green")
        ax.set_title("%s | v18 walk-forward OOS: 收益 %+.2f%% | 夏普 %.2f | 笔数 %d  "
                     "[OOS 从未参与选择]"
                     % (side, m["total_return"] * 100, m["sharpe"], int(m["n_trades"])))
        ax.set_ylabel("Equity (USDT)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("bar index (4h)")
    fig.tight_layout()
    fig.savefig(OUT / "equity_curve_v18.png", dpi=130)
    plt.close(fig)

    # ---------------- 落盘
    meta = {"config": {"SYMBOL": C.SYMBOL, "INTERVAL": C.INTERVAL, "SOURCE": C.BACKTEST_SOURCE,
                       "DATA_START": C.DATA_START, "HORIZON": C.HORIZON,
                       "SCHEME": "rolling_walk_forward",
                       "V18_TRAIN_BARS": C.V18_TRAIN_BARS, "V18_VAL_BARS": C.V18_VAL_BARS,
                       "V18_TEST_BARS": C.V18_TEST_BARS, "V18_N_STEPS": C.V18_N_STEPS,
                       "V18_RET_DROP": C.V18_RET_DROP,
                       "SELECT_ON": "val(step)", "TEST_IS_PURE_OOS": True,
                       "V15_LONG_REGIME_GRID": list(C.V15_LONG_REGIME_GRID),
                       "V15_LONG_TP_GRID": list(C.V15_LONG_TP_GRID),
                       "V5_REGIME_GRID": list(C.V5_REGIME_GRID),
                       "FEE_RATE": C.FEE_RATE, "SLIP_RATE": C.SLIP_RATE,
                       "INIT_CAPITAL": C.INIT_CAPITAL, "TRADE_NOTIONAL": C.TRADE_NOTIONAL,
                       "V4_MAX_VIF": C.V4_MAX_VIF, "V4_MIN_FACTORS": C.V4_MIN_FACTORS},
            "windows": [{"step": w["step"],
                         "train": [w["train"].start, w["train"].stop],
                         "val": [w["val"].start, w["val"].stop],
                         "test": [w["test"].start, w["test"].stop]} for w in wins],
            "oos": [oos_lo, oos_hi],
            "steps": json.loads(steps_df.to_json(orient="records")),
            "aggregate": {s: {k: float(v) for k, v in agg[s]["metrics"].items()}
                          for s in ("long", "short", "combined")},
            "val2test_gap": per_step_gap,
            "consistency": {"n_fail": n_fail, "checks": cons}}
    (OUT / "metrics_v18.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                          encoding="utf-8")

    tr_all_rows = []
    for s in steps:
        for t in s["trades"]:
            tr_all_rows.append(dict(step=s["step"], side=t.side, entry_time=str(t.entry_time),
                                    exit_time=str(t.exit_time), entry_price=t.entry_price,
                                    exit_price=t.exit_price, exit_reason=t.exit_reason,
                                    bars_held=t.bars_held, gross_ret=t.gross_ret,
                                    net_ret=t.net_ret, pnl_usdt=t.pnl_usdt))
    pd.DataFrame(tr_all_rows).to_csv(OUT / "trades_v18.csv", index=False)

    # ---------------- 控制台汇总
    print("\n" + "=" * 116)
    print("结果汇总 (v18 滚动 walk-forward | 连续 OOS[%d,%d) %d 根 | 前测窗从未参与选择)"
          % (oos_lo, oos_hi, oos_hi - oos_lo))
    print("=" * 116)
    print("%-9s %10s %8s %8s %8s %8s %9s %10s" %
          ("方向", "OOS收益%", "夏普", "卡玛", "胜率%", "盈亏比", "开仓数", "最大回撤%"))
    for side in ("long", "short", "combined"):
        m = agg[side]["metrics"]
        print("%-9s %+10.2f %8.2f %8.2f %8.1f %8.2f %9d %10.2f" %
              (side, m["total_return"] * 100, m["sharpe"], m["calmar"], m["win_rate"] * 100,
               m["payoff_ratio"], int(m["n_trades"]), m["max_drawdown"] * 100))

    print("\n前后一致性(选参窗 -> 前测窗 落差):")
    for side in ("long", "short"):
        g = per_step_gap[side]
        print("  %-5s 选参窗均值 %+.2f%%/夏普%+.2f  ->  前测窗均值 %+.2f%%/夏普%+.2f | "
              "落差 %.2fpp / %.2f | 前测正收益 %d/%d 折"
              % (side, g["val_ret_mean"] * 100, g["val_sharpe_mean"],
                 g["test_ret_mean"] * 100, g["test_sharpe_mean"], g["ret_gap"] * 100,
                 g["sharpe_gap"], g["n_pos_test"], g["n_steps"]))

    if p15.exists():
        m15 = json.loads(p15.read_text(encoding="utf-8"))
        print("\n%-6s %-22s %-22s %-22s" % ("方向", "v15 OOF(被选择污染)", "v15 OOC(15%单窗)",
                                            "v18 OOS(连续, 未参与选择)"))
        for side in ("long", "short"):
            a = m15["sides"][side]["metrics"]
            b = agg[side]["metrics"]
            print("%-6s %+13.2f%%/夏普%5.2f %+14.2f%%/夏普%5.2f %+14.2f%%/夏普%5.2f"
                  % (side, a["oof"]["total_return"] * 100, a["oof"]["sharpe"],
                     a["ooc"]["total_return"] * 100, a["ooc"]["sharpe"],
                     b["total_return"] * 100, b["sharpe"]))

    print("\n一致性检查: 失败项 %d / %d" % (n_fail, len(cons)))
    for c in cons:
        if not c["pass"]:
            print("  FAIL: %s | %s" % (c["check"], c["detail"]))
    print("\n产出目录: %s" % OUT)
    print("  backtest_report_v18.md / metrics_v18.json / equity_curve_v18.png / "
          "walk_forward_steps.csv / trades_v18.csv / selection_grid_step*_*.csv")


# ================================================================ 入口
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    from_state = "--from-state" in sys.argv
    print("=" * 116)
    print("SOL/USDT %s LGBM 回测 v18 | 滚动 walk-forward | 基线 v15 | 数据 >= %s%s"
          % (C.INTERVAL, C.DATA_START, "  [--from-state 仅重出报告]" if from_state else ""))
    print("窗口: 滚动训练 %d 根 / 选参 %d 根 / 前测 %d 根 / %d 步 | "
          "选优: v15 收益锚+让步<=%.2fpp 夏普择优"
          % (C.V18_TRAIN_BARS, C.V18_VAL_BARS, C.V18_TEST_BARS, C.V18_N_STEPS,
             C.V18_RET_DROP * 100))
    print("成本 单边 %.1fbp | 本金 %.0f / 单笔 %.0f USDT | 前测窗从不参与选择"
          % (C.FEE_RATE * 1e4, C.INIT_CAPITAL, C.TRADE_NOTIONAL))
    print("=" * 116)

    df = load_clean()
    df = F_lib.add_label(df, C.HORIZON)
    Fmat, _ = F_lib.build_factors(df)
    y = df["label"]
    n = len(df)
    ohlc = {k: df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close")}
    atr = df["atr"].to_numpy(dtype=float)
    times = df["datetime"].to_numpy()
    wins = walk_forward_windows(n)
    oos_lo, oos_hi = wins[0]["test"].start, wins[-1]["test"].stop

    print("\n数据: %d 根  %s ~ %s | 因子库 %d 个"
          % (n, str(df["datetime"].iloc[0])[:16], str(df["datetime"].iloc[-1])[:16],
             Fmat.shape[1]))
    print("滚动窗口(%d 步, 首尾相接, 前测窗从不参与选择):" % len(wins))
    for w in wins:
        a, b, c = w["train"], w["val"], w["test"]
        print("  step%d: train[%4d,%4d) %4d根 | val[%4d,%4d) %4d根 | test[%4d,%4d) %4d根 | "
              "test 区间 %s ~ %s | 基准 %+.2f%%"
              % (w["step"], a.start, a.stop, a.stop - a.start, b.start, b.stop, b.stop - b.start,
                 c.start, c.stop, c.stop - c.start, str(df["datetime"].iloc[c.start])[:10],
                 str(df["datetime"].iloc[c.stop - 1])[:10], buy_hold(df, c.start, c.stop) * 100))
    print("连续 OOS 段: [%d,%d) 共 %d 根(约占全样本 %.0f%%) | 基准 %+.2f%%"
          % (oos_lo, oos_hi, oos_hi - oos_lo, (oos_hi - oos_lo) / n * 100,
             buy_hold(df, oos_lo, oos_hi) * 100))

    if from_state:
        if not STATE.exists():
            raise SystemExit("缺少缓存状态: %s" % STATE)
        steps = pickle.loads(STATE.read_bytes())
        print("\n[--from-state] 载入逐折结果缓存: %s (%d 条)" % (STATE, len(steps)))
    else:
        masks = {rule: regime.regime_masks(df, rule) for rule in C.V5_REGIME_GRID}
        steps = compute(df, Fmat, y, ohlc, atr, times, wins, masks)
        STATE.write_bytes(pickle.dumps(steps))
        print("\n已缓存逐折结果: %s" % STATE)

    masks = {rule: regime.regime_masks(df, rule) for rule in C.V5_REGIME_GRID}
    report(df, wins, masks, steps)


if __name__ == "__main__":
    main()
