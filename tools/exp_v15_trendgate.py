# -*- coding: utf-8 -*-
"""v15 探索: long 门控从「价在均线上/下」扩到「趋势强度/质量」。

门控全部用**约定绝对阈值**(非段内分位, 避免未来函数):
  none / sma200 / sma200_slope (v14 已有)
  adx20, adx25        : ADX14 > 20 / 25          —— 趋势强度
  adx_up              : ADX14 > 20 且 ADX 上升    —— 趋势增强
  er30                : 有向效率比 ER20 >= 0.30   —— 走势"直"度
  tq30                : 有向趋势质量 R2*sign >= 0.30
  conf                : close>SMA200 且 ER20>=0.30 且 ADX14>20
其余口径与 v14 long 逐位一致(去冗余网格/7 个模型候选/1476 组执行网格/VIF 剪枝/选优规则)。
选择只在 OOF; OOC 仅观察。
"""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "/workspace/solusdt_repo")
import config as C
from src import data_clean, execution, factor_select, regime, models
from src import factors as F_lib
from src import optimize
from src.cv import time_split

t0 = time.time()
raw = pd.read_parquet(C.BACKTEST_RAW)
df = data_clean.clean(raw).reset_index(drop=True)
dt = pd.to_datetime(df["datetime"], utc=True)
df = df[dt.to_numpy() >= pd.Timestamp(C.DATA_START, tz="UTC")].reset_index(drop=True)
df["log_ret"] = np.log(df["close"]).diff()
df = F_lib.add_label(df, C.HORIZON)
Fmat, _ = F_lib.build_factors(df)
y = df["label"]
tr, oof, ooc = time_split(len(df))
ohlc = {k: df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close")}
atr = df["atr"].to_numpy(dtype=float)
times = df["datetime"].to_numpy()

def gate_col(name, thr):
    x = Fmat[name].to_numpy(dtype=float)
    return np.isfinite(x) & (x > thr)

s200 = regime.regime_masks(df, "sma200")["long"]
s200s = regime.regime_masks(df, "sma200_slope")["long"]
adx20 = gate_col("adx_14", 20.0); adx25 = gate_col("adx_14", 25.0)
G = {
    "none": np.ones(len(df), bool),
    "sma200": s200,
    "sma200_slope": s200s,
    "adx20": adx20,
    "adx25": adx25,
    "adx_up": adx20 & gate_col("adx_slope_14", 0.0),
    "er30": np.isfinite(Fmat["eff_ratio_20"].to_numpy(float)) & (Fmat["eff_ratio_20"].to_numpy(float) >= 0.30),
    "tq30": np.isfinite(Fmat["trend_qual_20"].to_numpy(float)) & (Fmat["trend_qual_20"].to_numpy(float) >= 0.30),
    "conf": s200 & np.isfinite(Fmat["eff_ratio_20"].to_numpy(float)) & (Fmat["eff_ratio_20"].to_numpy(float) >= 0.30) & adx20,
}
print("gates=%d  OOF bars=%d  exec=%d  models=%d  dedup=%d" %
      (len(G), oof.stop - oof.start,
       len(optimize._exec_grid_oof(tp_gt_sl=C.V3_ENFORCE_TP_GT_SL, thr_grid=C.V14_LONG_THR_GRID,
                                   tp_grid=C.V14_LONG_TP_GRID, sl_grid=C.V14_LONG_SL_GRID,
                                   hold_grid=C.V14_LONG_HOLD_GRID)),
       len(C.V14_LONG_MODEL_GRID), len(C.V14_LONG_DEDUP_GRID)), flush=True)

Ftr = Fmat.iloc[tr].reset_index(drop=True); y_tr = y.iloc[tr].reset_index(drop=True)
_, fs = factor_select.factor_sets_by_dedup(Ftr, y_tr, C.V14_LONG_DEDUP_GRID)
pruned = {d: factor_select.vif_prune(Ftr, fs[d]["long"])[0] for d in C.V14_LONG_DEDUP_GRID}
EXEC = optimize._exec_grid_oof(tp_gt_sl=C.V3_ENFORCE_TP_GT_SL, thr_grid=C.V14_LONG_THR_GRID,
                               tp_grid=C.V14_LONG_TP_GRID, sl_grid=C.V14_LONG_SL_GRID,
                               hold_grid=C.V14_LONG_HOLD_GRID)
cache, rows = {}, []
for d in C.V14_LONG_DEDUP_GRID:
    for mc in C.V14_LONG_MODEL_GRID:
        tm = models.train_side(Fmat.iloc[tr.start:tr.stop], y.iloc[tr.start:tr.stop],
                               pruned[d], "long", params=models.resolve_params(mc), verbose=False)
        pred = tm.predict(Fmat)
        cache[(float(d), mc["name"])] = pred
        for g, m in G.items():
            for ec in EXEC:
                r = execution.evaluate_with_params(pred, ohlc, atr, times, "long", oof.start, oof.stop,
                                                   ec["thr_q"], ec["tp_mult"], ec["sl_mult"],
                                                   max_hold=ec["max_hold"], regime=m)
                mm = r["metrics"]
                rows.append(dict(gate=g, dedup=float(d), model=mc["name"], tp=float(ec["tp_mult"]),
                                 sl=float(ec["sl_mult"]), hold=int(ec["max_hold"]),
                                 thr_q=float(ec["thr_q"]), thr_abs=float(r["thr_abs"]),
                                 n=int(mm["n_trades"]), total_return=float(mm["total_return"]),
                                 sharpe=float(mm["sharpe"]), tp_rate=float(mm["tp_rate"]),
                                 win=float(mm["win_rate"]), payoff=float(mm["payoff_ratio"])))
    print("   dedup %.2f done (%.0fs)" % (d, time.time() - t0), flush=True)
raw = pd.DataFrame(rows)
raw.to_csv("/workspace/solusdt_repo/reports/v14/v15_gate_grid.csv", index=False)

print("\n" + "=" * 122)
print("%-13s %-34s | %-34s" % ("gate", "OOF 选中", "OOC 观察"))
print("=" * 122)
res = []
for g in G:
    sub = raw[raw.gate == g]
    best = optimize.select_return_anchor_sharpe(optimize._select_best(sub), C.V14_RET_DROP).iloc[0]
    pred = cache[(float(best.dedup), best.model)]
    ro = execution.evaluate_with_threshold(pred, ohlc, atr, times, "long", ooc.start, ooc.stop,
                                           float(best.thr_abs), float(best.tp), float(best.sl),
                                           max_hold=int(best.hold), regime=G[g])
    mo = ro["metrics"]
    res.append(dict(gate=g, oof_ret=float(best.total_return), oof_sharpe=float(best.sharpe), oof_n=int(best.n),
                    ooc_ret=float(mo["total_return"]), ooc_sharpe=float(mo["sharpe"]), ooc_n=int(mo["n_trades"]),
                    tp=float(best.tp), sl=float(best.sl), hold=int(best.hold), dedup=float(best.dedup),
                    model=best.model, thr_abs=float(best.thr_abs)))
    print("%-13s | 收益%+6.2f%% 夏普%+5.2f 笔%3d 胜%2.0f%% | 收益%+6.2f%% 夏普%+5.2f 笔%3d 胜%2.0f%% | tp%.1f sl%.1f h%2d d%.2f %s"
          % (g, best.total_return * 100, best.sharpe, best.n, best.win * 100,
             mo["total_return"] * 100, mo["sharpe"], mo["n_trades"], mo["win_rate"] * 100,
             best.tp, best.sl, best.hold, best.dedup, best.model), flush=True)
res = pd.DataFrame(res)
res.to_csv("/workspace/solusdt_repo/reports/v14/v15_gate_summary.csv", index=False)

print("\n---- 跨门控用 v14 选优规则(收益锚+夏普)选最终解 ----")
wins = res.rename(columns={"gate": "regime", "oof_ret": "total_return", "oof_sharpe": "sharpe"})
anchor = wins.total_return.max()
sel = optimize.select_return_anchor_sharpe(wins[["regime", "total_return", "sharpe"]], C.V14_RET_DROP).iloc[0]
print("收益锚 %+.2f%% | 让步带 >= %+.2f%% | 选定门控: %s (OOF 收益 %+.2f%% / 夏普 %.2f | OOC %+.2f%% / 夏普 %.2f)"
      % (anchor * 100, (anchor - C.V14_RET_DROP) * 100, sel.regime,
         sel.total_return * 100, sel.sharpe,
         res.set_index("gate").loc[sel.regime, "ooc_ret"] * 100,
         res.set_index("gate").loc[sel.regime, "ooc_sharpe"]))
print("\n耗时 %.0fs" % (time.time() - t0))
