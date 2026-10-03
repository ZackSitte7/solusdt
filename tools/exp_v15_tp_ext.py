# -*- coding: utf-8 -*-
"""v15 第二轮: v14 报告指出 tp=6.0 一直贴在上界(4 个制度一致选它)。
本实验把 tp 网格外扩到 12, 看是否为「被网格截断」的机会 —— 与门控一起扫。
gate ∈ {none, sma200, adx_up, sma200_slope}; 其余口径同 v14。
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

def gc(n, t):
    x = Fmat[n].to_numpy(dtype=float); return np.isfinite(x) & (x > t)
G = {"none": np.ones(len(df), bool),
     "sma200": regime.regime_masks(df, "sma200")["long"],
     "sma200_slope": regime.regime_masks(df, "sma200_slope")["long"],
     "adx_up": gc("adx_14", 20.0) & gc("adx_slope_14", 0.0)}

TP = (2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0)
SL = (1.0, 1.5, 2.0, 2.5, 3.0)
HLD = (12, 24, 36, 48, 72)
EXEC = optimize._exec_grid_oof(tp_gt_sl=True, thr_grid=C.V14_LONG_THR_GRID,
                               tp_grid=TP, sl_grid=SL, hold_grid=HLD)
print("tp 外扩网格 exec=%d gates=%d" % (len(EXEC), len(G)), flush=True)

Ftr = Fmat.iloc[tr].reset_index(drop=True); y_tr = y.iloc[tr].reset_index(drop=True)
_, fs = factor_select.factor_sets_by_dedup(Ftr, y_tr, C.V14_LONG_DEDUP_GRID)
pruned = {d: factor_select.vif_prune(Ftr, fs[d]["long"])[0] for d in C.V14_LONG_DEDUP_GRID}
cache, rows = {}, []
for d in C.V14_LONG_DEDUP_GRID:
    for mc in C.V14_LONG_MODEL_GRID:
        tm = models.train_side(Fmat.iloc[tr.start:tr.stop], y.iloc[tr.start:tr.stop],
                               pruned[d], "long", params=models.resolve_params(mc), verbose=False)
        pred = tm.predict(Fmat); cache[(float(d), mc["name"])] = pred
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
    print("  dedup %.2f (%.0fs)" % (d, time.time() - t0), flush=True)
raw = pd.DataFrame(rows); raw.to_csv("/workspace/solusdt_repo/reports/v14/v15_tpext_grid.csv", index=False)

base_ret, base_sh = 0.09179612840019225, 8.593892798504415
print("\nv14 基线 OOF: 收益 %+.2f%% 夏普 %.2f" % (base_ret * 100, base_sh))
print("=" * 120)
res = []
for g in G:
    sub = raw[raw.gate == g]
    best = optimize.select_return_anchor_sharpe(optimize._select_best(sub), C.V14_RET_DROP).iloc[0]
    pred = cache[(float(best.dedup), best.model)]
    ro = execution.evaluate_with_threshold(pred, ohlc, atr, times, "long", ooc.start, ooc.stop,
                                           float(best.thr_abs), float(best.tp), float(best.sl),
                                           max_hold=int(best.hold), regime=G[g])
    mo = ro["metrics"]
    mr = sub.loc[sub.total_return.idxmax()]; ms = sub.loc[sub.sharpe.idxmax()]
    dom = sub[(sub.total_return > base_ret) & (sub.sharpe > base_sh)]
    print("[%s] 规则选中: 收益%+.2f%% 夏普%.2f 笔%d tp%.1f sl%.1f h%d | OOC %+.2f%%/%.2f | "
          "最高收益%+.2f%%(夏普%.2f) 最高夏普%.2f(收益%+.2f%%) | 支配v14的解=%d"
          % (g, best.total_return * 100, best.sharpe, int(best.n), best.tp, best.sl, int(best.hold),
             mo["total_return"] * 100, mo["sharpe"], mr.total_return * 100, mr.sharpe,
             ms.sharpe, ms.total_return * 100, len(dom)), flush=True)
    res.append(dict(gate=g, oof_ret=float(best.total_return), oof_sharpe=float(best.sharpe),
                    oof_n=int(best.n), ooc_ret=float(mo["total_return"]), ooc_sharpe=float(mo["sharpe"]),
                    tp=float(best.tp), sl=float(best.sl), hold=int(best.hold),
                    dedup=float(best.dedup), model=best.model, thr_abs=float(best.thr_abs)))
    if len(dom):
        print("     支配解 top3:", [(round(r.total_return*100,2), round(r.sharpe,2), r.tp, r.sl, r.hold,
              r.model) for r in dom.sort_values("sharpe", ascending=False).head(3).itertuples()])
pd.DataFrame(res).to_csv("/workspace/solusdt_repo/reports/v14/v15_tpext_summary.csv", index=False)
print("\n耗时 %.0fs" % (time.time() - t0))
