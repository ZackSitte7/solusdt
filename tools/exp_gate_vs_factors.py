# -*- coding: utf-8 -*-
"""实验: long 的「趋势门控」vs「趋势因子」—— 替代还是叠加?

2 个因子臂 × 4 个制度门控, 其余口径与 v14 long 完全一致(去冗余网格/模型候选/执行网格/
VIF 剪枝/成本/切分/选优规则)。选择只在 OOF, OOC 仅观察。
  arm FULL : 因子池含 v14 的 7 个趋势因子(即 v14 现状)
  arm NO7  : 因子池排除这 7 个趋势因子
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
print("bars=%d factors=%d | train[%d,%d) oof[%d,%d) ooc[%d,%d)" %
      (len(df), Fmat.shape[1], tr.start, tr.stop, oof.start, oof.stop, ooc.start, ooc.stop), flush=True)

Ftr = Fmat.iloc[tr].reset_index(drop=True)
y_tr = y.iloc[tr].reset_index(drop=True)
GATES = list(C.V5_REGIME_GRID)
masks = {r: regime.regime_masks(df, r)["long"] for r in GATES}
EXEC = optimize._exec_grid_oof(tp_gt_sl=C.V3_ENFORCE_TP_GT_SL,
                               thr_grid=C.V14_LONG_THR_GRID, tp_grid=C.V14_LONG_TP_GRID,
                               sl_grid=C.V14_LONG_SL_GRID, hold_grid=C.V14_LONG_HOLD_GRID)
print("exec=%d model=%d dedup=%s" % (len(EXEC), len(C.V14_LONG_MODEL_GRID), C.V14_LONG_DEDUP_GRID), flush=True)

ARMS = {"FULL(v14)": None, "NO7": list(C.V14_LONG_TREND_FACTORS)}
cache, raw_rows = {}, []
for arm, exc in ARMS.items():
    _, fs = factor_select.factor_sets_by_dedup(Ftr, y_tr, C.V14_LONG_DEDUP_GRID, exclude=exc)
    pruned = {d: factor_select.vif_prune(Ftr, fs[d]["long"])[0] for d in C.V14_LONG_DEDUP_GRID}
    print("\n== arm %s 因子数 %s" % (arm, {d: len(v) for d, v in pruned.items()}), flush=True)
    for d in C.V14_LONG_DEDUP_GRID:
        for mc in C.V14_LONG_MODEL_GRID:
            tm = models.train_side(Fmat.iloc[tr.start:tr.stop], y.iloc[tr.start:tr.stop],
                                   pruned[d], "long", params=models.resolve_params(mc), verbose=False)
            pred = tm.predict(Fmat)
            cache[(arm, float(d), mc["name"])] = pred
            for g in GATES:
                for ec in EXEC:
                    r = execution.evaluate_with_params(pred, ohlc, atr, times, "long",
                                                       oof.start, oof.stop, ec["thr_q"],
                                                       ec["tp_mult"], ec["sl_mult"],
                                                       max_hold=ec["max_hold"], regime=masks[g])
                    m = r["metrics"]
                    raw_rows.append(dict(arm=arm, gate=g, dedup=float(d), model=mc["name"],
                                         tp=float(ec["tp_mult"]), sl=float(ec["sl_mult"]),
                                         hold=int(ec["max_hold"]), thr_q=float(ec["thr_q"]),
                                         thr_abs=float(r["thr_abs"]), n=int(m["n_trades"]),
                                         ret=float(m["total_return"]), sharpe=float(m["sharpe"]),
                                         payoff=float(m["payoff_ratio"]),
                                         tp_rate=float(m["tp_rate"]), win=float(m["win_rate"])))
        print("   %s | dedup %.2f  (%.0fs)" % (arm, d, time.time() - t0), flush=True)

raw = pd.DataFrame(raw_rows); raw.to_csv("/workspace/solusdt_repo/reports/v14/exp_gate_raw.csv", index=False)

rows = []
print("\n" + "=" * 126)
print("%-11s %-13s | %-38s | %-42s" % ("arm", "gate", "OOF 选中", "OOC 观察(同一套参数)"))
print("=" * 126)
for arm in ARMS:
    for g in GATES:
        sub = raw[(raw.arm == arm) & (raw.gate == g)]
        best = optimize.select_return_anchor_sharpe(optimize._select_best(sub), C.V14_RET_DROP).iloc[0]
        pred = cache[(arm, float(best.dedup), best.model)]
        ro = execution.evaluate_with_threshold(pred, ohlc, atr, times, "long", ooc.start, ooc.stop,
                                               float(best.thr_abs), float(best.tp), float(best.sl),
                                               max_hold=int(best.hold), regime=masks[g])
        mo = ro["metrics"]
        print("%-11s %-13s | 收益%+6.2f%% 夏普%+5.2f 笔%3d 胜%2.0f%% 盈亏%4.2f | 收益%+6.2f%% 夏普%+5.2f 笔%3d 胜%2.0f%% 盈亏%4.2f | %.1f/%.1f/%d d%.2f %s"
              % (arm, g, best.ret * 100, best.sharpe, best.n, best.win * 100, best.payoff,
                 mo["total_return"] * 100, mo["sharpe"], mo["n_trades"], mo["win_rate"] * 100,
                 mo["payoff_ratio"], best.tp, best.sl, best.hold, best.dedup, best.model))
        rows.append(dict(arm=arm, gate=g, oof_ret=float(best.ret), oof_sharpe=float(best.sharpe),
                         oof_n=int(best.n), ooc_ret=float(mo["total_return"]),
                         ooc_sharpe=float(mo["sharpe"]), ooc_n=int(mo["n_trades"]),
                         tp=float(best.tp), sl=float(best.sl), hold=int(best.hold),
                         dedup=float(best.dedup), model=best.model, thr_abs=float(best.thr_abs)))
pd.DataFrame(rows).to_csv("/workspace/solusdt_repo/reports/v14/exp_gate_summary.csv", index=False)
print("\n耗时 %.0fs" % (time.time() - t0))
