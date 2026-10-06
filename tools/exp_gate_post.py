# -*- coding: utf-8 -*-
"""实验(后处理): 读 exp_gate_raw.csv 选出各 (arm,gate) 的 OOF 最优, 重训该模型后在 OOC 上观察。"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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
masks = {r: regime.regime_masks(df, r)["long"] for r in C.V5_REGIME_GRID}

Ftr = Fmat.iloc[tr].reset_index(drop=True)
y_tr = y.iloc[tr].reset_index(drop=True)
ARMS = {"FULL(v14)": None, "NO7": list(C.V14_LONG_TREND_FACTORS)}
pruned = {}
for arm, exc in ARMS.items():
    _, fs = factor_select.factor_sets_by_dedup(Ftr, y_tr, C.V14_LONG_DEDUP_GRID, exclude=exc)
    pruned[arm] = {d: factor_select.vif_prune(Ftr, fs[d]["long"])[0] for d in C.V14_LONG_DEDUP_GRID}

raw = pd.read_csv(C.REPORT_DIR / "v14" / "exp_gate_raw.csv").rename(columns={"ret": "total_return"})
MG = {m["name"]: m for m in C.V14_LONG_MODEL_GRID}
rows = []
print("=" * 132)
print("%-11s %-13s | %-40s | %-40s" % ("arm", "gate", "OOF 选中", "OOC 观察(同参数)"))
print("=" * 132)
for arm in ARMS:
    for g in C.V5_REGIME_GRID:
        sub = raw[(raw.arm == arm) & (raw.gate == g)]
        best = optimize.select_return_anchor_sharpe(optimize._select_best(sub), C.V14_RET_DROP).iloc[0]
        facs = pruned[arm][float(best.dedup)]
        tm = models.train_side(Fmat.iloc[tr.start:tr.stop], y.iloc[tr.start:tr.stop], facs, "long",
                               params=models.resolve_params(MG[best.model]), verbose=False)
        pred = tm.predict(Fmat)
        ro = execution.evaluate_with_threshold(pred, ohlc, atr, times, "long", ooc.start, ooc.stop,
                                               float(best.thr_abs), float(best.tp), float(best.sl),
                                               max_hold=int(best.hold), regime=masks[g])
        mo = ro["metrics"]
        print("%-11s %-13s | 收益%+6.2f%% 夏普%+5.2f 笔%3d 胜%2.0f%% | 收益%+6.2f%% 夏普%+5.2f 笔%3d 胜%2.0f%% | tp%.1f sl%.1f hold%2d d%.2f nf%2d %s"
              % (arm, g, best.total_return * 100, best.sharpe, best.n, best.win * 100,
                 mo["total_return"] * 100, mo["sharpe"], mo["n_trades"], mo["win_rate"] * 100,
                 best.tp, best.sl, best.hold, best.dedup, len(facs), best.model), flush=True)
        rows.append(dict(arm=arm, gate=g, oof_ret=float(best.total_return), oof_sharpe=float(best.sharpe),
                         oof_n=int(best.n), ooc_ret=float(mo["total_return"]), ooc_sharpe=float(mo["sharpe"]),
                         ooc_n=int(mo["n_trades"]), tp=float(best.tp), sl=float(best.sl),
                         hold=int(best.hold), dedup=float(best.dedup), model=best.model,
                         thr_abs=float(best.thr_abs), n_factors=len(facs)))
pd.DataFrame(rows).to_csv(C.REPORT_DIR / "v14" / "exp_gate_summary.csv", index=False)
print("\n耗时 %.0fs" % (time.time() - t0))
