# -*- coding: utf-8 -*-
"""OOF / OOC 亏损归因分析。

用**冻结后的**参数(见 reports/metrics.json)在 train / OOF / OOC 三段上重放成交,
把每段的净盈亏拆成: 毛收益(扣成本前) / 成本 / 盈亏交易 / 出场原因(止盈/止损/超时),
并检查信号本身是否还有边际(gross_ret 的均值与 t 值)、冻结阈值在各段的相对位置。

运行: python3 scripts/analysis/analyze_trades.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]      # 仓库根(脚本位于 scripts/backtest|analysis/)
sys.path.insert(0, str(BASE_DIR))
import config as C                                        # noqa: E402
from src import data_clean, execution                     # noqa: E402
from src import factors as F_lib                          # noqa: E402
from src import models                                    # noqa: E402
from src.cv import time_split                             # noqa: E402

SEGS = ("train", "oof", "ooc")
NOTIONAL = C.TRADE_NOTIONAL
COST = 2.0 * (C.FEE_RATE + C.SLIP_RATE)      # 一个来回的总成本率


def load() -> tuple:
    raw = pd.read_parquet(C.BACKTEST_RAW)
    df = data_clean.clean(raw).reset_index(drop=True)
    df = F_lib.add_label(df, C.HORIZON)
    Fmat, _ = F_lib.build_factors(df)
    return df, Fmat


def retrain(frozen: dict, Fmat: pd.DataFrame, y: pd.Series, tr: slice) -> models.TrainedModel:
    mc = next(m for m in models.MODEL_GRID if m["name"] == frozen["model_name"])
    return models.train_side(Fmat.iloc[tr].reset_index(drop=True),
                             y.iloc[tr].reset_index(drop=True),
                             frozen["factors"], frozen["side"],
                             params=models.resolve_params(mc), verbose=False)


def decompose(trades: list, seg: slice, pred: np.ndarray, y: pd.Series) -> dict:
    """单段归因。"""
    n = len(trades)
    out = {"n_trades": n, "n_signals": int(np.isfinite(pred[seg]).sum())}
    if n == 0:
        return out
    gross = np.array([t.gross_ret for t in trades], dtype=float)
    pnl = np.array([t.pnl_usdt for t in trades], dtype=float)
    reason = np.array([t.exit_reason for t in trades])

    out["gross_pnl"] = float(gross.sum() * NOTIONAL)
    out["cost_pnl"] = float(-COST * NOTIONAL * n)
    out["net_pnl"] = float(pnl.sum())
    out["gross_per_trade_bp"] = float(gross.mean() * 1e4)
    out["cost_per_trade_bp"] = float(COST * 1e4)
    tt = gross.mean() / (gross.std(ddof=1) / np.sqrt(n)) if gross.std(ddof=1) > 0 else 0.0
    out["gross_t"] = float(tt)
    out["win_rate"] = float((pnl > 0).mean())
    out["sum_win"] = float(pnl[pnl > 0].sum())
    out["sum_loss"] = float(pnl[pnl < 0].sum())
    out["profit_factor"] = float(pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum())) \
        if (pnl < 0).any() else float("inf")
    out["avg_bars"] = float(np.mean([t.bars_held for t in trades]))
    for r in ("tp", "sl", "timeout"):
        m = reason == r
        out["%s_n" % r] = int(m.sum())
        out["%s_pnl" % r] = float(pnl[m].sum())
        out["%s_wr" % r] = float((pnl[m] > 0).mean()) if m.any() else 0.0
    # 冻结阈值在该段预测分布中的位置(越低 -> 越容易触发)
    seg_pred = pred[seg][np.isfinite(pred[seg])]
    out["thr_pct_in_seg"] = float((seg_pred <= C_THR).mean())
    out["pred_mean"] = float(seg_pred.mean())
    out["pred_std"] = float(seg_pred.std(ddof=1))
    # 全段信号质量
    ok = np.isfinite(pred[seg]) & y.iloc[seg].notna().to_numpy()
    if ok.sum() > 30:
        from scipy.stats import spearmanr
        out["ic"] = float(spearmanr(pred[seg][ok], y.iloc[seg].to_numpy()[ok]).correlation)
        out["auc"] = _auc(y.iloc[seg].to_numpy()[ok] > 0, pred[seg][ok])
    return out


def _auc(lab: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(lab)) < 2:
        return float("nan")
    return float(roc_auc_score(lab, score))


def block_breakdown(trades: list, seg: slice, df: pd.DataFrame, k: int = 4) -> list:
    """把段内按时间 k 等分, 看净盈亏与亏损是否集中在某一段。"""
    edges = np.linspace(seg.start, seg.stop, k + 1).astype(int)
    out = []
    for b in range(k):
        lo, hi = edges[b], edges[b + 1]
        pnl = sum(t.pnl_usdt for t in trades if lo <= t.entry_idx < hi)
        n = sum(1 for t in trades if lo <= t.entry_idx < hi)
        px = df["close"].iloc[hi - 1] / df["close"].iloc[lo] - 1.0
        out.append({"block": b + 1, "lo": lo, "hi": hi, "n": n,
                    "pnl": float(pnl), "mkt": float(px),
                    "t0": str(df["datetime"].iloc[lo])[:10],
                    "t1": str(df["datetime"].iloc[hi - 1])[:10]})
    return out


C_THR = 0.0
FROZEN = {}


def main() -> None:
    global C_THR, FROZEN
    df, Fmat = load()
    y = df["label"]
    tr, oof, ooc = time_split(len(df))
    ohlc = {k: df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close")}
    atr = df["atr"].to_numpy(dtype=float)
    times = df["datetime"].to_numpy()

    rep = json.loads((C.REPORT_DIR / "metrics.json").read_text(encoding="utf-8"))
    segs = {"train": tr, "oof": oof, "ooc": ooc}

    print("=" * 100)
    print("每笔成本 = 2 × (手续费 %.1fbp + 滑点 %.1fbp) = %.1fbp = %.3f USDT/笔(名义 %.0f)"
          % (C.FEE_RATE * 1e4, C.SLIP_RATE * 1e4, COST * 1e4, COST * NOTIONAL, NOTIONAL))
    print("各段买入持有: " + "  ".join(
        "%s %+.2f%%" % (s, (df["close"].iloc[segs[s].stop - 1] / df["close"].iloc[segs[s].start] - 1) * 100)
        for s in SEGS))
    print("=" * 100)

    rows = []
    for side in ("long", "short"):
        info = rep["sides"][side]
        fz = dict(side=side, model_name=info["model_name"], factors=info["factors"],
                  **info["frozen_params"])
        FROZEN[side] = fz
        C_THR = fz["thr_abs"]
        print("\n### %s | 模型 %s | 因子 %d | 阈值(绝对) %.6f | tp %.2fATR sl %.2fATR hold %d"
              % (side, fz["model_name"], len(fz["factors"]), fz["thr_abs"],
                 fz["tp_mult"], fz["sl_mult"], fz["max_hold"]))
        tm = retrain(fz, Fmat, y, tr)
        pred = tm.predict(Fmat)
        res = {}
        for s in SEGS:
            seg = segs[s]
            r = execution.evaluate_with_threshold(
                pred, ohlc, atr, times, side, seg.start, seg.stop,
                fz["thr_abs"], fz["tp_mult"], fz["sl_mult"], max_hold=fz["max_hold"])
            res[s] = r
            d = decompose(r["trades"], seg, pred, y)
            d.update(side=side, seg=s)
            rows.append(d)
            print("  [%-5s] 笔数%4d | 毛 %+8.2f | 成本 %+7.2f | 净 %+8.2f USDT | 每笔毛 %+6.1fbp"
                  " (t=%+5.2f) vs 成本 %.1fbp" %
                  (s, d["n_trades"], d.get("gross_pnl", 0), d.get("cost_pnl", 0),
                   d.get("net_pnl", 0), d.get("gross_per_trade_bp", 0), d.get("gross_t", 0),
                   COST * 1e4))
            print("         出场: 止盈%3d笔(%+8.2f, 胜率%4.0f%%) 止损%3d笔(%+8.2f)"
                  " 超时%3d笔(%+8.2f) | 盈%+7.2f 亏%+7.2f PF %.2f | IC %+.3f AUC %.3f | 阈值位于该段 %.0f%% 分位"
                  % (d.get("tp_n", 0), d.get("tp_pnl", 0), d.get("tp_wr", 0) * 100,
                     d.get("sl_n", 0), d.get("sl_pnl", 0), d.get("timeout_n", 0),
                     d.get("timeout_pnl", 0), d.get("sum_win", 0), d.get("sum_loss", 0),
                     d.get("profit_factor", 0), d.get("ic", float("nan")),
                     d.get("auc", float("nan")), d.get("thr_pct_in_seg", 0) * 100))
        for s in ("oof", "ooc"):
            bl = block_breakdown(res[s]["trades"], segs[s], df)
            print("  %s 时间四等分: " % s.upper() +
                  " | ".join("B%d %s~%s 笔%3d 净%+7.2f (行情%+6.1f%%)"
                             % (b["block"], b["t0"], b["t1"], b["n"], b["pnl"], b["mkt"] * 100)
                             for b in bl))

    pd.DataFrame(rows).to_csv(C.REPORT_DIR / "loss_attribution.csv", index=False)
    print("\n明细已落盘: reports/loss_attribution.csv")


if __name__ == "__main__":
    main()
