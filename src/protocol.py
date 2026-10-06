# -*- coding: utf-8 -*-
"""抗过拟合选择协议 —— 把"选择"从 OOF 移回 train 内部的前推折。

与 src/optimize.py 的分工:
  - optimize.py: v1 路径, 在 **OOF** 上选优(需求7/8/9 的原始实现)。
  - protocol.py: v2 路径, 在 **train 内部的前推折**上选优, OOF 与 OOC 一并成为纯检验段。

为什么这么改(v1 的实测证据见 scripts/analysis/analyze_trades.py / DESIGN.md 第 4 节):
  v1 在 1969 根的 OOF 上做了 6480 次比较, 而入选组合每笔毛收益的 t 值只有 +1.53,
  统计上无法区分于噪声 -> 选出来的必然是噪声的最优实现。多头 4765 组里只有 10.9%
  为正、均值 -3.64%, 入选的 +7.12% 是 +3.53σ 的孤点, OOC 随即回到 -7.23%(≈参数面均值)。
  把选择集换成 train 内部的前推折后, 选择样本从 1969 根升到 ~6892 根, 且每一折都是
  真正的样本外前推, 选择结果才有可能外推。

前推折结构(WF_FOLDS=3, n_train=9189):
  折1: 训练 [0, 2297-gap)  验证 [2297, 4594)
  折2: 训练 [0, 4594-gap)  验证 [4594, 6892)
  折3: 训练 [0, 6892-gap)  验证 [6892, 9189)      gap = HORIZON + WF_EMBARGO
每折都在**自己的前缀**上重新筛因子并重新训练, 因此不存在跨折的信息泄漏。
"""
from __future__ import annotations

from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import config as C
from src import factor_select, models
from src.execution import evaluate_with_params


# ================================================================ 前推折
def wf_folds(n_train: int, n_folds: int = None, embargo: int = None,
             horizon: int = None) -> List[Tuple[slice, slice]]:
    """按时间递增前缀切分训练段, 返回 [(train_slice, valid_slice), ...]。

    第 k 折用前 k/(F+1) 根的**前缀**训练, 在紧随其后的那一段上验证;
    前缀与验证块之间挖掉 horizon + embargo 根(purge 标签重叠 + 阻断自相关残留)。
    """
    n_folds = C.WF_FOLDS if n_folds is None else n_folds
    embargo = C.WF_EMBARGO if embargo is None else embargo
    horizon = C.HORIZON if horizon is None else horizon
    gap = horizon + embargo
    out: List[Tuple[slice, slice]] = []
    for k in range(1, n_folds + 1):
        cut = int(n_train * k / (n_folds + 1))
        nxt = int(n_train * (k + 1) / (n_folds + 1))
        tr_end = max(1, cut - gap)
        out.append((slice(0, tr_end), slice(cut, nxt)))
    return out


def proto_grid() -> List[dict]:
    """预注册候选网格(模型 × 阈值 × 止盈 × 止损 × 持有 × 跟踪止损)。"""
    return [dict(model=m, thr_q=q, tp_mult=tp, sl_mult=sl, max_hold=h, trail_mult=tr)
            for m, q, tp, sl, h, tr in product(
                C.PROTO_MODELS, C.PROTO_THR_GRID, C.PROTO_TP_GRID,
                C.PROTO_SL_GRID, C.PROTO_HOLD_GRID, C.PROTO_TRAIL_GRID)]


def freeze_threshold(pred_parts: List[np.ndarray], side: str, thr_q: float) -> float:
    """由**选择阶段的前推折样本外预测**冻结出绝对阈值。

    v1 用评估段自身的预测分位当阈值(等于看了评估段的分布); v2 一律用选择阶段的
    样本外预测分布冻结成绝对值, OOF/OOC 只按这个绝对值发信号, 不再回看自己的分布。
    """
    if not pred_parts:
        return float("nan")
    v = np.concatenate([p[np.isfinite(p)] for p in pred_parts])
    if v.size < 50:
        return float("nan")
    q = 1.0 - thr_q if side == "long" else thr_q
    return float(np.quantile(v, q))


# ================================================================ 选择主流程
def select_on_wf(F: pd.DataFrame, y: pd.Series, side: str, ohlc: dict, atr: np.ndarray,
                 times: np.ndarray, train_slice: slice,
                 verbose: bool = True) -> Dict[str, object]:
    """在 train 内部的**前推折**上选优, 返回冻结参数。OOF/OOC 全程不参与。

    返回 dict:
      model_name / thr_q / tp_mult / sl_mult / max_hold / trail_mult / thr_abs  冻结配置
      model / factors                  在**整个 train** 上重训的最终模型与其因子
      agg / raw                        选优表与逐折明细(供落盘审阅)
      folds                            各折区间与各折因子集(展示"因子也在前推")
    """
    ts, te = train_slice.start, train_slice.stop
    n_train = te - ts
    folds = wf_folds(n_train)
    grid = proto_grid()
    by_model = {m: [g for g in grid if g["model"] == m] for m in C.PROTO_MODELS}

    rows: List[dict] = []
    oos_preds: Dict[str, List[np.ndarray]] = {m: [] for m in C.PROTO_MODELS}
    fold_info: List[dict] = []

    for fi, (tr_l, va_l) in enumerate(folds, 1):
        fs = factor_select.select_factors_robust(F.iloc[tr_l], y.iloc[tr_l], side)
        facs = list(fs["selected"])
        fold_info.append(dict(fold=fi, train_end=int(tr_l.stop), val_start=int(va_l.start),
                              val_end=int(va_l.stop), n_train=int(tr_l.stop - tr_l.start),
                              n_val=int(va_l.stop - va_l.start), n_factors=len(facs),
                              factors=", ".join(facs)))
        if not facs:
            if verbose:
                print("    [%s] 折%d 无可用因子, 跳过" % (side, fi))
            continue
        for name in C.PROTO_MODELS:
            mc = next(m for m in models.MODEL_GRID if m["name"] == name)
            tm = models.train_side(F.iloc[tr_l], y.iloc[tr_l], facs, side,
                                   params=models.resolve_params(mc), verbose=False)
            pred = tm.predict(F)
            oos_preds[name].append(pred[va_l].copy())
            for ec in by_model[name]:
                r = evaluate_with_params(pred, ohlc, atr, times, side, va_l.start, va_l.stop,
                                         ec["thr_q"], ec["tp_mult"], ec["sl_mult"],
                                         max_hold=ec["max_hold"], trail_mult=ec["trail_mult"])
                m = r["metrics"]
                rows.append(dict(fold=fi, **ec, n=int(m["n_trades"]),
                                 sharpe=float(m["sharpe"]),
                                 total_return=float(m["total_return"]),
                                 max_drawdown=float(m["max_drawdown"]),
                                 win_rate=float(m["win_rate"]), payoff=float(m["payoff_ratio"]),
                                 tp_rate=float(m["tp_rate"]), sl_rate=float(m["sl_rate"]),
                                 timeout_rate=float(m["timeout_rate"])))
        if verbose:
            print("    [%s] 前推折 %d/%d: 训练 %5d 根 / 验证 %4d 根 | 因子 %2d 个 %s"
                  % (side, fi, len(folds), tr_l.stop - tr_l.start, va_l.stop - va_l.start,
                     len(facs), facs[:4] + (["..."] if len(facs) > 4 else [])))

    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError("前推折评估为空: %s" % side)

    keys = ["model", "thr_q", "tp_mult", "sl_mult", "max_hold", "trail_mult"]
    g = raw[raw["n"] >= C.PROTO_MIN_TRADES_PER_FOLD]            # 逐折最少笔数门槛
    if g.empty:
        g = raw
    agg = (g.groupby(keys)
             .agg(n_folds=("fold", "nunique"),
                  worst_return=("total_return", "min"),
                  mean_return=("total_return", "mean"),
                  worst_sharpe=("sharpe", "min"),
                  mean_sharpe=("sharpe", "mean"),
                  worst_dd=("max_drawdown", "max"),
                  mean_n=("n", "mean"), tp_rate=("tp_rate", "mean"),
                  win_rate=("win_rate", "mean"), payoff=("payoff", "mean"))
             .reset_index())
    # 必须覆盖绝大多数折, 否则"某折几乎不交易"会伪装成稳健
    need = max(2, len(folds) - 1)
    agg = agg[agg["n_folds"] >= need].reset_index(drop=True)
    if agg.empty:
        agg = (g.groupby(keys).agg(n_folds=("fold", "nunique"),
                                   worst_return=("total_return", "min"),
                                   mean_return=("total_return", "mean"),
                                   worst_sharpe=("sharpe", "min"),
                                   mean_sharpe=("sharpe", "mean"),
                                   worst_dd=("max_drawdown", "max"),
                                   mean_n=("n", "mean"), tp_rate=("tp_rate", "mean"),
                                   win_rate=("win_rate", "mean"), payoff=("payoff", "mean"))
                 .reset_index())
    # 稳健准则: 主看**最差折**收益, 次看折间平均夏普(而非 v1 的"选尖峰")
    if C.SELECT_RULE == "min_fold":
        agg = agg.sort_values(["worst_return", "mean_sharpe"], ascending=False)
    else:
        agg = agg.sort_values(["mean_return", "mean_sharpe"], ascending=False)
    agg = agg.reset_index(drop=True)
    best = agg.iloc[0]

    # 用**整个 train** 重训最终模型: 因子集在全 train 上重筛(仍只用 train)
    fs_full = factor_select.select_factors_robust(F.iloc[train_slice], y.iloc[train_slice], side)
    facs_full = list(fs_full["selected"])
    mc = next(m for m in models.MODEL_GRID if m["name"] == best["model"])
    tm_full = models.train_side(F.iloc[train_slice], y.iloc[train_slice], facs_full, side,
                                params=models.resolve_params(mc), verbose=False)
    # 阈值只用**入选模型**在选择阶段的样本外预测分布冻结, 不混入其它模型的分布
    thr_abs = freeze_threshold(oos_preds[str(best["model"])], side, float(best["thr_q"]))

    return dict(model_name=str(best["model"]), model=tm_full, factors=facs_full,
                thr_q=float(best["thr_q"]), thr_abs=thr_abs,
                tp_mult=float(best["tp_mult"]), sl_mult=float(best["sl_mult"]),
                max_hold=int(best["max_hold"]), trail_mult=float(best["trail_mult"]),
                cv_metrics={k: float(best[k]) for k in
                            ("worst_return", "mean_return", "worst_sharpe", "mean_sharpe",
                             "mean_n", "tp_rate", "win_rate", "payoff", "n_folds")},
                agg=agg, raw=raw, folds=fold_info, factor_report=fs_full,
                n_combos=len(grid), n_folds=len(folds))


# ================================================================ 诚实性工具
def block_bootstrap_ci(pnl: np.ndarray, n_boot: int = None, block: int = None,
                       seed: int = None) -> Dict[str, float]:
    """对成交盈亏做**循环分块自助**(circular block bootstrap), 给出总盈亏的 95% 区间。

    用途: OOC 是单次实现, 一个 -72.26 不该被当成"策略必然亏损"的证据。分块自助保留
    交易之间的时序依赖(连亏/连赢), 给出"若重抽这段行情"的区间。**仅用于观察段披露**。
    """
    n_boot = C.BOOT_N if n_boot is None else n_boot
    block = C.BOOT_BLOCK if block is None else block
    seed = C.BOOT_SEED if seed is None else seed
    x = np.asarray(pnl, dtype=float)
    n = len(x)
    if n < 5:
        return dict(n=n, total=float(x.sum()) if n else 0.0, lo=float("nan"),
                    hi=float("nan"), p_pos=float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offs = np.arange(block)
    tot = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)[:, None]
        idx = ((starts + offs) % n).ravel()[:n]
        tot[b] = x[idx].sum()
    return dict(n=n, total=float(x.sum()),
                lo=float(np.percentile(tot, 2.5)), hi=float(np.percentile(tot, 97.5)),
                p_pos=float((tot > 0).mean()))


def disclosure(info: dict) -> str:
    n_eff = info["n_combos"]
    return ("选择协议 v2: 在 train 内部 %d 个前推折上比较 %d 个预注册组合(共 %d 次评估), "
            "准则为「%s」。OOF 与 OOC **均未参与任何选择**, 仅以选择阶段冻结的绝对阈值评估一次。"
            % (info["n_folds"], n_eff, n_eff * info["n_folds"], C.SELECT_RULE))
