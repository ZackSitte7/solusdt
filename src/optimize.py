# -*- coding: utf-8 -*-
"""优化器 —— 模型层(超参候选) × 执行层(阈值/止盈/止损) 的选优。

本文件有两个入口:
  - `optimize_on_oof` (需求7/8/9 的**主路径**): 用 **train** 训练各模型候选,
    再在 **OOF** 上网格评估每个 (模型, 执行) 组合并选优; OOC 全程只观察。
    `optimize_full_on_oof` 是它的**扩展版**(主脚本使用): 外层再叠「去冗余阈值」,
    即把「因子集松紧」也交给 OOF 选优; 内层执行网格增加「最长持有期」。
  - `nested_select` (备用路径, 供旧流水线 run_pipeline.py 使用): 在 train 内部做
    嵌套 CV, 以各折均值选优, 不触碰 OOF/OOC。

为什么主路径把优化放在 OOF:
  用户规格明确要求「只在 OOF 上优化, OOC 只观察」。这时样本被切成
  train(拟合) / OOF(选择) / OOC(唯一纯净检验), 是标准的 train-valid-test 结构;
  OOC 因此是唯一未被任何选择过程污染的段, 其表现是对外可披露的诚实结果。
  代价: OOF 上做了 ~数百次比较, 最优解可能含噪声 -> 故设 OOF_MIN_TRADES 门槛,
  并要求 OOC 表现作为最终(而非可选)检验。design==runtime 见 DESIGN.md / tests。
"""
from __future__ import annotations

from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import config as C
from src import models
from src.cv import PurgedKFold
from src.execution import evaluate_with_params

MIN_TRADES_PER_FOLD = 8       # 单折最少成交笔数, 低于此该组合该折不计分


def _exec_grid() -> List[dict]:
    return [dict(thr_q=q, tp_mult=tp, sl_mult=sl)
            for q, tp, sl in product(C.EXEC_THR_GRID, C.TP_ATR_GRID, C.SL_ATR_GRID)]


def nested_select(F: pd.DataFrame, y: pd.Series, factors: List[str], side: str,
                  ohlc: dict, atr: np.ndarray, times: np.ndarray,
                  train_slice: slice, verbose: bool = True) -> Tuple[str, dict, pd.DataFrame]:
    """在训练段内部做嵌套 CV, 选出 (模型候选名, 执行参数)。"""
    ts, te = train_slice.start, train_slice.stop
    n_train = te - ts
    cands = models.MODEL_GRID
    grid = _exec_grid()
    rows: List[dict] = []

    folds = list(PurgedKFold(n_splits=C.NESTED_FOLDS, embargo=C.CV_EMBARGO_BARS,
                             horizon=C.HORIZON).split(n_train))
    for fi, (tr_l, va_l) in enumerate(folds, 1):
        tr_g, va_g = ts + tr_l, ts + va_l
        v0, v1 = int(va_g.min()), int(va_g.max()) + 1
        for mc in cands:
            booster = models.train_fold(F, y, factors, tr_g, va_g,
                                        params=models.resolve_params(mc))
            pred = models.predict_booster(booster, F, factors)
            for ec in grid:
                r = evaluate_with_params(pred, ohlc, atr, times, side, v0, v1,
                                         ec["thr_q"], ec["tp_mult"], ec["sl_mult"])
                m = r["metrics"]
                if m["n_trades"] < MIN_TRADES_PER_FOLD:
                    continue
                rows.append(dict(fold=fi, model=mc["name"], **ec,
                                 n=m["n_trades"], sharpe=m["sharpe"],
                                 total_return=m["total_return"], tp_rate=m["tp_rate"],
                                 win_rate=m["win_rate"], payoff=m["payoff_ratio"]))
        if verbose:
            print("    [%s] 外层折 %d/%d 完成 (验证 %d 根)"
                  % (side, fi, len(folds), v1 - v0))

    g = pd.DataFrame(rows)
    if g.empty:
        return cands[0]["name"], dict(thr_q=0.6, tp_mult=2.0, sl_mult=1.5), g

    # 退化参数淘汰: 止盈几乎不触发的组合直接作废
    g = g[g["tp_rate"] >= C.MIN_TP_RATE]
    if g.empty:
        g = pd.DataFrame(rows)

    agg = (g.groupby(["model", "thr_q", "tp_mult", "sl_mult"])
             .agg(n_folds=("sharpe", "size"), sharpe=("sharpe", "mean"),
                  total_return=("total_return", "mean"),
                  tp_rate=("tp_rate", "mean"), win_rate=("win_rate", "mean"),
                  payoff=("payoff", "mean"), n_trades=("n", "mean"))
             .reset_index()
             .sort_values([C.OBJECTIVE_PRIMARY, C.OBJECTIVE_SECONDARY], ascending=False)
             .reset_index(drop=True))
    best = agg.iloc[0]
    return best["model"], dict(thr_q=float(best["thr_q"]), tp_mult=float(best["tp_mult"]),
                               sl_mult=float(best["sl_mult"])), agg


def disclosure(n_combos: int, n_folds: int) -> str:
    return ("嵌套 CV: %d 个(模型×执行)组合 x %d 折 = %d 次评估, 选优依据为**各折均值**。"
            "聚合后比较的独立组合为 %d 个。OOF/OOC 均未参与选择。" %
            (n_combos, n_folds, n_combos * n_folds, n_combos))


# ================================================================ 主路径: OOF 优化
def _select_best(g: pd.DataFrame) -> pd.DataFrame:
    """选优规则(需求7/8/9): 成交数门槛 -> 退化参数淘汰 -> 主目标+次目标排序。"""
    usable = g[g["n"] >= C.OOF_MIN_TRADES]
    if usable.empty:                        # 门槛过严时退回全量, 保证总有解
        usable = g
    cand = usable[usable["tp_rate"] >= C.MIN_TP_RATE]
    if cand.empty:                          # 止盈几乎不触发的组合全部作废; 仍空则退回
        cand = usable
    return cand.sort_values([C.OBJECTIVE_PRIMARY, C.OBJECTIVE_SECONDARY],
                            ascending=False).reset_index(drop=True)


def select_return_anchor_sharpe(agg: pd.DataFrame, ret_drop: float) -> pd.DataFrame:
    """v13 选优规则: **收益锚 + 夏普择优**。

    在(已按 OOF_MIN_TRADES / MIN_TP_RATE 过滤后的)组合表 `agg` 里:
      1. 取 OOF 收益最高的解为**收益锚**, 记其收益为 anchor_ret;
      2. 在「OOF 收益 >= anchor_ret - ret_drop」的**有界让步带**内, 取 OOF 夏普最高者
         (夏普并列时取收益更高者)。

    ret_drop=0 时逐位退化为 (收益, 夏普) 字典序 —— 即 v5 的原规则。
    ret_drop>0 时, 用**上限为 ret_drop 的绝对收益让步**换取更高夏普, 从而同时优化两者。
    让步带用绝对值而非比例: OOF 收益可能为负, 按比例会给出方向错误(反而收紧)的门槛。

    返回重排后的 `agg`(第一行即选中), 便于既有下游直接 `iloc[0]` 复用。
    """
    if agg.empty:
        return agg
    ret = agg[C.OBJECTIVE_PRIMARY].to_numpy(dtype=float)
    anchor = float(np.nanmax(ret))
    band = agg[ret >= anchor - ret_drop].sort_values(
        [C.OBJECTIVE_SECONDARY, C.OBJECTIVE_PRIMARY], ascending=False)
    rest = agg.drop(index=band.index).sort_values(
        [C.OBJECTIVE_PRIMARY, C.OBJECTIVE_SECONDARY], ascending=False)
    return pd.concat([band, rest], ignore_index=True)


def optimize_on_oof(F: pd.DataFrame, y: pd.Series, factors: List[str], side: str,
                    ohlc: dict, atr: np.ndarray, times: np.ndarray,
                    train_slice: slice, oof_slice: slice,
                    verbose: bool = True) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """在 **OOF** 上优化 (模型候选 × 执行参数)。训练只用 train, 选优只用 OOF。

    返回 (best, agg, raw):
      best = dict(model_name, model, thr_q, thr_abs, tp_mult, sl_mult, oof_metrics)
      agg  = 按选优规则排序后的组合表(供落盘审阅)
      raw  = 所有 (模型×执行) 的原始 OOF 评估明细
    """
    ts, te = train_slice.start, train_slice.stop
    grid = _exec_grid()
    rows: List[dict] = []
    trained: Dict[str, models.TrainedModel] = {}
    for mc in models.MODEL_GRID:
        tm = models.train_side(F.iloc[ts:te], y.iloc[ts:te], factors, side,
                               params=models.resolve_params(mc), verbose=False)
        trained[mc["name"]] = tm
        pred = tm.predict(F)                       # 全序列预测, 仅取 OOF 段评估
        for ec in grid:
            r = evaluate_with_params(pred, ohlc, atr, times, side,
                                     oof_slice.start, oof_slice.stop,
                                     ec["thr_q"], ec["tp_mult"], ec["sl_mult"])
            m = r["metrics"]
            rows.append(dict(model=mc["name"], **ec, n=int(m["n_trades"]),
                             thr_abs=float(r["thr_abs"]), sharpe=float(m["sharpe"]),
                             total_return=float(m["total_return"]), calmar=float(m["calmar"]),
                             max_drawdown=float(m["max_drawdown"]), win_rate=float(m["win_rate"]),
                             payoff=float(m["payoff_ratio"]), tp_rate=float(m["tp_rate"])))
        if verbose:
            print("    [%s] 模型候选 %-9s 已评估 %d 组执行参数" % (side, mc["name"], len(grid)))

    raw = pd.DataFrame(rows)
    agg = _select_best(raw)
    best = agg.iloc[0]
    name = best["model"]
    out = dict(model_name=name, model=trained[name], thr_q=float(best["thr_q"]),
               thr_abs=float(best["thr_abs"]), tp_mult=float(best["tp_mult"]),
               sl_mult=float(best["sl_mult"]),
               oof_metrics={k: float(best[k]) for k in
                            ("n", "sharpe", "total_return", "calmar", "max_drawdown",
                             "win_rate", "payoff", "tp_rate")})
    return out, agg, raw


# ================================================ 主路径扩展: OOF 外层扫描(去冗余 × 持有期)
def _exec_grid_oof(tp_gt_sl: bool = False, thr_grid=None, tp_grid=None,
                   sl_grid=None, hold_grid=None) -> List[dict]:
    """OOF 外层扫描用的执行网格: 阈值分位 × 止盈 × 止损 × **最长持有期**。

    tp_gt_sl=True 时只保留 `tp_mult > sl_mult` 的组合(v3 的风险报酬比 > 1 约束)。
    四条轴均可覆盖: v13 给 long 传入外扩后的专用网格, short 仍用全局默认网格。
    """
    thr_grid = C.EXEC_THR_GRID if thr_grid is None else thr_grid
    tp_grid = C.TP_ATR_GRID if tp_grid is None else tp_grid
    sl_grid = C.SL_ATR_GRID if sl_grid is None else sl_grid
    hold_grid = C.OOF_HOLD_GRID if hold_grid is None else hold_grid
    grid = [dict(thr_q=q, tp_mult=tp, sl_mult=sl, max_hold=h)
            for q, tp, sl, h in product(thr_grid, tp_grid, sl_grid, hold_grid)]
    if tp_gt_sl:
        grid = [g for g in grid if g["tp_mult"] > g["sl_mult"]]
    return grid


def optimize_full_on_oof(F: pd.DataFrame, y: pd.Series, factor_sets: Dict[float, List[str]],
                         side: str, ohlc: dict, atr: np.ndarray, times: np.ndarray,
                         train_slice: slice, oof_slice: slice,
                         tp_gt_sl: bool = False, regime: np.ndarray = None,
                         ret_drop: float = None,
                         thr_grid=None, tp_grid=None, sl_grid=None, hold_grid=None,
                         model_grid: List[Dict] = None,
                         verbose: bool = True, entry_hi: int = None
                         ) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """在 **OOF** 上做两阶段选优, 训练只用 train、选择只用 OOF。

    外层: 去冗余阈值(决定因子集) × 模型超参候选;
    内层: 阈值分位 × 止盈 × 止损 × 最长持有期。
    每个外层组合都用 train 段**重新筛因子并重新训练**, 避免"因子集"成为未优化的死参数。

    tp_gt_sl=True 时内层网格只保留 **止盈幅度 > 止损幅度** 的组合(v3 约束)。
    regime: 可选制度门控(全序列 bool 数组), 与阈值信号取交集(v5)。
    ret_drop: v13 选优规则。None = 沿用 (收益, 夏普) 字典序(v5); 给定数值则改用
              "收益锚 + 夏普择优"(见 select_return_anchor_sharpe)。

    factor_sets: {去冗余阈值: 该方向的因子列表}(由 factor_select 在 train 上产出)
    返回 (best, agg, raw); trained[(dedup, model_name)] 供冻结时取回模型。
    """
    ts, te = train_slice.start, train_slice.stop
    model_grid = models.MODEL_GRID if model_grid is None else model_grid
    grid = _exec_grid_oof(tp_gt_sl=tp_gt_sl, thr_grid=thr_grid, tp_grid=tp_grid,
                          sl_grid=sl_grid, hold_grid=hold_grid)
    rows: List[dict] = []
    trained: Dict[Tuple[float, str], models.TrainedModel] = {}

    for dedup, facs in factor_sets.items():
        if not facs:
            print("    [%s] 去冗余 |corr|<%.2f 无可用因子, 跳过" % (side, dedup))
            continue
        for mc in model_grid:
            tm = models.train_side(F.iloc[ts:te], y.iloc[ts:te], facs, side,
                                   params=models.resolve_params(mc), verbose=False)
            trained[(dedup, mc["name"])] = tm
            pred = tm.predict(F)                       # 全序列预测, 仅取 OOF 段评估
            for ec in grid:
                r = evaluate_with_params(pred, ohlc, atr, times, side,
                                         oof_slice.start, oof_slice.stop,
                                         ec["thr_q"], ec["tp_mult"], ec["sl_mult"],
                                         max_hold=ec["max_hold"], regime=regime,
                                         entry_hi=entry_hi)
                m = r["metrics"]
                rows.append(dict(dedup=float(dedup), n_factors=len(facs), model=mc["name"],
                                 thr_q=float(ec["thr_q"]), tp_mult=float(ec["tp_mult"]),
                                 sl_mult=float(ec["sl_mult"]), max_hold=int(ec["max_hold"]),
                                 thr_abs=float(r["thr_abs"]), n=int(m["n_trades"]),
                                 sharpe=float(m["sharpe"]), total_return=float(m["total_return"]),
                                 calmar=float(m["calmar"]),
                                 max_drawdown=float(m["max_drawdown"]),
                                 win_rate=float(m["win_rate"]), payoff=float(m["payoff_ratio"]),
                                 tp_rate=float(m["tp_rate"])))
        if verbose:
            print("    [%s] 去冗余 |corr|<%.2f -> 因子 %2d 个 | %d 个模型 × %d 组执行 = %d 组已评估"
                  % (side, dedup, len(facs), len(model_grid), len(grid),
                     len(model_grid) * len(grid)))

    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError("OOF 网格为空: %s" % side)
    agg = _select_best(raw)
    if ret_drop is not None:
        agg = select_return_anchor_sharpe(agg, float(ret_drop))
    best = agg.iloc[0]
    name = best["model"]
    out = dict(dedup=float(best["dedup"]), model_name=name,
               model=trained[(float(best["dedup"]), name)],
               thr_q=float(best["thr_q"]), thr_abs=float(best["thr_abs"]),
               tp_mult=float(best["tp_mult"]), sl_mult=float(best["sl_mult"]),
               max_hold=int(best["max_hold"]),
               oof_metrics={k: float(best[k]) for k in
                            ("n", "sharpe", "total_return", "calmar", "max_drawdown",
                             "win_rate", "payoff", "tp_rate")})
    return out, agg, raw


# ================================================ v6: 因子池 × 制度 × 去冗余 × 模型 × 执行
def _exec_grid_v6(thr_grid, hold_grid, trail_grid, sl_grid=None,
                  tp_gt_sl: bool = True) -> List[dict]:
    """v6 执行网格: 阈值分位(加密) × 止盈 × 止损 × 最长持有期 × **ATR 跟踪止损**。

    tp_gt_sl=True 时只保留 `tp_mult > sl_mult` 的组合(沿用 v3 的风险报酬比 > 1 约束)。
    sl_grid: 止损候选网格, 默认用 v5 的 SL_ATR_GRID; v7 传 V7_SL_GRID(下探到 0.35)。
    """
    sl_grid = C.SL_ATR_GRID if sl_grid is None else sl_grid
    grid = [dict(thr_q=q, tp_mult=tp, sl_mult=sl, max_hold=h, trail_mult=tr)
            for q, tp, sl, h, tr in product(thr_grid, C.TP_ATR_GRID, sl_grid,
                                            hold_grid, trail_grid)]
    if tp_gt_sl:
        grid = [g for g in grid if g["tp_mult"] > g["sl_mult"]]
    return grid


def optimize_v6_on_oof(F: pd.DataFrame, y: pd.Series,
                       factor_sets: Dict[Tuple[str, float], List[str]],
                       side: str, ohlc: dict, atr: np.ndarray, times: np.ndarray,
                       train_slice: slice, oof_slice: slice,
                       regimes: Dict[str, np.ndarray],
                       thr_grid=None, hold_grid=None, trail_grid=None, sl_grid=None,
                       tp_gt_sl: bool = True,
                       verbose: bool = True) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """v6 主优化: 在 **OOF** 上联合择优 (因子池 × 制度规则 × 去冗余阈值 × 模型 × 执行参数)。

    相对 `optimize_full_on_oof` 的三点改进:
      1. **模型只训练一次**: 制度门控只作用于信号掩码, 不改变模型预测, 故每个
         (因子池, 去冗余, 模型) 只训练一次, 随后在所有制度规则上复用同一预测 —— 省去重复训练;
      2. 执行网格**加入 ATR 跟踪止损**(trail_mult), 并支持自定义阈值/持有/跟踪网格;
      3. 外层新增**因子池**维度(core / expanded), 让"该用原有因子还是扩充因子"由 OOF 决定。

    训练只用 train、选择只用 OOF; OOC 全程不参与。
    factor_sets: {(因子池, 去冗余阈值): 该方向因子列表}(由 factor_select 在 train 上产出)
    regimes:     {制度规则名: 该方向的全序列 bool 掩码}
    返回 (best, agg, raw); best 含 pool / regime_rule / trail_mult。
    """
    ts, te = train_slice.start, train_slice.stop
    thr_grid = C.V6_THR_GRID if thr_grid is None else thr_grid
    hold_grid = C.V6_HOLD_GRID if hold_grid is None else hold_grid
    trail_grid = C.V6_TRAIL_GRID if trail_grid is None else trail_grid
    grid = _exec_grid_v6(thr_grid, hold_grid, trail_grid, sl_grid=sl_grid, tp_gt_sl=tp_gt_sl)
    rows: List[dict] = []
    trained: Dict[Tuple[Tuple[str, float], str], models.TrainedModel] = {}

    for (pool, dedup), facs in factor_sets.items():
        if not facs:
            print("    [%s] 池 %s |corr|<%.2f 无可用因子, 跳过" % (side, pool, dedup))
            continue
        for mc in models.MODEL_GRID:
            tm = models.train_side(F.iloc[ts:te], y.iloc[ts:te], facs, side,
                                   params=models.resolve_params(mc), verbose=False)
            trained[((pool, dedup), mc["name"])] = tm
            pred = tm.predict(F)                       # 全序列预测, 仅取 OOF 段评估
            for rule, mask in regimes.items():
                for ec in grid:
                    r = evaluate_with_params(pred, ohlc, atr, times, side,
                                             oof_slice.start, oof_slice.stop,
                                             ec["thr_q"], ec["tp_mult"], ec["sl_mult"],
                                             max_hold=ec["max_hold"],
                                             trail_mult=ec["trail_mult"], regime=mask)
                    m = r["metrics"]
                    rows.append(dict(pool=pool, regime=rule, dedup=float(dedup),
                                     n_factors=len(facs), model=mc["name"],
                                     thr_q=float(ec["thr_q"]),
                                     tp_mult=float(ec["tp_mult"]), sl_mult=float(ec["sl_mult"]),
                                     max_hold=int(ec["max_hold"]),
                                     trail_mult=float(ec["trail_mult"]),
                                     thr_abs=float(r["thr_abs"]), n=int(m["n_trades"]),
                                     sharpe=float(m["sharpe"]),
                                     total_return=float(m["total_return"]),
                                     calmar=float(m["calmar"]),
                                     max_drawdown=float(m["max_drawdown"]),
                                     win_rate=float(m["win_rate"]),
                                     payoff=float(m["payoff_ratio"]),
                                     tp_rate=float(m["tp_rate"])))
        if verbose:
            print("    [%s] 池 %-8s 去冗余 |corr|<%.2f -> 因子 %2d 个 | %d 模型 × %d 制度 × %d 执行"
                  % (side, pool, dedup, len(facs), len(models.MODEL_GRID), len(regimes), len(grid)))

    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError("OOF 网格为空: %s" % side)
    agg = _select_best(raw)
    best = agg.iloc[0]
    name = best["model"]
    tkey = (str(best["pool"]), float(best["dedup"]))
    out = dict(pool=str(best["pool"]), regime_rule=str(best["regime"]),
               dedup=float(best["dedup"]), model_name=name,
               model=trained[(tkey, name)],
               thr_q=float(best["thr_q"]), thr_abs=float(best["thr_abs"]),
               tp_mult=float(best["tp_mult"]), sl_mult=float(best["sl_mult"]),
               max_hold=int(best["max_hold"]), trail_mult=float(best["trail_mult"]),
               oof_metrics={k: float(best[k]) for k in
                            ("n", "sharpe", "total_return", "calmar", "max_drawdown",
                             "win_rate", "payoff", "tp_rate")})
    return out, agg, raw


# ================================================ v8: OOF 内部分折 + 稳健选优
def _v8_fold_slices(oof_slice: slice, n_folds: int) -> List[Tuple[int, int]]:
    """把 OOF 段切成 n_folds 个**连续**子折(时间序, 互不重叠), 返回 [(lo, hi), ...]。

    切分只发生在 OOF 内部: OOF 仍是从 train 之后、OOC 之前的那一段, 不触碰 OOC。
    """
    lo, hi = int(oof_slice.start), int(oof_slice.stop)
    edges = np.linspace(lo, hi, int(n_folds) + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(int(n_folds))
            if int(edges[i + 1]) - int(edges[i]) > 0]


# 配置身份(= 一组超参)的全部维度; 聚合/报告/复核都按它分组。
_V8_CONFIG_KEYS = ("pool", "regime", "dedup", "n_factors", "model",
                   "thr_q", "tp_mult", "sl_mult", "max_hold", "trail_mult")


def aggregate_v8(raw: pd.DataFrame) -> pd.DataFrame:
    """把"配置 × 子折"长表聚合为**稳健得分表**(已按稳健得分降序)。

    稳健得分 = 各折夏普均值 - V8_ROBUST_K × 各折夏普标准差;
    排序: 稳健得分 -> 最差折夏普(maximin) -> 平均收益。
    """
    g = (raw.groupby(list(_V8_CONFIG_KEYS), as_index=False)
            .agg(sharpe_mean=("sharpe", "mean"), sharpe_std=("sharpe", "std"),
                 sharpe_min=("sharpe", "min"), sharpe_max=("sharpe", "max"),
                 ret_mean=("total_return", "mean"), ret_min=("total_return", "min"),
                 n_total=("n", "sum"), n_min=("n", "min"), n_max=("n", "max"),
                 win_rate=("win_rate", "mean"), payoff=("payoff", "mean"),
                 tp_rate=("tp_rate", "mean"), calmar=("calmar", "mean"),
                 max_drawdown=("max_drawdown", "min"),
                 n_pos_folds=("total_return", lambda x: int((x > 0).sum())),
                 n_folds=("sharpe", "size")))
    g["sharpe_std"] = g["sharpe_std"].fillna(0.0)
    g["robust"] = g["sharpe_mean"] - C.V8_ROBUST_K * g["sharpe_std"]
    return g.sort_values(["robust", "sharpe_min", "ret_mean"],
                         ascending=False).reset_index(drop=True)


def optimize_v8_on_oof(F: pd.DataFrame, y: pd.Series,
                       factor_sets: Dict[Tuple[str, float], List[str]],
                       side: str, ohlc: dict, atr: np.ndarray, times: np.ndarray,
                       train_slice: slice, oof_slice: slice,
                       regimes: Dict[str, np.ndarray],
                       thr_grid=None, hold_grid=None, trail_grid=None, sl_grid=None,
                       tp_gt_sl: bool = True, n_folds=None,
                       verbose: bool = True,
                       out_models: dict = None) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """v8 主优化: 网格/训练与 `optimize_v6_on_oof` 完全相同, **只改选优纪律**。

    相对 v6/v7 的"整段 OOF 取最大 total_return":
      1. 把 OOF 切成 n_folds 个**连续子折**, 每组配置在**每个子折**上独立评估
         (阈值分位按子折自身预测计算; 子折仍在 OOF 内, 不触碰 OOC);
      2. 过滤: OOF 总笔数 >= OOF_MIN_TRADES, **单折笔数 >= V8_MIN_TRADES_PER_FOLD**,
         平均止盈率 >= MIN_TP_RATE;
      3. 稳健得分 = 各折夏普均值 - V8_ROBUST_K × 标准差, 次目标为最差折夏普。
    选定后**再在整段 OOF 上复评一次**, 取得与 v6/v7 同口径的冻结阈值(thr_abs)与指标。

    选择仍**只发生在 OOF**; OOC 全程不参与。
    返回 (best, agg, raw): raw 为"配置 × 子折"长表; agg 为稳健得分表(已排序)。
    """
    ts, te = train_slice.start, train_slice.stop
    thr_grid = C.V6_THR_GRID if thr_grid is None else thr_grid
    hold_grid = C.V6_HOLD_GRID if hold_grid is None else hold_grid
    trail_grid = C.V6_TRAIL_GRID if trail_grid is None else trail_grid
    folds = _v8_fold_slices(oof_slice, C.V8_N_FOLDS if n_folds is None else n_folds)
    grid = _exec_grid_v6(thr_grid, hold_grid, trail_grid, sl_grid=sl_grid, tp_gt_sl=tp_gt_sl)
    rows: List[dict] = []
    trained: Dict[Tuple[Tuple[str, float], str], models.TrainedModel] = {}

    for (pool, dedup), facs in factor_sets.items():
        if not facs:
            print("    [%s] 池 %s |corr|<%.2f 无可用因子, 跳过" % (side, pool, dedup))
            continue
        for mc in models.MODEL_GRID:
            tm = models.train_side(F.iloc[ts:te], y.iloc[ts:te], facs, side,
                                   params=models.resolve_params(mc), verbose=False)
            trained[((pool, dedup), mc["name"])] = tm
            if out_models is not None:          # v10: 供"重建候选逐折权益"复用, 免去重复训练
                out_models[(pool, float(dedup), mc["name"])] = (tm, tm.predict(F))
            pred = tm.predict(F)                       # 全序列预测, 只在 OOF 子折上评估
            for rule, mask in regimes.items():
                for ec in grid:
                    for fi, (flo, fhi) in enumerate(folds, 1):
                        r = evaluate_with_params(pred, ohlc, atr, times, side, flo, fhi,
                                                 ec["thr_q"], ec["tp_mult"], ec["sl_mult"],
                                                 max_hold=ec["max_hold"],
                                                 trail_mult=ec["trail_mult"], regime=mask)
                        m = r["metrics"]
                        rows.append(dict(pool=pool, regime=rule, dedup=float(dedup),
                                         n_factors=len(facs), model=mc["name"],
                                         fold=fi, fold_lo=flo, fold_hi=fhi,
                                         thr_q=float(ec["thr_q"]),
                                         tp_mult=float(ec["tp_mult"]), sl_mult=float(ec["sl_mult"]),
                                         max_hold=int(ec["max_hold"]),
                                         trail_mult=float(ec["trail_mult"]),
                                         n=int(m["n_trades"]), sharpe=float(m["sharpe"]),
                                         total_return=float(m["total_return"]),
                                         calmar=float(m["calmar"]),
                                         max_drawdown=float(m["max_drawdown"]),
                                         win_rate=float(m["win_rate"]),
                                         payoff=float(m["payoff_ratio"]),
                                         tp_rate=float(m["tp_rate"])))
        if verbose:
            print("    [%s] 池 %-8s 去冗余 |corr|<%.2f -> 因子 %2d 个 | %d 模型 × %d 制度 × %d 执行 × %d 子折"
                  % (side, pool, dedup, len(facs), len(models.MODEL_GRID), len(regimes),
                     len(grid), len(folds)))

    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError("OOF 网格为空: %s" % side)
    agg = aggregate_v8(raw)
    ok = ((agg["n_total"] >= C.OOF_MIN_TRADES)
          & (agg["n_min"] >= C.V8_MIN_TRADES_PER_FOLD)
          & (agg["tp_rate"] >= C.MIN_TP_RATE))
    usable = agg[ok]
    if usable.empty:                       # 门槛过严时逐级退回, 保证总有解
        usable = agg[agg["n_total"] >= C.OOF_MIN_TRADES]
    if usable.empty:
        usable = agg
    best = usable.iloc[0]

    name = str(best["model"])
    tm = trained[((str(best["pool"]), float(best["dedup"])), name)]
    pred = tm.predict(F)
    full = evaluate_with_params(pred, ohlc, atr, times, side, oof_slice.start, oof_slice.stop,
                                float(best["thr_q"]), float(best["tp_mult"]),
                                float(best["sl_mult"]), max_hold=int(best["max_hold"]),
                                trail_mult=float(best["trail_mult"]),
                                regime=regimes[str(best["regime"])])
    m = full["metrics"]
    out = dict(pool=str(best["pool"]), regime_rule=str(best["regime"]),
               dedup=float(best["dedup"]), model_name=name, model=tm,
               thr_q=float(best["thr_q"]), thr_abs=float(full["thr_abs"]),
               tp_mult=float(best["tp_mult"]), sl_mult=float(best["sl_mult"]),
               max_hold=int(best["max_hold"]), trail_mult=float(best["trail_mult"]),
               robust=float(best["robust"]), sharpe_mean=float(best["sharpe_mean"]),
               sharpe_std=float(best["sharpe_std"]), sharpe_min=float(best["sharpe_min"]),
               ret_mean=float(best["ret_mean"]), n_pos_folds=int(best["n_pos_folds"]),
               n_folds=int(best["n_folds"]),
               oof_metrics={"n": float(m["n_trades"]), "sharpe": float(m["sharpe"]),
                            "total_return": float(m["total_return"]),
                            "calmar": float(m["calmar"]),
                            "max_drawdown": float(m["max_drawdown"]),
                            "win_rate": float(m["win_rate"]),
                            "payoff": float(m["payoff_ratio"]),
                            "tp_rate": float(m["tp_rate"])})
    return out, usable.reset_index(drop=True), raw


# ================================================ v9: long+short 共同优化(合并总收益)
def candidate_pool_v9(raw: pd.DataFrame, k_cfg: int, robust_k: float = None) -> pd.DataFrame:
    """从单侧"配置×子折"长表里, 按**该侧稳健收益**(均值 - k×标准差)取前 k_cfg 个候选配置。

    门槛与 v8 一致(OOF_MIN_TRADES / V8_MIN_TRADES_PER_FOLD / MIN_TP_RATE),
    确保候选是"每个子折都真的在交易"的配置, 而不是只在个别子折蒙对的低频幻觉。
    """
    robust_k = C.V8_ROBUST_K if robust_k is None else robust_k
    keys = list(_V8_CONFIG_KEYS)
    g = (raw.groupby(keys, as_index=False)
            .agg(ret_mean=("total_return", "mean"), ret_std=("total_return", "std"),
                 ret_min=("total_return", "min"), n_total=("n", "sum"),
                 n_min=("n", "min"), tp_rate=("tp_rate", "mean")))
    g["ret_std"] = g["ret_std"].fillna(0.0)
    g["robust_ret"] = g["ret_mean"] - robust_k * g["ret_std"]
    ok = ((g["n_total"] >= C.OOF_MIN_TRADES)
          & (g["n_min"] >= C.V8_MIN_TRADES_PER_FOLD)
          & (g["tp_rate"] >= C.MIN_TP_RATE))
    u = g[ok]
    if u.empty:                                 # 门槛过严时退回全量, 保证总有候选
        u = g
    u = u.sort_values(["robust_ret", "ret_min", "ret_mean"], ascending=False)
    return u.head(int(k_cfg)).reset_index(drop=True)


def _fold_return_matrix(raw: pd.DataFrame, pool: pd.DataFrame) -> np.ndarray:
    """把候选池 × 子折的 total_return 整理成 (n_pool, n_folds) 矩阵(pool 顺序不变)。"""
    keys = list(_V8_CONFIG_KEYS)
    n_folds = int(raw["fold"].nunique())
    sub = pool[keys].merge(raw[keys + ["fold", "total_return"]], on=keys, how="left")
    sub = sub.sort_values(keys + ["fold"], kind="mergesort").reset_index(drop=True)
    if len(sub) != len(pool) * n_folds:
        raise RuntimeError("子折收益矩阵不完整: %d != %d x %d"
                           % (len(sub), len(pool), n_folds))
    return sub["total_return"].to_numpy(dtype=float).reshape(len(pool), n_folds)


def joint_select_v9(long_raw: pd.DataFrame, short_raw: pd.DataFrame, k_cfg: int,
                    robust_k: float = None, top_report: int = 30):
    """v9 联合选优: 两侧各取候选池, 再在**合并逐折收益**上联合排序。

    合并口径与 v7 一致: `equity_combined = eq_long + eq_short - INIT_CAPITAL`,
    故**合并收益 = 两侧收益之和**(逐折相加)。
    主目标 = mean(rc) - k×std(rc), 次目标 = 最差折 rc(maximin), 三目标 = mean(rc)。

    返回 (best, pairs, pool_long, pool_short):
      best  = dict(long=..., short=..., robust_combined, mean_combined, worst_combined,
                   std_combined, fold_returns=list)
      pairs = 前 top_report 个组合的明细表(含两侧配置与合并指标)
    """
    robust_k = C.V8_ROBUST_K if robust_k is None else robust_k
    pl = candidate_pool_v9(long_raw, k_cfg, robust_k)
    ps = candidate_pool_v9(short_raw, k_cfg, robust_k)
    RL = _fold_return_matrix(long_raw, pl)          # (nL, F)
    RS = _fold_return_matrix(short_raw, ps)         # (nS, F)

    rc = RL[:, None, :] + RS[None, :, :]            # (nL, nS, F) 合并逐折收益
    mean = np.nanmean(rc, axis=2)
    std = np.nanstd(rc, axis=2, ddof=1)
    worst = np.nanmin(rc, axis=2)
    robust = mean - robust_k * std

    flat = np.lexsort((-mean.ravel(), -worst.ravel(), -robust.ravel()))
    li, si = np.unravel_index(flat, robust.shape)

    keys = list(_V8_CONFIG_KEYS)
    lcols = {("long_" + k): pl[k].to_numpy()[li] for k in keys}
    scols = {("short_" + k): ps[k].to_numpy()[si] for k in keys}
    pairs = pd.DataFrame({**lcols, **scols,
                          "robust_combined": robust[li, si],
                          "worst_combined": worst[li, si],
                          "mean_combined": mean[li, si],
                          "std_combined": std[li, si]})
    for fi in range(rc.shape[2]):
        pairs["fold%d_ret" % (fi + 1)] = rc[li, si, fi]
    pairs = pairs.head(int(top_report)).reset_index(drop=True)

    i, j = int(li[0]), int(si[0])
    best = dict(long=pl.iloc[i].to_dict(), short=ps.iloc[j].to_dict(),
                robust_combined=float(robust[i, j]), mean_combined=float(mean[i, j]),
                worst_combined=float(worst[i, j]), std_combined=float(std[i, j]),
                fold_returns=[float(v) for v in rc[i, j]])
    return best, pairs, pl, ps


# ================================================ v10: 收益+夏普 双目标的联合选优
def _sharpe_rows(E: np.ndarray) -> np.ndarray:
    """批量算年化夏普, 与 `src.metrics.sharpe` 同口径(bar 收益 = diff/prev, ddof=1, 4h 年化)。

    metrics.sharpe 逐条 Python 调用在上万条曲线时太慢, 这里对 (m, T) 矩阵按行向量化。
    """
    if E.ndim == 1:
        E = E[None, :]
    if E.shape[1] < 3:
        return np.zeros(E.shape[0])
    prev = E[:, :-1]
    r = np.diff(E, axis=1) / np.where(prev == 0, np.nan, prev)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(r, axis=1)
        std = np.nanstd(r, axis=1, ddof=1)
    out = np.where(std > 0, mean / std * np.sqrt(365.0 * 6.0), 0.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _candidate_fold_equity(pool: pd.DataFrame, pred_map: dict, side: str, ohlc: dict,
                           atr: np.ndarray, times: np.ndarray, folds, masks: dict) -> List[np.ndarray]:
    """候选池内每个配置在**每个子折**上的 bar 级权益曲线 -> [ (n_cand, 折长), ... ]。

    评估口径与阶段1完全一致(分位阈值按子折自身预测计算), 因此重建的逐折收益应与原始长表一致。
    """
    out: List[np.ndarray] = []
    for (flo, fhi) in folds:
        rows = []
        for _, c in pool.iterrows():
            pred = pred_map[(c["pool"], float(c["dedup"]), c["model"])][1]
            r = evaluate_with_params(pred, ohlc, atr, times, side, flo, fhi,
                                     float(c["thr_q"]), float(c["tp_mult"]), float(c["sl_mult"]),
                                     max_hold=int(c["max_hold"]),
                                     trail_mult=float(c["trail_mult"]),
                                     regime=masks[c["regime"]][side])
            rows.append(r["equity"])
        out.append(np.vstack(rows))
    return out


def joint_select_v10(long_raw: pd.DataFrame, short_raw: pd.DataFrame,
                     pred_map_long: dict, pred_map_short: dict,
                     ohlc: dict, atr: np.ndarray, times: np.ndarray, folds, masks: dict,
                     k_cfg: int, robust_k: float = None, top_report: int = 30):
    """v10 联合选优: 在 v9 的"合并稳健收益"之上**加入真实 bar 级合并夏普**。

    用户口径: **收益优先, 夏普次之**; 两者都按跨折稳健(均值 - k×std)。
      主目标 = 合并稳健收益 = mean(rc) - k×std(rc),  rc = 逐折合并收益(两侧收益相加)
      次目标 = 合并稳健夏普 = mean(S)  - k×std(S),   S  = 逐折**合并组合**夏普
      三目标 = 最差折合并收益(maximin)
    合并夏普按真实口径重建: eq_c = eq_long + eq_short - INIT_CAPITAL, 再按 bar 收益年化 ——
    **不是**把两侧夏普相加(两侧交易独立, 组合夏普与单侧夏普不可线性合成)。
    选择只用 OOF 子折; OOC 不参与。

    返回 (best, pairs, pool_long, pool_short)。
    """
    robust_k = C.V8_ROBUST_K if robust_k is None else robust_k
    pl = candidate_pool_v9(long_raw, k_cfg, robust_k)
    ps = candidate_pool_v9(short_raw, k_cfg, robust_k)
    RL = _fold_return_matrix(long_raw, pl)          # (nL, F)
    RS = _fold_return_matrix(short_raw, ps)         # (nS, F)
    rc = RL[:, None, :] + RS[None, :, :]            # (nL, nS, F) 合并逐折收益
    ret_mean = np.nanmean(rc, axis=2)
    ret_std = np.nanstd(rc, axis=2, ddof=1)
    worst_ret = np.nanmin(rc, axis=2)
    robust_ret = ret_mean - robust_k * ret_std

    eqL = _candidate_fold_equity(pl, pred_map_long, "long", ohlc, atr, times, folds, masks)
    eqS = _candidate_fold_equity(ps, pred_map_short, "short", ohlc, atr, times, folds, masks)
    nL, nS, F = len(pl), len(ps), len(folds)
    sh = np.zeros((nL, nS, F))
    for f in range(F):
        base = eqS[f]                                     # (nS, T_f)
        for i in range(nL):
            sh[i, :, f] = _sharpe_rows(base + (eqL[f][i] - C.INIT_CAPITAL))
    sh_mean = np.nanmean(sh, axis=2)
    sh_std = np.nanstd(sh, axis=2, ddof=1)
    robust_sharpe = sh_mean - robust_k * sh_std

    order = np.lexsort((-worst_ret.ravel(), -robust_sharpe.ravel(), -robust_ret.ravel()))
    li, si = np.unravel_index(order, robust_ret.shape)

    keys = list(_V8_CONFIG_KEYS)
    pairs = pd.DataFrame({**{("long_" + k): pl[k].to_numpy()[li] for k in keys},
                          **{("short_" + k): ps[k].to_numpy()[si] for k in keys},
                          "robust_combined": robust_ret[li, si],
                          "sharpe_combined": robust_sharpe[li, si],
                          "worst_combined": worst_ret[li, si],
                          "mean_combined": ret_mean[li, si],
                          "std_combined": ret_std[li, si],
                          "sharpe_mean": sh_mean[li, si],
                          "sharpe_std": sh_std[li, si]})
    for fi in range(F):
        pairs["fold%d_ret" % (fi + 1)] = rc[li, si, fi]
        pairs["fold%d_sharpe" % (fi + 1)] = sh[li, si, fi]
    pairs = pairs.head(int(top_report)).reset_index(drop=True)

    i, j = int(li[0]), int(si[0])
    best = dict(long=pl.iloc[i].to_dict(), short=ps.iloc[j].to_dict(),
                robust_combined=float(robust_ret[i, j]), sharpe_combined=float(robust_sharpe[i, j]),
                mean_combined=float(ret_mean[i, j]), std_combined=float(ret_std[i, j]),
                worst_combined=float(worst_ret[i, j]), sharpe_mean=float(sh_mean[i, j]),
                sharpe_std=float(sh_std[i, j]),
                fold_returns=[float(v) for v in rc[i, j]],
                fold_sharpes=[float(v) for v in sh[i, j]])
    return best, pairs, pl, ps


# ================================================ v11: 强化选择器(双目标候选池 + Pareto 前沿)
def candidate_pool_v11(raw: pd.DataFrame, k_cfg: int, robust_k: float = None) -> pd.DataFrame:
    """v11 候选池: **双目标(收益/夏普)保留 + Pareto 前沿**, 而不是只按单一目标截断。

    v10 的 `candidate_pool_v9` 只按该侧"稳健收益"排序取前 k 个, 存在"夏普高、收益略低"的配置
    在**进入联合配对之前**就被丢掉的风险(实测在 v9 的 3 折长表上, 前 200 内已含该侧最高稳健夏普,
    故这是**风险**而非既成事实)。v11 改为:
      1. 分别在两个目标上排名(稳健收益 / 稳健夏普), 取 **min(两个排名)** 最优的 k_cfg 个 ——
         两个目标都不吃亏的配置优先入池;
      2. 强制并入 (稳健收益, 稳健夏普) 的 **Pareto 前沿**(非支配解): 按收益降序扫描、
         记录夏普的历史最高, 可保证"用另一个目标换收益/夏普"的解不会被单一排序误杀。

    门槛与 v8/v9/v10 完全一致(OOF_MIN_TRADES / V8_MIN_TRADES_PER_FOLD / MIN_TP_RATE)。
    返回列与 `candidate_pool_v9` 同构(另加 robust_sharpe / robust_rank / on_pareto), 下游无需改。
    """
    robust_k = C.V8_ROBUST_K if robust_k is None else robust_k
    keys = list(_V8_CONFIG_KEYS)
    g = (raw.groupby(keys, as_index=False)
            .agg(ret_mean=("total_return", "mean"), ret_std=("total_return", "std"),
                 ret_min=("total_return", "min"),
                 sharpe_mean=("sharpe", "mean"), sharpe_std=("sharpe", "std"),
                 n_total=("n", "sum"), n_min=("n", "min"), tp_rate=("tp_rate", "mean")))
    g["ret_std"] = g["ret_std"].fillna(0.0)
    g["sharpe_std"] = g["sharpe_std"].fillna(0.0)
    g["robust_ret"] = g["ret_mean"] - robust_k * g["ret_std"]
    g["robust_sharpe"] = g["sharpe_mean"] - robust_k * g["sharpe_std"]
    ok = ((g["n_total"] >= C.OOF_MIN_TRADES)
          & (g["n_min"] >= C.V8_MIN_TRADES_PER_FOLD)
          & (g["tp_rate"] >= C.MIN_TP_RATE))
    u = g[ok]
    if u.empty:                                  # 门槛过严时退回全量, 保证总有候选
        u = g
    u = u.reset_index(drop=True)

    # 1) 双目标排名: 取 min(两排名) 作入池优先级
    r_ret = u["robust_ret"].rank(ascending=False, method="first").to_numpy(dtype=float)
    r_shp = u["robust_sharpe"].rank(ascending=False, method="first").to_numpy(dtype=float)
    prio = np.minimum(r_ret, r_shp)

    # 2) Pareto 前沿(最大化 稳健收益 与 稳健夏普): 收益降序扫描, 夏普刷新历史最高即入前沿
    order = np.argsort(-u["robust_ret"].to_numpy(dtype=float), kind="mergesort")
    on_front = np.zeros(len(u), dtype=bool)
    best_shp = -np.inf
    for pos in order:
        v = float(u["robust_sharpe"].iloc[pos])
        if v > best_shp:
            on_front[pos] = True
            best_shp = v
    prio = np.where(on_front, 0.0, prio)         # 前沿解无条件优先入池

    u = u.assign(robust_rank=prio, on_pareto=on_front)
    sel = u.sort_values(["robust_rank", "robust_ret", "robust_sharpe"],
                        ascending=[True, False, False]).head(int(k_cfg))
    return sel.reset_index(drop=True)


def joint_select_v11(long_raw: pd.DataFrame, short_raw: pd.DataFrame,
                     pred_map_long: dict, pred_map_short: dict,
                     ohlc: dict, atr: np.ndarray, times: np.ndarray, folds, masks: dict,
                     k_cfg: int, robust_k: float = None, top_report: int = 30):
    """v11 联合选优: 排序目标与 `joint_select_v10` **逐位相同**,
    只把候选池换成 `candidate_pool_v11`(双目标保留 + Pareto 前沿)。

      主目标 = 合并稳健收益 = mean(rc) - k×std(rc),  rc = 逐折合并收益(两侧收益相加)
      次目标 = 合并稳健夏普 = mean(S)  - k×std(S),   S  = 逐折**合并组合**夏普(真实 bar 级)
      三目标 = 最差折合并收益(maximin)

    选择只用 OOF 子折; OOC 不参与。返回 (best, pairs, pool_long, pool_short)。
    """
    robust_k = C.V8_ROBUST_K if robust_k is None else robust_k
    pl = candidate_pool_v11(long_raw, k_cfg, robust_k)
    ps = candidate_pool_v11(short_raw, k_cfg, robust_k)
    RL = _fold_return_matrix(long_raw, pl)          # (nL, F)
    RS = _fold_return_matrix(short_raw, ps)         # (nS, F)
    rc = RL[:, None, :] + RS[None, :, :]            # (nL, nS, F) 合并逐折收益
    ret_mean = np.nanmean(rc, axis=2)
    ret_std = np.nanstd(rc, axis=2, ddof=1)
    worst_ret = np.nanmin(rc, axis=2)
    robust_ret = ret_mean - robust_k * ret_std

    eqL = _candidate_fold_equity(pl, pred_map_long, "long", ohlc, atr, times, folds, masks)
    eqS = _candidate_fold_equity(ps, pred_map_short, "short", ohlc, atr, times, folds, masks)
    nL, nS, F = len(pl), len(ps), len(folds)
    sh = np.zeros((nL, nS, F))
    for f in range(F):
        base = eqS[f]                                     # (nS, T_f)
        for i in range(nL):
            sh[i, :, f] = _sharpe_rows(base + (eqL[f][i] - C.INIT_CAPITAL))
    sh_mean = np.nanmean(sh, axis=2)
    sh_std = np.nanstd(sh, axis=2, ddof=1)
    robust_sharpe = sh_mean - robust_k * sh_std

    order = np.lexsort((-worst_ret.ravel(), -robust_sharpe.ravel(), -robust_ret.ravel()))
    li, si = np.unravel_index(order, robust_ret.shape)

    keys = list(_V8_CONFIG_KEYS)
    pairs = pd.DataFrame({**{("long_" + k): pl[k].to_numpy()[li] for k in keys},
                          **{("short_" + k): ps[k].to_numpy()[si] for k in keys},
                          "long_on_pareto": pl["on_pareto"].to_numpy()[li],
                          "short_on_pareto": ps["on_pareto"].to_numpy()[si],
                          "robust_combined": robust_ret[li, si],
                          "sharpe_combined": robust_sharpe[li, si],
                          "worst_combined": worst_ret[li, si],
                          "mean_combined": ret_mean[li, si],
                          "std_combined": ret_std[li, si],
                          "sharpe_mean": sh_mean[li, si],
                          "sharpe_std": sh_std[li, si]})
    for fi in range(F):
        pairs["fold%d_ret" % (fi + 1)] = rc[li, si, fi]
        pairs["fold%d_sharpe" % (fi + 1)] = sh[li, si, fi]
    pairs = pairs.head(int(top_report)).reset_index(drop=True)

    i, j = int(li[0]), int(si[0])
    best = dict(long=pl.iloc[i].to_dict(), short=ps.iloc[j].to_dict(),
                robust_combined=float(robust_ret[i, j]), sharpe_combined=float(robust_sharpe[i, j]),
                mean_combined=float(ret_mean[i, j]), std_combined=float(ret_std[i, j]),
                worst_combined=float(worst_ret[i, j]), sharpe_mean=float(sh_mean[i, j]),
                sharpe_std=float(sh_std[i, j]),
                fold_returns=[float(v) for v in rc[i, j]],
                fold_sharpes=[float(v) for v in sh[i, j]])
    return best, pairs, pl, ps


# ================================================ v12: 直接以"真实 OOF 段收益/夏普"为排序目标
def _candidate_full_equity(pool: pd.DataFrame, pred_map: dict, side: str, ohlc: dict,
                           atr: np.ndarray, times: np.ndarray, oof_slice: slice,
                           masks: dict) -> np.ndarray:
    """候选池内每个配置在**整段 OOF**(非子折)上的 bar 级权益曲线 -> (n_cand, T_oof)。

    阈值按**整段 OOF 的预测分位**计算 —— 与冻结评估(`threshold_from_quantile(..., oof.start,
    oof.stop)` + `evaluate_with_threshold`)逐位同口径, 因此这里重建的"真实 OOF 收益/夏普"
    就是最终报告的 OOF 段指标本身。
    """
    rows = []
    for _, c in pool.iterrows():
        pred = pred_map[(c["pool"], float(c["dedup"]), c["model"])][1]
        r = evaluate_with_params(pred, ohlc, atr, times, side, oof_slice.start, oof_slice.stop,
                                 float(c["thr_q"]), float(c["tp_mult"]), float(c["sl_mult"]),
                                 max_hold=int(c["max_hold"]),
                                 trail_mult=float(c["trail_mult"]),
                                 regime=masks[c["regime"]][side])
        rows.append(r["equity"])
    return np.vstack(rows)


def joint_select_v12(long_raw: pd.DataFrame, short_raw: pd.DataFrame,
                     pred_map_long: dict, pred_map_short: dict,
                     ohlc: dict, atr: np.ndarray, times: np.ndarray, oof_slice: slice,
                     folds, masks: dict, k_cfg: int, robust_k: float = None, top_report: int = 30):
    """v12 联合选优: **直接以真实 OOF 段的合并收益与夏普为排序目标**(字典序)。

    动机(v11 实测): 在 OOF 子折上做"均值 - K×标准差"的稳健排序, 与**真实 OOF 段收益脱钩** ——
    v11(3 折)折稳健收益略高于 v10, 真实 OOF 收益却从 8.90% 掉到 4.79%。用户因此选择直接优化
    最终要看的 OOF 段数字。仍在 OOF 内选择; OOC 全程不参与。

      主目标 = 真实 OOF 段**合并收益**  = (eq_long[-1] + eq_short[-1] - 2×INIT) / INIT
                                        (合并口径: equity = eq_long + eq_short - INIT, 收益可加)
      次目标 = 真实 OOF 段**合并夏普**  = _sharpe_rows(eq_long + eq_short - INIT)
      三目标 = 最差折合并收益(maximin, 仍取 OOF 子折, 作为"跨折不塌"兜底)

    候选池沿用 `candidate_pool_v11`(双目标保留 + Pareto 前沿, 每侧 k_cfg 个)。
    返回 (best, pairs, pool_long, pool_short)。键名与 v10/v11 保持同构, 便于下游复用, 但语义更新:
      robust_combined => 真实 OOF 段合并收益(主); sharpe_combined => 真实 OOF 段合并夏普(次);
      mean_combined/std_combined => 逐折合并收益的均值/标准差(仅参考)。
    """
    robust_k = C.V8_ROBUST_K if robust_k is None else robust_k
    pl = candidate_pool_v11(long_raw, k_cfg, robust_k)
    ps = candidate_pool_v11(short_raw, k_cfg, robust_k)

    # 真实 OOF 段逐 bar 权益(整段, 非子折)
    EL = _candidate_full_equity(pl, pred_map_long, "long", ohlc, atr, times, oof_slice, masks)
    ES = _candidate_full_equity(ps, pred_map_short, "short", ohlc, atr, times, oof_slice, masks)
    nL, nS = len(pl), len(ps)
    ret_L = (EL[:, -1] - C.INIT_CAPITAL) / C.INIT_CAPITAL          # (nL,)
    ret_S = (ES[:, -1] - C.INIT_CAPITAL) / C.INIT_CAPITAL          # (nS,)
    ret_real = ret_L[:, None] + ret_S[None, :]                     # 收益可加 -> 直接广播

    base_S = ES - C.INIT_CAPITAL                                   # (nS, T)
    sh_real = np.zeros((nL, nS))
    for i in range(nL):
        sh_real[i, :] = _sharpe_rows(EL[i] + base_S)               # 合并权益 = eqL + eqS - INIT

    # 三目标: 最差折合并收益(OOF 子折内)
    RL = _fold_return_matrix(long_raw, pl)                         # (nL, F)
    RS = _fold_return_matrix(short_raw, ps)                        # (nS, F)
    rc_fold = RL[:, None, :] + RS[None, :, :]
    fold_mean = np.nanmean(rc_fold, axis=2)
    fold_std = np.nanstd(rc_fold, axis=2, ddof=1)
    worst_fold = np.nanmin(rc_fold, axis=2)

    order = np.lexsort((-worst_fold.ravel(), -sh_real.ravel(), -ret_real.ravel()))
    li, si = np.unravel_index(order, ret_real.shape)

    keys = list(_V8_CONFIG_KEYS)
    pairs = pd.DataFrame({**{("long_" + k): pl[k].to_numpy()[li] for k in keys},
                          **{("short_" + k): ps[k].to_numpy()[si] for k in keys},
                          "long_on_pareto": pl["on_pareto"].to_numpy()[li],
                          "short_on_pareto": ps["on_pareto"].to_numpy()[si],
                          "robust_combined": ret_real[li, si],
                          "sharpe_combined": sh_real[li, si],
                          "worst_combined": worst_fold[li, si],
                          "mean_combined": fold_mean[li, si],
                          "std_combined": fold_std[li, si]})
    for fi in range(rc_fold.shape[2]):
        pairs["fold%d_ret" % (fi + 1)] = rc_fold[li, si, fi]
    pairs = pairs.head(int(top_report)).reset_index(drop=True)

    i, j = int(li[0]), int(si[0])
    best = dict(long=pl.iloc[i].to_dict(), short=ps.iloc[j].to_dict(),
                robust_combined=float(ret_real[i, j]), sharpe_combined=float(sh_real[i, j]),
                mean_combined=float(fold_mean[i, j]), std_combined=float(fold_std[i, j]),
                worst_combined=float(worst_fold[i, j]),
                ret_real=float(ret_real[i, j]), sharpe_real=float(sh_real[i, j]),
                fold_returns=[float(v) for v in rc_fold[i, j]])
    return best, pairs, pl, ps
