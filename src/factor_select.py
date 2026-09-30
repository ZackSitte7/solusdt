# -*- coding: utf-8 -*-
"""因子筛选 —— IC 过滤 + 相关性阈值贪心去冗余。

流程(全部只使用训练段数据, 杜绝用 OOF/OOC 选因子):
  1. 逐因子算 IC(Spearman 秩相关, 对异常值稳健)
  2. 按 |IC| 阈值过滤
  3. 按 |IC| 降序贪心去冗余: 与任一已保留因子 |corr| >= DEDUP_MAX_CORR 则丢弃
     (信息已被 IC 更强的因子代表) -> 数量由相关性结构决定, 不设固定上限
  4. 多头/空头分别筛选: 多头偏好 IC>0 的因子, 空头偏好 IC<0 的因子
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import config as C


def compute_ic(F: pd.DataFrame, y: pd.Series, method: str = "spearman") -> pd.Series:
    """逐因子 IC。F/y 必须已对齐且仅来自训练段。"""
    ic = {}
    for c in F.columns:
        s = pd.concat([F[c], y], axis=1).dropna()
        ic[c] = s.iloc[:, 0].corr(s.iloc[:, 1], method=method) if len(s) > 30 else np.nan
    return pd.Series(ic).dropna().sort_values(key=lambda s: s.abs(), ascending=False)


def select_by_side(ic: pd.Series, side: str, min_abs: float = None) -> List[str]:
    """按方向筛选: long 取 IC>0, short 取 IC<0。
    注意 short 端用「IC<0」而不是「取反」: 负 IC 因子天然预示下跌。"""
    min_abs = C.IC_MIN_ABS if min_abs is None else min_abs
    if side == "long":
        sel = ic[(ic >= min_abs)]
    elif side == "short":
        sel = ic[(ic <= -min_abs)]
    else:
        sel = ic[ic.abs() >= min_abs]
    return list(sel.index)


def corr_greedy_prune(F: pd.DataFrame, ic: pd.Series, candidates: List[str],
                      max_corr: float = None) -> Tuple[List[str], Dict[str, str]]:
    """相关性阈值贪心去冗余(取代固定簇数的 KMeans)。

    候选按 |IC| 降序扫描; 若某因子与**已保留**的任一因子 |corr| >= max_corr,
    说明它只是更强因子的同义复制, 丢弃; 否则保留。
    因子数量由相关性结构决定, 不再被"簇数"这一魔法数硬性截断。

    返回 (保留因子列表(|IC| 降序), {被丢弃因子: 导致其被丢弃的已保留因子})。
    """
    max_corr = C.DEDUP_MAX_CORR if max_corr is None else max_corr
    cand = [c for c in candidates if c in F.columns]
    if len(cand) <= 1:
        return cand, {}

    order = sorted(cand, key=lambda c: abs(ic.get(c, 0.0)), reverse=True)
    corr = F[cand].corr().abs().fillna(0.0)
    keep: List[str] = []
    dropped_by: Dict[str, str] = {}
    for c in order:
        conflict = next((k for k in keep if corr.loc[c, k] >= max_corr), None)
        if conflict is None:
            keep.append(c)
        else:
            dropped_by[c] = conflict
    return keep, dropped_by


def factor_sets_by_dedup(F_train: pd.DataFrame, y_train: pd.Series,
                         dedup_grid) -> Tuple[pd.Series, Dict[float, Dict[str, List[str]]]]:
    """一次算 IC, 再对**多个去冗余阈值**分别产出多空因子集。

    仅供 OOF 选优时扫描用: 因子与 IC 依旧只来自 train 段, 被扫描的是"去冗余松紧"这一超参,
    其取值由 OOF 表现决定(与模型/执行参数同一套选优纪律)。
    返回 (ic, {dedup: {"long": [...], "short": [...]}})。
    """
    ic = compute_ic(F_train, y_train)
    out: Dict[float, Dict[str, List[str]]] = {}
    for d in dedup_grid:
        out[d] = {}
        for side in ("long", "short"):
            cand = select_by_side(ic, side)
            keep, _ = corr_greedy_prune(F_train, ic, cand, max_corr=d)
            out[d][side] = keep
    return ic, out


def select_factors(F_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, object]:
    """对多头与空头分别产出因子集合与 IC 表。返回结构化结果, 便于落盘审阅。"""
    ic = compute_ic(F_train, y_train)
    out: Dict[str, object] = {"ic": ic, "selected": {}, "dropped_by_ic": {}, "dropped_by_corr": {}}
    for side in ("long", "short"):
        cand = select_by_side(ic, side)
        dropped = [c for c in ic.index if c not in cand]
        keep, dropped_by = corr_greedy_prune(F_train, ic, cand)
        out["selected"][side] = keep
        out["dropped_by_ic"][side] = dropped
        out["dropped_by_corr"][side] = dropped_by
    return out
