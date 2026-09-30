# -*- coding: utf-8 -*-
"""因子筛选 —— IC 过滤 + 相关性聚类去冗余。

流程(全部只使用训练段数据, 杜绝用 OOF/OOC 选因子):
  1. 逐因子算 IC(Spearman 秩相关, 对异常值稳健)
  2. 按 |IC| 阈值过滤
  3. 用 1-|corr| 作距离做 KMeans 聚类, 每簇只保留 |IC| 最高的因子 -> 去掉同义因子
  4. 多头/空头分别筛选: 多头偏好 IC>0 的因子, 空头偏好 IC<0 的因子
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

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


def cluster_prune(F: pd.DataFrame, ic: pd.Series, candidates: List[str],
                  max_clusters: int = None) -> Tuple[List[str], Dict[str, int]]:
    """相关性聚类去冗余: 距离 = 1 - |corr|, 每簇保留 |IC| 最大者。"""
    max_clusters = C.KMEANS_MAX_CLUSTERS if max_clusters is None else max_clusters
    cand = [c for c in candidates if c in F.columns]
    if len(cand) <= 1:
        return cand, {c: 0 for c in cand}

    X = F[cand].copy()
    corr = X.corr().abs().fillna(0.0).to_numpy()
    dist = 1.0 - corr
    np.fill_diagonal(dist, 0.0)

    k = int(min(max_clusters, max(2, len(cand) // 3), len(cand)))
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(dist)

    keep, cluster_of = [], {}
    for ci in range(k):
        members = [cand[i] for i in range(len(cand)) if labels[i] == ci]
        if not members:
            continue
        best = max(members, key=lambda c: abs(ic.get(c, 0.0)))
        keep.append(best)
        for m in members:
            cluster_of[m] = ci
    keep = sorted(keep, key=lambda c: abs(ic.get(c, 0.0)), reverse=True)
    return keep, cluster_of


def select_factors(F_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, object]:
    """对多头与空头分别产出因子集合与 IC 表。返回结构化结果, 便于落盘审阅。"""
    ic = compute_ic(F_train, y_train)
    out: Dict[str, object] = {"ic": ic, "selected": {}, "dropped_by_ic": {}, "clusters": {}}
    for side in ("long", "short"):
        cand = select_by_side(ic, side)
        dropped = [c for c in ic.index if c not in cand]
        keep, clus = cluster_prune(F_train, ic, cand)
        out["selected"][side] = keep
        out["dropped_by_ic"][side] = dropped
        out["clusters"][side] = clus
    return out
