# -*- coding: utf-8 -*-
"""模型层 —— LightGBM 回归, **多头/空头分开训练**。

多头模型: 使用「正向 IC」因子集, 预测未来收益; 预测值 > 阈值 时开多。
空头模型: 使用「负向 IC」因子集, 预测未来收益; 预测值 < -阈值 时开空。
两者各用 Purged K-Fold + Embargo 训练, 每折用早停确定轮数, 最终**集成为多折平均**,
避免单次随机划分带来的不稳定。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

import config as C
from src.cv import PurgedKFold


@dataclass
class TrainedModel:
    side: str
    factors: List[str]
    models: List[lgb.Booster] = field(default_factory=list)
    best_iters: List[int] = field(default_factory=list)
    cv_rmse: List[float] = field(default_factory=list)

    def predict(self, F: pd.DataFrame) -> np.ndarray:
        X = F[self.factors].to_numpy(dtype=float)
        preds = [m.predict(X, num_iteration=it if it and it > 0 else None)
                 for m, it in zip(self.models, self.best_iters)]
        return np.mean(preds, axis=0) if preds else np.full(len(F), np.nan)


def valid_indices(F: pd.DataFrame, y: pd.Series, factors: List[str]) -> np.ndarray:
    """因子与标签都非缺失的样本索引(全局位置)。"""
    ok = y.notna() & F[factors].notna().all(axis=1)
    return np.flatnonzero(ok.to_numpy())


def train_fold(F: pd.DataFrame, y: pd.Series, factors: List[str],
               tr_idx: np.ndarray, va_idx: np.ndarray,
               params: Optional[dict] = None, rounds: int = None,
               early_stop: int = None) -> lgb.Booster:
    """在指定索引上训练单个 LightGBM; 早停用 va_idx。用于嵌套 CV 的外层折。"""
    params = dict(C.LGBM_PARAMS_COMMON if params is None else params)
    rounds = C.LGBM_ROUNDS if rounds is None else rounds
    early_stop = C.LGBM_EARLY_STOP if early_stop is None else early_stop
    X = F[factors].to_numpy(dtype=float)
    yy = y.to_numpy(dtype=float)
    dtr = lgb.Dataset(X[tr_idx], label=yy[tr_idx])
    dva = lgb.Dataset(X[va_idx], label=yy[va_idx], reference=dtr)
    return lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(early_stop, verbose=False),
                                lgb.log_evaluation(0)])


def predict_booster(booster: lgb.Booster, F: pd.DataFrame, factors: List[str]) -> np.ndarray:
    return booster.predict(F[factors].to_numpy(dtype=float),
                           num_iteration=booster.best_iteration or None)


def train_side(F: pd.DataFrame, y: pd.Series, factors: List[str], side: str,
               params: Optional[dict] = None, rounds: int = None,
               early_stop: int = None, verbose: bool = True) -> TrainedModel:
    """在训练段上用 Purged K-Fold 训练一侧模型(集成)。"""
    params = dict(C.LGBM_PARAMS_COMMON if params is None else params)
    rounds = C.LGBM_ROUNDS if rounds is None else rounds
    early_stop = C.LGBM_EARLY_STOP if early_stop is None else early_stop

    F = F.reset_index(drop=True)
    y = y.reset_index(drop=True)
    valid = y.notna() & F[factors].notna().all(axis=1)
    idx_all = np.flatnonzero(valid.to_numpy())
    X_all = F[factors].to_numpy(dtype=float)
    y_all = y.to_numpy(dtype=float)

    tm = TrainedModel(side=side, factors=factors)
    folds = list(PurgedKFold().split(len(F)))
    # 映射到有效样本索引
    pos = {g: k for k, g in enumerate(idx_all)}
    for fi, (tr_g, va_g) in enumerate(folds, 1):
        tr = np.array([pos[g] for g in tr_g if g in pos], dtype=int)
        va = np.array([pos[g] for g in va_g if g in pos], dtype=int)
        if len(tr) < 200 or len(va) < 50:
            continue
        dtr = lgb.Dataset(X_all[idx_all[tr]], label=y_all[idx_all[tr]])
        dva = lgb.Dataset(X_all[idx_all[va]], label=y_all[idx_all[va]], reference=dtr)
        booster = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(early_stop, verbose=False),
                                       lgb.log_evaluation(0)])
        tm.models.append(booster)
        tm.best_iters.append(int(booster.best_iteration or rounds))
        tm.cv_rmse.append(float(booster.best_score["valid_0"]["rmse"]))
        if verbose:
            print("      [%s] fold %d: n_train=%d n_val=%d best_iter=%d rmse=%.6f"
                  % (side, fi, len(tr), len(va), tm.best_iters[-1], tm.cv_rmse[-1]))
    if verbose and tm.cv_rmse:
        print("      [%s] 集成 %d 折, 平均 RMSE=%.6f" % (side, len(tm.models), np.mean(tm.cv_rmse)))
    return tm


def feature_importance(tm: TrainedModel) -> pd.DataFrame:
    """多折平均增益重要性, 便于审阅空/多头因子差异。"""
    if not tm.models:
        return pd.DataFrame(columns=["factor", "gain"])
    g = np.mean([m.feature_importance(importance_type="gain") for m in tm.models], axis=0)
    return (pd.DataFrame({"factor": tm.factors, "gain": g})
            .sort_values("gain", ascending=False).reset_index(drop=True))


# 模型层候选(在 OOF 上比较后冻结; 数量刻意很少, 抑制多重比较)
MODEL_GRID: List[Dict] = [
    dict(name="base", learning_rate=0.03, num_leaves=31, feature_fraction=0.7,
         min_child_samples=60, lambda_l2=1.0),
    dict(name="shallow", learning_rate=0.03, num_leaves=15, feature_fraction=0.6,
         min_child_samples=100, lambda_l2=5.0),
    dict(name="deep_slow", learning_rate=0.015, num_leaves=63, feature_fraction=0.5,
         min_child_samples=80, lambda_l2=10.0),
]


def resolve_params(cand: Dict) -> Dict:
    p = dict(C.LGBM_PARAMS_COMMON)
    p.update({k: v for k, v in cand.items() if k != "name"})
    return p
