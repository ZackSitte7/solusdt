#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""因子逐个累加「解释方差」分析。

口径(已确认):
  - 度量: 对**标签**(未来 3 根 ATR 标准化收益)的累计解释方差 R², 用含截距的 OLS;
  - 样本: 仅**训练段 train**(与因子筛选一致, 不含未来信息);
  - 因子: 已筛选集, **多头/空头分别**;
  - 顺序: 按 |IC| 从大到小逐个加入。
  - 关键: 所有 k 用**同一**样本(所选因子与标签均非缺失), 否则不同 k 的 R² 不可比。

输出: reports/factor_cumvar_{side}.csv 与 reports/factor_cumvar.png

运行: python3 analyze_factors.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib import font_manager as _fm  # noqa: E402
_CJK_FONT = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
if Path(_CJK_FONT).exists():             # 注册中文字体, 供逐字形回退
    _fm.fontManager.addfont(_CJK_FONT)
plt.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]   # 拉丁 + 中文回退
plt.rcParams["axes.unicode_minus"] = False
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
import config as C                                          # noqa: E402
from src import factor_select                               # noqa: E402
from src import factors as F_lib                            # noqa: E402
from src.cv import time_split                               # noqa: E402
from run_backtest_okx import load_clean                     # noqa: E402


def cumulative_r2(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """按列顺序逐个加入因子, 返回累计 R²(含截距 OLS)。R² 随 k 单调不减。"""
    n = len(y)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    out = []
    for k in range(1, X.shape[1] + 1):
        A = np.column_stack([np.ones(n), X[:, :k]])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        ss_res = float(((y - A @ coef) ** 2).sum())
        out.append(1.0 - ss_res / ss_tot)
    return np.asarray(out)


def analyse_side(F: pd.DataFrame, y: pd.Series, ic: pd.Series,
                 factors: list, seg: slice) -> pd.DataFrame:
    """单方向: 固定共同样本, 按 |IC| 降序累加, 返回逐因子明细。"""
    facs = sorted(factors, key=lambda c: abs(ic[c]), reverse=True)
    sub = pd.concat([F.loc[seg, facs], y.loc[seg].rename("__y__")], axis=1).dropna()
    X = sub[facs].to_numpy(dtype=float)
    X = (X - X.mean(0)) / np.where(X.std(0) == 0, 1.0, X.std(0))   # 标准化(不改变 R²)
    yy = sub["__y__"].to_numpy(dtype=float)

    cum = cumulative_r2(X, yy)
    delta = np.diff(np.concatenate([[0.0], cum]))
    table = pd.DataFrame({"order": np.arange(1, len(facs) + 1), "factor": facs,
                          "ic": [ic[c] for c in facs], "abs_ic": [abs(ic[c]) for c in facs],
                          "cum_r2": cum, "delta_r2": delta,
                          "delta_r2_pct_of_total": delta / cum[-1] if cum[-1] > 0 else delta,
                          "n_samples": len(sub)})
    return table, X, yy


def main() -> None:
    print("=" * 96)
    print("因子逐个累加「解释方差」分析 | 数据源: 欧易(OKX) | 样本: 仅训练段 train")
    print("度量: 对标签(%d 根 ATR 标准化收益)的累计 R²(OLS) | 顺序: |IC| 降序" % C.HORIZON)
    print("=" * 96)

    df = F_lib.add_label(load_clean(), C.HORIZON)
    Fmat, _ = F_lib.build_factors(df)
    y = df["label"]
    tr, _, _ = time_split(len(df))
    sel = factor_select.select_factors(Fmat.iloc[tr].reset_index(drop=True),
                                       y.iloc[tr].reset_index(drop=True))
    ic = sel["ic"]

    tables = {}
    for side in ("long", "short"):
        facs = sel["selected"][side]
        if not facs:
            print("\n[%s] 无筛选因子, 跳过" % side)
            continue
        t, Xs, ys = analyse_side(Fmat, y, ic, facs, tr)
        tables[side] = t
        t.to_csv(C.REPORT_DIR / ("factor_cumvar_%s.csv" % side), index=False)

        # 运行时一致性: (1) 累计 R² 单调不减; (2) 末点 R² 与独立实现(sklearn)一致
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        assert np.all(np.diff(t["cum_r2"]) >= -1e-12), "累计 R² 非单调"
        ref = r2_score(ys, LinearRegression().fit(Xs, ys).predict(Xs))
        assert abs(t["cum_r2"].iloc[-1] - ref) < 1e-9, "末点与全因子集 R² 不一致"

        print("\n=== %s: %d 个因子, 共同样本 n=%d, 全因子集 R²=%.4f ===" %
              (side, len(facs), t["n_samples"].iloc[0], t["cum_r2"].iloc[-1]))
        print("  %-4s %-20s %8s %10s %10s %10s" %
              ("序", "因子", "|IC|", "ΔR²", "累计R²", "占总R²%"))
        for _, r in t.iterrows():
            print("  %-4d %-20s %8.4f %+10.5f %10.4f %9.1f%%" %
                  (int(r["order"]), r["factor"], r["abs_ic"], r["delta_r2"],
                   r["cum_r2"], r["delta_r2_pct_of_total"] * 100))

    # ---------------- 出图
    if tables:
        fig, axes = plt.subplots(1, len(tables), figsize=(7 * len(tables), 5.2), squeeze=False)
        for ax, side in zip(axes[0], tables):
            t = tables[side]
            x = np.arange(1, len(t) + 1)
            ax.bar(x, t["delta_r2"], width=.55, color="#9ecae1", label="单因子 ΔR²")
            ax.set_ylabel("ΔR² (左)", color="#3b7bbf")
            ax.tick_params(axis="y", colors="#3b7bbf")
            ax2 = ax.twinx()
            ax2.plot(x, t["cum_r2"], "o-", color="#d62728", lw=1.8, label="累计 R²")
            ax2.set_ylabel("累计解释方差 R² (右)", color="#d62728")
            ax2.tick_params(axis="y", colors="#d62728")
            ax2.set_ylim(0, max(0.001, t["cum_r2"].max() * 1.15))
            ax.set_xticks(x)
            ax.set_xticklabels(t["factor"], rotation=45, ha="right", fontsize=8)
            ax.set_xlabel("按 |IC| 降序逐个加入因子")
            ax.set_title("%s  (n=%d, 全因子 R²=%.4f)" %
                         (side, t["n_samples"].iloc[0], t["cum_r2"].iloc[-1]))
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
            ax.grid(alpha=.25)
        plt.tight_layout()
        plt.savefig(C.REPORT_DIR / "factor_cumvar.png", dpi=130)
        plt.close(fig)
        print("\n曲线已保存: %s" % (C.REPORT_DIR / "factor_cumvar.png"))
        print("明细已保存: %s" % C.REPORT_DIR)


if __name__ == "__main__":
    main()
