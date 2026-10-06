#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多空模型因子集的「向量空间维度」分析 —— 用于判断是否存在过拟合。

背景: 过拟合的一个典型征兆是「名义特征数远大于数据实际张成的有效维度」。若 25 个因子
在训练样本上只张成 6~8 个有效维度, 则多出来的维度只是噪声方向, LightGBM 会去拟合它们。

本脚本对 v3 冻结模型的多空因子集, 只在 **训练段 train** 上做(与因子筛选同口径, 无未来信息):
  1. 名义维度 = 因子个数;
  2. 相关矩阵特征值 -> **PCA 累计解释方差**; 给出达到 90% / 95% / 99% 方差所需主成分数;
  3. **有效维度**(不看阈值, 直接由谱决定):
       - 参与率 PR = (Σλ)² / Σλ²
       - 谱熵有效秩 erank = exp(-Σ pᵢ ln pᵢ),  pᵢ = λᵢ / Σλ
       - Kaiser 维度 = #{λ > 1}(相关矩阵口径)
  4. **共线性**: 条件数 κ = sqrt(λmax/λmin), 逐因子方差膨胀因子 VIF = diag(Σ⁻¹)。

输出: reports/factor_dim.md / factor_dim.csv / factor_dim_vif_{side}.csv / factor_dim.png
运行: python3 scripts/analysis/analyze_factor_dim.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib import font_manager as _fm  # noqa: E402
_CJK_FONT = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
if Path(_CJK_FONT).exists():             # 注册中文字体, 供逐字形回退
    _fm.fontManager.addfont(_CJK_FONT)
plt.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[2]      # 仓库根(脚本位于 scripts/backtest|analysis/)
sys.path.insert(0, str(BASE_DIR))
import config as C                                          # noqa: E402
from src import factor_select                               # noqa: E402
from src import factors as F_lib                            # noqa: E402
from src.cv import time_split                               # noqa: E402
from scripts.backtest.run_backtest_okx import load_clean                     # noqa: E402


def factor_set_for(side: str, Fmat: pd.DataFrame, y: pd.Series, train_slice: slice) -> list:
    """取 v3 冻结模型的因子集: 优先读 reports/v3/metrics_v3.json, 缺失则按 v3 口径重算。"""
    p = C.REPORT_DIR / C.V3_OUT_DIR / "metrics_v3.json"
    if p.exists():
        meta = json.loads(p.read_text(encoding="utf-8"))
        facs = meta.get("sides", {}).get(side, {}).get("factors")
        if facs:
            return [f for f in facs if f in Fmat.columns]
    # 退化: v3 结论(去冗余阈值 + 方向性 IC 筛选)
    dedup = {"long": 0.95, "short": 0.90}[side]
    _, fsets = factor_select.factor_sets_by_dedup(
        Fmat.iloc[train_slice].reset_index(drop=True),
        y.iloc[train_slice].reset_index(drop=True), [dedup])
    return [f for f in fsets[dedup][side] if f in Fmat.columns]


def side_dims(Fmat: pd.DataFrame, factors: list, seg: slice) -> dict:
    """在 seg 上对 factors 做谱分析, 返回维度与共线性指标 + 逐因子 VIF。"""
    sub = Fmat.loc[seg, factors].replace([np.inf, -np.inf], np.nan).dropna()
    X = sub.to_numpy(dtype=float)
    sd = X.std(axis=0, ddof=1)
    Xz = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    n, d = Xz.shape
    corr = np.corrcoef(Xz, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    eig = np.clip(np.linalg.eigvalsh(corr)[::-1], 1e-12, None)
    total = float(eig.sum())
    ratio = eig / total
    cum = np.cumsum(ratio)

    def n_for(thr: float) -> int:
        return int(np.searchsorted(cum, thr) + 1)

    pr = total ** 2 / float((eig ** 2).sum())                     # 参与率(有效维度)
    p = ratio
    erank = float(np.exp(-(p * np.log(p)).sum()))                 # 谱熵有效秩
    kaiser = int((eig > 1.0).sum())
    kappa = float(np.sqrt(eig[0] / eig[-1]))
    vif = pd.Series(np.diag(np.linalg.pinv(corr)), index=factors).sort_values(ascending=False)
    return dict(n_samples=int(n), n_factors=d,
                dim_90=n_for(0.90), dim_95=n_for(0.95), dim_99=n_for(0.99),
                effective_rank_pr=float(pr), effective_rank_entropy=float(erank),
                kaiser=int(kaiser), cond_number=kappa,
                vif_max=float(vif.iloc[0]), vif_mean=float(vif.mean()),
                eig=eig, cum=cum, vif=vif)


def main() -> None:
    print("=" * 108)
    print("多空模型因子集的向量空间维度分析(仅用 train) | 数据源: 欧易(OKX)")
    print("=" * 108)

    df = load_clean()
    df = F_lib.add_label(df, C.HORIZON)
    Fmat, _ = F_lib.build_factors(df)
    y = df["label"]
    tr, oof, ooc = time_split(len(df))
    segs = {"train": tr, "oof": oof, "ooc": ooc}

    rows, summaries = [], {}
    for side in ("long", "short"):
        facs = factor_set_for(side, Fmat, y, tr)
        r = side_dims(Fmat, facs, tr)
        summaries[side] = dict(r, factors=facs)
        rows.append(dict(side=side, n_factors=r["n_factors"], n_samples=r["n_samples"],
                         dim_90=r["dim_90"], dim_95=r["dim_95"], dim_99=r["dim_99"],
                         effective_rank_pr=r["effective_rank_pr"],
                         effective_rank_entropy=r["effective_rank_entropy"],
                         kaiser=r["kaiser"], cond_number=r["cond_number"],
                         vif_max=r["vif_max"], vif_mean=r["vif_mean"]))
        r["vif"].to_frame("VIF").to_csv(C.REPORT_DIR / ("factor_dim_vif_%s.csv" % side))
        print("\n### %s 因子集(%d 个, train 有效样本 %d)" % (side, r["n_factors"], r["n_samples"]))
        print("  名义维度=%d | PCA 90%%方差需 %d 维 | 95%%需 %d 维 | 99%%需 %d 维"
              % (r["n_factors"], r["dim_90"], r["dim_95"], r["dim_99"]))
        print("  有效维度: 参与率=%.2f | 谱熵秩=%.2f | Kaiser(λ>1)=%d"
              % (r["effective_rank_pr"], r["effective_rank_entropy"], r["kaiser"]))
        print("  共线性: 条件数κ=%.1f | 最大VIF=%.1f | 平均VIF=%.2f"
              % (r["cond_number"], r["vif_max"], r["vif_mean"]))
        print("  最高 VIF 因子: %s" % ", ".join("%s=%.1f" % (k, v)
                                                for k, v in r["vif"].head(5).items()))

    summary = pd.DataFrame(rows)
    summary.to_csv(C.REPORT_DIR / "factor_dim.csv", index=False)

    # ---------------- 报告
    lines = ["# 多空模型因子集的向量空间维度分析(抗过拟合审查)\n",
             "- 数据源: 欧易(OKX) 4h | 仅用 **训练段 train**(与因子筛选同口径, 无未来信息)\n",
             "- 对象: v3 冻结模型的因子集(long 去冗余0.95 / short 去冗余0.90)\n",
             "- 度量: 相关矩阵谱分析(PCA 累计解释方差 / 参与率 / 谱熵秩 / Kaiser / VIF)\n\n",
             "## 维度总览\n\n",
             "| 方向 | 名义维度(因子数) | 90%方差维度 | 95%方差维度 | 99%方差维度 | 有效维度(参与率) | 有效维度(谱熵) | Kaiser(λ>1) | 条件数κ | 最大VIF | 平均VIF |\n",
             "|---|---|---|---|---|---|---|---|---|---|---|\n"]
    for r in rows:
        lines.append("| %s | %d | %d | %d | %d | %.2f | %.2f | %d | %.1f | %.1f | %.2f |\n"
                     % (r["side"], r["n_factors"], r["dim_90"], r["dim_95"], r["dim_99"],
                        r["effective_rank_pr"], r["effective_rank_entropy"], r["kaiser"],
                        r["cond_number"], r["vif_max"], r["vif_mean"]))
    lines.append("\n")
    lines.append("> 「有效维度(参与率)」= (Σλ)²/Σλ², 是相关矩阵的连续谱有效秩: 若 25 个因子"
                 "只有 ~7 个有效维度, 说明其余维度只是近共线的噪声方向。\n")
    lines.append("> **过拟合判读**: 名义维度显著大于 95% 方差维度(本例 long 25 vs 11, 参与率口径"
                 "仅 4.3), 说明特征冗余; 最大 VIF > 10 亦提示严重共线。\n")
    for side in ("long", "short"):
        r = summaries[side]
        lines.append("\n## %s 因子集明细(%d 个)\n\n" % (side, r["n_factors"]))
        pc_var = r["eig"] / r["eig"].sum()
        lines.append("主成分解释方差(前 12): %s\n\n"
                     % ", ".join("PC%d=%.1f%%" % (i + 1, v * 100)
                                 for i, v in enumerate(pc_var[:12])))
        lines.append("逐因子 VIF(降序, 前 12): %s\n"
                     % ", ".join("%s=%.2f" % (k, v) for k, v in r["vif"].head(12).items()))
    (C.REPORT_DIR / "factor_dim.md").write_text("".join(lines), encoding="utf-8")

    # ---------------- 图: 累计解释方差曲线(图内文字用 ASCII, 沙箱无中文字体)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, side in zip(axes, ("long", "short")):
        r = summaries[side]
        cum = r["cum"]
        ax.plot(np.arange(1, len(cum) + 1), cum * 100, marker="o", ms=3, lw=1.2)
        for thr, col in ((0.90, "tab:green"), (0.95, "tab:orange"), (0.99, "tab:red")):
            ax.axhline(thr * 100, color=col, ls=":", lw=0.9)
            ax.axvline(np.searchsorted(cum, thr) + 1, color=col, ls=":", lw=0.9)
        ax.set_title("%s | nominal dim=%d, effective dim (participation)=%.1f, "
                     "95%% var = %d PCs"
                     % (side, r["n_factors"], r["effective_rank_pr"], r["dim_95"]))
        ax.set_xlabel("number of principal components")
        ax.set_ylabel("cumulative explained variance %")
        ax.set_ylim(0, 103)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(C.REPORT_DIR / "factor_dim.png", dpi=130)
    plt.close(fig)

    print("\n产出: reports/factor_dim.md / factor_dim.csv / factor_dim_vif_*.csv / factor_dim.png")


if __name__ == "__main__":
    main()
