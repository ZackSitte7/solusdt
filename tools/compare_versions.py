#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""版本横向对比工具 —— 汇总 reports/ 下各版本 metrics_v*.json, 生成对照表与对照图。

用法
----
    python3 tools/compare_versions.py
    python3 tools/compare_versions.py --out reports/version_compare --no-png

产出
----
    <out>.md   版本对照表(markdown)
    <out>.png  对照图: 左 = OOF/OOC 合并总收益; 右 = OOF/OOC 合并夏普

口径说明
--------
- 版本目录 reports/vN/metrics_vN.json; reports/metrics.json(遗留)视为 v1。
- 合并收益: 优先取 metrics 里的 `combined`(v9 起有); 更早版本按
  **多头收益 + 空头收益**(两侧独立、固定名义金, 收益可加)推算, 并在表内标注 *。
- 合并夏普: 只有实测过组合权益曲线的版本(v9 起)才有, 更早版本标注 —。
  两侧夏普**不可相加**(组合波动 != 两侧波动之和), 这里不做近似。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import config as C                                                    # noqa: E402

SEGS = ("oof", "ooc")


def discover(report_dir: Path) -> list:
    """-> [(label, path), ...], 按 v1..vN 数值序排列。"""
    found = []
    for d in sorted(report_dir.glob("v*")):
        if not d.is_dir():
            continue
        f = d / ("metrics_%s.json" % d.name)
        if f.exists():
            found.append((d.name, f))
    legacy = report_dir / "metrics.json"
    if legacy.exists():
        found.append(("v1", legacy))

    def key(item):
        lab = item[0]
        return (1, int(lab[1:])) if lab[1:].isdigit() else (0, 0)
    return sorted(found, key=key)


def extract(path: Path) -> dict:
    """把一个 metrics json 压平成 {seg: {long, short, comb, sharpe, mdd, n}} + 元信息。"""
    d = json.loads(path.read_text(encoding="utf-8"))
    cfg = d.get("config", {}) or {}
    sides = d.get("sides", {}) or {}
    comb = d.get("combined", {}) or {}
    row = dict(
        source=str(cfg.get("SOURCE", "—")),
        data_start=str(cfg.get("DATA_START", "—")),
        select_on=str(cfg.get("SELECT_ON", "—")),
        n_fail=(d.get("consistency", {}) or {}).get("n_fail", None),
    )
    for seg in SEGS:
        lm = (sides.get("long", {}).get("metrics", {}) or {}).get(seg, {}) or {}
        sm = (sides.get("short", {}).get("metrics", {}) or {}).get(seg, {}) or {}
        cm = comb.get(seg, {}) or {}
        lr, sr = lm.get("total_return"), sm.get("total_return")
        if cm:
            cr = cm.get("total_return")
            exact = True
        elif lr is not None and sr is not None:
            cr, exact = lr + sr, False            # 两侧收益可加
        else:
            cr, exact = None, False
        row[seg] = dict(
            long=lr, short=sr, comb=cr, comb_exact=exact,
            sharpe=cm.get("sharpe"), mdd=cm.get("max_drawdown"),
            n=cm.get("n_trades"),
        )
    return row


def _fmt(v, pct: bool = True, digits: int = 2) -> str:
    if v is None:
        return "—"
    return ("%+.*f%%" % (digits, 100 * v)) if pct else ("%.*f" % (digits, v))


def table(rows: list) -> str:
    out = []
    for seg, title in (("oof", "OOF(选择段)"), ("ooc", "OOC(观察段)")):
        out.append("### %s\n" % title)
        out.append("| 版本 | 数据源 | 数据起点 | 多头收益 | 空头收益 | **合并收益** | 合并夏普 | 合并最大回撤 | 合并笔数 | 一致性失败 |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for label, row in rows:
            m = row[seg]
            star = "" if m["comb_exact"] else ("*" if m["comb"] is not None else "")
            out.append("| %s | %s | %s | %s | %s | **%s**%s | %s | %s | %s | %s |" % (
                label, row["source"], row["data_start"],
                _fmt(m["long"]), _fmt(m["short"]), _fmt(m["comb"]), star,
                _fmt(m["sharpe"], pct=False), _fmt(m["mdd"]),
                ("%d" % m["n"]) if m["n"] is not None else "—",
                "—" if row["n_fail"] is None else str(row["n_fail"])))
        out.append("")
    out.append("> `*` = 合并收益由两侧收益相加推算(该版本未实测组合权益曲线)。")
    out.append("> 合并夏普/回撤仅 v9 起有实测值; 更早版本为 `—` —— 两侧夏普不可相加。")
    out.append("> `数据起点` 不同的版本不可直接横比(见各版本报告)。")
    return "\n".join(out) + "\n"


def figure(rows: list, out_png: Path) -> None:
    # 图上用英文标注: 本沙箱无 CJK 字体(与 run_backtest_v*.py 的图保持一致), 中文只在 md 表里。
    labels = [r[0] for r in rows]
    x = np.arange(len(labels))
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

    ax = axes[0]
    for off, seg, color, name in ((-w / 2, "oof", "#2f6fca", "OOF"), (w / 2, "ooc", "#c0392b", "OOC")):
        vals = [100.0 * (r[1][seg]["comb"] or 0.0) if r[1][seg]["comb"] is not None else 0.0
                for r in rows]
        bars = ax.bar(x + off, vals, w, label=name, color=color, alpha=0.85)
        for b, v, r in zip(bars, vals, rows):
            if r[1][seg]["comb"] is None:
                continue
            ax.text(b.get_x() + b.get_width() / 2, v + (0.25 if v >= 0 else -0.55),
                    "%.2f" % v, ha="center", fontsize=7.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Combined total return (%)")
    ax.set_title("Combined (long+short) total return: OOF vs OOC", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    for off, seg, color, name in ((-w / 2, "oof", "#2f6fca", "OOF"), (w / 2, "ooc", "#c0392b", "OOC")):
        vals = [(r[1][seg]["sharpe"] if r[1][seg]["sharpe"] is not None else np.nan) for r in rows]
        ax.bar(x + off, vals, w, label=name, color=color, alpha=0.85)
        for xi, v in zip(x + off, vals):
            if np.isnan(v):
                ax.text(xi, 0.02, "n/a", ha="center", fontsize=6.5, rotation=90, color="#888")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Combined Sharpe")
    ax.set_title("Combined Sharpe (only versions with measured combined equity)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    fig.suptitle("SOL/USDT 4h multi-factor backtest - version comparison "
                 "(versions with different DATA_START are not directly comparable)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="版本横向对比")
    ap.add_argument("--reports", default=str(C.REPORT_DIR))
    ap.add_argument("--out", default=None, help="输出前缀(默认 <reports>/version_compare)")
    ap.add_argument("--no-png", action="store_true")
    a = ap.parse_args(argv)

    report_dir = Path(a.reports)
    out = Path(a.out) if a.out else report_dir / "version_compare"
    found = discover(report_dir)
    if not found:
        print("未找到任何 metrics_v*.json: %s" % report_dir)
        return 1
    rows = [(lab, extract(p)) for lab, p in found]

    md = "# 版本对照(自动生成: tools/compare_versions.py)\n\n" + table(rows)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    print("表格已保存: %s" % out.with_suffix(".md"))
    if not a.no_png:
        figure(rows, out.with_suffix(".png"))
        print("对照图已保存: %s" % out.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
