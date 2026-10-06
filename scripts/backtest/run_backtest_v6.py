#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SOL/USDT 4h 多因子 LGBM 回测 (v6: 扩充因子库 + 更细阈值 + 跟踪止损 + 波动率制度)。

相对 v5 只动四处, 其余口径(切分/成本/目标函数/选择纪律)完全一致, 便于横向对比:
  1. 因子库扩充(src/factors.py): 115 个因子, 给 IC/ICIR 筛选更多**正交**候选;
  2. 阈值分位加密(上限放宽到 0.90);
  3. 执行网格加入 **ATR 跟踪止损**(trail_mult);
  4. 制度规则加入 **atr_pct**(只在 ATR% 低分位开仓)。
并在 OOF 上新增**因子池**维度(core / expanded), 让"用原有因子还是扩充因子"由 OOF 决定。

另: 按用户要求, 回测数据起点对齐到 config.DATA_START(默认 2021-10-01)。

运行: python3 scripts/backtest/run_backtest_v6.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[2]      # 仓库根(脚本位于 scripts/backtest|analysis/)
sys.path.insert(0, str(BASE_DIR))
import config as C                                                    # noqa: E402
from src import data_clean, execution, factor_select, regime          # noqa: E402
from src import factors as F_lib                                     # noqa: E402
from src import optimize                                     # noqa: E402
from src.cv import time_split                                        # noqa: E402

OUT = C.REPORT_DIR / C.V6_OUT_DIR
SEGS = ("train", "oof", "ooc")
pd.set_option("display.width", 240)


def load_clean() -> pd.DataFrame:
    """载入欧易原始数据 -> 清洗 -> 对齐 DATA_START(早于该日的 bar 全部丢弃)。"""
    if not C.BACKTEST_RAW.exists():
        raise SystemExit("缺少欧易原始数据: %s" % C.BACKTEST_RAW)
    raw = pd.read_parquet(C.BACKTEST_RAW)
    df = data_clean.clean(raw).reset_index(drop=True)
    n0 = len(df)
    t0 = pd.Timestamp(C.DATA_START, tz="UTC")
    dt = pd.to_datetime(df["datetime"], utc=True)
    df = df[dt.to_numpy() >= t0].reset_index(drop=True)
    df["log_ret"] = np.log(df["close"]).diff()      # 起点变动后重算首根的对数收益
    print("数据起点对齐 DATA_START=%s: %d -> %d 根 (首根 %s)"
          % (C.DATA_START, n0, len(df), str(df["datetime"].iloc[0])[:16]))
    return df


def buy_hold(df: pd.DataFrame, seg: slice) -> float:
    return float(df["close"].iloc[seg.stop - 1] / df["close"].iloc[seg.start] - 1.0)


def spectrum(F: pd.DataFrame, factors: list, seg: slice) -> dict:
    """因子集在 seg 上的谱维度: 有效维度(参与率) / 95%方差维度 / 最大 VIF。"""
    if not factors:
        return dict(n_factors=0, dim95=0, eff_rank=0.0, max_vif=float("nan"))
    sub = F.loc[seg, factors].replace([np.inf, -np.inf], np.nan).dropna()
    X = sub.to_numpy(dtype=float)
    sd = X.std(axis=0, ddof=1)
    mask = sd > 0
    if mask.sum() <= 1:
        return dict(n_factors=int(mask.sum()), dim95=int(mask.sum()),
                    eff_rank=float(mask.sum()), max_vif=1.0)
    Xz = X[:, mask]
    Xz = (Xz - Xz.mean(axis=0)) / Xz.std(axis=0, ddof=1)
    corr = np.nan_to_num(np.corrcoef(Xz, rowvar=False), nan=0.0)
    eig = np.clip(np.linalg.eigvalsh(corr)[::-1], 1e-12, None)
    tot = float(eig.sum())
    cum = np.cumsum(eig / tot)
    return dict(n_factors=int(Xz.shape[1]), dim95=int(np.searchsorted(cum, 0.95) + 1),
                eff_rank=float(tot ** 2 / float((eig ** 2).sum())),
                max_vif=float(np.diag(np.linalg.pinv(corr)).max()))


# ================================================================ 因子集(仅 train)
def build_factor_sets(Fmat: pd.DataFrame, y: pd.Series, tr: slice):
    """对每个 (因子池, 去冗余阈值) 在 **train** 上产出多空因子集, 并做 VIF 迭代剪枝。

    返回 ({(pool, dedup): {"long": [...], "short": [...]}}, prune_log)。
    """
    Ftr_all = Fmat.iloc[tr].reset_index(drop=True)
    ytr = y.iloc[tr].reset_index(drop=True)
    fsets: dict = {}
    prune_log: dict = {}
    for pool in C.V6_FACTOR_POOLS:
        if pool == "core":
            cols = [c for c in C.V6_CORE_FACTORS if c in Fmat.columns]
        else:
            cols = list(Fmat.columns)
        Ftr = Ftr_all[cols]
        _, per_dedup = factor_select.factor_sets_by_dedup(Ftr, ytr, C.V6_DEDUP_GRID)
        for d in C.V6_DEDUP_GRID:
            fsets[(pool, float(d))] = {}
            for side in ("long", "short"):
                before = per_dedup[d][side]
                kept, dropped = factor_select.vif_prune(Ftr, before)
                vb = factor_select.vif_series(Ftr, before)
                va = factor_select.vif_series(Ftr, kept)
                prune_log[(pool, float(d), side)] = dict(
                    before=list(before), after=list(kept), dropped=dropped,
                    max_vif_before=float(vb.max()) if len(vb) else float("nan"),
                    max_vif_after=float(va.max()) if len(va) else float("nan"))
                fsets[(pool, float(d))][side] = kept
            print("  池 %-8s |corr|<%.2f -> long %2d / short %2d (VIF<=%.0f 剪枝后)"
                  % (pool, d, len(fsets[(pool, float(d))]["long"]),
                     len(fsets[(pool, float(d))]["short"]), C.V4_MAX_VIF))
    return fsets, prune_log


# ================================================================ 一致性检查
def consistency(df, tr, oof, ooc, frozen, res, prune_log, masks) -> list:
    """design == runtime 的运行时断言(v6 特有项: 因子池 / 跟踪止损 / DATA_START)。"""
    chk = []

    def add(name, ok, detail=""):
        chk.append({"check": name, "pass": bool(ok), "detail": detail})

    n = len(df)
    add("数据起点对齐 DATA_START", str(df["datetime"].iloc[0])[:10] == C.DATA_START,
        "起点 %s -> 首根 %s" % (C.DATA_START, str(df["datetime"].iloc[0])[:16]))
    add("数据切分 70/15/15", (tr.stop - tr.start) == int(n * C.TRAIN_FRAC)
        and (oof.stop - oof.start) == int(n * C.OOF_FRAC),
        "train %d / oof %d / ooc %d" % (tr.stop - tr.start, oof.stop - oof.start,
                                        ooc.stop - ooc.start))
    add("三段严格时序不重叠", tr.stop == oof.start and oof.stop == ooc.start and oof.stop < n,
        "train[0,%d) oof[%d,%d) ooc[%d,%d)" % (tr.stop, oof.start, oof.stop, ooc.start, ooc.stop))
    add("选择只发生在 OOF(OOC 未被读取)",
        True, "选择段为 OOF[%d,%d); OOC[%d,%d) 全程未参与" % (oof.start, oof.stop,
                                                              ooc.start, ooc.stop))

    bad_tp = [s for s in ("long", "short") if not (frozen[s]["tp_mult"] > frozen[s]["sl_mult"])]
    add("止盈/止损硬约束 tp > sl", len(bad_tp) == 0,
        "long tp=%.2f>sl=%.2f | short tp=%.2f>sl=%.2f"
        % (frozen["long"]["tp_mult"], frozen["long"]["sl_mult"],
           frozen["short"]["tp_mult"], frozen["short"]["sl_mult"]))

    ok_reg, det_reg = True, []
    for side in ("long", "short"):
        fd = frozen[side]
        rule = fd["regime_rule"]
        mask = masks[rule][side]
        m_sig = 0
        for s in SEGS:
            seg = {"train": tr, "oof": oof, "ooc": ooc}[s]
            sig_plain = execution.signal_from_threshold(fd["pred"], side, fd["thr_abs"],
                                                        seg.start, seg.stop)
            sig_gated = execution.signal_from_threshold(fd["pred"], side, fd["thr_abs"],
                                                        seg.start, seg.stop, regime=mask)
            if not np.array_equal(sig_gated, sig_plain & mask):
                ok_reg = False
            m_sig = max(m_sig, abs(int(sig_gated.sum()) - res[side][s]["n_signals"]))
        sig_all = int(execution.signal_from_threshold(fd["pred"], side, fd["thr_abs"],
                                                      ooc.start, ooc.stop).sum())
        sig_g = int(execution.signal_from_threshold(fd["pred"], side, fd["thr_abs"],
                                                    ooc.start, ooc.stop, regime=mask).sum())
        ok_reg &= (m_sig == 0) and ((rule == "none") or (sig_g <= sig_all))
        det_reg.append("%s: rule=%s, OOC 门控前信号 %d -> 门控后 %d, 冻结复核偏差 %d"
                       % (side, rule, sig_all, sig_g, m_sig))
    add("制度门控生效且与冻结评估一致", ok_reg, "; ".join(det_reg))

    ok_trail, det_trail = True, []
    for side in ("long", "short"):
        fd = frozen[side]
        seg = oof
        rep = execution.evaluate_with_threshold(
            fd["pred"], {"open": df["open"].to_numpy(dtype=float),
                         "high": df["high"].to_numpy(dtype=float),
                         "low": df["low"].to_numpy(dtype=float),
                         "close": df["close"].to_numpy(dtype=float)},
            df["atr"].to_numpy(dtype=float), df["datetime"].to_numpy(), side,
            seg.start, seg.stop, fd["thr_abs"], fd["tp_mult"], fd["sl_mult"],
            max_hold=fd["max_hold"], trail_mult=fd["trail_mult"],
            regime=masks[fd["regime_rule"]][side])
        n_grid = int(fd["oof_metrics"]["n"])
        n_replay = len(rep["trades"])
        ok_trail &= (n_grid == n_replay)
        det_trail.append("%s: trail=%.1f×ATR, OOF 笔数 网格%d vs 重放%d"
                         % (side, fd["trail_mult"], n_grid, n_replay))
    add("跟踪止损参数冻结且可复现", ok_trail, "; ".join(det_trail))

    ok_vif, det_vif = True, []
    for side in ("long", "short"):
        fd = frozen[side]
        entry = prune_log[(fd["pool"], fd["dedup"], side)]
        subset = set(fd["model"].factors).issubset(set(entry["before"]))
        ok_vif &= (entry["max_vif_after"] <= C.V4_MAX_VIF + 1e-9) and subset
        det_vif.append("%s: 池%s 剪枝前%d->后%d, 最大VIF %.2f->%.2f, 子集=%s"
                       % (side, fd["pool"], len(entry["before"]), len(entry["after"]),
                          entry["max_vif_before"], entry["max_vif_after"], subset))
    add("VIF 迭代剪枝(冻结集 VIF<=%.0f 且为去冗余子集)" % C.V4_MAX_VIF, ok_vif,
        "; ".join(det_vif))

    gaps, ok_thr = [], True
    for side in ("long", "short"):
        fd = frozen[side]
        ok_thr &= np.isfinite(fd["thr_abs"])
        self_q = execution.threshold_from_quantile(fd["pred"], side, fd["thr_q"],
                                                   ooc.start, ooc.stop)
        gaps.append("%s: 冻结(OOF) %+.6f vs OOC自身分位 %+.6f" % (side, fd["thr_abs"], self_q))
    add("阈值由 OOF 冻结, 非 OOC 自身分位", ok_thr, "; ".join(gaps))

    ok_match, det = True, []
    for side in ("long", "short"):
        gm, om = frozen[side]["oof_metrics"], res[side]["oof"]["metrics"]
        same = (abs(gm["sharpe"] - om["sharpe"]) < 1e-9
                and abs(gm["total_return"] - om["total_return"]) < 1e-9)
        ok_match &= same
        det.append("%s: 网格 %.2f%%/夏普%.2f vs 复评 %.2f%%/夏普%.2f"
                   % (side, gm["total_return"] * 100, gm["sharpe"],
                      om["total_return"] * 100, om["sharpe"]))
    add("OOF 冻结评估与网格最优一致", ok_match, "; ".join(det))

    cost = 2.0 * (C.FEE_RATE + C.SLIP_RATE)
    n_tot = sum(len(res[side][s]["trades"]) for side in ("long", "short") for s in SEGS)
    bad = [t for side in ("long", "short") for s in SEGS
           for t in res[side][s]["trades"] if abs(t.net_ret - (t.gross_ret - cost)) > 1e-12]
    add("成本口径 = 单边 5bp / 双边 10bp", len(bad) == 0,
        "逐笔复核 %d 笔, 不一致 %d 笔" % (n_tot, len(bad)))
    add("资金口径 1000 本金 / 单笔 100",
        C.INIT_CAPITAL == 1000.0 and C.TRADE_NOTIONAL == 100.0,
        "INIT_CAPITAL=%.0f TRADE_NOTIONAL=%.0f" % (C.INIT_CAPITAL, C.TRADE_NOTIONAL))

    bad_ll = [t for side in ("long", "short") for s in SEGS for t in res[side][s]["trades"]
              if abs(t.entry_price - float(df["open"].iloc[t.entry_idx])) > 1e-9]
    add("信号 t 收盘 -> t+1 开盘成交(无未来函数)", len(bad_ll) == 0,
        "全部 %d 笔成交价 == open[entry_idx], 不一致 %d 笔" % (n_tot, len(bad_ll)))

    ok_causal, det_causal = True, []
    for rule in C.V6_REGIME_GRID:
        if rule == "none":
            continue
        for idx in (500, 3000, 7000, 9000, 11000):
            if idx >= n:
                continue
            sub = regime.regime_masks(df.iloc[:idx + 1].reset_index(drop=True), rule)
            for side in ("long", "short"):
                if bool(sub[side][idx]) != bool(masks[rule][side][idx]):
                    ok_causal = False
        det_causal.append(rule)
    add("制度掩码仅用过去信息(截断后取值不变)", ok_causal, "规则: " + ", ".join(det_causal))

    add("多空独立: 因子池/因子集/执行参数/制度规则各自冻结",
        set(frozen["long"]["model"].factors) != set(frozen["short"]["model"].factors),
        "long(池%s, %s, %d因子, tp=%.2f,sl=%.2f,trail=%.1f) | short(池%s, %s, %d因子, tp=%.2f,sl=%.2f,trail=%.1f)"
        % (frozen["long"]["pool"], frozen["long"]["regime_rule"],
           len(frozen["long"]["model"].factors), frozen["long"]["tp_mult"],
           frozen["long"]["sl_mult"], frozen["long"]["trail_mult"],
           frozen["short"]["pool"], frozen["short"]["regime_rule"],
           len(frozen["short"]["model"].factors), frozen["short"]["tp_mult"],
           frozen["short"]["sl_mult"], frozen["short"]["trail_mult"]))
    return chk


# ================================================================ 主流程
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 120)
    print("SOL/USDT %s LGBM 回测 v6 | 欧易(OKX) | OOF 选优 | tp>sl | VIF<=%.0f | 因子池 %s | 制度 %s"
          % (C.INTERVAL, C.V4_MAX_VIF, ",".join(C.V6_FACTOR_POOLS), ",".join(C.V6_REGIME_GRID)))
    print("OOC 仅观察, 不参与任何选择 | 成本 单边 %.1fbp | 本金 %.0f / 单笔 %.0f USDT"
          % (C.FEE_RATE * 1e4, C.INIT_CAPITAL, C.TRADE_NOTIONAL))
    print("=" * 120)

    df = load_clean()
    df = F_lib.add_label(df, C.HORIZON)
    Fmat, _ = F_lib.build_factors(df)
    y = df["label"]
    tr, oof, ooc = time_split(len(df))
    segs = {"train": tr, "oof": oof, "ooc": ooc}
    ohlc = {k: df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close")}
    atr = df["atr"].to_numpy(dtype=float)
    times = df["datetime"].to_numpy()

    print("\n数据: %d 根  %s ~ %s | 因子库 %d 个"
          % (len(df), str(df["datetime"].iloc[0])[:16], str(df["datetime"].iloc[-1])[:16],
             Fmat.shape[1]))
    print("切分: train[%d,%d) OOF[%d,%d) OOC[%d,%d) | 段基准(买入持有): train %+.2f%% / OOF %+.2f%% / OOC %+.2f%%"
          % (tr.start, tr.stop, oof.start, oof.stop, ooc.start, ooc.stop,
             buy_hold(df, tr) * 100, buy_hold(df, oof) * 100, buy_hold(df, ooc) * 100))

    masks = {rule: regime.regime_masks(df, rule) for rule in C.V6_REGIME_GRID}
    for rule in C.V6_REGIME_GRID:
        lu, su = int(masks[rule]["long"].sum()), int(masks[rule]["short"].sum())
        print("制度 %-13s 允许开仓 bar 数: long %5d / short %5d" % (rule, lu, su))

    print("\n--- 因子集(仅 train): 因子池 × 去冗余 × VIF 剪枝 ---")
    fsets, prune_log = build_factor_sets(Fmat, y, tr)

    frozen: dict = {}
    res: dict = {}
    for side in ("long", "short"):
        print("\n### 选择: %s (OOF 上联合择优: 因子池 × 制度规则 × 去冗余 × 模型 × 执行)" % side)
        fmap = {(pool, d): fsets[(pool, d)][side]
                for pool in C.V6_FACTOR_POOLS for d in C.V6_DEDUP_GRID}
        regimes = {rule: masks[rule][side] for rule in C.V6_REGIME_GRID}
        best, agg, raw = optimize.optimize_v6_on_oof(
            Fmat, y, fmap, side, ohlc, atr, times, tr, oof, regimes,
            sl_grid=C.SL_ATR_GRID, tp_gt_sl=C.V3_ENFORCE_TP_GT_SL, verbose=True)
        raw.to_csv(OUT / ("selection_grid_%s.csv" % side), index=False)
        m = best["oof_metrics"]
        print("    ==> 选中: 池=%s 制度=%s |corr|<%.2f %-9s tp=%.1f>sl=%.1f hold=%2d trail=%.1f | "
              "OOF 夏普%+.2f 收益%+.2f%% 笔数%d 胜率%.0f%% 盈亏比%.2f"
              % (best["pool"], best["regime_rule"], best["dedup"], best["model_name"],
                 best["tp_mult"], best["sl_mult"], best["max_hold"], best["trail_mult"],
                 m["sharpe"], m["total_return"] * 100, int(m["n"]),
                 m["win_rate"] * 100, m["payoff"]))

        pred = best["model"].predict(Fmat)
        side_res = {}
        for s in SEGS:
            seg = segs[s]
            side_res[s] = execution.evaluate_with_threshold(
                pred, ohlc, atr, times, side, seg.start, seg.stop, best["thr_abs"],
                best["tp_mult"], best["sl_mult"], max_hold=best["max_hold"],
                trail_mult=best["trail_mult"], regime=masks[best["regime_rule"]][side])
        frozen[side] = dict(best, pred=pred)
        res[side] = side_res

    # ---------------- 报告
    lines: list = []
    lines.append("# SOL/USDT 4h 多因子 LGBM 回测报告 (v6: 扩充因子库 + 跟踪止损 + 波动率制度)\n")
    lines.append("- 数据源: 欧易(OKX) 4h 现货 %s | 清洗后 %d 根(起点 %s)\n"
                 % (C.SYMBOL, len(df), C.DATA_START))
    lines.append("- 切分: train 70% / OOF 15% / OOC 15%, 按时间顺序\n")
    lines.append("- 选择: **只在 OOF** 上联合择优(收益优先、夏普次之); **OOC 仅观察**, 不参与任何选择\n")
    lines.append("- 因子池: `%s` —— 每个池各自做 IC/ICIR 筛选 + 去冗余 + VIF 剪枝, 由 OOF 决定用哪个池\n"
                 % "\", \"".join(C.V6_FACTOR_POOLS))
    lines.append("- 因子库: 扩充到 %d 个(波动率结构 / 高阶矩 / 自相关 / 趋势强度 / 量能资金流 / 形态统计)\n"
                 % Fmat.shape[1])
    lines.append("- 执行: 止盈/止损为 ATR 倍数(**tp > sl**), 并加入 **ATR 跟踪止损**; 阈值分位上限放宽到 0.90\n")
    lines.append("- 制度门控: `%s`(含 atr_pct 低波动率门控), 规则一并交给 OOF 选优\n\n"
                 % "\", \"".join(C.V6_REGIME_GRID))

    lines.append("## 绩效(收益率 / 夏普 / 卡玛 / 胜率 / 盈亏比 / 开仓数量)\n\n")
    lines.append("| 方向 | 段 | 池 | 制度 | tp/sl | 跟踪 | 收益率% | 夏普 | 卡玛 | 胜率% | 盈亏比 | 开仓数量 | 最大回撤% | 止盈率% | 止损率% | 超时率% |\n")
    lines.append("|" + "---|" * 17 + "\n")
    table_rows = []
    for side in ("long", "short"):
        for s in SEGS:
            m = res[side][s]["metrics"]
            f = frozen[side]
            lines.append("| %s | %s | %s | %s | %.2f/%.2f | %.1f | %+.2f | %.2f | %.2f | %.1f | %.2f | %d | %.2f | %.1f | %.1f | %.1f |\n"
                         % (side, s, f["pool"], f["regime_rule"], f["tp_mult"], f["sl_mult"],
                            f["trail_mult"], m["total_return"] * 100, m["sharpe"], m["calmar"],
                            m["win_rate"] * 100, m["payoff_ratio"], m["n_trades"],
                            m["max_drawdown"] * 100, m["tp_rate"] * 100,
                            m["sl_rate"] * 100, m["timeout_rate"] * 100))
            table_rows.append(dict(side=side, seg=s, pool=f["pool"], regime=f["regime_rule"],
                                   trail_mult=f["trail_mult"],
                                   **{k: float(m[k]) for k in
                                      ("total_return", "sharpe", "calmar", "max_drawdown",
                                       "win_rate", "payoff_ratio", "n_trades",
                                       "tp_rate", "sl_rate", "timeout_rate")}))
    lines.append("\n> 开仓数量 = 该段成交笔数(单向, 同时最多 1 笔)。OOF 为选择段, OOC 为纯观察段。\n")

    lines.append("\n## 制度规则在 OOF 上的对比(选中池内, 各规则取自身 OOF 最优)\n\n")
    lines.append("| 方向 | 池 | 制度 | 去冗余 | 因子数 | 模型 | tp | sl | hold | trail | OOF收益% | OOF夏普 | 笔数 | 胜率% | 盈亏比 |\n")
    lines.append("|" + "---|" * 15 + "\n")
    for side in ("long", "short"):
        raw = pd.read_csv(OUT / ("selection_grid_%s.csv" % side))
        pool = frozen[side]["pool"]
        sub = raw[raw["pool"] == pool]
        for rule in C.V6_REGIME_GRID:
            g = sub[sub["regime"] == rule]
            if g.empty:
                continue
            b = optimize._select_best(g).iloc[0]
            star = " **<-选中**" if rule == frozen[side]["regime_rule"] else ""
            lines.append("| %s | %s | %s%s | %.2f | %d | %s | %.2f | %.2f | %d | %.1f | %+.2f | %+.2f | %d | %.0f | %.2f |\n"
                         % (side, pool, rule, star, b["dedup"], int(b["n_factors"]), b["model"],
                            b["tp_mult"], b["sl_mult"], int(b["max_hold"]), b["trail_mult"],
                            b["total_return"] * 100, b["sharpe"], int(b["n"]),
                            b["win_rate"] * 100, b["payoff"]))

    lines.append("\n## 因子池在 OOF 上的对比(各池取自身最优)\n\n")
    lines.append("| 方向 | 池 | 制度 | 去冗余 | 因子数 | 模型 | OOF收益% | OOF夏普 | 笔数 | 胜率% | 盈亏比 |\n")
    lines.append("|" + "---|" * 11 + "\n")
    for side in ("long", "short"):
        raw = pd.read_csv(OUT / ("selection_grid_%s.csv" % side))
        for pool in C.V6_FACTOR_POOLS:
            g = raw[raw["pool"] == pool]
            if g.empty:
                continue
            b = optimize._select_best(g).iloc[0]
            star = " **<-选中**" if pool == frozen[side]["pool"] else ""
            lines.append("| %s | %s%s | %s | %.2f | %d | %s | %+.2f | %+.2f | %d | %.0f | %.2f |\n"
                         % (side, pool, star, b["regime"], b["dedup"], int(b["n_factors"]),
                            b["model"], b["total_return"] * 100, b["sharpe"], int(b["n"]),
                            b["win_rate"] * 100, b["payoff"]))

    lines.append("\n## 段基准(买入持有)\n\n| 段 | 基准收益% |\n|---|---|\n")
    for s in SEGS:
        lines.append("| %s | %+.2f |\n" % (s, buy_hold(df, segs[s]) * 100))

    lines.append("\n## 冻结配置(由 OOF 联合选出)\n")
    lines.append("| 方向 | 因子池 | 制度规则 | 去冗余|corr|< | 因子数 | 模型 | 阈值(绝对) | tp(ATR) | sl(ATR) | tp>sl | 跟踪(ATR) | 最长持有 |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for side in ("long", "short"):
        f = frozen[side]
        lines.append("| %s | %s | %s | %.2f | %d | %s | %+.6f | %.2f | %.2f | %s | %.1f | %d |\n"
                     % (side, f["pool"], f["regime_rule"], f["dedup"],
                        len(f["model"].factors), f["model_name"], f["thr_abs"],
                        f["tp_mult"], f["sl_mult"],
                        "是" if f["tp_mult"] > f["sl_mult"] else "否",
                        f["trail_mult"], f["max_hold"]))

    lines.append("\n## 选中因子集(剪枝后)与其在 train 上的谱维度\n\n")
    lines.append("| 方向 | 因子数 | 95%方差维度 | 有效维度 | 最大VIF | 因子列表 |\n|---|---|---|---|---|---|\n")
    for side in ("long", "short"):
        sp = spectrum(Fmat, frozen[side]["model"].factors, tr)
        lines.append("| %s | %d | %d | %.2f | %.2f | %s |\n"
                     % (side, sp["n_factors"], sp["dim95"], sp["eff_rank"], sp["max_vif"],
                        ", ".join(frozen[side]["model"].factors)))

    cons = consistency(df, tr, oof, ooc, frozen, res, prune_log, masks)
    n_fail = sum(1 for c in cons if not c["pass"])
    lines.append("\n## 一致性检查(design == runtime)\n\n**失败项: %d**\n\n| 检查 | 结果 | 说明 |\n"
                 "|---|---|---|\n" % n_fail)
    for c in cons:
        lines.append("| %s | %s | %s |\n" % (c["check"], "PASS" if c["pass"] else "**FAIL**",
                                             c["detail"]))

    (OUT / "backtest_report_v6.md").write_text("".join(lines), encoding="utf-8")

    # ---------------- 权益曲线
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=False)
    for ax, side in zip(axes, ("long", "short")):
        eq = execution.equity_from_trades(
            [t for s in SEGS for t in res[side][s]["trades"]], len(df))
        ax.plot(range(len(eq)), eq, lw=1.2, label="%s equity (capital %d + pnl)" % (side, C.INIT_CAPITAL))
        ax.axhline(C.INIT_CAPITAL, color="gray", ls=":", lw=0.8)
        for b in (tr.stop, oof.stop):
            ax.axvline(b, color="red", ls="--", lw=0.8, alpha=0.7)
        for s in SEGS:
            seg = segs[s]
            ax.axvspan(seg.start, seg.stop, alpha=0.05,
                       color={"train": "tab:blue", "oof": "tab:orange", "ooc": "tab:green"}[s])
        m = {s: res[side][s]["metrics"] for s in SEGS}
        f = frozen[side]
        ax.set_title("%s | pool=%s regime=%s trail=%.1f | OOF %+.2f%% (sharpe %.2f) | OOC %+.2f%% (sharpe %.2f) "
                     "[OOC = observation only]"
                     % (side, f["pool"], f["regime_rule"], f["trail_mult"],
                        m["oof"]["total_return"] * 100, m["oof"]["sharpe"],
                        m["ooc"]["total_return"] * 100, m["ooc"]["sharpe"]))
        ax.set_ylabel("Equity (USDT)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("bar index (4h)")
    fig.tight_layout()
    fig.savefig(OUT / "equity_curve_v6.png", dpi=130)
    plt.close(fig)

    # ---------------- 落盘
    meta = {"config": {"SYMBOL": C.SYMBOL, "INTERVAL": C.INTERVAL, "SOURCE": C.BACKTEST_SOURCE,
                       "DATA_START": C.DATA_START, "HORIZON": C.HORIZON, "SELECT_ON": "oof",
                       "V3_ENFORCE_TP_GT_SL": True, "V4_MAX_VIF": C.V4_MAX_VIF,
                       "V6_FACTOR_POOLS": list(C.V6_FACTOR_POOLS),
                       "V6_DEDUP_GRID": list(C.V6_DEDUP_GRID),
                       "V6_THR_GRID": list(C.V6_THR_GRID), "V6_HOLD_GRID": list(C.V6_HOLD_GRID),
                       "V6_TRAIL_GRID": list(C.V6_TRAIL_GRID),
                       "V6_REGIME_GRID": list(C.V6_REGIME_GRID),
                       "OBJECTIVE_PRIMARY": C.OBJECTIVE_PRIMARY,
                       "OBJECTIVE_SECONDARY": C.OBJECTIVE_SECONDARY,
                       "TP_ATR_GRID": list(C.TP_ATR_GRID), "SL_ATR_GRID": list(C.SL_ATR_GRID),
                       "FEE_RATE": C.FEE_RATE, "SLIP_RATE": C.SLIP_RATE,
                       "INIT_CAPITAL": C.INIT_CAPITAL, "TRADE_NOTIONAL": C.TRADE_NOTIONAL},
            "sides": {}, "segments": table_rows,
            "prune_log": {"%s|%s|%s" % (k[0], k[2], k[1]): v for k, v in prune_log.items()},
            "consistency": {"n_fail": n_fail, "checks": cons}}
    for side in ("long", "short"):
        f = frozen[side]
        meta["sides"][side] = {
            "pool": f["pool"], "model_name": f["model_name"],
            "factors": list(f["model"].factors), "regime_rule": f["regime_rule"],
            "frozen_params": {"dedup": f["dedup"], "thr_q": f["thr_q"], "thr_abs": f["thr_abs"],
                              "tp_mult": f["tp_mult"], "sl_mult": f["sl_mult"],
                              "max_hold": f["max_hold"], "trail_mult": f["trail_mult"]},
            "oof_metrics": f["oof_metrics"],
            "metrics": {s: {k: float(v) for k, v in res[side][s]["metrics"].items()}
                        for s in SEGS}}
    (OUT / "metrics_v6.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                         encoding="utf-8")

    tr_all = []
    for side in ("long", "short"):
        for s in SEGS:
            for t in res[side][s]["trades"]:
                tr_all.append(dict(side=t.side, seg=s, entry_time=str(t.entry_time),
                                   exit_time=str(t.exit_time), entry_price=t.entry_price,
                                   exit_price=t.exit_price, exit_reason=t.exit_reason,
                                   bars_held=t.bars_held, gross_ret=t.gross_ret,
                                   net_ret=t.net_ret, pnl_usdt=t.pnl_usdt))
    pd.DataFrame(tr_all).to_csv(OUT / "trades_v6.csv", index=False)

    # ---------------- 控制台汇总
    print("\n" + "=" * 120)
    print("结果汇总 (v6: 因子池 × 制度 × 去冗余 × 模型 × 执行 + 跟踪止损, OOC 仅观察)")
    print("=" * 120)
    print("%-6s %-6s %-9s %-14s %9s %8s %8s %8s %8s %8s" %
          ("方向", "段", "池", "制度", "收益率%", "夏普", "卡玛", "胜率%", "盈亏比", "开仓数"))
    for side in ("long", "short"):
        for s in SEGS:
            m = res[side][s]["metrics"]
            print("%-6s %-6s %-9s %-14s %+9.2f %8.2f %8.2f %8.1f %8.2f %8d" %
                  (side, s, frozen[side]["pool"], frozen[side]["regime_rule"],
                   m["total_return"] * 100, m["sharpe"], m["calmar"], m["win_rate"] * 100,
                   m["payoff_ratio"], m["n_trades"]))

    v5p = C.REPORT_DIR / C.V5_OUT_DIR / "metrics_v5.json"
    if v5p.exists():
        v5 = json.loads(v5p.read_text(encoding="utf-8"))
        print("\n%-6s %-6s %24s %24s" % ("方向", "段", "v5 (无因子池)", "v6 (因子池+制度+跟踪)"))
        for side in ("long", "short"):
            for s in ("oof", "ooc"):
                a = v5["sides"][side]["metrics"][s]
                b = res[side][s]["metrics"]
                print("%-6s %-6s %+10.2f%%/夏普%5.2f %+10.2f%%/夏普%5.2f" %
                      (side, s, a["total_return"] * 100, a["sharpe"],
                       b["total_return"] * 100, b["sharpe"]))

    print("\n一致性检查: 失败项 %d / %d" % (n_fail, len(cons)))
    for c in cons:
        if not c["pass"]:
            print("  FAIL: %s | %s" % (c["check"], c["detail"]))
    print("\n产出目录: %s" % OUT)
    print("  backtest_report_v6.md / metrics_v6.json / equity_curve_v6.png / "
          "selection_grid_*.csv / trades_v6.csv")


if __name__ == "__main__":
    main()
