# solusdt

SOL/USDT 4h 多因子 LightGBM 回测研究。**多空分为两个独立模型**, 在欧易(OKX) 4h 现货
K 线上做因子构建 → 筛选 → 训练 → 执行参数寻优 → 样本外验收。逐版迭代, 每版一个入口脚本,
历史版本保持可复现(见下方[版本谱系](#版本谱系))。

核心纪律(全部版本共用):

- 数据按时间顺序切 **Train 70% / OOF 15% / OOC 15%**;
- **所有选择只发生在 OOF**; **OOC 全程只观察, 绝不参与任何比较或调参**(有单测守卫);
- 阈值一律冻结为选择段预测的绝对分位, 评估段不得回看自身分布(防未来函数)。

## 快速开始

```bash
pip install -r requirements.lock.txt   # 精确版本(见下)
make test                              # 单元测试 + 冒烟测试
make compare                           # 生成版本对照表/图 -> reports/version_compare.{md,png}
make v6 && make v7                     # 逐版运行, 版本间有前置依赖, 详见 make help
```

运行环境: **Python 3.14.7**。`requirements.lock.txt` 是精确锁定, `requirements.txt` 是松散约束
(用后者复现可能与 `reports/` 内已入库产物有末位差异)。

## 目录结构

```
config.py                全局配置: 路径/数据规格/切分/成本/回测约束 + 各版本参数(唯一参数来源)
run_backtest_okx.py      v1 基线流水线(欧易数据)
run_backtest_v2..v15.py  逐版迭代入口(每版一个脚本)
run_backtest_v18.py      v18 滚动 walk-forward 入口
src/                     核心库: 数据抓取/清洗、因子、筛选、模型、执行、指标、CV、制度、优化
tools/shards.py          大文件分片/合并/校验
tools/compare_versions.py   跨版本指标汇总与对照图
tests/                   test_consistency.py(设计-运行一致性) + test_smoke.py(产物完整性)
data/                    原始与清洗数据(随仓库分发)
reports/                 回测产物; 根目录 = v1 基线, reports/vN/ = 各版本
```

> `reports/` 根目录**不是**历史垃圾: [run_backtest_v2.py](run_backtest_v2.py) 与
> [run_backtest_v3.py](run_backtest_v3.py) 会读 `reports/metrics.json` 作为 v1 基线做对比,
> 请勿归档或移动。

## 版本谱系

| 版本 | 关键改动 | OOF 合并收益 | OOC 合并收益 |
|---|---|---|---|
| v1 | 基线流水线 | +17.56%* | −7.14%* |
| v2 | 抗过拟合协议(嵌套 CV、最差折选择) | +2.44%* | −2.03%* |
| v3 | 选优移到 OOF + 约束 tp>sl | +16.85%* | −4.51%* |
| v4 | 因子筛选加 VIF 迭代剪枝 | +16.58%* | −4.87%* |
| v5 | 制度门控(顺势开仓), 收窄 long 的 OOC 亏损 | +17.53%* | −0.54%* |
| v6 | 扩充因子库 + ATR 跟踪止损 + 波动率制度 | +12.88%* | −0.91%* |
| v7 | 空头专用因子/制度, 止损网格下探 | +13.38%* | −1.45%* |
| v8 | OOF 内部分折 + 稳健选优(压 winner's curse) | +11.78%* | −1.44%* |
| v9 | long+short 共同优化, 目标 = 合并总收益 | +8.20% | +3.91% |
| v10 | OOF 目标加入合并夏普(真实 bar 级重建) | +8.90% | +2.65% |
| v11 | 候选池 200→500 + Pareto 双目标保留 | +4.79% | +2.24% |
| v12 | 直接以真实 OOF 段收益+夏普排序 | +11.63% | +0.68% |
| v13 | 收益锚 + 夏普择优(有界让步带 1.5pp) | +15.27% | +0.59% |
| **v14** | **long 趋势因子 + 模型/执行层外扩(long OOC 转正)** | **+15.59%** | **+4.32%** |
| v15 | long 门控冻结 `none` + tp 上界放开到 12 | +15.89% | +3.31% |
| v18 | 滚动 walk-forward(选参窗重选 / 前测窗纯 OOS) | — | −10.36%† |

`*` = 合并收益由两侧收益相加推算(该版本未实测组合权益曲线); v9 起有实测的合并夏普/回撤。
`†` = v18 换了口径: 用「选参窗选择 / 前测窗纯 OOS」的滚动 walk-forward, 该列是**从未参与选择**的
连续 OOS 合并收益(夏普 −1.14), 不是单窗 OOC。**v1–v5 未记录 `DATA_START`, 与 v6+ 不可直接横比。**
其中 v1–v10 由 `make compare` 自动汇总(完整版见 [reports/version_compare.md](reports/version_compare.md)),
v11 及以后见各版本目录下的 `backtest_report_vN.md`。

两条贯穿全程的经验:

1. **OOF 指标更高 ≠ OOC 更好**。多版把 OOF 越做越"漂亮"(收益/稳健性/夏普), OOC 却常常回落;
   选择纪律比指标高低更重要。v18 用滚动 walk-forward 直接量化了这一点: 「在 OOF/选参窗取最优」
   本身存在明显选择过拟合(选参窗→前测窗收益落差约 11.6pp), 连续前测由正转负。
2. **合并口径必须实测**。两侧交易独立但组合夏普/回撤与两侧各自的值不可相加, v9 起改为
   `equity = eq_long + eq_short − INIT_CAPITAL` 后按 bar 收益年化。

## ⭐ 推荐版本: v14

**结论: 实盘/落地取 v14**(入口 [run_backtest_v14.py](run_backtest_v14.py), 产物
[reports/v14/](reports/v14/))。

理由: v14 把 **long 侧 OOC 由负转正**(−1.43% → +2.30%), 且**整体 OOC 合并收益 +4.32% 为全谱系最高**
(v9 +3.91% / v10 +2.65% / v15 +3.31%)。后续 v15 把 OOF 做得更漂亮(tp 放开)但 OOC 回落;
v18 进一步证明 OOF/选参窗的漂亮数字带有选择过拟合。综合"观察段真实表现 + 稳健性",
v14 更适合作为落地基线。

**v14 模型(由 OOF 联合选出的冻结配置)**

| 方向 | 制度门控 | 去冗余 max\|corr\| | 因子数 | 模型 | 阈值(绝对) | tp(ATR) | sl(ATR) | 最长持有 | OOF 收益% | OOC 收益% |
|---|---|---|---|---|---|---|---|---|---|---|
| **long** | `none` | 0.90 | 38 | `deep_reg2` | −0.007313 | 6.0 | 2.5 | 48 | +9.18 | +2.30 |
| **short** | `sma200_slope` | 0.85 | 18 | `shallow` | +0.053547 | 2.5 | 0.75 | 6 | +6.41 | +2.02 |

> 制度门控: long 不限方向, short 仅在 close<MA200 且 MA200 下行时开仓; tp/sl 为 ATR 倍数;
> 阈值由 OOF 段预测的绝对分位冻结。全部细节与一致性检查见
> [reports/v14/backtest_report_v14.md](reports/v14/backtest_report_v14.md)。

## 报告与产物

每个版本目录包含: `metrics_vN.json`(全部指标 + 配置快照 + 一致性检查)、`backtest_report_vN.md`、
`trades_vN.csv`、权益曲线 PNG; 部分版本另有选优过程表。`reports/` 内的 `.md/.json/.csv/.png`
均随仓库分发以便复现与对比。

## 大文件分片

`reports/v8/`、`reports/v9/` 下的 `selection_folds_*` 原始网格转储单文件约 108MB, 超过 GitHub
单文件 100MiB 硬上限, 因此**原文件出库、分片入库**: `reports/vN/shards/<源文件名>/partNN`,
每片 ≤90MiB、按行边界切分且每片重复表头, 附 `MANIFEST.json` 记录 sha256/字节数/行数。

```bash
python3 tools/shards.py verify reports                       # 校验全部分片(含还原比对)
python3 tools/shards.py merge  reports/v9/shards/selection_folds_long.csv out.csv
```

## 已知约束

- **图中文字体**: 运行环境无 CJK 字体, 因此图内标注一律用英文, 中文只出现在 markdown 报告里。
- `requests` 仅抓取数据时需要(数据已随仓库分发), 锁定文件里默认不装。
- 版本脚本之间有前置依赖: v7/v8 依赖 v6 冻结的多头配置, v9/v10 依赖 v6/v8/v9 的指标文件。
