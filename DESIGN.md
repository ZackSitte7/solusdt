# SOL/USDT 4h 多因子 LGBM 回测(欧易数据) — 设计文档

本文件定义系统的**设计**；`run_backtest_okx.py` 的实现必须与本文件逐条一致。
运行后由 `reports/consistency_report.md` 自动复核关键不变量，`tests/test_consistency.py`
用单元测试固化（需求15：确保设计与运行的同一性）。

## 1. 数据（需求1、4）
- 数据源：**欧易 OKX** 现货 `SOL-USDT` 4h K线，落盘为 `data/solusdt_4h_raw_okx.parquet`
  （13,146 根，2020-09-30 12:00 ~ 2026-09-30 04:00 UTC）。
- 清洗（`src/data_clean.py`，可复现、留痕、不插值）：
  1. 按 `open_time` 排序、去重（保留末条）；
  2. 价格 ≤ 0 剔除；单根振幅 `high/low-1 > 35%` 剔除；单根 `|收益| > 35%` 剔除；零量剔除；
  3. `high/low` 与 `open/close` 取极值修正，保证 ATR/振幅自洽；
  4. 缺口只报告不填充（避免未来信息）。
- 清洗后 **13,128 根** 4h K线，写入 `data/solusdt_4h_clean_okx.parquet`。
- **字段约束**：欧易公开 K 线**不含** `trades / taker_buy_base / taker_buy_quote`，
  因子库据此自动跳过订单流类因子（`src/factors.py` 的 `ORDER_FLOW_COLS` 守卫）。

## 2. 样本切分（需求2、9）
按 K 线数时序切分，**严格先后、互不重叠**（`src/cv.py: time_split`）：
- train = 前 70%（9,189 根，2020-09-30 ~ 2024-12-12）→ 因子筛选、模型拟合
- OOF  = 中间 15%（1,969 根，2024-12-12 ~ 2025-11-05）→ **仅**用于优化
- OOC  = 最后 15%（1,970 根，2025-11-06 ~ 2026-09-30）→ **只观察**，任何环节均不使用

## 3. 因子（需求5、14）
`src/factors.py: build_factors`，全部只用 t ≤ t 的信息（单元测试 `test_factors_no_lookahead`）。
本次数据源可用因子 **61 个**，分组：
- **趋势/动量（需求14）**：`ret_{1,2,3,6,12,24,48}`、`roc_{6,12,24}`、`mom_accel`、
  `ma_ratio_{20,50,100,200}`、`ma_slope_{20,50}`、`ma_align`、`macd/macd_signal/macd_hist`、
  **`adx_14`**、`trend_strength_50`、`lr_slope_{20,50}`、`pos_in_range_{20,50,100}`、
  `hh_count_20`、`ll_count_20`
- 波动：`atr_ratio`、`atr_ratio_chg`、`rv_{12,20,50}`、`vol_ratio_5_20`、`bb_width_20`、
  `bb_pos_20`、`parkinson_20`、`range_ratio`
- 超买超卖：`rsi_14`、`rsi_28`、`rsi_div`、`stoch_k_14`、`cci_20`
- 量能：`vol_z_20`、`vol_ma_ratio`、`quote_vol_z_20`、`obv_slope_20`、`vwap_dev_20`
- 形态/时间：`close_pos_in_bar`、`upper/lower_shadow`、`body_ratio`、`gap`、
  `consec_up/consec_dn`、`hour_sin/cos`、`dow_sin/cos`
- 订单流（欧易缺字段 → 自动跳过）：`taker_buy_*`、`trades_z_20`、`avg_trade_size*`

**筛选**（`src/factor_select.py`，仅在 train 上，需求5、9、13）：
1. 逐因子 IC（Spearman 秩相关，对异常值稳健）；
2. `|IC| ≥ IC_MIN_ABS(0.01)` 过滤；多头取 `IC>0`、空头取 `IC<0`（方向性，而非取反）；
3. **相关性阈值贪心去冗余**：候选按 `|IC|` 降序扫描，与任一已保留因子
   `|corr| ≥ DEDUP_MAX_CORR(0.90)` 即丢弃（只是更强因子的同义复制）。
   因子数量由相关性结构决定，**不设固定上限**。
   （旧方案用固定簇数 KMeans，`k=min(20, 候选数//3)` 会把数量硬压到候选的 1/3，
   实测多头 32→10、空头 13→4，与信号强弱无关。）
- 多头/空头**各自筛选** → 得到不同因子集（需求13）；本次结果：多头 19 个、空头 12 个。

## 4. 标签（需求3）
`src/factors.py: add_label`，预测目标 = **未来 3 根（12 小时）对数收益 / (ATR/close)**。
除以 ATR 是为消除「训练段大牛市 vs 验证段熊市」的制度差异，避免模型学成「永远做多」；
且执行层止盈止损本就是 ATR 倍数，标签与执行同量纲。

## 5. 时序折叠（需求6）
`src/cv.py: PurgedKFold`（López de Prado）：
- 按时间切 5 个连续折；每折验证时，从训练集挖掉与验证集**标签窗重叠**的样本（purge=HORIZON 根），
  并在验证区后侧再挖 `embargo=12` 根；
- 用途：`models.train_side` 在 train 内做 K 折训练并按折**集成**，防止单次划分的不稳定。
- 单元测试 `test_purged_kfold_no_leak` 校验无重叠、标签窗与 embargo 均被挖空。

## 6. 模型层（需求3、13）
`src/models.py`：多头/空头**两个独立 LightGBM 回归模型**，各自使用**自己的筛选因子**。
- 超参候选（`MODEL_GRID`，刻意很少以抑制多重比较）：`base / shallow / deep_slow`；
- 每折用早停确定轮数，最终以多折**集成平均**预测。

## 7. 执行层（需求8、11、12、13）
`src/execution.py`：信号 → 成交的模拟。
- 资金：初始 **1000 USDT**，每笔名义 **100 USDT**，每方向同时最多 1 笔（需求11）。
- 成本：手续费**单边 5bp，双边 10bp**（`FEE_RATE=0.0005`, `SLIP_RATE=0`，需求11）。
- 入场：bar t 收盘出信号 → **bar t+1 开盘**成交；阈值取预测值分位对应的**绝对阈值**。
- 出场：止盈/止损（ATR 倍数），同根同时触发按**止损**（保守）；最长持有 12 根后按收盘平仓。
- 多空**各自一套**执行参数，独立优化（需求13）。
- **冻结阈值**：OOF 优化出的绝对阈值被冻结，OOC 直接套用，**不用 OOC 自身分布取分位**
  （否则等于偷看 OOC），见 `signal_from_threshold` 与 `test_frozen_threshold_no_ooc_peek`。

## 8. 优化（需求7、8、9）
`src/optimize.py: optimize_full_on_oof` —— **只在 OOF 上**优化，分内外两层：
1. **外层**：去冗余阈值 × 模型候选。每档去冗余阈值都会用 **train** 段重新筛因子、
   重新训练模型，因此"因子集松紧"本身也是被 OOF 选出的超参（而非写死的 0.90）；
2. **内层**：阈值分位 × 止盈 × 止损 × **最长持有期**；
3. 组合网格 = 去冗余 4 × 模型 3 × 阈值分位 5 × 止盈 6 × 止损 6 × 持有期 3
   = **6,480 组/方向**（旧版为 225 组）；
4. 在 **OOF** 上评估每组，得到收益/夏普/卡玛/回撤/胜率/笔数；
5. 选优：成交数 ≥ `OOF_MIN_TRADES(20)` 且止盈率 ≥ `MIN_TP_RATE(0.15)`（淘汰退化参数），
   再按 **收益优先、夏普次之**（`OBJECTIVE_PRIMARY/SECONDARY`）排序取第一；
6. 冻结 (去冗余阈值, 模型, 阈值, 止盈, 止损, 最长持有期)，OOF 评估结果与网格最优行
   以 assert 校验一致。
- OOC 段**只做一次观察评估**，不参与任何选择。
- 代价：OOF 上比较次数由 225 升至 6,480 组/方向，选优噪声随之上升；
  OOC 因此更关键 —— 它才是唯一未被污染的检验。

## 9. 输出（需求10）
`reports/` 下输出：
- `equity_curve.png`：OOF（优化段）与 OOC（观察段）的多头/空头/合计收益曲线；
- `backtest_report.md`、`metrics.json`：总收益、**胜率、盈亏比、夏普、卡玛、最大回撤**、
  成交数、出场构成、买入持有基准；
- `trades_*.csv`、`oof_grid_*.csv`、`factor_ic.csv`、`selected_factors.json`、
  `importance_*.csv`、`consistency_report.md`。

## 10. 一致性保证（需求15）
- 所有可调参数集中于 `config.py`，实现只读 config，杜绝硬编码漂移；
- `tests/test_consistency.py`（15 个用例）覆盖：切分完整性、因子无未来函数、
  订单流因子自适应、趋势因子存在、标签 ATR 口径、时序折叠无泄漏、
  执行层次根开盘/止盈止损/保守假设/成本/权益核算、冻结阈值不偷看、多空因子方向、
  模型确定性、资金口径；
- 运行时 `consistency_report.md` 复核：切分比例与不重叠、训练/优化用段、OOC 阈值来源、
  成本口径、本金口径、多空因子集不同、OOF 冻结评估与网格最优一致。

## 已知结果与局限（诚实披露）
本次运行（2020-09-30 ~ 2026-09-30，13128 根 4h K线）：
- 段基准（买入持有）：train **+7915.94%**（大牛市），OOF **-28.54%**，OOC **-27.08%**（均深熊）。
- 多头（模型 base，阈值分位 0.50，止盈 3.0 ATR，止损 1.5 ATR）：
  train +76.55%/夏普3.25；**OOF +3.39%/夏普1.10（优化目标段）**；OOC **-5.07%/夏普 -2.02**。
- 空头（模型 shallow，阈值分位 0.60，止盈 2.5 ATR，止损 2.5 ATR）：
  train -49.82%/夏普 -2.26（训练段为牛市，做空在样本内亏损属正常）；
  **OOF +7.13%/夏普2.04**；OOC **-2.31%/夏普 -0.82**。
- 局限：
  1. **优化在 OOF 上进行**（用户明确要求），OOF 上比较 225 组/方向，最优解含噪声；
     OOC 是唯一未被选择污染段，其上多头/空头均为负 —— 说明 OOF 的优良表现**未能外推**，
     这是本次最诚实的结论，未做任何美化。
  2. 欧易数据无订单流字段，因子数较币安口径少 5 个，可能损失部分截面信息。
  3. 标签 12 小时、最长持有 48 小时；加密 4h 级别信噪比低，成本（双边 10bp）对高频交易侵蚀明显。
  4. 样本段制度差异极大（train 大牛市 vs OOF/OOC 熊市），标签虽做了 ATR 标准化，
     但模型仍可能在方向上带有制度偏好。
