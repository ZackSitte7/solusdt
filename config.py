# -*- coding: utf-8 -*-
"""全局配置 —— 路径 / 数据规格 / 切分 / 成本 / 回测约束。

设计原则:
  - 本文件**不含任何密钥**。数据源是欧易(OKX)/币安(Binance)公开行情接口, 无需认证。
  - 所有会随实验调整的参数集中在此, 保证「设计与运行的同一性」:
    脚本只从 config 读参数, 不允许在业务代码里散落魔法数字。
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- 路径
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"
DATA_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# 主数据源: "binance" | "okx"。默认币安 —— 字段更全(含 trades/taker_buy_*,
# 下游 factors.py 依赖), 历史更早, 且本沙箱可直连。
SOURCE = "binance"

RAW_PARQUET = DATA_DIR / "solusdt_4h_raw.parquet"      # 原始抓取(当前主数据源产物)
CLEAN_PARQUET = DATA_DIR / "solusdt_4h_clean.parquet"  # 清洗后
PRED_PARQUET = DATA_DIR / "solusdt_4h_pred.parquet"    # 含预测与回测结果

# ---------------------------------------------------------------- 数据源
# 欧易(OKX)公开行情接口(无需 API Key)。
# 注意: 本沙箱网络走白名单, www.okx.com 不可达(连接被拒); 数据由外网主机抓取后,
# 以原始 JSON 转存到 data/solusdt_4h_okx_raw.json, 再离线导入(见 src/data_fetch.py)。
OKX_BASE = "https://www.okx.com"
OKX_INST = "SOL-USDT"        # OKX 现货交易对
OKX_BAR = "4H"               # OKX K线周期(4 小时)
OKX_LIMIT = 100              # history-candles 单次请求上限(OKX 硬限制)
OKX_DUMP = DATA_DIR / "solusdt_4h_okx_raw.json"   # 外网主机抓取的原始 JSON
OKX_RAW_PARQUET = DATA_DIR / "solusdt_4h_raw_okx.parquet"  # 欧易规整后(离线导入产物)

# 币安(Binance)官方公开数据镜像(无需 API Key; 沙箱白名单内, **可直连**)。
# 字段比 OKX 更全(含 trades / taker_buy_*), 历史更早(SOL 上市 2020-08-11 起)。
# 注意: api.binance.com 在本环境不可达, 只有 data-api.binance.vision 可达。
BINANCE_BASE = "https://data-api.binance.vision"
KLINES_LIMIT = 1000          # 单次请求上限(Binance 硬限制)
BINANCE_RAW_PARQUET = DATA_DIR / "solusdt_4h_raw_binance.parquet"  # 币安原始抓取

# ---------------------------------------------------------------- 本次回测数据源(需求1)
# 本次回测使用**欧易(OKX)** 4h 数据, 入口脚本 scripts/backtest/run_backtest_okx.py。
# 注意: 欧易公开 K 线**不含** trades / taker_buy_base / taker_buy_quote 订单流字段,
# 因子库会自动跳过依赖这些字段的因子(见 src/factors.py ORDER_FLOW_COLS)。
BACKTEST_SOURCE = "okx"
BACKTEST_RAW = OKX_RAW_PARQUET                              # 欧易原始(规整后 Parquet)
BACKTEST_CLEAN = DATA_DIR / "solusdt_4h_clean_okx.parquet"  # 欧易清洗后

SYMBOL = "SOLUSDT"
INTERVAL = "4h"
REQUEST_SLEEP = 0.2          # 请求间隔(秒), 礼貌限速
REQUEST_TIMEOUT = 25
REQUEST_RETRY = 4            # 单次请求重试次数

# ---------------------------------------------------------------- 清洗
# 4h 单根最大允许振幅(high/low-1)。超过视为数据错误(交易所故障/脏数据), 剔除。
MAX_BAR_RANGE = 0.35
# 单根最大允许 |收益|, 超过视为异常跳变, 剔除(加密市场 4h 极端波动罕见 >35%)
MAX_BAR_RET = 0.35
# 缺失 K 线容忍: 连续缺失超过该根数则视为数据缺口, 在报告中告警
MAX_GAP_BARS = 3

# ---------------------------------------------------------------- 数据切分(按时间顺序)
# 前 70% 训练 / 中间 15% OOF(样本外优化) / 最后 15% OOC(只观察, 绝不用于选择)
TRAIN_FRAC = 0.70
OOF_FRAC = 0.15
OOC_FRAC = 0.15

# ---------------------------------------------------------------- 标签
# 预测目标: 未来 HORIZON 根的**对数收益**(回归)
# 取 3 根(4h*3 = 12 小时): 与执行层最长持有(12 根)同一量级,
# 使「预测期」与「持仓期」尺度匹配; 用 1 根会因信噪比过低而难以学到东西。
HORIZON = 3

# 标签口径: "atr" = 未来对数收益 / (ATR/close), 即**以 ATR 为单位的未来涨跌**。
# 为什么必须标准化: 训练段(2020-2024)是 +7941% 大牛市, 而验证段是 -18.7% / -40.4% 熊市。
# 用原始收益当标签, 模型会学成「永远做多」这一制度特征; 除以 ATR 后标签变成
# 「相对自身波动幅度的涨跌倍数」, 消除了波动率与漂移的制度差异。
# 且执行层的止盈止损本就是 ATR 倍数 -> 标签与执行同一量纲, 阈值可直接解释。
LABEL_MODE = "atr"          # "atr" | "raw"
LABEL_IS_LOG_RET = True

# ---------------------------------------------------------------- 时序交叉验证(金融折叠)
# Purged K-Fold + Embargo: 训练集与验证集之间挖掉重叠与缓冲, 防标签泄漏。
CV_N_SPLITS = 5
CV_EMBARGO_BARS = 12         # 验证集前后各挖掉多少根(>=标签跨度, 防重叠泄漏)
NESTED_FOLDS = 5             # 嵌套 CV: 在训练段内部做外层折, 用于选模型/执行参数

# ---------------------------------------------------------------- 因子筛选
IC_MIN_ABS = 0.01            # |IC| 低于此值的因子剔除
IC_WINDOW = 0                # 0=用全训练集算 IC; >0 则用滚动窗口
# 去冗余(相关性阈值贪心): 候选按 |IC| 降序扫描, 与任一已保留因子 |corr| >= 该值即丢弃。
# 取代原先的「固定簇数 KMeans」—— 原方案 k=min(20, 候选数//3) 会把因子数硬压到候选的
# 1/3(实测多头 32->10、空头 13->4), 数量与信号强弱无关。阈值化后数量由相关性结构决定。
DEDUP_MAX_CORR = 0.90

# ---------------------------------------------------------------- LightGBM(多空分开)
# 多头模型与空头模型**分别训练、分别筛因子、分别调参**(见 models.py)
LGBM_PARAMS_COMMON = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=60,
    feature_fraction=0.7,
    bagging_fraction=0.7,
    bagging_freq=1,
    lambda_l1=0.0,
    lambda_l2=1.0,
    verbosity=-1,
    n_jobs=-1,
    seed=42,
)
LGBM_ROUNDS = 1500
LGBM_EARLY_STOP = 100

# ---------------------------------------------------------------- 成本(单边, 需求11)
# 需求11: 手续费**单边万分之五(5bp), 双边千分之一(10bp = 开+平各 5bp)**。
# 规格未要求滑点, 故 SLIP_RATE 置 0; 若需保守假设可调回 0.0005。
FEE_RATE = 0.0005            # 单边手续费 5 bp
SLIP_RATE = 0.0              # 单边滑点(本规格不启用)

# ---------------------------------------------------------------- 回测
INIT_CAPITAL = 1000.0        # 初始本金(USDT)
TRADE_NOTIONAL = 100.0       # 每次开仓名义金额(USDT) —— 使用者明确要求
MAX_CONCURRENT = 1           # 同时最多持仓笔数

# 止盈止损候选网格(按 ATR 倍数表达, 避免固定百分点在不同波动率制度下失效)
ATR_WINDOW = 14
# 网格已放宽: 加入 tp=4ATR / sl=0.5ATR 两端, 阈值分位加密到 5 档。
# 退化组合由 MIN_TP_RATE(止盈率) 与 OOF_MIN_TRADES(笔数) 两道门槛淘汰, 不会因放宽而入选。
TP_ATR_GRID = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
SL_ATR_GRID = (0.5, 0.75, 1.0, 1.5, 2.0, 2.5)
EXEC_THR_GRID = (0.4, 0.5, 0.6, 0.7, 0.8)   # 信号阈值取预测值的分位
MAX_HOLD_BARS = 12           # 最长持有根数(4h * 12 = 48h), 到期按收盘平仓
# 退化参数淘汰线: 止盈率低于此值说明「止盈」并未真正工作, 该组合作废
MIN_TP_RATE = 0.15

# ---------------------------------------------------------------- OOF 优化(需求7/8/9)
# **只在 OOF 上**优化模型层(超参候选)与执行层(阈值/止盈/止损); OOC 全程只观察。
OOF_MIN_TRADES = 20          # 单组合 OOF 最少成交笔数(防止选到"几乎不交易"的退化参数)

# 外层扫描(同样**只在 OOF 上**选优): 去冗余阈值 × 最长持有期。
# 这两个参数原先被写死(0.90 / 12), 但它们直接决定「因子集」与「持仓上限」, 属一阶参数,
# 因此一并交给 OOF 选优。每档都会用 train 段重新筛因子并重新训练模型。
OOF_DEDUP_CORR_GRID = (0.80, 0.85, 0.90, 0.95)
OOF_HOLD_GRID = (6, 12, 24)

# ---------------------------------------------------------------- 目标函数
# 只在 OOF 上优化。主目标为**收益**, 次目标为**夏普**(见 src/optimize.py optimize_full_on_oof)。
# 为什么收益优先: 多头的最优点收益与夏普同时最优(帕累托占优), 次序无所谓;
# 但空头存在真实取舍(收益 +10.44%/夏普 3.11 vs 收益 +9.15%/夏普 3.23), 按收益优先取前者。
OBJECTIVE_PRIMARY = "total_return"
OBJECTIVE_SECONDARY = "sharpe"

# ================================================================ 抗过拟合选择协议 (v2)
# 入口脚本 scripts/backtest/run_backtest_v2.py。设计依据是对 v1 的亏损归因(scripts/analysis/analyze_trades.py), 逐条对症:
#   病症1 选择偏差: v1 在 OOF 上比较 6480 组, 而 OOF 只有 1969 根/~175 笔, 每笔毛收益
#         t=+1.53 -> 统计上无法区分。多头 4765 组里仅 10.9% 为正, 选中的 +7.12% 是 +3.53σ 孤点。
#   药方1 把选择集从 OOF 移回 train 内部的**前推折**(WF_FOLDS), 选择样本 1969 -> 6892 根;
#         OOF 与 OOC 一并降级为纯检验段(比需求9更严: 连 OOF 也不参与选择)。
#   病症2 选尖峰: 各折取均值会奖励"某一折暴利"的组合。
#   药方2 SELECT_RULE="min_fold" -> 以**最差折**的收益为准, 选折间稳定的。
#   病症3 系统性偏好最多因子: v1 收益前 20 组的因子数全是 25(最宽去冗余)。
#   药方3 因子数硬上限 MAX_FACTORS_PER_SIDE, 去冗余阈值固定不再作旋钮。
#   病症4 因子排序用 |IC|, 高 |IC| 常是单期噪声尖峰。
#   药方4 改用 |ICIR|(按 ICIR_BLOCKS 个时间块算 IC 的均值/标准差)排序。
#   病症5 long OOC 净亏 72.26 里 96% 来自"超时"桶(-69.63): hold=6 太短, 72% 仓位
#         没走出方向就按收盘平仓。
#   药方5 执行层加 ATR 跟踪止损(PROTO_TRAIL_GRID), 并放宽 max_hold。
#   病症6 v1 的阈值取自**评估段自身**的预测分位, 等于看了评估段的分布。
#   药方6 阈值一律冻结为**选择阶段样本外预测**的绝对分位, OOF/OOC 只读不调。

SELECT_ON = "train_wf"           # "train_wf"(默认) | "oof"(退回 v1 行为, 严格照原需求9)
WF_FOLDS = 3                     # train 内部前推折数: 第 k 折用前 k/(F+1) 训练、下一段验证
WF_EMBARGO = 12                  # 前缀与验证块之间再挖掉的根数(叠加 HORIZON 的 purge)
SELECT_RULE = "min_fold"         # "min_fold"(最差折, 默认) | "mean"
PROTO_MIN_TRADES_PER_FOLD = 15   # 单折最少成交笔数, 低于此该组合作废

# 因子(药方3/4)
MAX_FACTORS_PER_SIDE = 10        # 因子数硬上限 —— 直接针对"偏好最多因子"的过拟合特征
FACTOR_RANK = "icir"             # "icir"(默认) | "ic"
ICIR_MIN_ABS = 0.10              # |ICIR| 下限
ICIR_BLOCKS = 6                  # 计算 ICIR 的时间分块数
DEDUP_MAX_CORR_V2 = 0.85         # 去冗余阈值固定(不再作为可调旋钮, 少一层选择)

# 预注册候选网格(药方1: 收缩到两位数, 抑制多重比较)
# 2 模型 × 2 阈值分位 × 2 止盈 × 2 止损 × 2 持有 × 2 跟踪止损 = 64 组/方向
PROTO_MODELS = ("base", "shallow")
PROTO_THR_GRID = (0.5, 0.7)
PROTO_TP_GRID = (2.0, 3.0)
PROTO_SL_GRID = (1.5, 2.5)
PROTO_HOLD_GRID = (12, 24)
PROTO_TRAIL_GRID = (0.0, 2.0)    # 0 = 关闭跟踪止损(v1 行为); 2.0 = 2×ATR 跟踪

# OOC 置信区间(仅用于**观察**段的诚实披露, 不参与任何选择)
BOOT_N = 2000                    # 自助抽样次数
BOOT_BLOCK = 5                   # 分块自助的块长(笔), 保留交易间的时序依赖
BOOT_SEED = 20260930

# ================================================================ v3: OOF 选优 + ATR 止盈止损(tp>sl)
# 入口脚本 scripts/backtest/run_backtest_v3.py。用户明确要求, 相对 v1/v2 的差异只有两点:
#   1. 选择集回到 **OOF**(与 v1 同口径, 以把 OOF 收益做回 v1 水准), OOC 仍**只观察**;
#   2. 止盈/止损的**硬约束 tp > sl**(风险报酬比 > 1) —— 只保留止盈幅度大于止损幅度的
#      组合, 提高单笔盈亏比、削掉"止盈比止损还近"的退化配置, 从而改善夏普。
# 其余口径(数据/因子/标签/成本/资金/指标/目标函数)与 v1 完全一致, 保证可比。
V3_ENFORCE_TP_GT_SL = True       # True = 只扫描 tp_mult > sl_mult 的止盈止损组合
V3_OUT_DIR = "v3"                # 产出目录 reports/v3

# ================================================================ v4: VIF 迭代剪枝(治理共线过拟合)
# 入口脚本 scripts/backtest/run_backtest_v4.py。依据 scripts/analysis/analyze_factor_dim.py 的诊断:
#   long 名义 25 个因子, 但有效维度(参与率)仅 4.3、最大 VIF 71.5 —— 多出来的维度是近共线的
#   噪声方向, 交给 LGBM 只会被拟合。pairwise 去冗余(|corr|>=0.95)挡不住**多元共线**, 故引入
#   方差膨胀因子(VIF)迭代剪枝: 每次删除 VIF 最大的因子, 直至全部 VIF <= V4_MAX_VIF。
# 其余口径与 v3 完全一致(OOF 选优 + ATR 止盈止损 + tp>sl), 便于直接对比。
V4_MAX_VIF = 10.0                # VIF 上限(经验阈值: >10 视为严重共线)
V4_MIN_FACTORS = 5               # 剪枝后至少保留的因子数(安全下限, 避免剪成单因子)
V4_OUT_DIR = "v4"                # 产出目录 reports/v4

# ================================================================ v5: 制度门控(治理制度错配)
# 入口脚本 scripts/backtest/run_backtest_v5.py。依据 v4 结论: OOC 亏损主因不是因子冗余, 而是 train(牛市)
# 与 OOC(熊市)的制度错配。做法: 用**只用过去信息**的均线制度做开关, 只在顺势制度里开仓。
#   规则集(预注册, 直接进入 OOF 选优):
#     none         : 不门控(回落 v4 行为)
#     sma100       : long 需 close>MA100; short 需 close<MA100
#     sma200       : long 需 close>MA200; short 需 close<MA200
#     sma200_slope : 在上者基础上再要求 MA200 本身朝该方向倾斜(过滤假突破)
# 门控只作用于**信号 bar**, 成交仍在 t+1 开盘; 规则同样只在 OOF 上择优, OOC 仅观察。
V5_REGIME_GRID = ("none", "sma100", "sma200", "sma200_slope")
V5_REGIME_MA = {"sma100": 100, "sma200": 200, "sma200_slope": 200}
V5_SLOPE_LOOKBACK = 12           # 均线倾斜的观察窗(根)
V5_OUT_DIR = "v5"                # 产出目录 reports/v5

# ================================================================ v6: 扩充因子库 + 更细阈值 + 跟踪止损 + 波动率制度
# 入口脚本 scripts/backtest/run_backtest_v6.py。相对 v5 只动四处, 其余口径(切分/成本/目标函数/选择纪律)完全一致:
#   1. **因子库扩充**(src/factors.py): 在原 61 个因子之外补足「波动率结构 / 高阶矩 / 自相关 /
#      趋势强度(Aroon/Vortex/Keltner) / 量能资金流(MFI/CMF/Amihud) / 形态统计」等方向,
#      因子总数增至 115 —— 给 IC/ICIR 筛选更多**正交**候选, 提高模型上限。
#   2. **阈值分位加密**: 上限放宽到 0.90(v5 最优常顶在网格上边界, 更选择性的方向没被搜到)。
#   3. **执行网格加入 ATR 跟踪止损**: 让走不出方向的仓位被截断、走出方向的仓位继续持有。
#   4. **制度规则加入 atr_pct(低波动率门控)**: 只在 ATR% 处于过去 V6_VOL_WIN 根低分位时开仓。
# 选择仍**只发生在 OOF**; OOC 全程只观察。
# 另: 按用户要求, 回测数据起点对齐到 2021-10-01(此前的 2020-09~2021-09 段不参与)。
DATA_START = "2021-10-01"        # 数据起始日期(UTC), 早于该日的 bar 全部丢弃

V6_DEDUP_GRID = (0.80, 0.85, 0.90)          # 去冗余阈值(v5 中 0.95 从未入选, 收敛)
V6_THR_GRID = (0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.90)   # 阈值分位(上限放宽到 0.90)
V6_HOLD_GRID = (12, 24)                     # 最长持有根数
V6_TRAIL_GRID = (0.0, 1.5, 2.5)             # ATR 跟踪止损倍数; 0 = 关闭(回落 v5 行为)
V6_REGIME_GRID = ("none", "sma100", "sma200_slope", "atr_pct")
V6_VOL_WIN = 240                            # 波动率分位的回看窗(根)
V6_VOL_MIN = 60                             # 波动率分位的最少样本
V6_VOL_Q = 0.70                             # 只放行 ATR% <= 该分位的 bar
V6_OUT_DIR = "v6"                           # 产出目录 reports/v6

# 因子池(v6 新增一个 OOF 维度): 扩充后因子库 115 个, 但"多"不等于"好" —— 对某些方向,
# 新因子可能挤掉原有更有效的因子。故把"候选池"本身作为一个**预注册超参**交给 OOF 择优:
#   "core"     : v5 的原有 61 因子(横向对比的基线池)
#   "expanded" : 扩充后的 115 因子
# 每个池各自做 IC/ICIR 筛选 + 去冗余 + VIF 剪枝, 各自训练。OOF 上哪个池好就用哪个池。
V6_FACTOR_POOLS = ("core", "expanded")
# v5 原有因子名(用于构造 "core" 池; 与 factors.py 中同名因子定义完全一致)。
V6_CORE_FACTORS = (
    "ret_1", "ret_2", "ret_3", "ret_6", "ret_12", "ret_24", "ret_48",
    "roc_6", "roc_12", "roc_24", "mom_accel",
    "ma_ratio_20", "ma_ratio_50", "ma_ratio_100", "ma_ratio_200",
    "ma_slope_20", "ma_slope_50", "ma_align", "macd", "macd_signal", "macd_hist",
    "adx_14", "trend_strength_50", "lr_slope_20", "lr_slope_50",
    "pos_in_range_20", "pos_in_range_50", "pos_in_range_100", "hh_count_20", "ll_count_20",
    "atr_ratio", "atr_ratio_chg", "rv_12", "rv_20", "rv_50", "vol_ratio_5_20",
    "bb_width_20", "bb_pos_20", "parkinson_20", "range_ratio",
    "rsi_14", "rsi_28", "rsi_div", "stoch_k_14", "cci_20",
    "vol_z_20", "vol_ma_ratio", "quote_vol_z_20", "obv_slope_20", "vwap_dev_20",
    "close_pos_in_bar", "upper_shadow", "lower_shadow", "body_ratio", "gap",
    "consec_up", "consec_dn", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
)

# ================================================================ v7: 空头专用优化
# 入口脚本 scripts/backtest/run_backtest_v7.py。依据对 v6 的归因 —— 空头在 OOC 由 +5.94% 崩到 -0.85%,
# 且分月看是"下跌市里空头亏钱"的结构性失败。v7 **只重做空头**, 多头沿用 v6 冻结配置:
#   1. **空头专用因子**(src/factors.py): 下行动量分解、破位/支撑阻力距离、结构走弱
#      (高点和低点同时降低)、相对自身历史的超卖、跳空低开统计 —— 都是"空头逻辑"专属,
#      多头用不到; 因子库 115 -> 126。
#   2. **空头专用制度**: sma_align(空头排列) / ema_bear / breakdown_20(破 20 根新低) /
#      bear_vol(空头排列 + 低波动); 制度网格 4 -> 8。
#   3. **止损下探到 0.35×ATR**: v6 的 OOF 把 0.5 选成最优(网格下界), 说明真实最优可能在
#      更紧的一侧 —— 把边界让开, 由 OOF 自己判断, 而不是由网格边界替它决定。
# 选择仍**只发生在 OOF**; OOC 全程只观察。
V7_SL_GRID = (0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5)      # 止损网格(下探到 0.35)
V7_REGIME_GRID = ("none", "sma100", "sma200_slope", "atr_pct",
                  "sma_align", "ema_bear", "breakdown_20", "bear_vol")
V7_OUT_DIR = "v7"                           # 产出目录 reports/v7
V7_ALIGN_SMA = (20, 50, 200)                # 空头排列用的均线组: close < MA20 < MA50 < MA200
V7_EMA_PAIR = (21, 55)                      # ema_bear: close < EMA21 < EMA55
V7_BREAKDOWN_WIN = 20                       # breakdown_20: 跌破过去 20 根的最低价

# ================================================================ v8: OOF 内部分折 + 稳健选优
# 入口脚本 scripts/backtest/run_backtest_v8.py。依据对 v7 的归因: v7 空头 OOF +6.31% 而 OOC 仅 +1.11%,
# 且 selection_grid_short.csv 在 OOF 上比较了 193,536 组 —— "最优"只是 19 万个含噪估计的
# **最大值**(winner's curse, 天然高估); v7 的 5 个执行参数里更有 3 个落在网格边界。
# v8 **不动选择段(仍是 OOF, OOC 仍只观察)**, 只改"怎么选"这条纪律:
#   1. 把 OOF 段再切成 V8_N_FOLDS 个**连续子折**, 每组配置在每个子折上独立评估
#      (阈值分位按子折自身预测计算; 子折仍在 OOF 内, 不触碰 OOC);
#   2. 稳健得分 = 各折夏普**均值 - V8_ROBUST_K × 标准差**, 次目标为**最差折夏普**,
#      替代原来的"整段 OOF 取最大 total_return" —— 用折间一致性压低 max 效应;
#   3. 单折成交数门槛 V8_MIN_TRADES_PER_FOLD, 淘汰"只在个别子折蒙对"的低频组合。
# 除选择纪律外, 因子/训练/网格/成本/资金/选择段与 v7 完全一致, 便于横向对比。
V8_N_FOLDS = 3                  # OOF 内部连续子折数
V8_ROBUST_K = 1.0               # 稳健得分对"折间标准差"的惩罚系数
V8_MIN_TRADES_PER_FOLD = 15     # 单折最少成交笔数(低于此该组合作废)
V8_OUT_DIR = "v8"               # 产出目录 reports/v8

# ================================================================ v9: long+short 共同优化
# 入口脚本 scripts/backtest/run_backtest_v9.py。用户要求: long / short 只是两个模型, **要一起优化看总收益**;
# 不再像 v7/v8 那样把 long 冻结自 v6, 而是**两侧都重做**, 并在 OOF 上对"组合"联合选优。
#
# 为什么不能直接对"配置长表 × 配置长表"做全枚举: 每侧 24,192 组, 组合数 ~5.8 亿, 不可行。
# 故采用**两阶段联合**(V9_CAND_PER_SIDE 控制规模):
#   阶段1(每侧): 用 v8 同一套 OOF 内部分折纪律, 把每侧的长表聚合成"该侧稳健收益"
#                = 各折收益均值 - V8_ROBUST_K × 标准差, 取前 V9_CAND_PER_SIDE 个候选;
#   阶段2(联合): 对 (long候选 × short候选) 的笛卡尔积, 计算**合并**逐折收益
#                rc = r_long + r_short(v7 的合并口径: 合并权益 = eq_long + eq_short - INIT),
#                以合并稳健分 = mean(rc) - V8_ROBUST_K×std(rc) 为主目标,
#                最差折合并收益(maximin)为次目标, 平均合并收益为三目标。
#
# 为什么"联合"不是"各自最优再相加": 单看收益, 相加是可分的(max 之和 = 和之 max);
# 但折间标准差惩罚 std(r_long + r_short) 依赖两侧在各折上的**搭配**, 通常 != 两侧各自 std 之和,
# 于是两侧真正耦合 —— 联合选优会偏好"两侧不在同一折一起回撤"的组合。
# 选择仍**只发生在 OOF 子折**; OOC 全程只观察。两侧同用 v7 的制度/止损网格。
V9_CAND_PER_SIDE = 200          # 每侧进入联合配对的候选配置数(按该侧"稳健收益"降序)
V9_TOP_PAIRS = 30               # 落盘/报告的联合候选组合数
V9_OUT_DIR = "v9"               # 产出目录 reports/v9

# ================================================================ v10: OOF 收益 + 夏普 双目标
# 入口脚本 scripts/backtest/run_backtest_v10.py。用户要求: **优化 OOF 的收益和夏普**, 并给出 OOF / OOC 总体收益曲线。
# v9 的选择目标里**完全没有夏普**(阶段1 只看该侧稳健收益, 阶段2 只看合并稳健收益);
# v10 保持 v9 的全部选择纪律(仍只在 OOF 子折; OOC 只观察), 只把夏普加进排序:
#   主目标 = 合并稳健收益 = 各折合并收益均值 - V8_ROBUST_K×标准差   (收益优先, 沿用仓库既有约定)
#   次目标 = 合并稳健夏普 = 各折**合并组合**夏普均值 - V8_ROBUST_K×标准差
#   三目标 = 最差折合并收益(maximin)
# 合并夏普按**真实 bar 级**重建: eq_c = eq_long + eq_short - INIT_CAPITAL 后再按 bar 收益年化;
# 不是把两侧夏普相加 —— 两侧交易独立, 组合夏普与单侧夏普不可线性合成。
V10_OUT_DIR = "v10"             # 产出目录 reports/v10
# 阶段1(因子池×制度×去冗余×模型×执行×子折)的网格与 v9 **逐位相同**;
# 若 reports/<该值>/selection_folds_{long,short}.csv 已存在则直接复用(省去约 18 分钟重算),
# 否则脚本自行计算并落盘。设为空字符串 "" 则强制重算。
V10_REUSE_GRID_FROM = "v9"

# ================================================================ v11: 强化选择器(把 OOF 收益+夏普再推高)
# 入口脚本 scripts/backtest/run_backtest_v11.py。**数据窗口(DATA_START=2021-10-01 起)、因子池、训练、
# 执行网格、成本、资金、合并口径、排序目标、折数都与 v10 逐位相同**, 只强化"怎么选"这条纪律:
# v10 的候选池**只按单一目标截断**: 每侧仅按"该侧稳健收益"取前 200, 存在"夏普更高、收益略低"
# 的解在**进入联合配对之前**就被丢掉的风险(注意: 实测在 v9 的 3 折长表上, 前 200 内已含
# 该侧最高稳健夏普, 故这里说的是**风险**而非既成事实)。
# v11 只改一处 —— 候选池改**双目标保留**:
#   1. V11_CAND_PER_SIDE = 500: 候选池扩大(执行网格不动 -> 不新增假设空间, 只少丢好解);
#   2. 入池优先级 = min(稳健收益排名, 稳健夏普排名), 并强制并入 (稳健收益, 稳健夏普) 的
#      **Pareto 前沿**(非支配解)。
# 关键性质: 折数与 v10 **保持 3 折不变**, 目标取值尺度逐位相同 -> v11 的候选池**严格包含**
# v10 的池(v11 ⊇ v10), 于是"选出的配对在稳健收益这一主目标上不会比 v10 差"是选择集上的硬保证。
# (教训: 曾把折数 3 改 5, 会改变目标本身尺度、使该包含性失效, 反而选出更差的配对 —— 已回退。)
# 排序目标不变: 主 = 合并稳健收益, 次 = 合并稳健夏普(真实 bar 级), 三 = 最差折合并收益;
# 选择只发生在 OOF 子折, OOC 全程只观察。
V11_N_FOLDS = 3                 # OOF 内部连续子折数(**与 v10 一致**, 保证候选池包含 v10 池)
V11_CAND_PER_SIDE = 500         # 每侧候选池规模(v10 为 200)
V11_OUT_DIR = "v11"             # 产出目录 reports/v11
# 折数与 v9/v10 相同 -> 阶段1 长表(配置×子折)可直接复用 v9 的网格, 秒级完成。
V11_REUSE_GRID_FROM = "v9"

# ================================================================ v12: 直接以真实 OOF 收益+夏普为排序目标
# 入口脚本 scripts/backtest/run_backtest_v12.py。v11 的实测结论: "逐折稳健收益"这一目标与**真实 OOF 段收益脱钩** ——
# v11(3 折)的折稳健收益略高于 v10, 但真实 OOF 段收益反而从 8.90% 降到 4.79%。即:
# 在 OOF 子折上做"均值 - K×标准差"的稳健排序, 并不能可靠地抬升最终要看的 OOF 段数字。
# 用户明确选择: **直接以真实 OOF 段的合并收益与夏普为排序目标**(仍在 OOF 内选择, OOC 全程不碰)。
# v12 的排序目标(字典序):
#   主 = 真实 OOF 段**合并收益**(两侧冻结阈值下逐 bar 重建的真实值, 非折均值)
#   次 = 真实 OOF 段**合并夏普**(同上, bar 级)
#   三 = 最差折合并收益(maximin, 仍是 OOF 内子折, 作为"跨折不塌"的兜底)
# 候选池沿用 v11 的双目标保留 + Pareto 前沿(每侧 V12_CAND_PER_SIDE 个), 折数 3 与 v10 一致。
# 口径提醒: 选出来的就是"在 OOF 段上收益/夏普最高的那一对", 对 OOF 的选择压力比 v10/v11 更大;
# OOC 仍**只观察**, 不参与任何一步。冻结阈值仍按**整段 OOF 分位**计算(与 v6~v11 同口径)。
V12_N_FOLDS = 3                 # 与 v10/v11 一致的 OOF 子折数(仅用于"最差折"兜底项)
V12_CAND_PER_SIDE = 500         # 每侧候选池规模(沿用 v11 的双目标 + Pareto 池)
V12_OUT_DIR = "v12"             # 产出目录 reports/v12
V12_REUSE_GRID_FROM = "v9"      # 阶段1 长表与 v9 结构一致, 直接复用

# ================================================================ v13: 在 v5 口径上优化 OOF 收益与夏普
# 入口脚本 scripts/backtest/run_backtest_v13.py。基线是 **v5**(制度门控 + VIF 迭代剪枝 + OOF 选优 +
# ATR 止盈止损且 tp>sl; 因子库/执行网格/成本/资金/切分比例与 v5 完全一致), 只改两处:
#   1. **数据窗口对齐 DATA_START=2021-10-01** —— v5 当时的 load_clean() 未做此对齐
#      (起点对齐是 v6 才引入的), 因此 v13 与 v5 的**数据窗口不同**, 这是用户明确要求的口径变化。
#   2. **选择规则改为"收益锚 + 夏普择优"** —— v5 的规则是 (收益, 夏普) 严格字典序, 夏普只在
#      收益完全相等时才起作用, 实际上等于"只看收益"。v13 先取 OOF 收益最高的配置为**收益锚**,
#      再在「OOF 收益 >= 锚收益 - V13_RET_DROP」的**有界让步带**内取 OOF 夏普最高者:
#          V13_RET_DROP = 0 时逐位退化为 v5 的字典序;
#          V13_RET_DROP > 0 时用**上限明确**的少量收益让步换更高夏普 -> **同时**优化收益与夏普。
# 约束: 让步带用**绝对值**而非比例 —— 收益可能为负, 按比例(×0.98)会给出错误方向的门槛。
# 选择仍**只在 OOF**(真实 OOF 段, 与 v5 同口径, 不用子折); OOC 全程只观察。
# 取值依据(实测): 各制度规则的最优 OOF 收益彼此相差 1~2 个百分点。0.5pp 的让步带**够不到**
# 其他制度规则, 会让本规则逐位退化为 v5 的"只看收益"; 1.5pp 才让 long 侧真正做一次取舍
# (收益 +7.32%/夏普 2.73  ->  +6.59%/夏普 3.72, 让出 0.73pp 换 +0.99 夏普), short 侧不变。
V13_RET_DROP = 0.015            # 允许让出的 OOF 收益上限(绝对值, 1.5 个百分点)
V13_OUT_DIR = "v13"             # 产出目录 reports/v13

# ---- v13 多头专用扩展(只作用于 long; short 逐位不变) ----
# 依据: v13 的 long 帕累托前沿 6 个点**全部贴在网格边界上** —— max_hold 5/6 顶到上界 24、
# sl 3/6 顶到 2.5、tp 2/6 顶到 4.0、dedup 2/6 顶到 0.95, 而 thr_q 6/6 贴在下界 0.4。
# 是网格边界而不是数据在决定 long 的最优解, 且"当前点已是当前空间内的帕累托最优"(支配检验 0 个)。
# 因此 v13 把 long 的这几条轴整体让开, 并把模型层与因子层也一并专属化, 由 OOF 自己重选:
#   1. 执行网格外扩(下/上界各让开一档);
#   2. 模型层候选 3 -> 5(补"更深更强正则"与"更慢更平滑"两个方向);
#   3. 因子层新增 7 个**多头专用因子**(上涨结构走强/突破/相对超买/跳空高开/上涨效率),
#      并在 short 侧显式排除, 保证空头口径不变。
V13_LONG_DEDUP_GRID = (0.80, 0.85, 0.90, 0.95, 0.97)
V13_LONG_THR_GRID = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
V13_LONG_TP_GRID = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)
V13_LONG_SL_GRID = (0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0)
V13_LONG_HOLD_GRID = (6, 12, 24, 36, 48)
V13_LONG_FACTORS = ("higher_high_20", "higher_low_20", "bull_persist", "break_res_20",
                    "rsi_bull_z_50", "gap_up_20", "up_eff_20")
# 模型层候选(多头)。前 3 个与 src/models.py 的 MODEL_GRID 逐位相同(便于对照), 后 2 个为新增:
#   deep_reg: 更深(127 叶) + 更强 L1/L2 正则 + 更大叶最小样本  -> 抓更复杂的上涨结构但不过拟合
#   smooth  : 学习率 0.01 + 200 最小叶样本 + 强 L2          -> 更慢更平滑的拟合
V13_LONG_MODEL_GRID = (
    dict(name="base", learning_rate=0.03, num_leaves=31, feature_fraction=0.7,
         min_child_samples=60, lambda_l2=1.0),
    dict(name="shallow", learning_rate=0.03, num_leaves=15, feature_fraction=0.6,
         min_child_samples=100, lambda_l2=5.0),
    dict(name="deep_slow", learning_rate=0.015, num_leaves=63, feature_fraction=0.5,
         min_child_samples=80, lambda_l2=10.0),
    dict(name="deep_reg", learning_rate=0.015, num_leaves=127, feature_fraction=0.4,
         min_child_samples=150, lambda_l1=1.0, lambda_l2=20.0),
    dict(name="smooth", learning_rate=0.01, num_leaves=31, feature_fraction=0.5,
         min_child_samples=200, lambda_l2=15.0),
)

# ================================================================ v14: long 趋势因子 + 模型层/执行层再优化
# 入口脚本 scripts/backtest/run_backtest_v14.py。基线 = v13(其 long 多头专属扩展版), 用户要求:
#   1. **为 long 补趋势类因子** —— 在因子库已有的趋势因子之外, 补 5 个**新的趋势维度**:
#      趋势质量(LR-R² × 斜率符号)、Kaufman 效率比、多周期斜率同向度、趋势加速度、ADX 斜率;
#   2. **优化 long 模型** + **优化模型层与执行层** —— 模型候选与执行网格继续外扩, 由 OOF 重选;
#   3. 目标仍是 **long 的 OOF 收益率与夏普**(收益锚 + 夏普择优); **OOC 仍只观察**; short 逐位不变。
# 依据: v13 的 long 最优点又贴在网格边界(hold 48/48、sl 3.0/3.0、dedup 0.97/0.97),
#       仍是"空间"而非"数据"在决定 long 的解 -> v14 把这几条轴再让开一档, 并给趋势因子更大发挥空间。
V14_LONG_TREND_FACTORS = (
    "eff_ratio_20", "eff_ratio_50",     # Kaufman 效率比: 净移动/路径长度, 趋势 vs 震荡
    "trend_qual_20", "trend_qual_50",   # 线性回归 R² × 斜率符号 = 有向趋势质量
    "mtf_trend_align",                  # 多周期(20/50/100 均线斜率)同向度 -1~1
    "ma_slope_accel",                   # 趋势加速度(短均线斜率的增量)
    "adx_slope_14",                     # ADX 斜率(趋势强度增强/衰减)
)
V14_LONG_DEDUP_GRID = (0.80, 0.85, 0.90, 0.95, 0.97, 0.99)
V14_LONG_THR_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
V14_LONG_TP_GRID = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)
V14_LONG_SL_GRID = (0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0)
V14_LONG_HOLD_GRID = (6, 12, 24, 36, 48, 72)
# 模型层候选(多头)7 个: 前 5 个与 v13 逐位相同(便于对照), 后 2 个为 v14 新增:
#   deep_reg2: 更深(255 叶) + 更强 L1/L2 + 更大叶最小样本 -> 更大容量但强正则约束
#   mid_reg  : 中等深度(95 叶) + 中等正则                    -> 介于 deep_slow 与 deep_reg 之间
V14_LONG_MODEL_GRID = (
    dict(name="base", learning_rate=0.03, num_leaves=31, feature_fraction=0.7,
         min_child_samples=60, lambda_l2=1.0),
    dict(name="shallow", learning_rate=0.03, num_leaves=15, feature_fraction=0.6,
         min_child_samples=100, lambda_l2=5.0),
    dict(name="deep_slow", learning_rate=0.015, num_leaves=63, feature_fraction=0.5,
         min_child_samples=80, lambda_l2=10.0),
    dict(name="deep_reg", learning_rate=0.015, num_leaves=127, feature_fraction=0.4,
         min_child_samples=150, lambda_l1=1.0, lambda_l2=20.0),
    dict(name="smooth", learning_rate=0.01, num_leaves=31, feature_fraction=0.5,
         min_child_samples=200, lambda_l2=15.0),
    dict(name="deep_reg2", learning_rate=0.01, num_leaves=255, feature_fraction=0.3,
         min_child_samples=300, lambda_l1=2.0, lambda_l2=30.0),
    dict(name="mid_reg", learning_rate=0.02, num_leaves=95, feature_fraction=0.45,
         min_child_samples=120, lambda_l1=0.5, lambda_l2=12.0),
)
V14_RET_DROP = 0.015            # 同 v13: 允许让出的 OOF 收益上限(绝对值, 1.5 个百分点)
V14_OUT_DIR = "v14"             # 产出目录 reports/v14

# ================================================================ v15: 放开 long 止盈上限 + 冻结 long 门控
# 入口脚本 scripts/backtest/run_backtest_v15.py。基线 = v14。依据 v14 的 OOF 归因 + 两组 OOF 实验
# (tools/exp_v15_trendgate.py / tools/exp_v15_tp_ext.py), 结论逐条对症:
#   1. **趋势门控无效**: long 从「价在均线上/下」扩到 ADX/效率比/趋势质量等 9 种门控后,
#      各门控在 OOF 上的最优收益都低于「不门控(none)」 -> long 的门控**冻结为 none**,
#      不再作为 long 的可调旋钮(趋势信息仍由 v14 的趋势类因子承载)。
#   2. **真正的瓶颈在执行层**: v14 的 tp 网格上界 6.0 被 4 个制度一致顶到 -> tp 是"被空间截断"
#      而非"被数据选优"。v15 把 long 的 tp 网格外扩到 12.0(同时放开 tp>sl 下的 sl 下界到 1.0)。
#      放开后 long 的 OOF 收益 +9.18% -> +9.48%、夏普 8.59 -> 9.64, 选中解为 tp=10/sl=2/hold=72。
# 选优规则、成本、资金、切分、标签、VIF 纪律与 v14 **逐位一致**(收益锚 + 夏普择优, 让步带同 v14);
# **short 逐位不变**(仍用 v13/v14 口径与网格); OOC 全程只观察。
V15_LONG_REGIME_GRID = ("none",)     # long 门控冻结(实验: 任一门控都不提升 OOF 收益)
V15_LONG_DEDUP_GRID = (0.80, 0.85, 0.90, 0.95, 0.97, 0.99)
V15_LONG_THR_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
V15_LONG_TP_GRID = (2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0)   # 上界 6.0 -> 12.0
V15_LONG_SL_GRID = (1.0, 1.5, 2.0, 2.5, 3.0)
V15_LONG_HOLD_GRID = (12, 24, 36, 48, 72)
V15_LONG_MODEL_GRID = V14_LONG_MODEL_GRID     # 模型候选沿用 v14(7 个), 便于对照
V15_LONG_FACTORS = tuple(V13_LONG_FACTORS) + tuple(V14_LONG_TREND_FACTORS)  # long 专用因子(v14 口径)
V15_RET_DROP = V14_RET_DROP          # 选优让步带同 v14(1.5 个百分点)
V15_OUT_DIR = "v15"                  # 产出目录 reports/v15

# ================================================================ v18: 滚动 walk-forward
# 入口脚本 scripts/backtest/run_backtest_v18.py。基线 = v15。动机: v15/v16/v17 的诊断一致指向**选择层**
# 才是过拟合主因 —— 单次 train/OOF/OOC 切分下, 「在 OOF 网格里取最大值」是噪声选择
# (long 网格两万余组里前 8 名相差 <1pp、冠亚军仅差 0.09pp)。v17 把选优搬进 train 内部
# 嵌套 CV 虽消除了选择污染, 却牺牲了**制度适配**(train 段以旧牛市为主, 选出的参数不适配
# 近期制度)。v18 因此改为**滚动 walk-forward**:
#   - 训练窗 = 固定长度**滚动**窗(约 2 年), 只含最近制度, 不再把五年前的行情带进来;
#   - 选参窗 = 紧邻前测窗之前的 6 个月(样本外), 每一步都在其上**重选一轮**全部配置
#     (去冗余 × 模型 × 制度 × 阈值 × 止盈 × 止损 × 持有期), 选优规则仍是 v15 的
#     「收益锚 + 夏普择优」;
#   - 前测窗 = 选参窗之后的 6 个月, 只用冻结配置前测, **从不参与选择**;
#   - V18_N_STEPS 步首尾相接, 前测窗拼成**连续的 OOS 段**(约 50% 数据), 不需要一个
#     脆弱的 15% OOC 单窗; 每步的阈值 thr_abs 由**该步选参窗**预测分位冻结后应用到前测窗。
#   long 门控沿用 v15 的实验结论冻结为 none; 因子/标签/成本/网格/训练口径与 v15 一致,
#   唯一差异是取成交上界 entry_hi=窗口 stop(不含), 使成交严格落在窗内、不穿越折边界。
V18_TRAIN_BARS = 4380        # 滚动训练窗长度(约 2 年: 4h*6 根/天 * 365)
V18_VAL_BARS = 1095          # 选参窗长度(约 6 个月), 紧邻前测窗之前
V18_TEST_BARS = 1095         # 前测窗长度(约 6 个月), 与选参窗等长
V18_N_STEPS = 5              # 滚动步数(前测折数): OOS = 5*1095 ≈ 50% 数据
V18_RET_DROP = V15_RET_DROP  # 选优让步带沿用 v15(1.5 个百分点)
V18_MIN_TRAIN_BARS = 2000    # 首步训练窗安全下限(滚动窗不足时告警)
V18_REF_METRICS = REPORT_DIR / V15_OUT_DIR / "metrics_v15.json"   # 对照基线 v15
V18_OUT_DIR = "v18"          # 产出目录 reports/v18
