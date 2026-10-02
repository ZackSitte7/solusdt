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
# 本次回测使用**欧易(OKX)** 4h 数据, 入口脚本 run_backtest_okx.py。
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
# 入口脚本 run_backtest_v2.py。设计依据是对 v1 的亏损归因(analyze_trades.py), 逐条对症:
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
# 入口脚本 run_backtest_v3.py。用户明确要求, 相对 v1/v2 的差异只有两点:
#   1. 选择集回到 **OOF**(与 v1 同口径, 以把 OOF 收益做回 v1 水准), OOC 仍**只观察**;
#   2. 止盈/止损的**硬约束 tp > sl**(风险报酬比 > 1) —— 只保留止盈幅度大于止损幅度的
#      组合, 提高单笔盈亏比、削掉"止盈比止损还近"的退化配置, 从而改善夏普。
# 其余口径(数据/因子/标签/成本/资金/指标/目标函数)与 v1 完全一致, 保证可比。
V3_ENFORCE_TP_GT_SL = True       # True = 只扫描 tp_mult > sl_mult 的止盈止损组合
V3_OUT_DIR = "v3"                # 产出目录 reports/v3

# ================================================================ v4: VIF 迭代剪枝(治理共线过拟合)
# 入口脚本 run_backtest_v4.py。依据 analyze_factor_dim.py 的诊断:
#   long 名义 25 个因子, 但有效维度(参与率)仅 4.3、最大 VIF 71.5 —— 多出来的维度是近共线的
#   噪声方向, 交给 LGBM 只会被拟合。pairwise 去冗余(|corr|>=0.95)挡不住**多元共线**, 故引入
#   方差膨胀因子(VIF)迭代剪枝: 每次删除 VIF 最大的因子, 直至全部 VIF <= V4_MAX_VIF。
# 其余口径与 v3 完全一致(OOF 选优 + ATR 止盈止损 + tp>sl), 便于直接对比。
V4_MAX_VIF = 10.0                # VIF 上限(经验阈值: >10 视为严重共线)
V4_MIN_FACTORS = 5               # 剪枝后至少保留的因子数(安全下限, 避免剪成单因子)
V4_OUT_DIR = "v4"                # 产出目录 reports/v4

# ================================================================ v5: 制度门控(治理制度错配)
# 入口脚本 run_backtest_v5.py。依据 v4 结论: OOC 亏损主因不是因子冗余, 而是 train(牛市)
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
# 入口脚本 run_backtest_v6.py。相对 v5 只动四处, 其余口径(切分/成本/目标函数/选择纪律)完全一致:
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
# 入口脚本 run_backtest_v7.py。依据对 v6 的归因 —— 空头在 OOC 由 +5.94% 崩到 -0.85%,
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
