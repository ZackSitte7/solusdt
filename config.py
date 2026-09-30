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
# 收紧网格: 上一版最优解跑到 tp=6ATR, 实际止盈率只有 1.5~6.3%, 参数已退化。
TP_ATR_GRID = (1.0, 1.5, 2.0, 2.5, 3.0)
SL_ATR_GRID = (0.75, 1.0, 1.5, 2.0, 2.5)
EXEC_THR_GRID = (0.5, 0.6, 0.7)     # 信号阈值取预测值的分位
MAX_HOLD_BARS = 12           # 最长持有根数(4h * 12 = 48h), 到期按收盘平仓
# 退化参数淘汰线: 止盈率低于此值说明「止盈」并未真正工作, 该组合作废
MIN_TP_RATE = 0.15

# ---------------------------------------------------------------- OOF 优化(需求7/8/9)
# **只在 OOF 上**优化模型层(超参候选)与执行层(阈值/止盈/止损); OOC 全程只观察。
OOF_MIN_TRADES = 20          # 单组合 OOF 最少成交笔数(防止选到"几乎不交易"的退化参数)

# ---------------------------------------------------------------- 目标函数
# 只在 OOF 上优化。主目标为夏普, 次目标为收益(见 src/optimize.py optimize_on_oof)
OBJECTIVE_PRIMARY = "sharpe"
OBJECTIVE_SECONDARY = "total_return"
