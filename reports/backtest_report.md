# SOL/USDT 4h 多因子 LGBM 回测报告(欧易数据)

- 数据: okx, 13128 根 4h K线, 2020-09-30 12:00 ~ 2026-09-30 04:00
- 切分: train 9189 / OOF 1969 / OOC 1970 (70%/15%/15%)
- 本金 1000 USDT, 单笔 100 USDT, 手续费单边 5.0bp(双边 10.0bp)
- 标签: 未来 3 根收益 / ATR(制度中性)

## 因子(多空独立筛选, 仅用 train; 去冗余阈值由 OOF 选定)

- long(去冗余 |corr|<0.95, 25 个): vol_ratio_5_20, range_ratio, atr_ratio_chg, rv_12, atr_ratio, parkinson_20, ma_align, vol_ma_ratio, rv_50, bb_width_20, vol_z_20, pos_in_range_100, ret_24, ma_slope_50, macd_signal, pos_in_range_50, rsi_28, lr_slope_50, ma_ratio_200, ma_slope_20, ma_ratio_100, ma_ratio_50, rsi_14, hour_cos, bb_pos_20
- short(去冗余 |corr|<0.90, 12 个): mom_accel, hour_sin, gap, ret_6, close_pos_in_bar, ret_1, body_ratio, dow_cos, dow_sin, lower_shadow, consec_up, consec_dn

## 执行层与模型(OOF 优化后冻结)

- long: 模型 shallow, 阈值分位 0.40(绝对 0.020269), 止盈 2.50 ATR, 止损 2.50 ATR, 最长持有 6 根
- short: 模型 shallow, 阈值分位 0.50(绝对 0.022171), 止盈 3.00 ATR, 止损 2.50 ATR, 最长持有 24 根

## 绩效(train 为样本内参考; OOF 为优化段; OOC 仅观察)

| 方向 | 段 | 总收益% | 夏普 | 卡玛 | 最大回撤% | 胜率% | 盈亏比 | 笔数 |
|---|---|---|---|---|---|---|---|---|
| long | train | +136.42 | 5.14 | 8.08 | -2.82 | 59.1 | 1.26 | 915 |
| long | oof | +7.12 | 2.36 | 2.36 | -3.36 | 50.6 | 1.23 | 172 |
| long | ooc | -7.23 | -3.07 | -0.82 | -9.77 | 46.7 | 0.85 | 180 |
| short | train | -16.90 | -0.79 | -0.11 | -40.13 | 47.6 | 1.02 | 515 |
| short | oof | +9.15 | 3.23 | 2.35 | -4.36 | 53.4 | 1.15 | 118 |
| short | ooc | +0.60 | 0.33 | 0.13 | -4.94 | 50.4 | 1.01 | 115 |

### 各段买入持有基准

| 段 | 买入持有% |
|---|---|
| train | +7915.94 |
| oof | -28.54 |
| ooc | -27.08 |

## 说明

- 优化**只在 OOF** 上进行: 外层 4 档去冗余阈值 × 3 个模型候选, 内层 5 阈值分位 × 6 止盈 × 6 止损 × 3 最长持有期; 多空独立; OOC 全程只观察。
- 目标函数: 主目标 sharpe, 次目标 total_return(门槛: 笔数 >= 20, 止盈率 >= 0.15)。
- OOC 使用由 OOF 冻结的**绝对阈值**, 不使用 OOC 自身分布。
- 因子库含趋势类(ADX、均线斜率、MACD、线性回归斜率、区间位置等); 欧易数据无订单流字段, 相关因子自动跳过。
- 注意: OOF 上比较次数已从 225 组升至 6480 组/方向, 选优噪声随之上升 -> OOC 才是最终检验。