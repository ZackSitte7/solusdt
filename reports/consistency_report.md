# 设计与运行一致性报告

> 由 run_backtest_okx.py 运行时自动生成; 详细设计见 DESIGN.md; 单元测试见 tests/test_consistency.py

| 检查项 | 结果 | 说明 |
|---|---|---|
| 切分比例 70/15/15 | PASS | train 9189 / oof 1969 / ooc 1970 of 13128 |
| 切分严格时序不重叠 | PASS | train[0,9189) oof[9189,11158) ooc[11158,13128) |
| 训练/筛选只使用 train 段 | PASS | 因子筛选 factor_sets_by_dedup(F[tr], y[tr]); 模型 train_side(F[tr], y[tr]) |
| 优化只使用 OOF 段 | PASS | optimize_full_on_oof: 外层(去冗余阈值 [0.8, 0.85, 0.9, 0.95] × 模型) + 内层(阈值×止盈×止损×持有期 [6, 12, 24]); 网格评估区间 = oof 切片; 目标 total_return > sharpe |
| 因子只来自 train, 去冗余阈值由 OOF 选定 | PASS | {"long": {"dedup_max_corr": 0.95, "max_hold": 6, "n_factors": 25}, "short": {"dedup_max_corr": 0.9, "max_hold": 12, "n_factors": 12}} |
| OOC 阈值来自 OOF(冻结绝对阈值, 未偷看 OOC 分布) | PASS | {"long": 0.020269, "short": 0.020354} |
| 成本口径 = 单边手续费 5.0bp × 2 | PASS | 来回成本率 = 0.0010 |
| 本金/单笔名义 = 1000 / 100 USDT | PASS |  |
| 多空因子集不同(独立筛选) | PASS | long 25 个 / short 12 个 |
| OOF 冻结评估 == 网格最优行 | PASS | 已在主流程中 assert 校验(差值 < 1e-9) |

失败项: 0