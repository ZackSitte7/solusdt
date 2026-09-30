# 设计与运行一致性报告

> 由 run_backtest_okx.py 运行时自动生成; 详细设计见 DESIGN.md; 单元测试见 tests/test_consistency.py

| 检查项 | 结果 | 说明 |
|---|---|---|
| 切分比例 70/15/15 | PASS | train 9189 / oof 1969 / ooc 1970 of 13128 |
| 切分严格时序不重叠 | PASS | train[0,9189) oof[9189,11158) ooc[11158,13128) |
| 训练/筛选只使用 train 段 | PASS | 因子筛选 select_factors(F[tr], y[tr]); 模型 train_side(F[tr], y[tr]) |
| 优化只使用 OOF 段 | PASS | optimize_on_oof 网格评估区间 = oof 切片; 目标 sharpe>total_return |
| OOC 阈值来自 OOF(冻结绝对阈值, 未偷看 OOC 分布) | PASS | {"long": 0.004667, "short": 0.020354} |
| 成本口径 = 单边手续费 5.0bp × 2 | PASS | 来回成本率 = 0.0010 |
| 本金/单笔名义 = 1000 / 100 USDT | PASS |  |
| 多空因子集不同(独立筛选) | PASS | long 19 个 / short 12 个 |
| OOF 冻结评估 == 网格最优行 | PASS | 已在主流程中 assert 校验(差值 < 1e-9) |

失败项: 0