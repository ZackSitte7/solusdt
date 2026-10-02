# ============================================================ Makefile
# SOL/USDT 4h 多因子回测 —— 常用命令。查看全部目标: make help
SHELL := /bin/bash
PY    := python3

.DEFAULT_GOAL := help
.PHONY: help install test lint smoke compare shards-verify shards-merge \
        v6 v7 v8 v9 v10 v11 clean

help:  ## 列出所有可用目标
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | sort | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- 环境
install:  ## 按精确版本安装依赖
	$(PY) -m pip install -r requirements.lock.txt

# ---------------------------------------------------------------- 质量
test:  ## 运行全部测试(单元 + 冒烟)
	$(PY) -m pytest tests -q

lint:  ## ruff 静态检查
	$(PY) -m ruff check .

smoke:  ## 只跑冒烟测试: 校验入库产物(分片完整性 / 指标自洽)
	$(PY) -m pytest tests/test_smoke.py -q

# ---------------------------------------------------------------- 工具
compare:  ## 重新生成版本对照表 + 对照图
	$(PY) tools/compare_versions.py

shards-verify:  ## 校验全部大文件分片(含还原后与源文件 sha256 比对)
	$(PY) tools/shards.py verify reports

# ---------------------------------------------------------------- 回测
# 版本间有前置依赖: v7/v8 依赖 v6 的冻结多头, v9/v10 依赖 v6/v8/v9 的指标。
v6:  ## 运行 v6(扩充因子库 + 波动率制度), 是 v7/v8 的前置
	$(PY) run_backtest_v6.py

v7:  ## 运行 v7(空头专用优化, 需先 make v6)
	$(PY) run_backtest_v7.py

v8:  ## 运行 v8(OOF 内部分折 + 稳健选优, 需先 make v6)
	$(PY) run_backtest_v8.py

v9:  ## 运行 v9(long+short 共同优化合并总收益, 需先 make v6 v8)
	$(PY) run_backtest_v9.py

v10:  ## 运行 v10(OOF 收益+夏普双目标, 复用 v9 网格约 2 分钟)
	$(PY) run_backtest_v10.py

v11:  ## 运行 v11(强化选择器: 5 折 + 双目标候选池 + Pareto; 需重算阶段1 网格)
	$(PY) run_backtest_v11.py

# ---------------------------------------------------------------- 清理
clean:  ## 清理 Python 缓存
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
