# -*- coding: utf-8 -*-
"""冒烟测试 —— 不重跑回测, 只校验**入库产物**的完整性与自洽性。

与 `test_consistency.py`(设计/运行一致性单元测试)互补: 这里守的是"仓库里已入库的
东西是否完好、可读、可还原", 适合在 CI 或提交前快速跑(`make smoke`)。

覆盖:
  1. 大文件分片: 逐片 sha256 / 字节数 / 行数 / 首行表头与 MANIFEST 一致;
  2. 配置自洽: 训练/OOF/OOC 切分比例之和为 1, 且顺序非负;
  3. src 全部模块可导入(不触发任何 IO 副作用);
  4. 各版本 metrics_v*.json 可解析, 且其内部一致性检查为 0 失败。
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import config as C                       # noqa: E402

REPORTS = BASE / "reports"


def _load_shards_tool():
    """tools/ 不是包, 用 spec 直接加载 shards.py 以复用其校验逻辑。"""
    spec = importlib.util.spec_from_file_location("_shards_tool", BASE / "tools" / "shards.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SHARDS = _load_shards_tool()
SHARD_DIRS = SHARDS.shard_dirs(REPORTS) if REPORTS.exists() else []
METRICS_FILES = sorted(REPORTS.glob("v*/metrics_v*.json"))


def test_split_fractions_sum_to_one():
    total = C.TRAIN_FRAC + C.OOF_FRAC + C.OOC_FRAC
    assert abs(total - 1.0) < 1e-9, "切分比例之和应为 1, 实际 %.6f" % total
    assert 0 < C.OOF_FRAC < 1 and 0 < C.OOC_FRAC < 1
    # 注: "OOC 只观察、绝不参与选择" 由 test_consistency.py::test_frozen_threshold_no_ooc_peek 覆盖
    #     (config.SELECT_ON 是 v1 旧流水线的选择段, 不描述 v6+ 的 OOF 选优流程)


@pytest.mark.parametrize("name", ["cv", "data_clean", "execution", "factor_select",
                                  "factors", "metrics", "models", "optimize",
                                  "protocol", "regime"])
def test_src_module_importable(name):
    importlib.import_module("src.%s" % name)


@pytest.mark.skipif(not SHARD_DIRS, reason="仓库内无分片")
@pytest.mark.parametrize("shard_dir", SHARD_DIRS, ids=lambda p: str(p.relative_to(BASE)))
def test_shards_intact(shard_dir):
    """逐片校验, 但**不做还原比对**(避免冒烟测试读 300MB+)。"""
    assert SHARDS.verify(shard_dir, check_source=False, verbose=False), \
        "分片校验失败: %s" % shard_dir
    man = SHARDS.read_manifest(shard_dir)
    assert man["n_parts"] >= 1
    for p in man["parts"]:
        assert p["bytes"] <= SHARDS.DEFAULT_MAX_MB * 1024 * 1024, \
            "分片 %s 超过 %d MiB 上限" % (p["part"], SHARDS.DEFAULT_MAX_MB)


@pytest.mark.skipif(not METRICS_FILES, reason="仓库内无版本指标")
@pytest.mark.parametrize("path", METRICS_FILES, ids=lambda p: p.parent.name)
def test_metrics_parse_and_consistent(path):
    d = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(d, dict) and "sides" in d, "指标文件结构异常: %s" % path
    for side in ("long", "short"):
        m = d["sides"][side]["metrics"]
        for seg in ("train", "oof", "ooc"):
            assert "total_return" in m[seg] and "sharpe" in m[seg]
    n_fail = (d.get("consistency", {}) or {}).get("n_fail")
    assert n_fail == 0, "%s 的设计-运行一致性检查有 %s 项失败" % (path.parent.name, n_fail)
