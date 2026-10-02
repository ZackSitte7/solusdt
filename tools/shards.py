#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""大文件分片 / 合并 / 校验工具。

背景
----
`reports/v*/selection_folds_*.csv` 是"配置 × 子折"的原始网格转储, 单文件约 108MB,
超过 GitHub 单文件 100MiB 硬上限, 直接入库会被拒推。本工具把它们按**行边界**切成
多个 `partNN`, 每片不超过 --max-mb(默认 90MiB), 且在**每片顶部重复表头**,
使每一片都能被 pandas 独立读取。

目录约定
--------
每个源文件拥有**独立的**分片目录, 避免同名清单互相覆盖:

    reports/v9/selection_folds_long.csv           <- 源文件(不入库, .gitignore 排除)
    reports/v9/shards/selection_folds_long.csv/   <- 分片目录(入库)
        ├── MANIFEST.json
        ├── part00
        └── part01

清单 `MANIFEST.json` 记录源文件路径/sha256/字节数/数据行数, 以及每片的
文件名/sha256/字节数/数据行数, 供 verify 逐片校验、merge 还原。

用法
----
    python3 tools/shards.py split  reports/v9/selection_folds_long.csv
    python3 tools/shards.py verify reports/v9/shards            # 递归校验该根下所有分片集
    python3 tools/shards.py verify reports/v9/shards/selection_folds_long.csv
    python3 tools/shards.py merge  reports/v9/shards/selection_folds_long.csv /tmp/restored.csv

前提
----
CSV 为**逐行记录**(字段内不含换行)。本仓库的网格转储均为 pandas `to_csv` 产物,
列名与取值都是数值/短字符串, 满足该前提。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

CHUNK = 1 << 20          # 读大文件时按 1MiB 分块算 sha256
MANIFEST = "MANIFEST.json"
DEFAULT_MAX_MB = 90


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(CHUNK), b""):
            h.update(blk)
    return h.hexdigest()


def shard_dir_for(src) -> Path:
    """源文件 -> 其分片目录(约定: <源文件所在目录>/shards/<源文件名>/)。"""
    src = Path(src)
    return src.parent / "shards" / src.name


def read_manifest(shard_dir) -> dict:
    p = Path(shard_dir) / MANIFEST
    if not p.exists():
        raise SystemExit("缺少清单文件: %s" % p)
    return json.loads(p.read_text(encoding="utf-8"))


def shard_dirs(target) -> list:
    """target 自身含 MANIFEST.json 则视作单个分片集; 否则递归收集其下所有分片集。"""
    target = Path(target)
    if (target / MANIFEST).exists():
        return [target]
    return sorted({p.parent for p in target.rglob(MANIFEST)})


def _seal(path: Path, rows: int) -> dict:
    return dict(part=path.name, rows=int(rows),
                bytes=int(path.stat().st_size), sha256=sha256_file(path))


def split(src, max_mb: float = DEFAULT_MAX_MB, out_dir=None, verbose: bool = True) -> dict:
    """把 src 按行边界切成多个分片, 返回清单 dict。

    保证: 每片字节数 <= max_mb MiB(除非单行本身超限); 每片首行为源文件表头。
    """
    src = Path(src)
    if not src.exists():
        raise SystemExit("源文件不存在: %s" % src)
    out_dir = Path(out_dir) if out_dir else shard_dir_for(src)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = int(float(max_mb) * 1024 * 1024)

    for old in sorted(out_dir.glob("part*")):          # 清掉旧分片, 防残留混入
        old.unlink()

    def part_path(i: int) -> Path:
        return out_dir / ("part%02d" % i)

    with open(src, "r", encoding="utf-8", newline="") as fin:
        header = fin.readline()
        if not header:
            raise SystemExit("空文件, 无可分片内容: %s" % src)
        header_bytes = len(header.encode("utf-8"))
        if header_bytes > max_bytes:
            raise SystemExit("表头单行即超过单片上限(%d 字节)" % max_bytes)

        parts: list = []
        idx, size, rows = 0, 0, 0
        fout = open(part_path(idx), "w", encoding="utf-8", newline="")
        fout.write(header)
        size = header_bytes
        for line in fin:
            lb = len(line.encode("utf-8"))
            if size + lb > max_bytes and rows > 0:      # 只在行边界切, 且不留空片
                fout.close()
                parts.append(_seal(part_path(idx), rows))
                idx += 1
                fout = open(part_path(idx), "w", encoding="utf-8", newline="")
                fout.write(header)
                size, rows = header_bytes, 0
            fout.write(line)
            size += lb
            rows += 1
        fout.close()
        if rows > 0 or not parts:                       # 末片(或整文件不足一片)
            parts.append(_seal(part_path(idx), rows))

    manifest = dict(
        source=src.as_posix(),
        source_bytes=int(src.stat().st_size),
        source_sha256=sha256_file(src),
        source_rows=int(sum(p["rows"] for p in parts)),
        header=header.rstrip("\r\n"),
        max_bytes=max_bytes,
        n_parts=len(parts),
        parts=parts,
    )
    (out_dir / MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if verbose:
        print("分片完成: %s" % src)
        print("  源文件 %.1f MB / %d 行 -> %d 片(max %.0f MiB)"
              % (manifest["source_bytes"] / 1048576.0, manifest["source_rows"],
                 manifest["n_parts"], max_mb))
        for p in manifest["parts"]:
            print("    %-10s %8.1f MB  %9d 行" % (p["part"], p["bytes"] / 1048576.0, p["rows"]))
        print("  分片目录: %s" % out_dir)
    return manifest


def merge(shard_dir, dest):
    """按清单顺序合并分片到 dest(每片重复的表头只保留一次), 返回 (dest, manifest)。"""
    shard_dir = Path(shard_dir)
    man = read_manifest(shard_dir)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8", newline="") as out:
        out.write(man["header"] + "\n")
        for p in man["parts"]:
            with open(shard_dir / p["part"], "r", encoding="utf-8", newline="") as f:
                f.readline()                            # 跳过每片重复的表头
                for line in f:
                    out.write(line)
    return dest, man


def verify(shard_dir, check_source: bool = True, verbose: bool = True) -> bool:
    """逐片校验 sha256 / 字节数 / 行数 / 首行表头; 可选还原后与源文件 sha256 比对。"""
    shard_dir = Path(shard_dir)
    man = read_manifest(shard_dir)
    errs: list = []
    for p in man["parts"]:
        f = shard_dir / p["part"]
        if not f.exists():
            errs.append("%s 缺失" % p["part"])
            continue
        if f.stat().st_size != p["bytes"]:
            errs.append("%s 字节数不符(期望 %d, 实际 %d)"
                        % (p["part"], p["bytes"], f.stat().st_size))
        if sha256_file(f) != p["sha256"]:
            errs.append("%s sha256 不符" % p["part"])
        with open(f, "r", encoding="utf-8", newline="") as fh:
            if fh.readline().rstrip("\r\n") != man["header"]:
                errs.append("%s 首行表头与清单不一致" % p["part"])
            n = sum(1 for _ in fh)
        if n != p["rows"]:
            errs.append("%s 数据行数不符(期望 %d, 实际 %d)" % (p["part"], p["rows"], n))

    if check_source:
        src = Path(man["source"])
        if not src.exists():
            if verbose:
                print("  [跳过] 源文件不在本地, 无法做还原比对: %s" % src)
        else:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td) / src.name
                merge(shard_dir, tmp)
                if tmp.stat().st_size != man["source_bytes"]:
                    errs.append("还原后字节数与源文件不符")
                if sha256_file(tmp) != man["source_sha256"]:
                    errs.append("还原后 sha256 与源文件不符")

    if verbose:
        tag = "通过" if not errs else "失败"
        print("校验[%s] %s: %d 片 / %d 行"
              % (tag, shard_dir, man["n_parts"], man["source_rows"]))
        for e in errs:
            print("    ! %s" % e)
    return not errs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="大文件分片 / 合并 / 校验")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("split", help="按行边界分片, 每片重复表头")
    sp.add_argument("src")
    sp.add_argument("--max-mb", type=float, default=DEFAULT_MAX_MB)
    sp.add_argument("--out-dir", default=None, help="分片目录(默认 <源目录>/shards/<源文件名>)")

    sv = sub.add_parser("verify", help="逐片校验并在本地还原比对(可传根目录递归校验)")
    sv.add_argument("target")
    sv.add_argument("--no-source", action="store_true", help="不做还原比对")

    sm = sub.add_parser("merge", help="按清单合并还原")
    sm.add_argument("shard_dir")
    sm.add_argument("dest")

    a = ap.parse_args(argv)
    if a.cmd == "split":
        split(a.src, max_mb=a.max_mb, out_dir=a.out_dir)
        return 0
    if a.cmd == "verify":
        dirs = shard_dirs(a.target)
        if not dirs:
            print("未找到任何分片集(缺少 %s): %s" % (MANIFEST, a.target))
            return 1
        ok = all([verify(d, check_source=not a.no_source) for d in dirs])
        print("汇总: %d 个分片集, %s" % (len(dirs), "全部通过" if ok else "存在失败"))
        return 0 if ok else 1
    dest, man = merge(a.shard_dir, a.dest)
    print("已还原: %s (%.1f MB / %d 行)"
          % (dest, dest.stat().st_size / 1048576.0, man["source_rows"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
