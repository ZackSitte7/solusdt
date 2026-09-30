# -*- coding: utf-8 -*-
"""SOL/USDT 4h K线抓取 —— 公开行情接口(无需 API Key)。

支持两个数据源, 由 `--source` 选择(config.SOURCE 为默认):

  * binance(默认, **本沙箱可直连**)
    `GET /api/v3/klines`, 从 0 号 K线起逐页向后翻, 覆盖 SOL 上市以来全部 4h K线。
    字段最全, 含 `trades` 与 `taker_buy_*`(下游 factors.py 的订单流因子依赖这些列),
    历史最早(2020-08-11 起)。镜像域名 data-api.binance.vision 在本环境可达。

  * okx
    `GET /api/v5/market/history-candles`, 逐页向前翻。
    注意: 本沙箱白名单拦截 www.okx.com, 因此实际数据需由一台可直连 OKX 的外网主机
    抓取后, 以原始 JSON 转存到 data/solusdt_4h_okx_raw.json, 再由本模块离线导入。
    OKX 不提供 trades/taker_buy_* 字段, 无法满足 factors.py 的订单流因子, 仅作备用。

币安 K线字段(/api/v3/klines, 官方顺序固定):
  [0] open_time       开盘时间(ms)
  [1] open  [2] high  [3] low  [4] close
  [5] volume          成交量(基础币, 即 SOL 数量)
  [6] close_time      收盘时间(ms)
  [7] quote_volume    成交量(计价币, 即 USDT 金额)
  [8] trades          成交笔数
  [9] taker_buy_base  主动买入量(基础币)
  [10] taker_buy_quote 主动买入量(计价币)
  [11] ignore         忽略字段

OKX 蜡烛字段(history-candles):
  [0] ts  [1] o  [2] h  [3] l  [4] c  [5] vol  [6] volCcy  [7] volCcyQuote  [8] confirm
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402

# ------------------------------------------------------------------ 列定义
# 下游清洗/因子所需的规范列(不含中间字段)。
BINANCE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
BINANCE_KEEP = [
    "open_time", "datetime", "open", "high", "low", "close", "volume",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote",
]

OKX_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "quote_volume", "vol_ccy_quote", "confirm",
]
OKX_NUMERIC = ["open", "high", "low", "close", "volume", "quote_volume", "vol_ccy_quote"]
BINANCE_NUMERIC = ["open", "high", "low", "close", "volume", "quote_volume",
                   "trades", "taker_buy_base", "taker_buy_quote"]

BAR_MS = {"4h": 4 * 3600 * 1000, "1h": 3600 * 1000, "1d": 24 * 3600 * 1000}


def _session() -> requests.Session:
    """requests 默认 trust_env=True, 会自动走 HTTP(S)_PROXY; 外网主机上直连即可。"""
    s = requests.Session()
    s.trust_env = True
    s.headers.update({"User-Agent": "solusdt-research/1.0"})
    return s


def _get_json(sess: requests.Session, url: str, params: dict):
    """带指数退避重试的 GET, 返回已解析 JSON。"""
    delay, last_err = 1.0, None
    for attempt in range(1, C.REQUEST_RETRY + 1):
        try:
            r = sess.get(url, params=params, timeout=C.REQUEST_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            last_err = "HTTP %d: %s" % (r.status_code, r.text[:120])
        except Exception as e:                      # 网络异常
            last_err = "%s: %s" % (type(e).__name__, e)
        if attempt < C.REQUEST_RETRY:
            print("    [重试 %d/%d] %s -> %.1fs 后重试" % (attempt, C.REQUEST_RETRY, last_err, delay))
            time.sleep(delay)
            delay = min(delay * 2, 20.0)
    raise RuntimeError("抓取失败(已重试 %d 次): %s" % (C.REQUEST_RETRY, last_err))


# ================================================================ 币安 Binance
def _binance_get_page(sess: requests.Session, start_time: int) -> list:
    """取一页 K线(start_time 为起点毫秒时间戳, 0 表示自上市起)。"""
    url = "%s/api/v3/klines" % C.BINANCE_BASE
    params = {"symbol": C.SYMBOL, "interval": C.INTERVAL,
              "startTime": start_time, "limit": C.KLINES_LIMIT}
    return _get_json(sess, url, params)


def _binance_to_frame(rows: list) -> pd.DataFrame:
    """原始行 -> 规整 DataFrame(按时间升序、去重、仅保留已收盘 K线)。"""
    df = pd.DataFrame(rows, columns=BINANCE_COLS)
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce").astype("int64")
    df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce").astype("int64")
    for c in BINANCE_NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

    # 只保留已收盘 K线: 最新一根的 close_time 仍在前方(进行中), 剔除。
    now_ms = int(time.time() * 1000)
    df = df[df["close_time"] < now_ms]

    df = (df.drop_duplicates("open_time")
            .sort_values("open_time")
            .reset_index(drop=True))
    return df[BINANCE_KEEP]


def fetch_binance_klines(verbose: bool = True) -> pd.DataFrame:
    """自上市起逐页向后抓取全部 4h K线(币安镜像域名, 本沙箱可直连)。"""
    sess = _session()
    rows, seen, page, start = [], set(), 0, 0
    while True:
        batch = _binance_get_page(sess, start)
        if not batch:
            break
        page += 1
        added = 0
        for r in batch:
            if r[0] not in seen:
                seen.add(r[0])
                rows.append(r)
                added += 1
        if verbose:
            print("    第 %2d 页: %4d 根, 累计 %5d, 至 %s" %
                  (page, len(batch), len(rows),
                   pd.to_datetime(int(batch[-1][0]), unit="ms", utc=True)))
        if len(batch) < C.KLINES_LIMIT or added == 0:
            break
        start = int(batch[-1][0]) + 1               # 下一页起点: 本页最新一根之后
        time.sleep(C.REQUEST_SLEEP)
    if not rows:
        raise RuntimeError("接口未返回任何数据, 请检查网络/白名单")
    return _binance_to_frame(rows)


# ================================================================ 欧易 OKX
def _okx_get_page(sess: requests.Session, after: str) -> list:
    """取一页 K线(after 为空则取最新一页)。带指数退避重试。"""
    url = "%s/api/v5/market/history-candles" % C.OKX_BASE
    params = {"instId": C.OKX_INST, "bar": C.OKX_BAR, "limit": C.OKX_LIMIT}
    if after:
        params["after"] = after
    j = _get_json(sess, url, params)
    if isinstance(j, dict):
        if j.get("code") == "0":
            return j.get("data", [])
        raise RuntimeError("OKX code=%s msg=%s" % (j.get("code"), j.get("msg")))
    return j or []


def _okx_to_frame(rows: list) -> pd.DataFrame:
    """原始行 -> 规整 DataFrame(按时间升序、去重、仅保留已收盘 K线)。"""
    df = pd.DataFrame(rows, columns=OKX_COLS)
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce").astype("int64")
    for c in OKX_NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

    # 只保留已收盘 K线(OKX 最新一根 confirm=0 为进行中)
    df = df[df["confirm"].astype(str) == "1"]

    df = (df.drop_duplicates("open_time")
            .sort_values("open_time")
            .reset_index(drop=True))
    return df[["open_time", "datetime"] + OKX_NUMERIC + ["confirm"]]


def fetch_okx_klines(verbose: bool = True) -> pd.DataFrame:
    """从最新一根开始, 逐页向前抓取全部 4h K线。仅在本机可直连 OKX 时可用。"""
    sess = _session()
    rows, seen, page, after = [], set(), 0, ""
    while True:
        batch = _okx_get_page(sess, after)
        if not batch:
            break
        page += 1
        added = 0
        for r in batch:
            if r[0] not in seen:
                seen.add(r[0])
                rows.append(r)
                added += 1
        if verbose:
            print("    第 %2d 页: %4d 根, 累计 %5d, 至 %s" %
                  (page, len(batch), len(rows),
                   pd.to_datetime(int(batch[-1][0]), unit="ms", utc=True)))
        if len(batch) < C.OKX_LIMIT or added == 0:
            break
        after = batch[-1][0]                        # 下一页起点: 本页最旧一根
        time.sleep(C.REQUEST_SLEEP)
    if not rows:
        raise RuntimeError("接口未返回任何数据, 请检查网络/白名单")
    return _okx_to_frame(rows)


def load_okx_dump(path: str | Path = None) -> pd.DataFrame:
    """离线导入: 读取外网主机抓取的原始 JSON(OKX data 数组的拼接)。"""
    path = Path(path) if path else C.OKX_DUMP
    if not path.exists():
        raise FileNotFoundError("未找到转存的原始 JSON: %s" % path)
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [r["value"] if isinstance(r, dict) else r for r in raw]
    return _okx_to_frame(rows)


# ================================================================ 主流程
def _fetch(source: str) -> tuple[pd.DataFrame, str]:
    """按数据源抓取, 返回 (DataFrame, 来源说明)。"""
    if source == "binance":
        return fetch_binance_klines(), "Binance API 直连 %s" % C.BINANCE_BASE
    if source == "okx":
        try:
            return fetch_okx_klines(), "OKX API 直连"
        except Exception as e:
            print("  直连 OKX 失败(%s), 回退到离线转存文件..." % e)
            return load_okx_dump(), "离线转存 %s" % C.OKX_DUMP
    raise ValueError("未知数据源: %s(可选 binance / okx)" % source)


def _report(df: pd.DataFrame, src: str, out_path: Path) -> None:
    step = BAR_MS[C.INTERVAL]
    span = df["datetime"].iloc[-1] - df["datetime"].iloc[0]
    print("\n=== 完成 (%s) ===" % src)
    print("K线根数 : %d" % len(df))
    print("时间范围 : %s ~ %s" % (df["datetime"].iloc[0], df["datetime"].iloc[-1]))
    print("跨度     : %.1f 天 (%.2f 年)" % (span.total_seconds() / 86400,
                                          span.total_seconds() / 86400 / 365.25))
    expect = int(span.total_seconds() * 1000 / step) + 1
    print("期望根数 : %d (按 %s 等间隔推算, 差值=缺口 %d 根)" %
          (expect, C.INTERVAL, expect - len(df)))
    print("字段     : %s" % ", ".join(df.columns))
    print("落盘     : %s" % out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="抓取 SOL/USDT 4h K线")
    ap.add_argument("--source", choices=("binance", "okx"), default=C.SOURCE,
                    help="数据源(默认 %s)" % C.SOURCE)
    args = ap.parse_args()

    print("=== 抓取 %s %s K线 (数据源: %s) ===" % (C.SYMBOL, C.INTERVAL, args.source))
    df, src = _fetch(args.source)

    # 各数据源另存一份带来源标识的副本, 便于对照。
    per_source = {"binance": C.BINANCE_RAW_PARQUET, "okx": C.OKX_RAW_PARQUET}[args.source]
    df.to_parquet(per_source, index=False)
    out = per_source
    # RAW_PARQUET 是"当前主数据源"的规范产物, 仅抓取默认源时刷新, 避免被备用源覆盖。
    if args.source == C.SOURCE:
        df.to_parquet(C.RAW_PARQUET, index=False)

    _report(df, src, out)


if __name__ == "__main__":
    main()
