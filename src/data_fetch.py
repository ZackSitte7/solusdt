# -*- coding: utf-8 -*-
"""SOL/USDT 4h K线抓取 —— Binance 官方公开数据接口(无需 API Key)。

为什么用 data-api.binance.vision:
  本沙箱网络走白名单, api.binance.com 不可达(连接被拒), 而
  data-api.binance.vision 可达且提供完全相同的 /api/v3/klines 公开数据,
  无需任何认证 —— 因此本项目**不涉及任何密钥**。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402

# Binance klines 列名(官方文档固定顺序)
KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "ignore",
]
NUMERIC_COLS = [c for c in KLINE_COLS if c not in ("open_time", "close_time", "ignore")]


def _session() -> requests.Session:
    """requests 默认 trust_env=True, 会自动走 HTTP(S)_PROXY; 沙箱必须依赖它。"""
    s = requests.Session()
    s.trust_env = True
    s.headers.update({"User-Agent": "solusdt-research/1.0"})
    return s


def _get_klines(sess: requests.Session, start_ms: int, limit: int = C.KLINES_LIMIT) -> list:
    """取一页 K线。带指数退避重试; 429/418 视为限速, 退避后重试。"""
    url = "%s/api/v3/klines" % C.BINANCE_BASE
    params = {"symbol": C.SYMBOL, "interval": C.INTERVAL,
              "limit": limit, "startTime": start_ms}
    delay = 1.0
    last_err = None
    for attempt in range(1, C.REQUEST_RETRY + 1):
        try:
            r = sess.get(url, params=params, timeout=C.REQUEST_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 418):
                last_err = "限速 HTTP %d" % r.status_code
            else:
                last_err = "HTTP %d: %s" % (r.status_code, r.text[:120])
        except Exception as e:                      # 网络异常
            last_err = "%s: %s" % (type(e).__name__, e)
        if attempt < C.REQUEST_RETRY:
            print("    [重试 %d/%d] %s -> %.1fs 后重试" % (attempt, C.REQUEST_RETRY, last_err, delay))
            time.sleep(delay)
            delay = min(delay * 2, 20.0)
    raise RuntimeError("抓取失败(已重试 %d 次): %s" % (C.REQUEST_RETRY, last_err))


def fetch_all_klines(verbose: bool = True) -> pd.DataFrame:
    """从上市首根开始, 逐页抓取全部 4h K线。返回按时间升序、已去重的 DataFrame。"""
    sess = _session()
    # 先探测最早时间(limit=1 + startTime=0)
    first = _get_klines(sess, 0, limit=1)
    if not first:
        raise RuntimeError("接口未返回任何数据, 请检查网络/白名单")
    start_ms = int(first[0][0])
    if verbose:
        print("首个 K线: %s" % pd.to_datetime(start_ms, unit="ms", utc=True))

    rows, page, cursor = [], 0, start_ms
    while True:
        batch = _get_klines(sess, cursor)
        if not batch:
            break
        rows.extend(batch)
        page += 1
        last_close_ms = int(batch[-1][6])
        if verbose:
            print("    第 %2d 页: %4d 根, 至 %s" %
                  (page, len(batch), pd.to_datetime(int(batch[-1][0]), unit="ms", utc=True)))
        if len(batch) < C.KLINES_LIMIT:
            break                                   # 已到最新
        cursor = last_close_ms + 1                  # 下一页起点
        time.sleep(C.REQUEST_SLEEP)

    df = pd.DataFrame(rows, columns=KLINE_COLS)
    df = df.drop(columns=["ignore"])
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("open_time", "close_time"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("int64")
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = (df.drop_duplicates("open_time")
            .sort_values("open_time")
            .reset_index(drop=True))
    return df


def main() -> None:
    print("=== 抓取 %s %s K线 (%s) ===" % (C.SYMBOL, C.INTERVAL, C.BINANCE_BASE))
    df = fetch_all_klines()
    df.to_parquet(C.RAW_PARQUET, index=False)

    span = df["datetime"].iloc[-1] - df["datetime"].iloc[0]
    print("\n=== 完成 ===")
    print("K线根数 : %d" % len(df))
    print("时间范围 : %s ~ %s" % (df["datetime"].iloc[0], df["datetime"].iloc[-1]))
    print("跨度     : %.1f 天 (%.2f 年)" % (span.total_seconds() / 86400,
                                          span.total_seconds() / 86400 / 365.25))
    expect = int(span.total_seconds() / (4 * 3600)) + 1
    print("期望根数 : %d (按 4h 等间隔推算, 差值=缺口 %d 根)" % (expect, expect - len(df)))
    print("落盘     : %s" % C.RAW_PARQUET)


if __name__ == "__main__":
    main()
