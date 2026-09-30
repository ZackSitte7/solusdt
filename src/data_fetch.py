# -*- coding: utf-8 -*-
"""SOL/USDT 4h K线抓取 —— 欧易(OKX)公开行情接口(无需 API Key)。

数据源: OKX `/api/v5/market/history-candles`, 逐页向前翻, 覆盖上市以来全部 4h K线。

网络说明:
  本沙箱网络走白名单, www.okx.com 不可达(ERR_CONNECTION_CLOSED), 因此实际数据由
  一台可直连 OKX 的外网主机抓取, 以原始 JSON 转存到 data/solusdt_4h_okx_raw.json,
  再由本模块离线导入 —— 全程无需任何密钥。

OKX 蜡烛字段(history-candles): [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
  ts          开盘时间(ms)
  vol         成交量(基础币, 即 SOL 数量)
  volCcy      成交量(计价币, 即 USDT 金额)
  volCcyQuote 成交量(计价币, 即 USDT 金额)
  confirm     1 = 已收盘, 0 = 进行中(未确认)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402

# OKX 蜡烛字段顺序(官方文档固定)
OKX_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "quote_volume", "vol_ccy_quote", "confirm",
]
NUMERIC_COLS = ["open", "high", "low", "close", "volume", "quote_volume", "vol_ccy_quote"]


def _session() -> requests.Session:
    """requests 默认 trust_env=True, 会自动走 HTTP(S)_PROXY; 外网主机上直连即可。"""
    s = requests.Session()
    s.trust_env = True
    s.headers.update({"User-Agent": "solusdt-research/1.0"})
    return s


def _get_page(sess: requests.Session, after: str) -> list:
    """取一页 K线(after 为空则取最新一页)。带指数退避重试。"""
    url = "%s/api/v5/market/history-candles" % C.OKX_BASE
    params = {"instId": C.OKX_INST, "bar": C.OKX_BAR, "limit": C.OKX_LIMIT}
    if after:
        params["after"] = after
    delay, last_err = 1.0, None
    for attempt in range(1, C.REQUEST_RETRY + 1):
        try:
            r = sess.get(url, params=params, timeout=C.REQUEST_TIMEOUT)
            if r.status_code == 200:
                j = r.json()
                if j.get("code") == "0":
                    return j.get("data", [])
                last_err = "OKX code=%s msg=%s" % (j.get("code"), j.get("msg"))
            else:
                last_err = "HTTP %d: %s" % (r.status_code, r.text[:120])
        except Exception as e:                      # 网络异常
            last_err = "%s: %s" % (type(e).__name__, e)
        if attempt < C.REQUEST_RETRY:
            print("    [重试 %d/%d] %s -> %.1fs 后重试" % (attempt, C.REQUEST_RETRY, last_err, delay))
            time.sleep(delay)
            delay = min(delay * 2, 20.0)
    raise RuntimeError("抓取失败(已重试 %d 次): %s" % (C.REQUEST_RETRY, last_err))


def _to_frame(rows: list) -> pd.DataFrame:
    """原始行 -> 规整 DataFrame(按时间升序、去重、仅保留已收盘 K线)。"""
    df = pd.DataFrame(rows, columns=OKX_COLS)
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce").astype("int64")
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

    # 只保留已收盘 K线(OKX 最新一根 confirm=0 为进行中)
    df = df[df["confirm"].astype(str) == "1"]

    df = (df.drop_duplicates("open_time")
            .sort_values("open_time")
            .reset_index(drop=True))
    return df[["open_time", "datetime"] + NUMERIC_COLS + ["confirm"]]


def fetch_all_klines(verbose: bool = True) -> pd.DataFrame:
    """从最新一根开始, 逐页向前抓取全部 4h K线。仅在本机可直连 OKX 时可用。"""
    sess = _session()
    rows, seen, page, after = [], set(), 0, ""
    while True:
        batch = _get_page(sess, after)
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
    return _to_frame(rows)


def load_dump(path: str | Path = None) -> pd.DataFrame:
    """离线导入: 读取外网主机抓取的原始 JSON(OKX data 数组的拼接)。"""
    path = Path(path) if path else C.OKX_DUMP
    if not path.exists():
        raise FileNotFoundError("未找到转存的原始 JSON: %s" % path)
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [r["value"] if isinstance(r, dict) else r for r in raw]
    return _to_frame(rows)


def main() -> None:
    print("=== 抓取 %s %s K线 (OKX history-candles, %s) ===" %
          (C.OKX_INST, C.OKX_BAR, C.OKX_BASE))
    try:
        df = fetch_all_klines()                     # 直连 OKX(外网主机)
        src = "OKX API 直连"
    except Exception as e:
        print("  直连 OKX 失败(%s), 回退到离线转存文件..." % e)
        df = load_dump()                            # 本沙箱: 离线导入
        src = "离线转存 %s" % C.OKX_DUMP
    df.to_parquet(C.RAW_PARQUET, index=False)

    span = df["datetime"].iloc[-1] - df["datetime"].iloc[0]
    print("\n=== 完成 (%s) ===" % src)
    print("K线根数 : %d" % len(df))
    print("时间范围 : %s ~ %s" % (df["datetime"].iloc[0], df["datetime"].iloc[-1]))
    print("跨度     : %.1f 天 (%.2f 年)" % (span.total_seconds() / 86400,
                                          span.total_seconds() / 86400 / 365.25))
    expect = int(span.total_seconds() / (4 * 3600)) + 1
    print("期望根数 : %d (按 4h 等间隔推算, 差值=缺口 %d 根)" % (expect, expect - len(df)))
    print("落盘     : %s" % C.RAW_PARQUET)


if __name__ == "__main__":
    main()
