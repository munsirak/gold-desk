"""Live data feeds. Free sources only; each feed fails soft so the app still runs."""
import io, os
import pandas as pd, requests, yfinance as yf

TICKERS = {"gold": "GC=F", "gld": "GLD", "dxy": "DX-Y.NYB", "tnx": "^TNX",
           "oil": "CL=F", "vix": "^VIX", "silver": "SI=F"}
GOLD_COT_CODE = "088691"  # COMEX gold, CFTC contract market code


def _clean(df):
    df = df.copy()
    idx = pd.to_datetime(df.index)
    df.index = idx.tz_localize(None) if idx.tz is not None else idx
    return df.dropna(how="all")


def prices(period="2y"):
    raw = yf.download(list(TICKERS.values()), period=period, interval="1d",
                      auto_adjust=False, progress=False, group_by="ticker")
    out = {}
    for k, t in TICKERS.items():
        try:
            out[k] = _clean(raw[t])
        except Exception:
            pass
    return out


def intraday(ticker="GC=F", period="5d", interval="5m"):
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return _clean(df)


def real_yield():
    """10Y TIPS real yield (FRED DFII10), no API key needed."""
    r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10", timeout=20)
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "ry"]
    df["date"] = pd.to_datetime(df["date"])
    df["ry"] = pd.to_numeric(df["ry"], errors="coerce")
    return df.dropna().set_index("date")["ry"]


def cot(n=120):
    """CFTC disaggregated futures-only, gold. Managed-money net position (weekly, Tuesday data)."""
    url = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
    q = {"$where": f"cftc_contract_market_code='{GOLD_COT_CODE}'",
         "$order": "report_date_as_yyyy_mm_dd DESC", "$limit": n}
    df = pd.DataFrame(requests.get(url, params=q, timeout=25).json())
    df["date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
    df["mm_net"] = df["m_money_positions_long_all"].astype(float) - df["m_money_positions_short_all"].astype(float)
    df["open_interest"] = df["open_interest_all"].astype(float)
    return df.sort_values("date").set_index("date")[["mm_net", "open_interest"]]


def option_walls(ratio, expiries=3, top=3):
    """Largest open-interest strikes in GLD options, converted to gold futures price.
    These act as dealer-hedging magnets; they are a proxy for resting order clusters."""
    t = yf.Ticker("GLD")
    calls, puts = [], []
    for e in t.options[:expiries]:
        ch = t.option_chain(e)
        calls.append(ch.calls[["strike", "openInterest"]])
        puts.append(ch.puts[["strike", "openInterest"]])
    c = pd.concat(calls).groupby("strike")["openInterest"].sum()
    p = pd.concat(puts).groupby("strike")["openInterest"].sum()
    conv = lambda s: sorted((s.nlargest(top).index.values * ratio).round(0).tolist())
    return {"call_walls": conv(c), "put_walls": conv(p),
            "put_call_oi": float(p.sum() / max(c.sum(), 1))}


def premium_order_flow():
    """Plug-in point for real order-book data (CME MDP via Databento, Polygon, Rithmic...).
    Return {"bid_depth":..., "ask_depth":..., "delta":...} and the engine will use it.
    Left empty on purpose: needs your paid key (set DATABENTO_API_KEY etc.)."""
    if not os.getenv("DATABENTO_API_KEY"):
        return None
    return None  # implement with your vendor's client


def collect(period="2y"):
    """Fetch everything; return dict with data + a status log per feed."""
    d, status = {}, {}
    for name, fn in [("px", lambda: prices(period)), ("ry", real_yield), ("cot", cot)]:
        try:
            d[name] = fn(); status[name] = "ok"
        except Exception as e:
            d[name] = None; status[name] = f"failed: {type(e).__name__}"
    try:
        g, l = d["px"]["gold"]["Close"].iloc[-1], d["px"]["gld"]["Close"].iloc[-1]
        d["walls"] = option_walls(float(g / l)); status["options"] = "ok"
    except Exception as e:
        d["walls"] = None; status["options"] = f"failed: {e}"
d["flow"] = premium_order_flow()
    status["order_flow"] = "ok" if d["flow"] else "not configured (needs paid feed)"
    d["status"] = status
    return d
