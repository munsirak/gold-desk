"""Scoring, levels and timing. Pure functions, no network."""
import datetime as dt
import numpy as np, pandas as pd

WEIGHTS = {"real_yield": 2.0, "dollar": 1.5, "oil_inflation": 1.0, "risk_vix": 0.5,
           "trend": 1.5, "rsi_reversion": 0.7, "cot": 1.0, "gld_flow": 1.0}
OIL_SIGN = -1   # -1: oil up = bearish gold (rate-hike regime). Use +1 in a safe-haven regime.
HISTORY_FACTORS = ["real_yield", "dollar", "oil_inflation", "risk_vix", "trend", "rsi_reversion"]


def _z(s, n=5, win=120):
    ch = s.diff(n)
    return (ch / ch.rolling(win, min_periods=30).std()).clip(-2, 2)


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df, n=14):
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - df["Close"].shift()).abs(),
                    (df["Low"] - df["Close"].shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def build_history(px, ry):
    X = pd.DataFrame({k: px[k]["Close"] for k in ("gold", "dxy", "oil", "vix") if k in px})
    X = X.dropna(subset=["gold"]).ffill()
    X["ry"] = ry.reindex(X.index.union(ry.index)).ffill().reindex(X.index)
    return X


def hist_factors(X):
    g = X["gold"]
    f = pd.DataFrame(index=X.index)
    f["real_yield"] = -_z(X["ry"])
    f["dollar"] = -_z(X["dxy"])
    f["oil_inflation"] = OIL_SIGN * _z(X["oil"])
    f["risk_vix"] = _z(X["vix"])
    f["trend"] = ((g > g.ewm(span=20).mean()).astype(int) * 2 - 1) + ((g > g.ewm(span=50).mean()).astype(int) * 2 - 1)
    f["rsi_reversion"] = ((50 - rsi(g)) / 15).clip(-2, 2)
    return f


def score(row):
    items = [(k, v) for k, v in dict(row).items() if k in WEIGHTS and pd.notna(v)]
    den = sum(2 * WEIGHTS[k] for k, _ in items)
    return 100 * sum(WEIGHTS[k] * float(v) for k, v in items) / den if den else 0.0


def cot_factor(c):
    if c is None or len(c) < 20:
        return np.nan
    d = c["mm_net"].diff()
    f = float(np.clip(d.iloc[-1] / d.std(), -2, 2))
    pct = c["mm_net"].rank(pct=True).iloc[-1]
    f += -1 if pct > .9 else 1 if pct < .1 else 0   # crowded positioning is a contrarian risk
    return float(np.clip(f, -2, 2))


def gld_flow_factor(gld):
    """Proxy only: on-balance volume momentum. Real ETF holdings need a paid or scraped source."""
    if gld is None or len(gld) < 60:
        return np.nan
    obv = (np.sign(gld["Close"].diff()) * gld["Volume"]).cumsum()
    return float(_z(obv).iloc[-1])


def live_factors(data):
    X = build_history(data["px"], data["ry"] if data["ry"] is not None else pd.Series(dtype=float))
    f = hist_factors(X).ffill().iloc[-1].to_dict()
    f["cot"] = cot_factor(data.get("cot"))
    f["gld_flow"] = gld_flow_factor(data["px"].get("gld"))
    return f


def levels(daily, spot):
    last = daily.index[-1].date()
    bar = daily.dropna().iloc[-2] if last == dt.datetime.utcnow().date() else daily.dropna().iloc[-1]
    H, L, C = float(bar["High"]), float(bar["Low"]), float(bar["Close"])
    p = (H + L + C) / 3
    return {"R2": p + (H - L), "R1": 2 * p - L, "Pivot": p, "S1": 2 * p - H, "S2": p - (H - L),
            "ATR": float(atr(daily).dropna().iloc[-1])}


def verdict(sc, spot, lv, walls=None):
    A = lv["ATR"]
    bias = "bull" if sc >= 25 else "bear" if sc <= -25 else "neutral"
    buy = lv["S2"] if bias == "bear" else lv["S1"]
    sell = lv["R2"] if bias == "bull" else lv["R1"]
    if walls:   # snap to nearby option-OI magnets
        near = lambda x, ws: min(ws, key=lambda w: abs(w - x)) if ws and min(abs(w - x) for w in ws) < A * .5 else x
        buy, sell = near(buy, walls["put_walls"]), near(sell, walls["call_walls"])
    sig = {"bear": "SELL RALLIES", "bull": "BUY DIPS", "neutral": "WAIT"}[bias]
    if bias != "bear" and abs(spot - buy) < A * .25: sig = "BUY ZONE NOW"
    if bias != "bull" and abs(spot - sell) < A * .25: sig = "SELL ZONE NOW"
    return {"bias": bias, "signal": sig, "score": round(sc),
            "buy_zone": (buy - A * .2, buy + A * .2), "buy_stop": buy - A * .5,
            "sell_zone": (sell - A * .2, sell + A * .2), "sell_stop": sell + A * .5,
            "targets_up": [lv["Pivot"], lv["R1"]], "targets_down": [lv["Pivot"], lv["S1"]]}


def probs(sc):
    u, d, g = np.exp(sc / 40), np.exp(-sc / 40), 1.1
    t = u + d + g
    return {"up": u / t, "range": g / t, "down": d / t}


# (name, start_utc_min, end_utc_min, note, kind)
WINDOWS = [("Asia", 0, 420, "thin, choppy", "wait"), ("London open", 420, 540, "sets direction", "key"),
           ("London AM fix", 630, 690, "benchmark flow", "key"), ("COMEX open + US data", 800, 900, "highest volume", "key"),
           ("London PM fix", 900, 960, "late reversals", "mid"), ("US afternoon", 960, 1260, "fades, profit-taking", "late")]


def rate_window(kind, bias):
    if kind == "wait": return "Avoid"
    if bias == "bear": return {"key": "Best to SELL rallies", "mid": "Sell at resistance", "late": "Best to BUY bounce / cover"}[kind]
    if bias == "bull": return {"key": "Best to BUY dips", "mid": "Buy at support", "late": "Take profit"}[kind]
    return "Wait for breakout" if kind == "key" else "Fade range edges"
