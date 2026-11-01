"""Run: streamlit run app.py"""
import datetime as dt
import pandas as pd, plotly.graph_objects as go, streamlit as st
import feeds, engine as E, backtest as B

st.set_page_config(page_title="Gold Timing Desk", layout="wide")
st.title("Gold Timing Desk (live)")


@st.cache_data(ttl=120)
def load():
    return feeds.collect()


@st.cache_data(ttl=60)
def live_bars():
    return feeds.intraday()


if st.sidebar.button("Refresh now"):
    st.cache_data.clear()
data = load()
st.sidebar.write("Feed status")
for k, v in data["status"].items():
    st.sidebar.write(("✅ " if v == "ok" else "⚠️ ") + f"{k}: {v}")
if not data["px"] or "gold" not in data["px"]:
    st.error("No price data. Check your internet connection."); st.stop()

bars = live_bars()
spot = float(bars["Close"].dropna().iloc[-1]) if len(bars) else float(data["px"]["gold"]["Close"].iloc[-1])
f = E.live_factors(data) if data["ry"] is not None else {}
sc = E.score(f); lv = E.levels(data["px"]["gold"], spot); p = E.probs(sc)
walls = None
if data["walls"]:   # keep only levels on the useful side of price and within 4 ATR
    A4 = 4 * lv["ATR"]
    walls = dict(data["walls"])
    walls["call_walls"] = [w for w in walls["call_walls"] if spot < w <= spot + A4]
    walls["put_walls"] = [w for w in walls["put_walls"] if spot - A4 <= w < spot]
v = E.verdict(sc, spot, lv, walls)
_n = dt.datetime.utcnow(); _m = _n.hour * 60 + _n.minute
if _n.weekday() == 5 or (_n.weekday() == 6 and _m < 23 * 60) or (_n.weekday() == 4 and _m >= 22 * 60):
    st.info("Gold futures are closed (weekend). Showing the last price; it reopens Sunday about 23:00 UTC.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Gold (GC=F)", f"{spot:,.0f}"); c2.metric("Signal", v["signal"]); c3.metric("Score", v["score"])
c4.metric("Up / Range / Down", f"{p['up']:.0%} / {p['range']:.0%} / {p['down']:.0%}")

fig = go.Figure(go.Candlestick(x=bars.index, open=bars["Open"], high=bars["High"], low=bars["Low"], close=bars["Close"]))
for n in ("R2", "R1", "Pivot", "S1", "S2"):
    fig.add_hline(y=lv[n], line_dash="dot", annotation_text=n, annotation_position="top left")
for w in (walls or {}).get("call_walls", []): fig.add_hline(y=w, line_color="red", annotation_position="top left", annotation_text=("call wall" if (walls or {}).get("source") == "options" else "resistance (volume)"))
for w in (walls or {}).get("put_walls", []): fig.add_hline(y=w, line_color="green", annotation_position="top left", annotation_text=("put wall" if (walls or {}).get("source") == "options" else "support (volume)"))
fig.add_hrect(y0=v["buy_zone"][0], y1=v["buy_zone"][1], fillcolor="green", opacity=.15, line_width=0)
fig.add_hrect(y0=v["sell_zone"][0], y1=v["sell_zone"][1], fillcolor="red", opacity=.15, line_width=0)
_A = lv["ATR"]
fig.update_yaxes(range=[min(spot - 2.5 * _A, v["buy_zone"][0] - .3 * _A), max(spot + 2.5 * _A, v["sell_zone"][1] + .3 * _A)])
fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, use_container_width=True)

a, b = st.columns(2)
with a:
    st.subheader("Trade plan")
    st.write(f"**Buy zone** {v['buy_zone'][0]:,.0f}-{v['buy_zone'][1]:,.0f}, stop {v['buy_stop']:,.0f}, targets {', '.join(f'{x:,.0f}' for x in v['targets_up'])}")
    st.write(f"**Sell zone** {v['sell_zone'][0]:,.0f}-{v['sell_zone'][1]:,.0f}, stop {v['sell_stop']:,.0f}, targets {', '.join(f'{x:,.0f}' for x in v['targets_down'])}")
    st.subheader("Factors (-2 bearish to +2 bullish)")
    st.dataframe(pd.DataFrame({"factor": list(f), "value": [round(x, 2) if pd.notna(x) else None for x in f.values()],
                               "weight": [E.WEIGHTS.get(k) for k in f]}), hide_index=True)
with b:
    st.subheader("Best time to buy and sell")
    now = dt.datetime.utcnow(); m = now.hour * 60 + now.minute; base = dt.datetime(now.year, now.month, now.day, tzinfo=dt.timezone.utc)
    rows = []
    for n, s, e, note, kind in E.WINDOWS:
        t = lambda x: (base + dt.timedelta(minutes=x)).astimezone().strftime("%H:%M")
        rows.append({"window": n + (" ← now" if s <= m < e else ""), "your time": f"{t(s)}-{t(e)}",
                     "action": E.rate_window(kind, v["bias"]), "note": note})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.subheader("Positioning and walls")
    c = data.get("cot")
    if c is not None: st.write(f"CFTC managed-money net: {c['mm_net'].iloc[-1]:,.0f} ({c.index[-1].date()}), change {c['mm_net'].diff().iloc[-1]:+,.0f}")
    if data["walls"]: st.write(data["walls"])
with st.expander("Backtest: does this score have an edge?"):
    st.write("Tests the 6 price-based factors on 10 years of daily data. COT, options and flow have no long free history, so they are not tested.")
    if st.button("Run backtest"):
        with st.spinner("Downloading 10 years and testing..."):
            g, stats, curve = B.run(E.build_history(feeds.prices("10y"), feeds.real_yield()))
        st.write("Average next-day gold move by score bucket")
        st.dataframe(g.round(3))
        st.json({k: round(float(v), 3) for k, v in stats.items()})
        st.line_chart(curve)
        edge = stats["corr(score,next-day)"] > 0.03 and stats["corr_first_half"] > 0 and stats["corr_second_half"] > 0 and stats["strategy_sharpe"] > stats["buy_hold_sharpe"]
        st.success("Some edge shown in both halves. Still treat it as weak and test costs.") if edge else st.warning("No reliable edge: the score does not consistently predict the next day. Do not trade it as a signal.")
st.caption("A decision aid, not a prediction or financial advice. Spot gold has no public order book; options walls, COT and volume are proxies.")
