"""Run: streamlit run app.py"""
import datetime as dt
import pandas as pd, plotly.graph_objects as go, streamlit as st
import feeds, engine as E

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
sc = E.score(f); lv = E.levels(data["px"]["gold"], spot); v = E.verdict(sc, spot, lv, data["walls"]); p = E.probs(sc)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Gold (GC=F)", f"{spot:,.0f}"); c2.metric("Signal", v["signal"]); c3.metric("Score", v["score"])
c4.metric("Up / Range / Down", f"{p['up']:.0%} / {p['range']:.0%} / {p['down']:.0%}")

fig = go.Figure(go.Candlestick(x=bars.index, open=bars["Open"], high=bars["High"], low=bars["Low"], close=bars["Close"]))
for n in ("R2", "R1", "Pivot", "S1", "S2"):
    fig.add_hline(y=lv[n], line_dash="dot", annotation_text=n)
for w in (data["walls"] or {}).get("call_walls", []): fig.add_hline(y=w, line_color="red", annotation_text="call wall")
for w in (data["walls"] or {}).get("put_walls", []): fig.add_hline(y=w, line_color="green", annotation_text="put wall")
fig.add_hrect(y0=v["buy_zone"][0], y1=v["buy_zone"][1], fillcolor="green", opacity=.15, line_width=0)
fig.add_hrect(y0=v["sell_zone"][0], y1=v["sell_zone"][1], fillcolor="red", opacity=.15, line_width=0)
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
st.caption("A decision aid, not a prediction or financial advice. Spot gold has no public order book; options walls, COT and volume are proxies.")
