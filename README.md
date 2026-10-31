# Gold Timing Desk (Python, live)
    pip install -r requirements.txt
    streamlit run app.py        # live dashboard, refreshes every 1-2 min or on button
    python backtest.py          # check whether the score has any edge, BEFORE trusting it

Feeds (free): Yahoo Finance (GC=F, GLD, DXY, oil, VIX, 5-min bars, GLD options), FRED 10Y real yield, CFTC COT.
Real order flow: implement `premium_order_flow()` in feeds.py with your paid vendor key.
Tune `WEIGHTS` and `OIL_SIGN` in engine.py. Spot gold has no public order book; these are proxies.
