"""Honest test of the price-based part of the score. Run: python backtest.py"""
import numpy as np, pandas as pd
import feeds, engine as E


def run(X):
    f = E.hist_factors(X)
    keep = f[E.HISTORY_FACTORS].copy()
    s = keep.apply(E.score, axis=1)
    r1 = X["gold"].pct_change().shift(-1) * 100
    df = pd.DataFrame({"score": s, "r1": r1}).dropna()
    df["bucket"] = pd.cut(df["score"], [-101, -40, -15, 15, 40, 101])
    g = df.groupby("bucket", observed=True)["r1"].agg(n="count", mean_next_day_pct="mean",
                                                        up_rate=lambda x: (x > 0).mean())
    pos = np.sign(df["score"]).where(df["score"].abs() > 15, 0)
    strat = pos * df["r1"] - (pos.diff().abs().fillna(0) * 0.01)   # 1bp cost per flip
    sharpe = strat.mean() / strat.std() * np.sqrt(252) if strat.std() else float("nan")
    half = len(df) // 2
    c1 = df["score"].iloc[:half].corr(df["r1"].iloc[:half], method="spearman")
    c2 = df["score"].iloc[half:].corr(df["r1"].iloc[half:], method="spearman")
    curve = pd.DataFrame({"strategy_%": strat.cumsum(), "buy_and_hold_%": df["r1"].cumsum()})
    return g, {"corr_first_half": c1, "corr_second_half": c2, "corr(score,next-day)": df["score"].corr(df["r1"], method="spearman"),
               "strategy_sharpe": sharpe, "buy_hold_sharpe": df["r1"].mean() / df["r1"].std() * np.sqrt(252),
               "days": len(df)}, curve


if __name__ == "__main__":
    px = feeds.prices("10y"); ry = feeds.real_yield()
    g, stats, _ = run(E.build_history(px, ry))
    print(g.to_string()); print({k: round(v, 3) for k, v in stats.items()})
    print("\nIf correlation is near 0 and Sharpe <= buy-and-hold, the score has no edge. Believe the numbers.")
