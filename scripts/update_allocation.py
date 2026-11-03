#!/usr/bin/env python3
"""Publish month-end MODEL weights (not orders or investment advice)."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

OUT = Path("data/allocations.json")
RISKY = ["SPY", "IWM", "QQQ", "VGK", "EWJ", "VWO", "VNQ", "GSG", "GLD", "TLT", "HYG", "LQD"]
DEFENSIVE = ["SHY", "IEF", "LQD"]
CANARIES = ["VWO", "BND"]
ALL = sorted(set(RISKY + DEFENSIVE + CANARIES + ["IWD", "BIL", "DBC"]))


def get_monthly_prices(cutoff):
    start = (cutoff - pd.DateOffset(months=18)).strftime("%Y-%m-%d")
    end = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    daily = yf.download(
        ALL, start=start, end=end, group_by="ticker",
        auto_adjust=True, threads=False, progress=False
    )
    if daily.empty:
        raise RuntimeError("Yahoo Finance returned no ETF prices")
    monthly = {}
    dates = {}
    for ticker in ALL:
        try:
            series = daily[ticker]["Close"].dropna().astype(float)
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Missing ETF price: {ticker}") from exc
        series = series.loc[series.index.tz_localize(None) <= cutoff]
        if series.empty or series.index[-1].to_period("M") != cutoff.to_period("M"):
            raise RuntimeError(f"{ticker}: missing completed month-end price")
        dates[ticker] = series.index[-1].date().isoformat()
        m = series.resample("ME").last().asfreq("ME")
        if len(m) < 13 or m.iloc[-13:].isna().any():
            raise RuntimeError(f"{ticker}: need 13 consecutive month-end closes")
        monthly[ticker] = m
    return monthly, dates


def get_unemployment(cutoff):
    # At the end of month M, M's unemployment is not yet published.
    # Use only observations dated no later than M-1; no look-ahead.
    response = requests.get(
        "https://fred.stlouisfed.org/graph/fredgraph.csv",
        params={"id": "UNRATE"}, timeout=45
    )
    response.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(response.text))
    if len(df.columns) < 2:
        raise RuntimeError("Unexpected FRED UNRATE CSV layout")
    observations = pd.Series(
        pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(),
        index=pd.to_datetime(df.iloc[:, 0], errors="coerce")
    ).dropna().sort_index()
    eligible = observations.loc[observations.index < cutoff.replace(day=1)]
    if len(eligible) < 12:
        raise RuntimeError("Need 12 published monthly unemployment readings")
    if eligible.index[-1].to_period("M") != (cutoff - pd.DateOffset(months=1)).to_period("M"):
        raise RuntimeError("Latest eligible unemployment release is unexpectedly stale")
    trailing = eligible.iloc[-12:]
    return {
        "period": eligible.index[-1].strftime("%Y-%m"),
        "value": round(float(trailing.iloc[-1]), 4),
        "mean_12": round(float(trailing.mean()), 4),
        "warning": bool(trailing.iloc[-1] > trailing.mean()),
    }


def momentum(monthly_close):
    p = monthly_close.iloc[-1]
    # 13612W: weighted 1/3/6/12-month percentage returns, proportional
    # to 12, 4, 2, 1. Equivalent for ranks/sign to weighted mean.
    return float(sum(w * (p / monthly_close.iloc[-(n + 1)] - 1)
                     for n, w in [(1, 12), (3, 4), (6, 2), (12, 1)]))


def check_weights(weights):
    if any(v < 0 for v in weights.values()) or abs(sum(weights.values()) - 1) > 1e-8:
        raise ValueError(f"Invalid allocation: {weights}")
    return {k: round(v, 8) for k, v in sorted(weights.items()) if v > 0}


def calculate(prices, unemployment):
    permanent = check_weights({"SPY": .25, "TLT": .25, "GLD": .25, "BIL": .25})
    all_weather = check_weights({"SPY": .30, "TLT": .40, "IEF": .15,
                                 "GLD": .075, "DBC": .075})
    ma = float(prices["SPY"].iloc[-10:].mean())
    spy = float(prices["SPY"].iloc[-1])
    spy_warning = spy < ma
    laa_defensive = unemployment["warning"] and spy_warning
    laa = check_weights({"IWD": .25, "GLD": .25, "IEF": .25,
                         ("SHY" if laa_defensive else "QQQ"): .25})
    scores = {ticker: momentum(prices[ticker]) for ticker in ALL}
    bad = [t for t in CANARIES if scores[t] <= 0]
    safe = max(DEFENSIVE, key=lambda t: scores[t])
    ordered = sorted(RISKY, key=lambda t: (-scores[t], t))
    n = 6 if len(bad) == 0 else 3 if len(bad) == 1 else 0
    daa = {ticker: 1 / 6 for ticker in ordered[:n]}
    if bad:
        daa[safe] = daa.get(safe, 0) + len(bad) / 2
    daa = check_weights(daa)
    return {
        "permanent": {"weights": permanent, "status": "static"},
        "all_weather": {"weights": all_weather, "status": "static"},
        "laa": {
            "weights": laa, "status": "computed",
            "signal": {
                "unemployment_warning": unemployment["warning"],
                "spy_below_10_month_mean": spy_warning,
                "sleeve": "SHY" if laa_defensive else "QQQ",
                "unemployment": unemployment,
                "spy_monthly_close": round(spy, 4),
                "spy_10_month_mean": round(ma, 4),
            },
        },
        "daa": {
            "weights": daa, "status": "computed",
            "signal": {
                "canary_momentum": {t: round(scores[t], 6) for t in CANARIES},
                "bad_canaries": bad,
                "defensive_fraction": len(bad) / 2,
                "best_defensive": safe,
                "offensive_ranking": ordered,
            },
        },
    }


def main():
    today = datetime.now(timezone.utc)
    cutoff = pd.Timestamp(today.date().replace(day=1)) - pd.Timedelta(days=1)
    cutoff = cutoff.normalize()
    prices, trading_dates = get_monthly_prices(cutoff)
    unemployment = get_unemployment(cutoff)
    calculated = calculate(prices, unemployment)
    old = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    old_history = [entry for entry in old.get("history", [])
                   if entry.get("period") != cutoff.strftime("%Y-%m")]
    if old.get("period") and old["period"] != cutoff.strftime("%Y-%m"):
        old_history.insert(0, {"period": old["period"], "strategies": old.get("strategies", {})})
    new = {
        "period": cutoff.strftime("%Y-%m"),
        "signal_as_of": cutoff.date().isoformat(),
        "updated_at_utc": today.isoformat(timespec="seconds"),
        "input_prices_through": trading_dates,
        "pricing_note": "Yahoo Finance adjusted ETF prices; distributions may differ from original total-return indices.",
        "signal_note": "US ETF signals drive both implementations. FRED unemployment is lagged one observation month.",
        "strategies": calculated,
        "history": old_history[:23],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Calculated all four strategies for {new['period']} ({len(trading_dates)} ETF inputs)")


if __name__ == "__main__":
    main()
