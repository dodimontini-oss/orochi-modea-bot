"""
Mode A ("Orochi"-inspired auction rejection / mean reversion) live bot -
the ONE variant from orochi_vwap_volprofile_lab.py (see futures_orb/) that
held up under walk-forward, a long/short split, AND a real 2022
bear-market stress test. This is a from-scratch reconstruction of the
PUBLICLY NAMED concepts behind a paid "Orochi framework" trading course
(Auction Market Theory, Volume Profile, VWAP, order flow) - NOT a copy of
that course's actual undisclosed rules, which were never seen.

Deployed at RR_RATIO=2.0 specifically - RR=1.5 was meaningfully less
robust in every check run against it (first-half profit factor 0.99,
weak short side) and was deliberately NOT deployed.

DEDICATED ALPACA PAPER ACCOUNT - as of 2026-09-08, its own account,
separate from every other bot. (History: originally deployed sharing the
crypto bot's account, since that account doesn't trade QQQ itself - fine
in isolation, but it meant this bot's account WAS shadowed by another
QQQ strategy the moment orderflow_cvd15min_bot was scoped, since sharing
still collides whenever ANY two QQQ-trading bots land on the same
account. Moved to its own account at the same time for that reason, once
a second Alpaca login provided the extra paper-account slots to do it
properly.) Both orb-bot-5min and orb-bot-15min ALSO trade QQQ on their
own separate accounts - sharing with any of them (or leaving this on a
shared account at all) risks netting positions together with no way to
tell which bot owns what shares, and one bot's stop closing out shares
another bot still thinks it holds. ALPACA_API_KEY/ALPACA_SECRET_KEY below
MUST point at this bot's own dedicated paper account.

Rules (exactly matching the validated backtest - orochi_vwap_volprofile_lab.py, Mode A):
1. Build the most recently COMPLETED session's Volume Profile from 5-min
   bars: bin by typical price (H+L+C)/3, ~40 bins across that day's
   range, POC = highest-volume bin, Value Area = 70% of volume expanded
   outward from POC one bin at a time -> VAH/VAL.
2. Build today's session VWAP + volume-weighted 2SD bands, computed
   causally (only bars up to and including "now" - no lookahead).
3. Scan today's bars from bar #6 onward (skips the noisy first ~30 min)
   for the FIRST occurrence of: a bar closing beyond yesterday's VAH/VAL
   AND beyond today's VWAP+-2SD band at the same time (an "excess"),
   immediately followed by the next bar closing back inside that level
   (a "rejection") -> confirms the entry, fading back toward VWAP. At
   most one entry attempt per session, matching the backtest.
4. Stop = the excess bar's high (short) / low (long). Target = entry +/-
   2x that risk distance. Real Alpaca BRACKET order (entry+stop+target in
   one call, GTC so the legs survive past today's close, matching "no
   session-close flatten" in the backtest).

Checks ALL of today's bars each run (not just the newest), so an
occasional missed/delayed GitHub Actions run doesn't cause a missed
signal - same resilience pattern as futures_orb/orb_live_bot.py, which
this file's structure otherwise closely follows.

Runs every 5 minutes via cron during market hours - see orb_live_bot.py's
docstring for the DST-safe approach this copies (checks real
America/New_York time itself rather than relying on a DST-adjusted cron
schedule).

Environment variables required (a DEDICATED Alpaca paper account - see above):
    ALPACA_API_KEY
    ALPACA_SECRET_KEY
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from trading_core import execution as safety

import logging
import os
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("orochi-modea-live")

ALPACA_API_KEY = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
TRADING_BASE_URL = "https://paper-api.alpaca.markets"  # paper only - never change without a deliberate decision
DATA_BASE_URL = "https://data.alpaca.markets"
HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}

SYMBOL = "QQQ"
RR_RATIO = 2.0  # the one variant that held up in every check - see docstring
RISK_PER_TRADE_PCT = 1.0
# Caps position notional at MAX_LEVERAGE x equity, matching a standard
# Robinhood Gold / Reg-T margin account's OVERNIGHT buying power (2x
# equity) - not Robinhood's 4x day-trade buying power, since that only
# applies to intraday round trips on a PDT-flagged account and evaporates
# by end of day; this bot holds positions overnight (GTC, no session-
# close flatten), so 2x is the honest real-broker comparison. Added
# 2026-09-08 after a live trade sized to ~4x equity (tight stop distance
# let 1%-risk sizing call for far more shares than the account's actual
# Alpaca paper buying power realistically should allow) - only ever
# shrinks qty, same as the buying-power cap below; never grows it.
MAX_LEVERAGE = 2.0
VALUE_AREA_PCT = 0.70
MIN_BARS_BEFORE_ENTRY = 6

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)


def market_is_open_now() -> bool:
    global SESSION_CLOSE
    broker=safety.Alpaca(TRADING_BASE_URL,HEADERS)
    session=broker.session()
    if session is None:
        return False
    opening,SESSION_CLOSE=session
    return opening <= pd.Timestamp.now(tz="America/New_York") < SESSION_CLOSE


def fetch_bars(start_et: datetime, end_et: datetime) -> pd.DataFrame:
    params = {
        "timeframe": "5Min",
        "start": start_et.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end_et.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": 2000,
        "feed": "iex",
    }
    resp = requests.get(f"{DATA_BASE_URL}/v2/stocks/{SYMBOL}/bars", headers=HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    bars = resp.json().get("bars", [])
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df["time_et"] = df["t"].dt.tz_convert(ET)
    df["session_date"] = df["time_et"].dt.date
    df["typical"] = (df["high"] + df["low"] + df["close"]) / 3
    return safety.closed_rth(df.sort_values("t").reset_index(drop=True))


def get_today_bars() -> pd.DataFrame:
    now_et = datetime.now(ET)
    start_of_day = now_et.replace(hour=0, minute=0, second=0, microsecond=0)
    return fetch_bars(start_of_day, now_et)


def _volume_profile(day_df: pd.DataFrame):
    return safety.value_area(day_df, VALUE_AREA_PCT)


def get_prior_session_profile():
    """(profile_dict_or_None, prior_date_or_None) for the most recently
    COMPLETED trading session before today - looks back 10 calendar days
    to skip weekends/holidays safely without a market-calendar dependency."""
    now_et = datetime.now(ET)
    today = now_et.date()
    lookback_start = (now_et - timedelta(days=10)).replace(hour=0, minute=0, second=0, microsecond=0)
    df = fetch_bars(lookback_start, now_et)
    if df.empty:
        return None, None
    prior_dates = sorted(d for d in df["session_date"].unique() if d < today)
    if not prior_dates:
        return None, None
    prior_date = prior_dates[-1]
    day_df = df[df["session_date"] == prior_date]
    return _volume_profile(day_df), prior_date


def session_vwap_bands(day_df: pd.DataFrame) -> pd.DataFrame:
    """Causal (no-lookahead) session VWAP + 2SD bands - identical formula
    to orochi_vwap_volprofile_lab.py's session_vwap_bands()."""
    v = day_df["volume"].to_numpy()
    p = day_df["typical"].to_numpy()
    cum_v = np.cumsum(v)
    cum_pv = np.cumsum(p * v)
    cum_pv2 = np.cumsum(p * p * v)
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = cum_pv / cum_v
        variance = cum_pv2 / cum_v - vwap ** 2
    std = np.sqrt(np.clip(variance, 0, None))
    out = day_df.copy()
    out["vwap"] = vwap
    out["vwap_up2"] = vwap + 2 * std
    out["vwap_dn2"] = vwap - 2 * std
    return out


def get_position():
    resp = requests.get(f"{TRADING_BASE_URL}/v2/positions/{SYMBOL}", headers=HEADERS, timeout=10)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def get_open_orders() -> list:
    resp = requests.get(f"{TRADING_BASE_URL}/v2/orders", headers=HEADERS,
                         params={"status": "open", "symbols": SYMBOL}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def already_traded_today() -> bool:
    """Any order (filled, open, or otherwise) for this symbol submitted
    today counts as 'already attempted' - matches the backtest's one-
    entry-attempt-per-session rule."""
    now_et = datetime.now(ET)
    start_of_day = now_et.replace(hour=0, minute=0, second=0, microsecond=0)
    resp = requests.get(f"{TRADING_BASE_URL}/v2/orders", headers=HEADERS,
                         params={"status": "all", "symbols": SYMBOL, "direction": "desc", "limit": 50,
                                 "after": start_of_day.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")},
                         timeout=10)
    resp.raise_for_status()
    return len(resp.json()) > 0


def get_account_info() -> dict:
    resp = requests.get(f"{TRADING_BASE_URL}/v2/account", headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_latest_trade_price() -> float:
    """Real-time last-trade price, used as a final freshness check right
    before order submission. `current_price` above (a 5-min bar's close)
    can itself be almost 5 minutes stale, and several more API calls
    (get_account_info, etc.) pass before the order actually reaches
    Alpaca - real incident 2026-09-08: the bar-close check passed, but by
    submission time live price had fallen further through the stop and
    Alpaca correctly rejected it (422, twice in a row, each crashing the
    run) since the rejected order was never recorded so the identical
    stale signal kept retrying every 5 minutes. This closes most of that
    gap by checking again against an actual live trade price."""
    resp = requests.get(f"{DATA_BASE_URL}/v2/stocks/{SYMBOL}/trades/latest", headers=HEADERS,
                         params={"feed": "iex"}, timeout=10)
    resp.raise_for_status()
    return float(resp.json()["trade"]["p"])


def place_bracket_order(direction: str, qty: int, stop: float, target: float, client_id=None) -> dict:
    if client_id is None:
        raise ValueError("Missing signal identity")
    return safety.Alpaca(TRADING_BASE_URL,HEADERS).submit({"symbol":SYMBOL,"qty":str(qty),
        "side":"buy" if direction=="LONG" else "sell","type":"market","time_in_force":"gtc",
        "order_class":"bracket","take_profit":{"limit_price":str(round(target,2))},
        "stop_loss":{"stop_price":str(round(stop,2))},"client_order_id":client_id})


def check_and_trade():
    if not market_is_open_now():
        log.info("Outside regular market hours (9:30-16:00 ET, weekdays). No action.")
        return

    position = get_position()
    if position is not None and float(position["qty"]) != 0:
        log.info("Already in a position (%s %s shares). Bracket order manages the exit. No action.",
                  position["side"], position["qty"])
        return

    if get_open_orders():
        log.info("Open order(s) already pending on %s. No action.", SYMBOL)
        return

    if already_traded_today():
        log.info("Already attempted an entry today. No action (one attempt per session).")
        return

    profile, prior_date = get_prior_session_profile()
    if profile is None:
        log.info("Could not build a usable prior-session Volume Profile (prior_date=%s). No action.", prior_date)
        return
    vah, val = profile["vah"], profile["val"]

    today_df = get_today_bars()
    if len(today_df) < MIN_BARS_BEFORE_ENTRY + 1:
        log.info("Only %d bars so far today - need at least %d before trusting the VWAP bands. No action.",
                  len(today_df), MIN_BARS_BEFORE_ENTRY + 1)
        return

    vwap_df = session_vwap_bands(today_df)

    direction = entry_ref = stop = None
    for i in range(MIN_BARS_BEFORE_ENTRY, len(vwap_df) - 1):
        row = vwap_df.iloc[i]
        nxt = vwap_df.iloc[i + 1]
        excess_short = row["close"] > vah and row["close"] > row["vwap_up2"]
        excess_long = row["close"] < val and row["close"] < row["vwap_dn2"]
        if excess_short and nxt["close"] < vah:
            direction, entry_ref, stop = "SHORT", nxt["close"], row["high"]
            break
        if excess_long and nxt["close"] > val:
            direction, entry_ref, stop = "LONG", nxt["close"], row["low"]
            break

    if direction is None:
        log.info("No rejection signal yet today (prior-session VAH=%.2f VAL=%.2f, latest close=%.2f). No action.",
                  vah, val, vwap_df.iloc[-1]["close"])
        return

    stop_distance = abs(entry_ref - stop)
    if stop_distance <= 0:
        log.warning("Zero-width stop distance - skipping.")
        return

    # Same staleness guard as orb_live_bot.py: a delayed run could act on a
    # signal that's no longer valid (price has already moved back past the
    # fixed stop level before the order is even submitted) - Alpaca rejects
    # a bracket order whose stop is already on the wrong side of price.
    STOP_SANITY_BUFFER = 0.01
    current_price = vwap_df.iloc[-1]["close"]
    if direction == "LONG" and current_price <= stop + STOP_SANITY_BUFFER:
        log.warning("Stale signal - price (%.2f) has fallen back through the stop (%.2f). Skipping, will "
                     "recheck next run.", current_price, stop)
        return
    if direction == "SHORT" and current_price >= stop - STOP_SANITY_BUFFER:
        log.warning("Stale signal - price (%.2f) has risen back through the stop (%.2f). Skipping, will "
                     "recheck next run.", current_price, stop)
        return

    target = entry_ref + stop_distance * RR_RATIO if direction == "LONG" else entry_ref - stop_distance * RR_RATIO

    account = get_account_info()
    equity = float(account["equity"])
    buying_power = float(account["buying_power"])
    risk_amount = equity * RISK_PER_TRADE_PCT / 100
    qty = int(risk_amount / stop_distance)
    # Risk-based sizing can call for more notional than the account can
    # actually pay for - cap qty at what's affordable so it never gets
    # rejected for insufficient buying power (same guard as orb_live_bot.py).
    max_affordable_qty = int(buying_power / current_price)
    qty = min(qty, max_affordable_qty)
    # Real incident 2026-09-08: a tight stop_distance let pure risk-based
    # sizing call for far more shares than reasonable leverage should allow
    # (a live trade sized to ~4x equity, only bounded by Alpaca's generous
    # paper buying power). Cap notional at MAX_LEVERAGE x equity too - same
    # "only ever shrinks qty" safety property as the buying-power cap above.
    max_leverage_qty = int((equity * MAX_LEVERAGE) / current_price)
    qty = min(qty, max_leverage_qty)
    if qty <= 0:
        log.warning("Computed qty <= 0 (risk_amount=%.2f stop_distance=%.4f buying_power=%.2f) - skipping.",
                     risk_amount, stop_distance, buying_power)
        return

    # Final freshness check, immediately before submission, against a real
    # live trade price rather than the (potentially minutes-stale) bar
    # close used for the checks above - see get_latest_trade_price()'s
    # docstring for the real incident this fixes.
    try:
        fresh_price = get_latest_trade_price()
    except requests.exceptions.RequestException as exc:
        log.warning("Could not fetch a live quote for the final freshness check (%s) - skipping this run, "
                     "will recheck next run.", exc)
        return
    if direction == "LONG" and fresh_price <= stop + STOP_SANITY_BUFFER:
        log.warning("Stale signal (final live-price check) - price (%.2f) has fallen back through the stop "
                     "(%.2f). Skipping, will recheck next run.", fresh_price, stop)
        return
    if direction == "SHORT" and fresh_price >= stop - STOP_SANITY_BUFFER:
        log.warning("Stale signal (final live-price check) - price (%.2f) has risen back through the stop "
                     "(%.2f). Skipping, will recheck next run.", fresh_price, stop)
        return

    stop_distance = abs(fresh_price-stop)
    qty = min(int(risk_amount/stop_distance), int(buying_power*.95/fresh_price),
              int(equity*MAX_LEVERAGE/fresh_price))
    if qty <= 0:
        return
    target = fresh_price + stop_distance*RR_RATIO if direction=="LONG" else fresh_price-stop_distance*RR_RATIO
    log.info("%s rejection signal confirmed (prior-session VAH=%.2f VAL=%.2f) - placing bracket: qty=%d "
              "stop=%.2f target=%.2f", direction, vah, val, qty, stop, target)
    try:
        result = place_bracket_order(direction, qty, stop, target, safety.signal_id("orochi", SYMBOL, pd.Timestamp.now(tz="America/New_York").normalize()))
    except requests.exceptions.HTTPError as exc:
        # Belt-and-suspenders: even the fresh-price check above has a
        # sub-second gap before the order actually lands. Alpaca rejecting
        # a bracket because price moved is an expected, recoverable
        # outcome (exactly like the local staleness skips above) - it must
        # NOT crash the script, or the rejected (never-recorded) order
        # just gets identically re-attempted and re-failed every 5 minutes
        # for the rest of the session (this is exactly what happened live
        # 2026-09-08, and to orb-bot-5min on 2026-08-28 before its own
        # equivalent guard was added - see project memory).
        log.warning("Alpaca still rejected the order despite the freshness checks (%s) - treating as a stale "
                     "signal, not a crash. Skipping, will recheck next run.", exc)
        return
    log.info("Alpaca response: %s", result)


if __name__ == "__main__":
    check_and_trade()
