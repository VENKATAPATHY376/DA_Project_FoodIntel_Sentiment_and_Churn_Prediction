"""Live Google Play Store review fetching (public app-store data).

Data source: Google Play Store publicly visible reviews via the
`google-play-scraper` library. This is app-store listing review data -
NOT internal Swiggy/Zomato/Zepto transaction or customer databases.
No authentication or access controls are bypassed.

Caching: results are saved to data/raw/live/ with a TTL so repeated runs
respect rate limits instead of hammering the source.
"""
import time
from datetime import datetime
from glob import glob
from pathlib import Path

import pandas as pd

from src.utils import DATA_RAW

APPS = {
    "Swiggy": "in.swiggy.android",
    "Zomato": "com.application.zomato",
    "Zepto": "com.zeptoconsumerapp",
}
COUNTRY, LANG = "in", "en"
CACHE_TTL_HOURS = 24
LIVE_DIR = DATA_RAW / "live"


def fetch_app_reviews(app_name, app_id, n, sleep_s=0.5):
    """Fetch up to n newest public reviews for one app (paged, rate-limited)."""
    from google_play_scraper import reviews, Sort
    out, token = [], None
    while len(out) < n:
        if token is None:
            batch, token = reviews(app_id, lang=LANG, country=COUNTRY,
                                   sort=Sort.NEWEST, count=min(200, n - len(out)))
        else:
            batch, token = reviews(app_id, continuation_token=token)
        if not batch:
            break
        out.extend(batch)
        if token is None:
            break
        time.sleep(sleep_s)
    d = pd.DataFrame(out[:n])
    if not d.empty:
        d["app"] = app_name
        d["appId"] = app_id
    return d


def _latest_cached(min_rows=0):
    """Newest cached file; optionally require a minimum row count (cache integrity)."""
    files = sorted(glob(str(LIVE_DIR / "playstore_reviews_*.csv")))
    files = [f for f in files if f.endswith(".csv")]
    if not files:
        return None
    if min_rows <= 0:
        return files[-1]
    # prefer the newest file that satisfies the size requirement
    for f in reversed(files):
        try:
            if sum(1 for _ in open(f, "rb")) - 1 >= min_rows:
                return f
        except OSError:
            continue
    return None


def get_live_reviews(force_refresh=False, per_app=1500):
    """Return (dataframe, status).

    Order: fresh cache (size-validated) -> live fetch -> stale cache -> RuntimeError.
    status keys: is_live, source, fetched_at, saved_file, errors
    """
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    status = {"is_live": False, "source": None, "fetched_at": None,
              "saved_file": None, "errors": []}
    # Cache integrity: a cached file must cover at least half of the requested volume
    min_rows = max(100, int(per_app * len(APPS) * 0.5))
    latest = _latest_cached(min_rows=min_rows)
    latest_any = _latest_cached()

    # 1. Fresh cache (respect TTL / rate limits)
    if latest and not force_refresh:
        try:
            mtime = datetime.fromtimestamp(Path(latest).stat().st_mtime)
            age_h = (datetime.now() - mtime).total_seconds() / 3600
            if age_h < CACHE_TTL_HOURS:
                df = pd.read_csv(latest, parse_dates=["at"])
                status.update(source="cached", fetched_at=mtime.isoformat(timespec="seconds"),
                              saved_file=latest)
                status["errors"].append(f"used cached file ({age_h:.1f}h old, TTL {CACHE_TTL_HOURS}h)")
                return df, status
        except Exception as e:
            status["errors"].append(f"cache read failed: {type(e).__name__}: {e}")

    # 2. Live fetch (public Play Store reviews)
    frames = []
    for name, pkg in APPS.items():
        try:
            d = fetch_app_reviews(name, pkg, per_app)
            if d.empty:
                raise ValueError("no reviews returned (network blocked or app id changed)")
            frames.append(d)
            status["errors"].append(f"fetched {len(d)} reviews for {name}")
        except Exception as e:
            status["errors"].append(f"{name}: {type(e).__name__}: {e}")
    if frames:
        raw = pd.concat(frames, ignore_index=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = LIVE_DIR / f"playstore_reviews_{stamp}.csv"
        raw.to_csv(path, index=False)
        status.update(is_live=True, source="live", saved_file=str(path),
                      fetched_at=datetime.now().isoformat(timespec="seconds"))
        return raw, status

    # 3. Stale cache fallback (any saved file; size-validated preferred)
    fallback = latest or latest_any
    if latest_any and latest is None:
        status["errors"].append("cached file(s) exist but are too small for the requested "
                                f"volume (need >= {min_rows} rows); attempting live fetch")
    if fallback:
        try:
            df = pd.read_csv(fallback, parse_dates=["at"])
            mtime = datetime.fromtimestamp(Path(fallback).stat().st_mtime)
            status.update(source="stale_cache", fetched_at=mtime.isoformat(timespec="seconds"),
                          saved_file=fallback)
            status["errors"].append("live fetch failed; using saved file")
            return df, status
        except Exception as e:
            status["errors"].append(f"stale cache read failed: {type(e).__name__}: {e}")

    raise RuntimeError("No live data and no saved file found. Check your internet connection.")
