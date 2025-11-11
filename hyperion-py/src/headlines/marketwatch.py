from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from gdeltdoc import GdeltDoc, Filters

def pull_marketwatch(ticker: str, years_back: int = 10, extra_terms=None,
                     window_months: int = 1, per_call_max: int = 250) -> pd.DataFrame:
    """
    Pull MarketWatch headlines mentioning `ticker` via GDELT DOC 2.0 using gdeltdoc.
    Sweeps backwards in monthly windows to cover long ranges.
    """
    print("Pulling MarketWatch articles for", ticker, "going back", years_back, "years")
    terms = [ticker]
    if extra_terms:
        terms.extend(extra_terms)

    gd = GdeltDoc()
    end = datetime.utcnow()
    start = end - relativedelta(years=years_back)

    rows = []
    seen = set()

    cur = start
    while cur < end:
        nxt = min(cur + relativedelta(months=window_months), end)

        f = Filters(
            start_date=cur,                # can be datetime
            end_date=nxt,
            num_records=per_call_max,      # DOC API returns up to ~250 per call
            keyword=terms,                 # list => OR across terms
            domain_exact="marketwatch.com" # lock to MarketWatch only
        )

        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', 2000)
        pd.set_option('display.max_columns', None)

        try:
            df = gd.article_search(f)      # returns pandas DataFrame
        except Exception as e:
            print(f"window {cur} -> {nxt} error: {e}")
            df = pd.DataFrame()

        if df.empty:
            print("No articles for window", cur.date(), "->", nxt.date())

        if not df.empty:
            # keep only unique URLs
            new = df[~df["url"].isin(seen)].copy()
            rows.append(new)
            seen.update(new["url"].tolist())
            print("Fetched", len(new), "new articles for window", cur.date(), "->", nxt.date())

        cur = nxt

    if rows:
        out = pd.concat(rows, ignore_index=True)
        # standardise and sort
        out.rename(columns={"seendate": "date"}, inplace=True)
        out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True)
        out = out.sort_values("date", ascending=False).reset_index(drop=True)
        # keep the useful columns
        cols = ["date", "title", "url", "domain", "language", "sourcecountry"]
        return out[[c for c in cols if c in out.columns]]
    else:
        return pd.DataFrame(columns=["date", "title", "url", "domain", "language", "sourcecountry"])
