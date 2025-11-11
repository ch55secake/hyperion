from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from gdeltdoc import GdeltDoc, Filters
from marketwatch import pull_marketwatch

def print_articles(df: pd.DataFrame):
    for _, r in df.iterrows():
        d = pd.to_datetime(r['date']).strftime('%Y-%m-%d %H:%M:%S %Z') if pd.notna(r['date']) else ''
        print(f"[{d}]")
        print(r.get('title', ''))
        print(r.get('url', ''))
        print(f"{r.get('language','')} | {r.get('sourcecountry','')} | {r.get('domain','')}")
        print("-" * 80)

#df = marketwatch_gdelt("AAPL", years_back=1, extra_terms=["Apple", "Apple Inc"])
#print_articles(df)

if __name__ == "__main__":
    articles = []
    articles.append(pull_marketwatch("AAPL", years_back=1, extra_terms=["Apple", "Apple Inc"]))
    articles.append(pull_marketwatch("MSFT", years_back=1, extra_terms=["Microsoft", "Microsoft Corp"]))
    articles.append(pull_marketwatch("GOOGL", years_back=1, extra_terms=["Alphabet", "Google"]))
    #print(df.to_string(index=False))
    for dataframe in articles:
        print_articles(dataframe)
    #df.to_csv("marketwatch_AAPL_gdelt.csv", index=False)