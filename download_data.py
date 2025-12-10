import yfinance as yf
import pandas as pd
from pathlib import Path
import glob


from config import UNIVERSE_CSV_PATH, PRICES_CSV_PATH, ASSETS, START, END



start = START
end = END

# Remove old generated CSVs so only the latest pair remains
def cleanup_old_csvs():
    prefixes = ["universe_", "prices_"]
    for prefix in prefixes:
        for path_str in glob.glob(f"{prefix}*.csv"):
            try:
                Path(path_str).unlink()
                print(f"Removed old file: {path_str}")
            except Exception as exc:
                print(f"Could not remove {path_str}: {exc}")


cleanup_old_csvs()

all_prices = pd.DataFrame()
success = []
failed = []

for ticker in ASSETS:
    print(f"\n=== Downloading {ticker} ===")
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True
    )
    print("Shape:", df.shape)

    if df.empty or df["Close"].dropna().empty and "Adj Close" not in df.columns:
        print(f"⚠️  {ticker}: EMPTY, skipping")
        failed.append(ticker)
        continue

    # Prefer adjusted close; fall back to close when needed
    if "Adj Close" in df.columns:
        series = df["Adj Close"]
    else:
        series = df["Close"]

    all_prices[ticker] = series
    success.append(ticker)
    print(f"✅ {ticker}: added to prices")

print("\n--- SUMMARY ---")
print("Successful tickers:", success)
print("Failed tickers:", failed)
print("Final prices shape:", all_prices.shape)

# Drop days where all tickers are NaN
all_prices = all_prices.dropna(how="all")
print("After dropna(all):", all_prices.shape)

all_prices.to_csv(PRICES_CSV_PATH, index_label="Date")
print("Saved to:", PRICES_CSV_PATH)


df = pd.DataFrame({"ticker" : ASSETS})
df.to_csv(UNIVERSE_CSV_PATH, index=False)
print("Saved to:", UNIVERSE_CSV_PATH)