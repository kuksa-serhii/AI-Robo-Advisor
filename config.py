import os
from dotenv import load_dotenv
from datetime import date
# Load environment variables from .env file
load_dotenv()


RISK_FREE_RATE = 0.03               # 3% annual risk-free rate
W_MAX = 0.25 

# we can change the set and start date here
START = "2015-01-01"
END = date.today().strftime("%Y-%m-%d")
ASSETS = [
    "SPY", "VTI", "QQQ", "IWM", "EFA",
    "EEM", "VT", "XLV", "XLF", "XLY",
    "VNQ", "GLD",
    "AGG", "BND", "TLT", "LQD", "HYG",
    "TIP", "BNDX", "SHY"
]


UNIVERSE_CSV_PATH = f"universe_{START}__{END}.csv"   # Universe of ETFs
PRICES_CSV_PATH = f"prices_{START}__{END}.csv"      # Historical prices

# Load OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY environment variable is not set. "
        "Please create a .env file with your OpenAI API key."
    )


