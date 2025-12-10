import os
import json
from openai import OpenAI
from config import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

client = OpenAI()


def llm_parse_user_message(user_message: str, default_capital, default_exclude):
    """
    LLM agent: extract capital (USD) and exclude_tickers from a free-text user message.
    If nothing is specified, use previous/default values.
    Used ONLY in 'New portfolio request' mode.
    """
    system_prompt = f"""
You are a financial assistant. Your job is to extract from the user's message:
- capital: numeric USD amount (or null if not mentioned and no default)
- exclude_tickers: list of ETF tickers in UPPERCASE (may be empty)

Current defaults:
- default_capital = {default_capital if default_capital is not None else "null"}
- default_exclude_tickers = {json.dumps(default_exclude or [])}

Logic:
1) If the user explicitly mentions a new amount (e.g., "15 000", "20000$", "40k"),
   use it as capital.
2) If not, use default_capital.
3) If the user asks to exclude tickers (e.g., "without HYG and GLD", "exclude HYG, GLD",
   "do not include gold"), update exclude_tickers accordingly.
4) If exclusions are not mentioned, keep default_exclude_tickers.

Respond with ONLY valid JSON:
{{
  "capital": <number or null>,
  "exclude_tickers": ["HYG", "GLD", ...]
}}
    """.strip()

    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    content = resp.choices[0].message.content.strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Fallback: return defaults
        data = {
            "capital": default_capital,
            "exclude_tickers": default_exclude or []
        }
    capital = data.get("capital", default_capital)
    exclude = data.get("exclude_tickers", default_exclude or [])
    # Normalize
    if capital is not None:
        try:
            capital = float(capital)
        except Exception:
            capital = default_capital
    exclude = [str(t).upper().strip() for t in exclude if str(t).strip()]
    return capital, exclude


def llm_explain_strategies(
    user_message: str,
    core_portfolios: dict,
    extra_strategies: dict | None,
    capital,
    exclude_tickers,
    risk_free_rate: float,
    w_max: float,
    mode: str,
):
    """
    LLM explains or discusses ALL strategies (core + additional)
    WITHOUT changing any numbers.

    mode:
      - "initial"  -> first explanation right after a new calculation
      - "qa"       -> follow-up questions about existing results
    """
    payload = {
        "core_portfolios": core_portfolios,
        "extra_strategies": extra_strategies or {},
        "capital": capital,
        "exclude_tickers": exclude_tickers or [],
        "risk_free_rate": risk_free_rate,
        "w_max": w_max,
        "mode": mode,
    }
    portfolios_json = json.dumps(payload, indent=2)

    system_prompt = """
You are a robo-advisor. A separate quant engine has already computed a set of portfolios
based on the user's settings. You are NOT allowed to change any numbers (returns, volatilities,
weights, Sharpe ratios). You only interpret and explain what is already calculated.

You are given:
- 3 core portfolios:
    - Conservative (Min Risk)
    - Aggressive (Max Return)
    - Balanced (Max Sharpe)
- Additional strategies (e.g. Equal Weight, Buy & Hold, ERC, Robust MV, Leveraged Max Sharpe).
- Risk-free rate and max weight constraints that were used.
- Capital amount (if set) and excluded tickers.

Your tasks:

1) Respect the 'mode':
   - If mode == "initial":
       * The user has just asked for a new portfolio, and you should give a structured overview:
         - Short intro: what was computed and what constraints (risk-free rate, w_max, exclusions).
         - Explain each core portfolio.
         - Then briefly compare additional strategies vs core ones:
           who is more conservative, who is more aggressive, typical use cases.
   - If mode == "qa":
       * The portfolios are already on screen.
       * The user is asking follow-up questions (e.g. "why is volatility so high?",
         "which strategy is most balanced?", "how does ERC differ from Min Risk?").
       * Answer their specific question(s), referencing the numbers from the JSON.
       * Do NOT recompute anything. Treat the JSON as ground truth.

2) When explaining strategies:
   - Always differentiate between:
       * Core portfolios (Conservative / Aggressive / Balanced).
       * Additional strategies (Equal Weight, Buy & Hold, ERC, Robust, Leveraged, etc.).
   - Use approximate percentages when quoting returns/volatility (e.g. "about 7%").
   - Use language like "historically", "expected", "could", "may", not "guaranteed".

3) No personalized investment advice:
   - You may explain trade-offs ("higher risk/higher volatility", "better diversified", etc.).
   - Do NOT say "you should invest in", instead say "this type of profile is often chosen by someone who...".

4) Never invent new strategies or change existing numeric values.
   Always stick to the JSON you are given.
""".strip()

    user_prompt = f"""
User message:
\"\"\"{user_message}\"\"\"

Below is the full JSON with current portfolios and strategies (core + additional)
and the parameters used to compute them:

{portfolios_json}

Please respond according to the 'mode' and instructions in the system prompt.
    """.strip()

    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.5,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content
