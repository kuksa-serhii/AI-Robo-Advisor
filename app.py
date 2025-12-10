import os
import streamlit as st

from agent import llm_parse_user_message, llm_explain_strategies
from portfolio_tool import compute_three_portfolios_and_frontier
from ui_components import show_portfolios_section
from data_layer import load_universe
from config import RISK_FREE_RATE, W_MAX


def run_app():
    st.set_page_config(page_title="LLM Robo-Advisor (Multi-Strategy)", layout="wide")
    st.title("LLM Robo-Advisor — Multi-Strategy with LLM")

    st.markdown(
        "This demo app uses a fixed ETF universe from `universe.csv` and historical prices "
        "from `prices.csv`.\n\n"
        "You can type something like: _\"Calculate portfolios for 15,000 dollars, "
        "without HYG and GLD\"_.\n\n"
        "In the sidebar you can choose which assets from the universe should be considered, "
        "adjust risk parameters (risk-free rate, max weight per asset), and control the conversation."
    )

    # -------------------------
    # Session state init
    # -------------------------
    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # list of {"role": "user"/"assistant", "content": str}
    if "capital" not in st.session_state:
        st.session_state["capital"] = None
    if "exclude_tickers" not in st.session_state:
        st.session_state["exclude_tickers"] = []
    if "portfolios" not in st.session_state:
        st.session_state["portfolios"] = None
    if "frontier" not in st.session_state:
        st.session_state["frontier"] = None
    if "include_tickers" not in st.session_state:
        st.session_state["include_tickers"] = None  # will be initialized from universe
    if "extra_strategies" not in st.session_state:
        st.session_state["extra_strategies"] = None
    if "risk_free_rate" not in st.session_state:
        st.session_state["risk_free_rate"] = RISK_FREE_RATE
    if "w_max" not in st.session_state:
        st.session_state["w_max"] = W_MAX
    if "chat_mode" not in st.session_state:
        st.session_state["chat_mode"] = "New portfolio request"
    if "trigger_explain_results" not in st.session_state:
        st.session_state["trigger_explain_results"] = False

    # -------------------------
    # Sidebar: controls
    # -------------------------
    with st.sidebar:
        st.header("Portfolio controls")

        # --- Universe / assets ---
        universe_df = load_universe()
        all_tickers = universe_df["ticker"].tolist()

        if st.session_state["include_tickers"] is None:
            st.session_state["include_tickers"] = all_tickers.copy()

        st.markdown("**Assets to include**")

        cols = st.columns(2)
        selected_assets = []
        for i, ticker in enumerate(all_tickers):
            col = cols[i % 2]
            with col:
                checked = st.checkbox(
                    ticker,
                    value=(ticker in st.session_state["include_tickers"]),
                    key=f"asset_{ticker}",
                )
            if checked:
                selected_assets.append(ticker)
        st.session_state["include_tickers"] = selected_assets

        st.markdown("---")
        st.markdown("**Risk parameters**")

        rf_input = st.number_input(
            "Risk-free rate (annual)",
            min_value=0.0,
            max_value=0.20,
            value=float(st.session_state["risk_free_rate"]),
            step=0.005,
            format="%.3f",
            help="Annual risk-free rate used in Sharpe ratio and leveraged strategies.",
        )
        w_max_input = st.number_input(
            "Max weight per asset",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state["w_max"]),
            step=0.05,
            format="%.2f",
            help="Upper bound for any single asset's portfolio weight.",
        )
        st.session_state["risk_free_rate"] = rf_input
        st.session_state["w_max"] = w_max_input

        st.markdown("---")

        current_cap = st.session_state.get("capital")
        current_excl = st.session_state.get("exclude_tickers", [])

        st.write(f"Current capital: **{current_cap if current_cap is not None else 'not set'}**")
        st.write("Excluded tickers: " + (", ".join(current_excl) if current_excl else "_none_"))

        new_capital = st.number_input(
            "Set capital, USD",
            value=float(current_cap or 10000.0),
            min_value=0.0,
            step=1000.0,
            key="capital_input",
        )
        if st.button("Update capital"):
            st.session_state["capital"] = new_capital
            st.success(f"Capital updated: {new_capital:,.0f} USD")

        if st.button("Clear exclusions"):
            st.session_state["exclude_tickers"] = []
            st.success("Exclusion list cleared.")

        if st.button("Clear chat history"):
            st.session_state["messages"] = []
            st.success("Chat history cleared.")

        if st.button("Recalculate portfolios"):
            capital = st.session_state.get("capital")
            exclude = st.session_state.get("exclude_tickers", [])
            include = st.session_state.get("include_tickers", [])
            rf = st.session_state.get("risk_free_rate", RISK_FREE_RATE)
            w_max = st.session_state.get("w_max", W_MAX)

            if not include:
                st.warning("Please select at least one asset from the universe.")
            elif capital is None or capital <= 0:
                st.warning("Please set a positive capital via the chat or the field above.")
            else:
                try:
                    portfolios, frontier_df, extra_strats = compute_three_portfolios_and_frontier(
                        capital,
                        exclude,
                        include_tickers=include,
                        frontier_points=40,
                        risk_free_rate=rf,
                        w_max=w_max,
                    )
                    st.session_state["portfolios"] = portfolios
                    st.session_state["frontier"] = frontier_df
                    st.session_state["extra_strategies"] = extra_strats
                    st.success("Portfolios recalculated.")
                except Exception as e:
                    st.error(f"Error during recalculation: {e}")

        # --- Conversation controls moved to the bottom ---
        st.markdown("---")
        st.markdown("**Conversation controls**")

        chat_mode = st.radio(
            "How should the assistant use your next message?",
            options=[
                "New portfolio request",
                "Explain current results",
            ],
            index=0 if st.session_state["portfolios"] is None else 1,
            help=(
                "- **New portfolio request**: your message will be parsed as instructions "
                "for a new optimization (capital, exclusions, etc.).\n"
                "- **Explain current results**: no recalculation, the assistant will only "
                "discuss the portfolios already shown on screen."
            ),
        )
        st.session_state["chat_mode"] = chat_mode

        # Button to auto-start QA explanation
        if st.button("Explain results"):
            st.session_state["trigger_explain_results"] = True

    # -------------------------
    # MAIN AREA
    # -------------------------

    # 1) Handle "Explain results" button (QA auto-start)
    if st.session_state.get("trigger_explain_results"):
        st.session_state["trigger_explain_results"] = False

        portfolios = st.session_state.get("portfolios")
        extra_strats = st.session_state.get("extra_strategies")
        capital = st.session_state.get("capital")
        exclude_tickers = st.session_state.get("exclude_tickers")
        rf = st.session_state.get("risk_free_rate", RISK_FREE_RATE)
        w_max = st.session_state.get("w_max", W_MAX)

        if portfolios is None:
            # Nothing to explain yet
            st.session_state["messages"].append({
                "role": "assistant",
                "content": "There are no portfolios to explain yet. Please run a new portfolio request first."
            })
        else:
            # Synthetic user message that starts QA
            user_msg = "Please explain the current portfolio results in more detail."
            st.session_state["messages"].append({"role": "user", "content": user_msg})

            explanation = llm_explain_strategies(
                user_message=user_msg,
                core_portfolios=portfolios,
                extra_strategies=extra_strats,
                capital=capital,
                exclude_tickers=exclude_tickers,
                risk_free_rate=rf,
                w_max=w_max,
                mode="qa",
            )
            st.session_state["messages"].append(
                {"role": "assistant", "content": explanation}
            )

    # 2) Chat input (manual user messages)
    user_message = st.chat_input("Ask about your portfolio or the strategies...")
    if user_message:
        st.session_state["messages"].append({"role": "user", "content": user_message})

        try:
            chat_mode = st.session_state.get("chat_mode", "New portfolio request")
            rf = st.session_state.get("risk_free_rate", RISK_FREE_RATE)
            w_max = st.session_state.get("w_max", W_MAX)

            if chat_mode == "New portfolio request" or st.session_state["portfolios"] is None:
                # -------- NEW PORTFOLIO REQUEST --------
                capital_prev = st.session_state["capital"]
                excl_prev = st.session_state["exclude_tickers"]

                capital, exclude_tickers = llm_parse_user_message(
                    user_message, capital_prev, excl_prev
                )
                st.session_state["capital"] = capital
                st.session_state["exclude_tickers"] = exclude_tickers

                include = st.session_state.get("include_tickers", [])

                if not include:
                    raise ValueError(
                        "No assets selected from the universe. "
                        "Please tick at least one asset in the sidebar."
                    )

                portfolios, frontier_df, extra_strats = compute_three_portfolios_and_frontier(
                    capital,
                    exclude_tickers,
                    include_tickers=include,
                    frontier_points=40,
                    risk_free_rate=rf,
                    w_max=w_max,
                )
                st.session_state["portfolios"] = portfolios
                st.session_state["frontier"] = frontier_df
                st.session_state["extra_strategies"] = extra_strats

                explanation = (
                    "> The explanation below describes the newly calculated portfolios and "
                    "all strategies based on your request. "
                    "Use *Explain current results* mode or the button in the sidebar for follow-up questions without recalculation.\n\n"
                )
                explanation += llm_explain_strategies(
                    user_message=user_message,
                    core_portfolios=portfolios,
                    extra_strategies=extra_strats,
                    capital=capital,
                    exclude_tickers=exclude_tickers,
                    risk_free_rate=rf,
                    w_max=w_max,
                    mode="initial",
                )

            else:
                # -------- Q&A ABOUT CURRENT RESULTS --------
                portfolios = st.session_state.get("portfolios")
                extra_strats = st.session_state.get("extra_strategies")
                capital = st.session_state.get("capital")
                exclude_tickers = st.session_state.get("exclude_tickers")

                if portfolios is None:
                    raise ValueError(
                        "No portfolios are available yet. "
                        "Switch chat mode to 'New portfolio request' and ask for a portfolio first."
                    )

                explanation = llm_explain_strategies(
                    user_message=user_message,
                    core_portfolios=portfolios,
                    extra_strategies=extra_strats,
                    capital=capital,
                    exclude_tickers=exclude_tickers,
                    risk_free_rate=rf,
                    w_max=w_max,
                    mode="qa",
                )

            st.session_state["messages"].append(
                {"role": "assistant", "content": explanation}
            )

        except Exception as e:
            error_msg = f"An error occurred: {e}"
            st.session_state["messages"].append(
                {"role": "assistant", "content": error_msg}
            )

    # 3) Show portfolios / tables / plots
    show_portfolios_section()

    # 4) Show conversation AFTER tables
    if st.session_state["messages"]:
        st.markdown("---")
        st.markdown("### Conversation about your portfolios")

        for msg in st.session_state["messages"]:
            role = "You" if msg["role"] == "user" else "Assistant"
            st.markdown(f"**{role}:** {msg['content']}")


if __name__ == "__main__":
    run_app()
