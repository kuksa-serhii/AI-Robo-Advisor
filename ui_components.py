import pandas as pd
import streamlit as st


def show_portfolios_section():
    """
    If portfolios are present in session_state, show:
    - summary table for core portfolios
    - risk/return scatter chart with efficient frontier and ALL strategies
    - detailed structure per core portfolio in tabs
    - additional strategies summary + composition
    """
    portfolios = st.session_state.get("portfolios")
    if not portfolios:
        return

    frontier_df = st.session_state.get("frontier")
    extra_strats = st.session_state.get("extra_strategies")

    st.subheader("Portfolio optimization results")

    # ---------------- Core portfolios ----------------
    mapping = [
        ("min_risk", "Conservative (Min Risk)",
         ),
        ("max_return", "Aggressive (Max Return)"),
        ("max_sharpe", "Balanced (Max Sharpe)"),
    ]

    rows = []
    for key, name in mapping:
        p = portfolios[key]
        rows.append({
            "portfolio": name,
            "return_annual": p["expected_return_annual"],
            "vol_annual": p["volatility_annual"],
            "sharpe": p["sharpe_ratio"],
            "return_pct": p["expected_return_annual"] * 100.0,
            "vol_pct": p["volatility_annual"] * 100.0,
        })

    summary_df = pd.DataFrame(rows)

    st.markdown("#### Core portfolio comparison (annual metrics)")
    st.dataframe(
        summary_df[["portfolio", "return_pct", "vol_pct", "sharpe"]]
        .rename(columns={
            "portfolio": "Portfolio",
            "return_pct": "Expected return, %",
            "vol_pct": "Volatility, %",
            "sharpe": "Sharpe ratio",
        })
        .style.format({
            "Expected return, %": "{:.2f}",
            "Volatility, %": "{:.2f}",
            "Sharpe ratio": "{:.2f}",
        }),
        use_container_width=True,
    )

    # ---------------- Risk–return chart (ALL strategies) ----------------
    st.markdown("#### Risk–return chart (core + additional strategies + efficient frontier)")

    chart_rows = []

    # 1) Core portfolios
    for row in summary_df.itertuples():
        chart_rows.append({
            "Series": row.portfolio,
            "Return, %": row.return_pct,
            "Risk (volatility), %": row.vol_pct,
        })

    # 2) Additional strategies
    if extra_strats:
        for key, strat in extra_strats.items():
            label = strat.get("label", key)
            chart_rows.append({
                "Series": label,
                "Return, %": strat["expected_return_annual"] * 100.0,
                "Risk (volatility), %": strat["volatility_annual"] * 100.0,
            })

    # 3) Efficient frontier points, if present
    if frontier_df is not None and not frontier_df.empty:
        for ret, vol in zip(frontier_df["return_annual"], frontier_df["vol_annual"]):
            chart_rows.append({
                "Series": "Efficient frontier",
                "Return, %": ret * 100.0,
                "Risk (volatility), %": vol * 100.0,
            })

    chart_df = pd.DataFrame(chart_rows)
    st.scatter_chart(
        chart_df,
        x="Risk (volatility), %",
        y="Return, %",
        color="Series",
        use_container_width=True,
    )

    # ---------------- Core portfolio composition ----------------
    st.markdown("#### Core portfolio composition")
    tab1, tab2, tab3 = st.tabs([
        "Conservative (Min Risk)",
        "Aggressive (Max Return)",
        "Balanced (Max Sharpe)",
    ])
    tabs = [tab1, tab2, tab3]
    for (key, name), tab in zip(mapping, tabs):
        with tab:
            p = portfolios[key]
            pos = p.get("positions", [])
            if not pos:
                st.write("No positions.")
            else:
                df_pos = pd.DataFrame(pos)
                if "weight_pct" in df_pos.columns:
                    df_pos["weight_pct"] = df_pos["weight_pct"].astype(float)
                if "amount_usd" in df_pos.columns:
                    df_pos["amount_usd"] = df_pos["amount_usd"].astype(float)

                df_pos_for_show = df_pos.rename(columns={
                    "ticker": "Ticker",
                    "weight_pct": "Weight, %",
                    "amount_usd": "Amount, USD",
                })
                st.dataframe(
                    df_pos_for_show.style.format({
                        "Weight, %": "{:.2f}",
                        "Amount, USD": "{:,.2f}",
                    }),
                    use_container_width=True,
                )

    # ---------------- Additional strategies ----------------
    if extra_strats:
        st.markdown("### Additional strategies")

        extra_rows = []
        for key, strat in extra_strats.items():
            label = strat.get("label", key)
            extra_rows.append({
                "Strategy": label,
                "Expected return, %": strat["expected_return_annual"] * 100.0,
                "Volatility, %": strat["volatility_annual"] * 100.0,
                "Sharpe ratio": strat["sharpe_ratio"],
            })

        extra_df = pd.DataFrame(extra_rows)
        st.markdown("#### Summary metrics")
        st.dataframe(
            extra_df
            .style.format({
                "Expected return, %": "{:.2f}",
                "Volatility, %": "{:.2f}",
                "Sharpe ratio": "{:.2f}",
            }),
            use_container_width=True,
        )

        st.markdown("#### Strategy compositions")
        tabs_extra = st.tabs([strat.get("label", key) for key, strat in extra_strats.items()])
        for (key, strat), tab in zip(extra_strats.items(), tabs_extra):
            with tab:
                pos = strat.get("positions", [])
                if not pos:
                    st.write("No positions.")
                else:
                    df_pos = pd.DataFrame(pos)
                    if "weight_pct" in df_pos.columns:
                        df_pos["weight_pct"] = df_pos["weight_pct"].astype(float)
                    if "amount_usd" in df_pos.columns:
                        df_pos["amount_usd"] = df_pos["amount_usd"].astype(float)

                    df_pos_for_show = df_pos.rename(columns={
                        "ticker": "Ticker",
                        "weight_pct": "Weight, %",
                        "amount_usd": "Amount, USD",
                    })
                    st.dataframe(
                        df_pos_for_show.style.format({
                            "Weight, %": "{:.2f}",
                            "Amount, USD": "{:,.2f}",
                        }),
                        use_container_width=True,
                    )
