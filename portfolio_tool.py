import numpy as np

from data_layer import load_universe, load_prices, compute_returns_mu_cov, apply_exclusions
from optimization_core import (
    optimize_min_variance,
    optimize_max_return,
    optimize_max_sharpe_grid,
    compute_efficient_frontier,
)
from strategies import (
    strat_equal_weight,
    strat_buy_and_hold,
    strat_min_variance_weights,
    strat_max_sharpe_weights,
    strat_robust_mean_variance,
    strat_equal_risk_contributions,
    strat_leveraged_max_sharpe,
)


def _format_core_portfolios(capital, mu_f, cov_f, tickers, risk_free_rate: float, w_max: float):
    """
    Compute the three core portfolios on a filtered mu/cov universe.
    Returns dict with keys: min_risk, max_return, max_sharpe.
    """
    # 1) Minimum risk
    w_min_risk, r_min, v_min, s_min = optimize_min_variance(mu_f, cov_f, w_max=w_max, risk_free_rate=risk_free_rate)

    # 2) Maximum return
    w_max_ret, r_max_ret, v_max_ret, s_max_ret = optimize_max_return(mu_f, cov_f, w_max=w_max, risk_free_rate=risk_free_rate)

    # 3) Maximum Sharpe ratio
    w_max_sharpe, r_max_sh, v_max_sh, s_max_sh = optimize_max_sharpe_grid(mu_f, cov_f, w_max=w_max, risk_free_rate=risk_free_rate)

    def fmt_port(weights, exp_ret, vol, sharpe):
        weights = np.array(weights).reshape(-1)
        weights_pct = 100 * weights
        positions = []
        for t, w_pct in zip(tickers, weights_pct):
            if w_pct < 0.01:
                continue
            entry = {"ticker": t, "weight_pct": float(w_pct)}
            if capital is not None and capital > 0:
                entry["amount_usd"] = float(capital * w_pct / 100.0)
            positions.append(entry)

        return {
            "expected_return_annual": float(exp_ret),
            "volatility_annual": float(vol),
            "sharpe_ratio": float(sharpe),
            "positions": positions,
        }

    return {
        "min_risk": fmt_port(w_min_risk, r_min, v_min, s_min),
        "max_return": fmt_port(w_max_ret, r_max_ret, v_max_ret, s_max_ret),
        "max_sharpe": fmt_port(w_max_sharpe, r_max_sh, v_max_sh, s_max_sh),
        "fmt_port": fmt_port,  # expose formatter for reuse
    }


def _format_extra_strategies(capital, mu_f, cov_f, tickers, fmt_port, risk_free_rate: float, w_max: float):
    """
    Compute additional strategies and format them similarly to core portfolios.
    Returns dict: strategy_key -> dict with metrics and positions.
    """
    extra = {}

    # 1) Buy & hold (approximated as equal-weight in this static setting)
    w_bh, r_bh, v_bh, s_bh = strat_buy_and_hold(mu_f, cov_f, risk_free_rate)
    extra["buy_and_hold"] = {
        "label": "Buy & Hold (static approx.)",
        **fmt_port(w_bh, r_bh, v_bh, s_bh),
    }

    # 2) Equally weighted
    w_eq, r_eq, v_eq, s_eq = strat_equal_weight(mu_f, cov_f, risk_free_rate)
    extra["equal_weight"] = {
        "label": "Equally Weighted",
        **fmt_port(w_eq, r_eq, v_eq, s_eq),
    }

    # 3) Min Variance (wrapper version; similar to core min_risk)
    w_mv, r_mv, v_mv, s_mv = strat_min_variance_weights(mu_f, cov_f, w_max=w_max, risk_free_rate=risk_free_rate)
    extra["min_variance_alt"] = {
        "label": "Min Variance (alt solver)",
        **fmt_port(w_mv, r_mv, v_mv, s_mv),
    }

    # 4) Max Sharpe (wrapper version; similar to core max_sharpe)
    w_ms, r_ms, v_ms, s_ms = strat_max_sharpe_weights(mu_f, cov_f, w_max=w_max, risk_free_rate=risk_free_rate)
    extra["max_sharpe_alt"] = {
        "label": "Max Sharpe (alt solver)",
        **fmt_port(w_ms, r_ms, v_ms, s_ms),
    }

    # 5) Equal Risk Contributions (ERC)
    try:
        w_erc, r_erc, v_erc, s_erc = strat_equal_risk_contributions(mu_f, cov_f, risk_free_rate)
        extra["erc"] = {
            "label": "Equal Risk Contributions (ERC)",
            **fmt_port(w_erc, r_erc, v_erc, s_erc),
        }
    except Exception:
        pass

    # 6) Robust mean-variance
    try:
        w_rob, r_rob, v_rob, s_rob = strat_robust_mean_variance(
            mu_f, cov_f, risk_free_rate, w_max=w_max
        )
        extra["robust_mv"] = {
            "label": "Robust Mean–Variance",
            **fmt_port(w_rob, r_rob, v_rob, s_rob),
        }
    except Exception:
        pass

    # 7) Leveraged Max Sharpe
    try:
        w_lev, r_lev, v_lev, s_lev = strat_leveraged_max_sharpe(
            mu_f, cov_f, risk_free_rate, w_max=w_max, leverage=2.0
        )
        extra["leveraged_max_sharpe"] = {
            "label": "Leveraged Max Sharpe (2x)",
            **fmt_port(w_lev, r_lev, v_lev, s_lev),
        }
    except Exception:
        pass

    return extra


def compute_three_portfolios_and_frontier(
    capital,
    exclude_tickers,
    include_tickers=None,
    frontier_points: int = 30,
    risk_free_rate: float = 0.02,
    w_max: float = 0.25,
):
    """
    Main tool:
    1) Load data
    2) Apply asset selection (include_tickers) from universe.csv
    3) Apply exclusions (exclude_tickers)
    4) Compute core portfolios (min_risk, max_return, max_sharpe)
    5) Compute additional strategies (buy & hold, equal-weight, ERC, robust, leveraged, etc.)
    6) Compute efficient frontier on the same reduced universe
    7) Optionally attach USD allocations

    Returns:
        core_portfolios_dict, frontier_df, extra_strategies_dict
    """
    universe_df = load_universe()
    prices_df = load_prices()

    universe_tickers = universe_df["ticker"].tolist()

    # Step 2: apply active asset universe (checkboxes)
    if include_tickers:
        include_tickers = [t.upper() for t in include_tickers]
        active_universe = [t for t in universe_tickers if t in include_tickers]
    else:
        active_universe = universe_tickers

    prices_df = prices_df.loc[:, [c for c in prices_df.columns if c in active_universe]]

    # Compute mu and cov on this active universe
    _, mu, cov = compute_returns_mu_cov(prices_df)

    # Step 3: apply exclusions
    mu_f, cov_f = apply_exclusions(mu, cov, exclude_tickers or [])
    tickers = list(mu_f.index)

    if len(tickers) == 0:
        raise ValueError("No assets left after applying asset selection and exclusions.")

    # 4) Core portfolios
    core = _format_core_portfolios(capital, mu_f, cov_f, tickers, risk_free_rate, w_max)
    fmt_port = core.pop("fmt_port")

    # 5) Additional strategies
    extra = _format_extra_strategies(capital, mu_f, cov_f, tickers, fmt_port, risk_free_rate, w_max)

    # 6) Efficient frontier on the same filtered mu/cov
    frontier_df = compute_efficient_frontier(
        mu_f, cov_f, n_points=frontier_points, w_max=w_max, risk_free_rate=risk_free_rate
    )

    return core, frontier_df, extra


def compute_three_portfolios(capital, exclude_tickers, include_tickers=None, risk_free_rate: float = 0.02, w_max: float = 0.25):
    """
    Backwards-compatible helper: return only the three portfolios,
    ignoring the frontier and extra strategies.
    """
    core, _, _ = compute_three_portfolios_and_frontier(
        capital,
        exclude_tickers,
        include_tickers=include_tickers,
        risk_free_rate=risk_free_rate,
        w_max=w_max,
    )
    return core
