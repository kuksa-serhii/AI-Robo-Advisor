import numpy as np
import cvxpy as cp
import pandas as pd  # used both for typing and for frontier output


def compute_portfolio_stats(w: np.ndarray, mu: pd.Series, cov: pd.DataFrame, risk_free_rate: float):
    """
    Compute annual expected return, volatility and Sharpe ratio for a given weight vector.
    """
    w = np.array(w).reshape(-1)
    mu_vec = mu.values.reshape(-1)
    cov_mat = cov.values

    exp_return = float(mu_vec @ w)
    vol = float(np.sqrt(w.T @ cov_mat @ w))
    sharpe = (exp_return - risk_free_rate) / vol if vol > 0 else 0.0

    return exp_return, vol, sharpe


def solve_qp(objective, constraints):
    """
    Helper wrapper around cvxpy optimization.
    """
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimization failed with status: {prob.status}")
    return prob


def optimize_min_variance(mu: pd.Series, cov: pd.DataFrame, w_max: float, risk_free_rate: float):
    """
    Minimize portfolio variance: min w^T Σ w
    subject to sum(w)=1, 0<=w<=w_max.
    """
    n = len(mu)
    w = cp.Variable(n)
    Sigma = cov.values

    objective = cp.Minimize(cp.quad_form(w, Sigma))
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= w_max
    ]
    solve_qp(objective, constraints)
    w_opt = w.value
    exp_return, vol, sharpe = compute_portfolio_stats(w_opt, mu, cov, risk_free_rate)
    return w_opt, exp_return, vol, sharpe


def optimize_max_return(mu: pd.Series, cov: pd.DataFrame, w_max: float, risk_free_rate: float):
    """
    Maximize expected return: max mu^T w
    subject to sum(w)=1, 0<=w<=w_max.
    """
    n = len(mu)
    w = cp.Variable(n)
    mu_vec = mu.values

    objective = cp.Maximize(mu_vec @ w)
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= w_max
    ]
    solve_qp(objective, constraints)
    w_opt = w.value
    exp_return, vol, sharpe = compute_portfolio_stats(w_opt, mu, cov, risk_free_rate)
    return w_opt, exp_return, vol, sharpe


def optimize_max_sharpe_grid(mu: pd.Series, cov: pd.DataFrame, w_max: float, risk_free_rate: float):
    """
    Approximate max Sharpe by scanning over lambda in:
       max mu^T w - λ w^T Σ w
    We pick the solution with the highest Sharpe ratio.
    """
    n = len(mu)
    mu_vec = mu.values
    Sigma = cov.values

    best_sharpe = -1e9
    best_w = None
    best_ret = None
    best_vol = None

    lambdas = np.logspace(-3, 3, 25)

    for lam in lambdas:
        w = cp.Variable(n)
        objective = cp.Maximize(mu_vec @ w - lam * cp.quad_form(w, Sigma))
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= w_max
        ]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, verbose=False)
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            continue

        w_val = w.value
        exp_return, vol, sharpe = compute_portfolio_stats(w_val, mu, cov, risk_free_rate)
        if sharpe > best_sharpe and vol > 0:
            best_sharpe = sharpe
            best_w = w_val
            best_ret = exp_return
            best_vol = vol

    if best_w is None:
        raise ValueError("No feasible solution for max Sharpe")

    return best_w, best_ret, best_vol, best_sharpe


def compute_efficient_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    n_points: int,
    w_max: float,
    risk_free_rate: float,
):
    """
    Compute an approximate efficient frontier by solving a sequence of
    min-variance problems with different target returns.

    For each target return r_target, we solve:
        min w^T Σ w
        s.t. mu^T w >= r_target
             sum(w) = 1
             0 <= w <= w_max

    Returns a DataFrame with columns:
        - return_annual
        - vol_annual
    """
    mu_vec = mu.values
    Sigma = cov.values
    min_ret = float(mu_vec.min())
    max_ret = float(mu_vec.max())

    if n_points < 2 or max_ret <= min_ret:
        # Degenerate case; return empty frontier
        return pd.DataFrame(columns=["return_annual", "vol_annual"])

    target_returns = np.linspace(min_ret, max_ret, n_points)

    frontier_returns = []
    frontier_vols = []

    n = len(mu)
    for r_target in target_returns:
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, Sigma))
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= w_max,
            mu_vec @ w >= r_target
        ]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, verbose=False)
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            # Skip infeasible targets
            continue

        w_val = w.value
        exp_return, vol, _ = compute_portfolio_stats(w_val, mu, cov, risk_free_rate)
        frontier_returns.append(exp_return)
        frontier_vols.append(vol)

    if not frontier_returns:
        return pd.DataFrame(columns=["return_annual", "vol_annual"])

    return pd.DataFrame({
        "return_annual": frontier_returns,
        "vol_annual": frontier_vols,
    })
