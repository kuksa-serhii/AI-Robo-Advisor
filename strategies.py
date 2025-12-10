import numpy as np
import pandas as pd
import cvxpy as cp

from optimization_core import (
    compute_portfolio_stats,
    optimize_min_variance,
    optimize_max_sharpe_grid,
)


def strat_equal_weight(mu: pd.Series, cov: pd.DataFrame, risk_free_rate: float):
    """
    Equally-weighted portfolio over the given universe.
    """
    n = len(mu)
    if n == 0:
        raise ValueError("No assets in universe for equal-weight strategy.")
    w = np.ones(n) / n
    exp_return, vol, sharpe = compute_portfolio_stats(w, mu, cov, risk_free_rate)
    return w, exp_return, vol, sharpe


def strat_buy_and_hold(mu: pd.Series, cov: pd.DataFrame, risk_free_rate: float):
    """
    Placeholder for 'buy & hold' in a static setting.
    In a full backtest, this would keep x_init.
    Here we approximate it by equal-weight (static allocation).
    """
    return strat_equal_weight(mu, cov, risk_free_rate)


def strat_min_variance_weights(mu: pd.Series, cov: pd.DataFrame, w_max: float, risk_free_rate: float):
    """
    Thin wrapper around our min-variance optimizer to keep naming similar
    to your original strat_min_variance.
    """
    return optimize_min_variance(mu, cov, w_max=w_max, risk_free_rate=risk_free_rate)


def strat_max_sharpe_weights(mu: pd.Series, cov: pd.DataFrame, w_max: float, risk_free_rate: float):
    """
    Thin wrapper around our grid-based max Sharpe optimizer.
    """
    return optimize_max_sharpe_grid(mu, cov, w_max=w_max, risk_free_rate=risk_free_rate)


def strat_robust_mean_variance(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float,
    delta: float = 0.05,
    gamma: float = 3.0,
    w_max: float = 0.25,
):
    """
    Simple robust mean-variance model:

      minimize   w^T (Q + δ I) w - γ μ^T w
      subject to sum(w)=1, 0 <= w <= w_max.

    δ controls robustness to covariance estimation error.
    γ controls emphasis on expected return.
    """
    n = len(mu)
    if n == 0:
        raise ValueError("No assets for robust mean-variance strategy.")

    w = cp.Variable(n)
    Sigma = cov.values
    mu_vec = mu.values
    Q_rob = Sigma + delta * np.eye(n)

    objective = cp.Minimize(cp.quad_form(w, Q_rob) - gamma * (mu_vec @ w))
    constraints = [cp.sum(w) == 1, w >= 0, w <= w_max]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if prob.status not in ["optimal", "optimal_inaccurate"] or w.value is None:
        # fallback to equal-weight
        return strat_equal_weight(mu, cov, risk_free_rate)

    w_star = np.array(w.value).reshape(-1)
    exp_return, vol, sharpe = compute_portfolio_stats(w_star, mu, cov, risk_free_rate)
    return w_star, exp_return, vol, sharpe


def _project_to_simplex(w: np.ndarray) -> np.ndarray:
    """
    Project vector w onto the probability simplex:
        { w : w_i >= 0, sum_i w_i = 1 }.
    """
    n = w.shape[0]
    u = np.sort(w)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0

    if not np.any(cond):
        theta = cssv[-1] / n
    else:
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / rho

    w_proj = np.maximum(w - theta, 0)
    return w_proj


def strat_equal_risk_contributions(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float,
    max_iter: int = 500,
    step_size: float = 0.01,
):
    """
    Equal Risk Contributions (ERC) portfolio via simple projected gradient descent.

    Risk contributions:
        RC_i = w_i * (Q w)_i

    Objective:
        f(w) = sum_i (RC_i - mean(RC))^2

    This is an approximate numerical solution, good enough for demo / intuition.
    """
    Sigma = cov.values
    n = len(mu)
    if n == 0:
        raise ValueError("No assets for ERC strategy.")

    w = np.ones(n) / n  # start from equal weights

    def objective_and_grad(w_vec: np.ndarray):
        """
        Compute f(w) and gradient ∇f(w).
        """
        v = Sigma @ w_vec         # shape (n,)
        rc = w_vec * v            # risk contributions
        avg_rc = rc.mean()
        e = rc - avg_rc           # deviations

        # J_RC(i,j) = δ_ij * v_i + w_i * Sigma_ij
        J_RC = np.diag(v) + (w_vec[:, None] * Sigma)

        col_sum = J_RC.sum(axis=0)           # sum over i
        J = J_RC - (1.0 / n) * np.ones((n, 1)) @ col_sum[None, :]

        grad = 2.0 * J.T @ e
        f_val = float((e ** 2).sum())
        return f_val, grad

    for _ in range(max_iter):
        f_val, grad = objective_and_grad(w)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-6:
            break
        w = w - step_size * grad
        w = _project_to_simplex(w)

    exp_return, vol, sharpe = compute_portfolio_stats(w, mu, cov, risk_free_rate)
    return w, exp_return, vol, sharpe


def strat_leveraged_max_sharpe(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float,
    w_max: float,
    leverage: float = 2.0,
):
    """
    Leveraged maximum-Sharpe strategy (static approximation).

    We:
      1) Compute a long-only max-Sharpe portfolio w_ms.
      2) Assume we lever it by 'leverage' times around the risk-free rate.

    In a static setting without explicit cash, we:
      - keep the same weights w_ms as composition,
      - scale return and volatility:
          R_lever ≈ rf + L * (R_ms - rf)
          σ_lever ≈ L * σ_ms
      - Sharpe remains approximately the same.
    """
    w_ms, r_ms, vol_ms, sharpe_ms = optimize_max_sharpe_grid(
        mu, cov, w_max=w_max, risk_free_rate=risk_free_rate
    )
    excess = r_ms - risk_free_rate
    r_lever = risk_free_rate + leverage * excess
    vol_lever = leverage * vol_ms
    sharpe_lever = (r_lever - risk_free_rate) / vol_lever if vol_lever > 0 else 0.0

    return w_ms, r_lever, vol_lever, sharpe_lever
