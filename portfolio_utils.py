import numpy as np
import cvxpy as cp

def portfolio_return(weights, returns):
    """
    Expected portfolio return
    """
    return np.dot(weights, returns)


def portfolio_risk(weights, cov_matrix):
    """
    Portfolio volatility (standard deviation)
    """
    return np.sqrt(weights.T @ cov_matrix @ weights)


def sharpe_ratio(portfolio_ret, portfolio_risk, risk_free_rate=0.05):
    """
    Sharpe Ratio calculation
    """
    return (portfolio_ret - risk_free_rate) / portfolio_risk


def optimize_portfolio(expected_returns, cov_matrix):
    """
    Mean-Variance Portfolio Optimization
    """
    n = len(expected_returns)
    weights = cp.Variable(n)

    portfolio_ret = expected_returns @ weights
    portfolio_var = cp.quad_form(weights, cov_matrix)

    problem = cp.Problem(
        cp.Maximize(portfolio_ret - 0.5 * portfolio_var),
        [
            cp.sum(weights) == 1,
            weights >= 0
        ]
    )

    problem.solve()

    return np.array(weights.value).flatten()

def efficient_frontier(expected_returns, cov_matrix, num_points=50):
    """
    Generate Efficient Frontier points.
    """
    n = len(expected_returns)
    target_returns = np.linspace(
        expected_returns.min(),
        expected_returns.max(),
        num_points
    )

    risks = []
    returns = []

    for r in target_returns:
        weights = cp.Variable(n)

        portfolio_var = cp.quad_form(weights, cov_matrix)

        constraints = [
            expected_returns @ weights == r,
            cp.sum(weights) == 1,
            weights >= 0
        ]

        problem = cp.Problem(cp.Minimize(portfolio_var), constraints)
        problem.solve()

        risks.append(np.sqrt(problem.value))
        returns.append(r)

    return np.array(risks), np.array(returns)


def min_variance_portfolio(cov_matrix):
    n = cov_matrix.shape[0]
    weights = cp.Variable(n)

    problem = cp.Problem(
        cp.Minimize(cp.quad_form(weights, cov_matrix)),
        [
            cp.sum(weights) == 1,
            weights >= 0
        ]
    )

    problem.solve()
    return np.array(weights.value).flatten()


def max_sharpe_portfolio(expected_returns, cov_matrix, target_risk=0.15):
    """
    Approximate maximum Sharpe portfolio by
    maximizing return under a risk constraint.
    """
    n = len(expected_returns)
    weights = cp.Variable(n)

    portfolio_return = expected_returns @ weights
    portfolio_variance = cp.quad_form(weights, cov_matrix)

    problem = cp.Problem(
        cp.Maximize(portfolio_return),
        [
            cp.sum(weights) == 1,
            weights >= 0,
            portfolio_variance <= target_risk**2
        ]
    )

    problem.solve()

    return np.array(weights.value).flatten()

