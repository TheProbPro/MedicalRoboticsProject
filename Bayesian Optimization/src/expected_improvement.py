import numpy as np
from scipy.stats import norm

def expected_improvement(X, model, y_best, xi=0.01):
    """
    Calculates the Expected Improvement (EI) for Bayesian optimization.

    Args:
        X (ndarray): Points where EI is evaluated.
        model: Trained Gaussian Process model.
        y_best (float): Best observed value.
        xi (float): Exploration-exploitation trade-off parameter.

    Returns:
        ndarray: EI values for the given points.
    """
    mu, sigma = model.predict(X, return_std=True)
    mu = mu.reshape(-1, 1)
    sigma = sigma.reshape(-1, 1)
    with np.errstate(divide='warn'):
        Z = (mu - y_best - xi) / sigma
        ei = (mu - y_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    return ei.ravel()
