import numpy as np
from scipy.stats import norm

# # Expected Improvement acquisition function
# def expected_improvement(X, model, y_best, xi=0.01):
#     #xi: exploration parameter to balance exploration and exploitation
#     #y_best: best observed value
#     #X: points to evaluate the acquisition function
#     mu, sigma = model.predict(X, return_std=True)
#     #mu: predicted mean
#     #sigma: predicted standard deviation
#     mu = mu.reshape(-1, 1)
#     # Reshape mu and sigma to ensure they are 2D arrays
#     # This is necessary for broadcasting in the next step
#     sigma = sigma.reshape(-1, 1)
#     with np.errstate(divide='warn'):
#         # warning when dividing by zero
#         # Calculate the expected improvement
#         Z = (mu - y_best - xi) / sigma
#         # Z: standard normal variable
#         ei = (mu - y_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
#     # ei: expected improvement
#         ei[sigma == 0.0] = 0.0  # Avoid NaNs when sigma = 0
#     return ei.ravel()

# Expected Improvement (EI)
# This function calculates the expected improvement at each point in the grid
def expected_improvement(X, model, y_best, xi=1): 
    # xi is a parameter that controls the trade-off between exploration and exploitation
    mu, sigma = model.predict(X, return_std=True)
    mu = mu.reshape(-1, 1)
    sigma = sigma.reshape(-1, 1)
    with np.errstate(divide='warn'):
        Z = (mu - y_best - xi) / sigma
        ei = (mu - y_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        # norm.cdf(Z) is the cumulative distribution function for a normal distribution
        # norm.pdf(Z) is the probability density function for a normal distribution
        ei[sigma == 0.0] = 0.0  
    return ei.ravel()