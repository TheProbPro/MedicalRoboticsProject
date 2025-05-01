import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from target import target_function
from acquisition import expected_improvement

# Main Bayesian Optimization loop
def run_bo_3d(n_init=5, n_iter=20, grid_size=50, seed=42):
    np.random.seed(seed)
    
    # Create 2D grid
    x = np.linspace(0, 6, grid_size)
    y = np.linspace(0, 6, grid_size)
    X1, X2 = np.meshgrid(x, y)
    X_grid = np.vstack([X1.ravel(), X2.ravel()]).T

    # Random initial samples
    init_idx = np.random.choice(len(X_grid), n_init, replace=False)
    X_sample = X_grid[init_idx]
    y_sample = target_function(X_sample)
   
    # Gaussian Process model
    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
    # alpha: noise level
    # normalize_y: normalize the target variable
    # Optimization iterations
    for i in range(n_iter):
        gp.fit(X_sample, y_sample)
        y_best = np.max(y_sample)
        ei = expected_improvement(X_grid, gp, y_best)
        next_idx = np.argmax(ei)
        next_x = X_grid[next_idx]
        next_y = target_function(next_x.reshape(1, -1))
        X_sample = np.vstack((X_sample, next_x))
        y_sample = np.append(y_sample, next_y)

    # Final GP model and outputs
    gp.fit(X_sample, y_sample)
    mu, std = gp.predict(X_grid, return_std=True)
    ei_final = expected_improvement(X_grid, gp, np.max(y_sample))
    Z_target = target_function(X_grid)

    return X1, X2, mu, std, ei_final, Z_target, X_sample
