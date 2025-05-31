# from Simulation import simulate_tissue  # Import simulate_tissue

# def run_bo_3d(n_init=5, n_iter=20, grid_size=50, seed=42, tissue_stiffness=None):
#     import numpy as np
#     from sklearn.gaussian_process import GaussianProcessRegressor
#     from sklearn.gaussian_process.kernels import Matern
#     from scipy.stats import norm
#     from Target_Function import target_function
#     from Expected_Improvement import expected_improvement

#     # Ensure tissue_stiffness is provided
#     if tissue_stiffness is None:
#         raise ValueError("tissue_stiffness must be provided as an argument to run_bo_3d.")
#     print("Tissue stiffness received successfully in run_bo_3d.")

#     np.random.seed(seed)
#     x = np.linspace(0, 6, grid_size)
#     y = np.linspace(0, 6, grid_size)
#     X1, X2 = np.meshgrid(x, y)
#     X_grid = np.vstack([X1.ravel(), X2.ravel()]).T

#     init_idx = np.random.choice(len(X_grid), n_init, replace=False)
#     X_sample = X_grid[init_idx]
#     y_sample = target_function(X_sample, tissue_stiffness)  # Add tissue_stiffness as argument

#     kernel = Matern(nu=2.5)
#     gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

#     for i in range(n_iter):
#         gp.fit(X_sample, y_sample)
#         y_best = np.max(y_sample)
#         ei = expected_improvement(X_grid, gp, y_best)
#         next_idx = np.argmax(ei)
#         next_x = X_grid[next_idx]
#         next_y = target_function(next_x.reshape(1, -1), tissue_stiffness)  # Add tissue_stiffness as argument
#         X_sample = np.vstack((X_sample, next_x))
#         y_sample = np.append(y_sample, next_y)

#     gp.fit(X_sample, y_sample)
#     mu, std = gp.predict(X_grid, return_std=True)
#     ei_final = expected_improvement(X_grid, gp, np.max(y_sample))
#     Z_target = target_function(X_grid, tissue_stiffness)  # Add tissue_stiffness as argument

#     return X1, X2, mu, std, ei_final, Z_target, X_sample

import numpy as np #calculations
import matplotlib.pyplot as plt #visualization
import matplotlib as mpl # control plot style
from scipy.stats import norm # calculation of expected improvement
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from mpl_toolkits.mplot3d import Axes3D # 3D plotting 
from matplotlib.cm import get_cmap # color map
from matplotlib.colors import Normalize # color bar

# Set plot style for LaTeX
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

# Simulated Tumor Function
def target_function(X, center=(5,5), sigma=1, amplitude=0.8, noise_std=0.1):
    # cnter: tumors placement
    # sigma: tumor size
    # amplitude: tumor height
    x, y = X[:, 0], X[:, 1]
    if sigma == 0 or amplitude == 0:
        return np.zeros(len(X))   # Return zeros if tumor is "off"

    tumor = amplitude * np.exp(-((x - center[0])**2 + (y - center[1])**2) / (2 * sigma**2))
    noise = np.random.normal(0, noise_std, size=len(X))
    return tumor + noise

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

# Run Bayesian Optimization
def run_bo_3d(n_init=10, n_iter=30, grid_size=60, seed=42):
    """
    Runs Bayesian Optimization to find the maximum of the simulated tumor function.
    Parameters:
        n_init: Number of initial random samples.
        n_iter: Number of Bayesian Optimization iterations.
        grid_size: Number of points along each axis for the search grid.
        seed: Random seed for reproducibility.
    Returns:
        Various arrays for plotting and analysis.
    """
    np.random.seed(seed)

    # Create a 2D grid for the target function
    x = np.linspace(0, 15, grid_size)
    y = np.linspace(0, 15, grid_size)
    X1, X2 = np.meshgrid(x, y)  # Create a meshgrid for 3D plotting
    X_grid = np.vstack([X1.ravel(), X2.ravel()]).T

    # Select random initial points (exploration)
    init_idx = np.random.choice(len(X_grid), n_init, replace=False)
    X_sample = X_grid[init_idx]
    y_sample = target_function(X_sample)

    # Gaussian Process model
    kernel = Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=1.5) 
    # automatic length scale selection
    # nu=1.5 controls the smoothness of the function
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
    gp.fit(X_sample, y_sample)
    print(gp.kernel_) 

    # Iterative Bayesian Optimization loop
    for i in range(n_iter):
        gp.fit(X_sample, y_sample)
        y_best = np.max(y_sample)
        ei = expected_improvement(X_grid, gp, y_best)
        next_idx = np.argmax(ei)
        next_x = X_grid[next_idx]
        next_y = target_function(next_x.reshape(1, -1)) 
        X_sample = np.vstack((X_sample, next_x))
        y_sample = np.append(y_sample, next_y)

    # Final model training and predictions 
    gp.fit(X_sample, y_sample)
    mu, std = gp.predict(X_grid, return_std=True)
    ei_final = expected_improvement(X_grid, gp, np.max(y_sample))
    Z_target = target_function(X_grid)

    return X1, X2, mu, std, ei_final, Z_target, X_sample

 # Plotting function
def plot_surface_3d(X1, X2, surfaces, titles, X_sample=None):
    fig = plt.figure(figsize=(18, 12))
    cmap = get_cmap("RdYlGn")

    for i, (Z, title) in enumerate(zip(surfaces, titles)):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        ax.plot_surface(X1, X2, Z.reshape(X1.shape), cmap='viridis', edgecolor='none', alpha=0.9)

        # Use color scale in the second subplot ("Phantom")
        if X_sample is not None and i == 1:
            num_points = len(X_sample)
            colors = cmap(np.linspace(0, 1, num_points))
            for idx, (x, y) in enumerate(X_sample):
                ax.scatter(x, y, np.max(Z), color=colors[idx], s=40, marker='o')

            # Add color bar
            norm = Normalize(vmin=0, vmax=num_points)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
    
            cbar_ax = fig.add_axes([0.91, 0.55, 0.015, 0.3])  # [left, bottom, width, height]
            cbar = plt.colorbar(sm, cax=cbar_ax, ticks=[0, num_points])
            cbar.ax.set_yticklabels(['Start', 'End'])
            cbar.set_label('Palpation Order', rotation=270, labelpad=15)

        # Black dots in the other plots
        elif X_sample is not None:
            ax.scatter(X_sample[:, 0], X_sample[:, 1], np.max(Z) * np.ones(len(X_sample)),
                       c='k', marker='D', label='Samples')

        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Value')

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()

# plotting the 3D surface one by one
""" def plot_surface_3d(X1, X2, surfaces, titles, X_sample=None):
    cmap = get_cmap("RdYlGn")

    for i, (Z, title) in enumerate(zip(surfaces, titles)):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X1, X2, Z.reshape(X1.shape), cmap='viridis', edgecolor='none', alpha=0.9)

        # Brug farveskala til Phantom-plot
        if X_sample is not None and i == 1:
            num_points = len(X_sample)
            colors = cmap(np.linspace(0, 1, num_points))
            for idx, (x, y) in enumerate(X_sample):
                ax.scatter(x, y, np.max(Z), color=colors[idx], s=40, marker='o')

            norm = Normalize(vmin=0, vmax=num_points)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, ticks=[0, num_points])
            cbar.ax.set_yticklabels(['Start', 'End'])
            cbar.set_label('Palpation Order', rotation=270, labelpad=15)

        elif X_sample is not None:
            ax.scatter(X_sample[:, 0], X_sample[:, 1], np.max(Z) * np.ones(len(X_sample)),
                       c='k', marker='D', label='Samples')

        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Value')

        plt.tight_layout()
        plt.show() """

# Run the full system
if __name__ == "__main__":
    X1, X2, mu, std, ei_final, Z_target, X_sample = run_bo_3d()
    plot_surface_3d(
        X1, X2,
        [mu, Z_target, std, ei_final],
        ["GP Predicted Mean", "Simulated Tumor in Tissue Phantom", "Uncertainty (Std. Dev.)", "Expected Improvement"],
        X_sample=X_sample
    )