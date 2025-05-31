import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from BaysianOptimization.Target import target_function
from BaysianOptimization.Acquisition import expected_improvement
from BaysianOptimization.PlotUtils import plot_surface_3d
# from Target import target_function
# from Acquisition import expected_improvement
# from PlotUtils import plot_surface_3d

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

class BayesianOptimizer3D:
    def __init__(self, boxSize, boxPosition, grid_size=60, n_init=10, seed=42):
        """
        Initialize the Bayesian Optimizer with the box size and position.
        """
        self.boxSize = boxSize * 100
        self.boxPosition = boxPosition * 100
        self.grid_size = grid_size
        self.n_init = n_init
        self.seed = seed
        self.X_grid = None
        self.X_sample = np.array([])
        self.y_sample = np.array([])
        self.gp = GaussianProcessRegressor(kernel=Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=1.5), alpha=1e-6, normalize_y=True)
        np.random.seed(self.seed)

    def init_random_samples(self):
        """
        Generate a 2D grid based on the box size and position.
        """
        # Generate a grid of points within the box
        x = np.linspace((self.boxPosition[0] - self.boxSize[0] / 2) + 1, (self.boxPosition[0] + self.boxSize[0] / 2) - 1, self.grid_size)
        y = np.linspace((self.boxPosition[1] - self.boxSize[1] / 2) + 1, (self.boxPosition[1] + self.boxSize[1] / 2) - 1, self.grid_size)
        self.X1, self.X2 = np.meshgrid(x, y)
        self.X_grid = np.vstack([self.X1.ravel(), self.X2.ravel()]).T
        # Randomly select n_init amount of points
        init_idx = np.random.choice(len(self.X_grid), self.n_init, replace=False)
        X_sample = self.X_grid[init_idx]
        print(f"x grid size: {self.X_grid}")
        return X_sample / 100

    def update_samples(self, new_point, force_reading):
        """
        Update the samples with a new point and its corresponding force reading.
        """
        if self.X_sample.size == 0 and self.y_sample.size == 0:
            self.X_sample = np.array([new_point*100])
            self.y_sample = np.array([force_reading])
        else:
            self.X_sample = np.vstack([self.X_sample, new_point*100])
            self.y_sample = np.append(self.y_sample, force_reading)
        
        print(f"Updated samples, X_sample: {new_point*100}, y_sample: {force_reading}")

    def get_next_sample(self):
        """
        Generate the next point to sample using Expected Improvement.
        """
        self.gp.fit(self.X_sample, self.y_sample)
        y_best = np.max(self.y_sample)
        ei = expected_improvement(self.X_grid, self.gp, y_best)
        next_idx = np.argmax(ei)
        
        next_sample = self.X_grid[next_idx]
        box_min = self.boxPosition - self.boxSize / 2
        box_max = self.boxPosition + self.boxSize / 2
        if not (box_min[0] <= next_sample[0] <= box_max[0] and
                box_min[1] <= next_sample[1] <= box_max[1]):
            raise ValueError("Next sample is outside the box boundaries.")
        return next_sample/100
    
    def get_number_of_samples(self):
        """
        Get the number of samples collected so far.
        """
        if self.X_sample is None:
            return 0    
        else:
            return len(self.X_sample)
    
    def plot_optimization(self):
        """
        Plots graphs
        """
        self.gp.fit(self.X_sample, self.y_sample)
        mu, std = self.gp.predict(self.X_grid, return_std=True)
        ei_final = expected_improvement(self.X_grid, self.gp, np.max(self.y_sample))
        Z_target = target_function(self.X_grid)
        plot_surface_3d(
            self.X1, self.X2,
            [mu, Z_target, std, ei_final],
            ["GP Predicted Mean", "True Function", "GP Std Dev (Uncertainty)", "Expected Improvement"],
            X_sample=self.X_sample
        )


# class BayesianOptimizer3D:
#     def __init__(self, boxSize, boxPosition, margin = 0.1, grid_size=50, n_init=5, seed=42):
#         """
#         Initialize the Bayesian Optimizer with the box size and position in meters.
#         Internally convert everything to centimeters.
#         """
#         self.boxSize = np.array(boxSize) * 100        # Convert to cm
#         self.boxPosition = np.array(boxPosition) * 100  # Convert to cm
#         self.margin = margin
#         self.grid_size = grid_size
#         self.n_init = n_init
#         self.seed = seed
#         self.X_grid = None
#         self.X_sample = np.array([])
#         self.y_sample = np.array([])
#         self.gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-6, normalize_y=True)
#         np.random.seed(self.seed)

#     def init_random_samples(self):
#         """
#         Generate a 2D grid based on the box size and position (in cm),
#         return initial samples converted to meters.
#         """
#         x = np.linspace(
#             self.boxPosition[0] - self.boxSize[0] / 2 + self.margin,
#             self.boxPosition[0] + self.boxSize[0] / 2 - self.margin,
#             self.grid_size
#         )
#         y = np.linspace(
#             self.boxPosition[1] - self.boxSize[1] / 2 + self.margin,
#             self.boxPosition[1] + self.boxSize[1] / 2 - self.margin,
#             self.grid_size
#         )

#         self.X1, self.X2 = np.meshgrid(x, y)
#         self.X_grid = np.vstack([self.X1.ravel(), self.X2.ravel()]).T

#         init_idx = np.random.choice(len(self.X_grid), self.n_init, replace=False)
#         X_sample_cm = self.X_grid[init_idx]

#         return X_sample_cm / 100  # Return in meters

#     def update_samples(self, new_point, force_reading):
#         """
#         Update the samples. Convert input point from meters to cm internally.
#         """
#         new_point_cm = np.array(new_point) * 100
#         force_reading = np.array(force_reading) * 100

#         if self.X_sample.size == 0 and self.y_sample.size == 0:
#             self.X_sample = np.array([new_point_cm])
#             self.y_sample = np.array([force_reading])
#         else:
#             self.X_sample = np.vstack([self.X_sample, new_point_cm])
#             self.y_sample = np.append(self.y_sample, force_reading)

#     def get_next_sample(self):
#         """
#         Generate the next point to sample using Expected Improvement.
#         Return point in meters.
#         """
#         self.gp.fit(self.X_sample, self.y_sample)
#         y_best = np.max(self.y_sample)
#         ei = expected_improvement(self.X_grid, self.gp, y_best)
#         next_idx = np.argmax(ei)

#         next_sample_cm = self.X_grid[next_idx]

#         # Validate it lies within box bounds
#         box_min = self.boxPosition - self.boxSize / 2
#         box_max = self.boxPosition + self.boxSize / 2

#         if not (box_min[0] <= next_sample_cm[0] <= box_max[0] and
#                 box_min[1] <= next_sample_cm[1] <= box_max[1]):
#             raise ValueError("Next sample is outside the box boundaries.")

#         return next_sample_cm / 100  # Return in meters

#     def get_number_of_samples(self):
#         """
#         Get the number of samples collected so far.
#         """
#         return 0 if self.X_sample is None else len(self.X_sample)

#     def plot_optimization(self):
#         """
#         Plot optimization results. All internal data in cm.
#         """
#         self.gp.fit(self.X_sample, self.y_sample)
#         mu, std = self.gp.predict(self.X_grid, return_std=True)
#         ei_final = expected_improvement(self.X_grid, self.gp, np.max(self.y_sample))
#         Z_target = target_function(self.X_grid)

#         plot_surface_3d(
#             self.X1, self.X2,
#             [mu, Z_target, std, ei_final],
#             ["GP Predicted Mean", "True Function", "GP Std Dev (Uncertainty)", "Expected Improvement"],
#             X_sample=self.X_sample
#         )
