import numpy as np

# Target function: models a tumor as a 2D Gaussian bump
# sigma controls the width of the bump
# center controls the location of the bump
# amplitude controls the height of the bump
def target_function(X, center=(3, 5), sigma=0.5, amplitude=1.0):
    x, y = X[:, 0], X[:, 1]
    return amplitude * np.exp(-((x - center[0])**2 + (y - center[1])**2) / (2 * sigma**2))

def target_function2(center, amplitude, sigma=0.5):
    return amplitude * np.exp(-((center[0])**2 + (center[1])**2) / (2 * sigma**2))