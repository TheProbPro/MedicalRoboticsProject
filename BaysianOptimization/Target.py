import numpy as np

# Target function: models a tumor as a 2D Gaussian bump
# sigma controls the width of the bump
# center controls the location of the bump
# amplitude controls the height of the bump
# def target_function(X, center=(3, 5), sigma=0.5, amplitude=1.0):
#     if X.ndim == 1:
#         # single sample
#         x, y = X[0], X[1]
#         # compute and return scalar
#     elif X.ndim == 2:
#         # batch of samples
#         x, y = X[:, 0], X[:, 1]
    
#     return amplitude * np.exp(-((x - center[0])**2 + (y - center[1])**2) / (2 * sigma**2))

def target_function(X, center=(5,5), sigma=1, amplitude=0.8, noise_std=0.1):
    # cnter: tumors placement
    # sigma: tumor size
    # amplitude: tumor height
    X = np.array(X)
    x = None
    y = None
    if X.ndim == 1:
        # single sample
        x, y = X[0], X[1]
        # compute and return scalar
        tumor = amplitude * np.exp(-((x - center[0])**2 + (y - center[1])**2) / (2 * sigma**2))
        noise = np.random.normal(0, noise_std)  # Just one value
        return tumor + noise
    elif X.ndim == 2:
        # batch of samples
        x, y = X[:, 0], X[:, 1]
        tumor = amplitude * np.exp(-((x - center[0])**2 + (y - center[1])**2) / (2 * sigma**2))
        noise = np.random.normal(0, noise_std, size=len(X))
        return tumor + noise

    if sigma == 0 or amplitude == 0:
        return np.zeros(len(X))   # Return zeros if tumor is "off"

    tumor = amplitude * np.exp(-((x - center[0])**2 + (y - center[1])**2) / (2 * sigma**2))
    noise = np.random.normal(0, noise_std, size=len(X))
    return tumor + noise

def target_function2(center, amplitude, sigma=0.5):
    return amplitude * np.exp(-((center[0])**2 + (center[1])**2) / (2 * sigma**2))