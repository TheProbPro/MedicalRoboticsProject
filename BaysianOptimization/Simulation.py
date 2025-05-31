import numpy as np

def simulate_tissue(grid_size=50, tumor_center=(3, 5), tumor_sigma=0.5, tumor_amplitude=1.0):
    """
    Simulates the tissue stiffness field with a tumor.

    Args:
        grid_size (int): Size of the tissue area (grid_size x grid_size).
        tumor_center (tuple): Tumor's (x, y) position.
        tumor_sigma (float): Tumor's spread (standard deviation).
        tumor_amplitude (float): Tumor's stiffness (amplitude).

    Returns:
        X1, X2 (ndarray): Meshgrid for the tissue area.
        tissue_stiffness (ndarray): Tissue stiffness field.
    """
    x = np.linspace(0, 6, grid_size)
    y = np.linspace(0, 6, grid_size)
    X1, X2 = np.meshgrid(x, y)
    tissue_stiffness = tumor_amplitude * np.exp(
        -((X1 - tumor_center[0])**2 + (X2 - tumor_center[1])**2) / (2 * tumor_sigma**2)
    )
    return X1, X2, tissue_stiffness