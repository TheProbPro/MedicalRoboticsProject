import numpy as np

def target_function(X, tissue_stiffness):
    """
    Returns stiffness values for the given points in the tissue area.

    Args:
        X (ndarray): Array of points (x, y) where stiffness is measured.
        tissue_stiffness (ndarray): Tissue stiffness field.

    Returns:
        ndarray: Stiffness values for the given points.
    """
    grid_size = tissue_stiffness.shape[0]
    x_idx = np.clip((X[:, 0] * (grid_size / 6)).astype(int), 0, grid_size - 1)
    y_idx = np.clip((X[:, 1] * (grid_size / 6)).astype(int), 0, grid_size - 1)
    return tissue_stiffness[x_idx, y_idx]
