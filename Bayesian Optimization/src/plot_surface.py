def plot_surface_3d(X1, X2, surfaces, titles, X_sample=None):
    """
    Plots 3D surfaces for visualization.

    Args:
        X1, X2 (ndarray): Meshgrid for the surface.
        surfaces (list): List of surfaces to plot.
        titles (list): Titles for each subplot.
        X_sample (ndarray, optional): Sample points to overlay on the plots.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    fig = plt.figure(figsize=(16, 12))
    for i, (Z, title) in enumerate(zip(surfaces, titles)):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        ax.plot_surface(X1, X2, Z.reshape(X1.shape), cmap='viridis', edgecolor='none', alpha=0.9)
        if X_sample is not None:
            ax.scatter(X_sample[:, 0], X_sample[:, 1], np.max(Z) * np.ones(len(X_sample)), 
                       c='k', marker='D', label='Samples')
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Value')
    plt.tight_layout()
    plt.show()
