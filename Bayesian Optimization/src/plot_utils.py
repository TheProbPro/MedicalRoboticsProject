import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plots
# Plot multiple 3D surfaces side-by-side
def plot_surface_3d(X1, X2, surfaces, titles, X_sample=None):
    fig = plt.figure(figsize=(16, 12))
    for i, (Z, title) in enumerate(zip(surfaces, titles)):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        ax.plot_surface(X1, X2, Z.reshape(X1.shape), cmap='viridis', edgecolor='none', alpha=0.9)
        
        # Add sample points
        if X_sample is not None:
            ax.scatter(X_sample[:, 0], X_sample[:, 1], np.max(Z) * np.ones(len(X_sample)), 
                       c='k', marker='D', label='Samples')
        
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Value')
    
    plt.tight_layout()
    plt.show()
