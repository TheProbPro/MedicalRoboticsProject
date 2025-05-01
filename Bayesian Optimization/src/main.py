from optimizer import run_bo_3d
from plot_utils import plot_surface_3d

# Entry point to run Bayesian Optimization and visualize the result
if __name__ == "__main__":
    X1, X2, mu, std, ei_final, Z_target, X_sample = run_bo_3d()
    plot_surface_3d(
        X1, X2,
        [mu, Z_target, std, ei_final],
        ["GP Predicted Mean", "True Function", "GP Std Dev (Uncertainty)", "Expected Improvement"],
        X_sample=X_sample
    )
