from bayesian_optimization import run_bo_3d
from plot_surface import plot_surface_3d
from simulation import simulate_tissue

if __name__ == "__main__":
    # Simulate the tissue stiffness field
    X1, X2, tissue_stiffness = simulate_tissue(grid_size=50)  # Ensure grid_size matches run_bo_3d
    print("Tissue stiffness generated successfully.")

    # Run Bayesian optimization
    X1, X2, mu, std, ei_final, Z_target, X_sample = run_bo_3d(
        n_init=5, n_iter=20, grid_size=50, seed=42, tissue_stiffness=tissue_stiffness
    )

    # Visualize the results
    plot_surface_3d(
        X1, X2,
        [mu, tissue_stiffness, std, ei_final],
        ["GP Predicted Mean", "True Tissue Stiffness", "GP Std Dev (Uncertainty)", "Expected Improvement"],
        X_sample=X_sample
    )