from BaysianOptimization.Optimizer import BayesianOptimizer3D
from BaysianOptimization.Target import target_function

import spatialgeometry as sg
from spatialmath import SE3, SO3
import numpy as np

if __name__ == "__main__":
    # Define the box size and position
    box_plane = SE3(0, 0, 0)
    box_size = np.array([0.15, 0.15, 0.15])

    # Initialize the optimizer
    optimizer = BayesianOptimizer3D(box_size, box_plane.t, grid_size=50)

    # Get the initial sample
    initial_sample = optimizer.init_random_samples()
    print(f"Initial Sample: {initial_sample}")

    for sample in initial_sample:
        force_reading = target_function(sample)
        print(f"Force Reading for {sample}: {force_reading}")
        optimizer.update_samples(sample, force_reading)

    for i in range(40):  # Simulate 30 iterations of optimization
        sample = optimizer.get_next_sample()
        force_reading = target_function(sample)
        print(f"Force Reading for {sample}: {force_reading}")
        optimizer.update_samples(sample, force_reading)
    
    #plot
    optimizer.plot_optimization()



    




    # # Update samples with a new point and its corresponding force reading
    # new_point = [0.1, 0.1]  # Example new point in meters
    # force_reading = 10.0  # Example force reading
    # optimizer.update_samples(new_point, force_reading)

    # # Get the next sample to optimize
    # next_sample = optimizer.get_next_sample()
    # print(f"Next Sample: {next_sample}")

    # # Get the number of samples collected so far
    # num_samples = optimizer.get_number_of_samples()
    # print(f"Number of Samples Collected: {num_samples}")

    # Plot the optimization results
    optimizer.plot_optimization()