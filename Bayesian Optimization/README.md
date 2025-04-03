# 3D Bayesian Optimization Project

This project implements a Bayesian optimization algorithm to optimize a target function representing a tumor modeled as a 2D Gaussian bump. The optimization process is visualized in 3D, showcasing the predicted mean, true function, standard deviation, and expected improvement.

## Project Structure

```
3d-bo-project
├── src
│   ├── main.py                # Entry point of the application
│   ├── target_function.py     # Contains the target function definition
│   ├── expected_improvement.py # Contains the expected improvement function
│   ├── bayesian_optimization.py# Implements the Bayesian optimization loop
│   ├── plot_surface.py        # Responsible for visualizing the results
│   └── __init__.py           # Marks the directory as a Python package
└── README.md                  # Documentation for the project
```

## Installation

To run this project, ensure you have Python installed along with the necessary libraries. You can install the required libraries using pip:

```bash
pip install numpy matplotlib scikit-learn scipy
```

## Usage

```bash
python src/main.py
```
This will execute the Bayesian optimization process and display the resulting 3D plots.

## libraries

- numpy
- matplotlib
- scikit-learn
- scipy

