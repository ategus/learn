# Load necessary libraries
library(ggplot2)
library(gridExtra)

# Define kernel functions
linear_kernel <- function(x1, x2) {
  sum(x1 * x2)
}

polynomial_kernel <- function(x1, x2, degree) {
  (1 + sum(x1 * x2))^degree
}

rbf_kernel <- function(x1, x2, gamma) {
  exp(-gamma * sum((x1 - x2)^2))
}

sigmoid_kernel <- function(x1, x2, gamma, c) {
  tanh(gamma * sum(x1 * x2) + c)
}

# Function to compute and plot kernels
plot_kernel <- function(kernel_func, params, title) {
  x <- seq(-5, 5, length.out = 50)
  y <- seq(-5, 5, length.out = 50)
  grid <- expand.grid(x = x, y = y)
  
  # Compute kernel values
  z <- matrix(apply(grid, 1, function(point) kernel_func(c(point[1], point[2]), c(1, 1), ...)), nrow = length(x), ncol = length(y))
  
  # Create the plot
  p <- ggplot() +
    geom_raster(data = as.data.frame(grid), aes(x = x, y = y, fill = as.vector(z))) +
    scale_fill_viridis_c() +
    labs(title = title, x = "X", y = "Y", fill = "Kernel Value") +
    theme_minimal()
  
  return(p)
}

# Parameters for kernels
params_polynomial <- list(degree = 3)
params_rbf <- list(gamma = 0.5)
params_sigmoid <- list(gamma = 0.5, c = 1)

# Create plots
plot_linear <- plot_kernel(function(x1, x2, ...) linear_kernel(x1, x2), NULL, "Linear Kernel")
plot_polynomial <- plot_kernel(function(x1, x2, degree) polynomial_kernel(x1, x2, degree), params_polynomial, "Polynomial Kernel")
plot_rbf <- plot_kernel(function(x1, x2, gamma) rbf_kernel(x1, x2, gamma), params_rbf, "RBF Kernel")
plot_sigmoid <- plot_kernel(function(x1, x2, gamma, c) sigmoid_kernel(x1, x2, gamma, c), params_sigmoid, "Sigmoid Kernel")

# Arrange plots in a grid
grid.arrange(plot_linear, plot_polynomial, plot_rbf, plot_sigmoid, ncol = 2)

