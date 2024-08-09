# Install and load the required packages
library(MASS)
library(kernlab)
library(ggplot2)

# Define the kernel functions
linear_kernel <- function(x, y) {
  sum(x * y)
}

polynomial_kernel <- function(x, y, degree = 2) {
  (sum(x * y) + 1) ^ degree
}

rbf_kernel <- function(x, y, sigma = 1) {
  exp(-sum((x - y) ^ 2) / (2 * sigma ^ 2))
}

sigmoid_kernel <- function(x, y, alpha = 1, beta = 1) {
  tanh(alpha * sum(x * y) + beta)
}

# Generate some sample data
set.seed(123)
x <- matrix(rnorm(100 * 2), nrow = 100, ncol = 2)
y <- ifelse(x[, 1] + x[, 2] > 1, 1, -1)

# Apply the kernel functions
linear_matrix <- matrix(0, nrow = nrow(x), ncol = nrow(x))
polynomial_matrix <- matrix(0, nrow = nrow(x), ncol = nrow(x))
rbf_matrix <- matrix(0, nrow = nrow(x), ncol = nrow(x))
sigmoid_matrix <- matrix(0, nrow = nrow(x), ncol = nrow(x))

for (i in 1:nrow(x)) {
  for (j in 1:nrow(x)) {
    linear_matrix[i, j] <- linear_kernel(x[i, ], x[j, ])
    polynomial_matrix[i, j] <- polynomial_kernel(x[i, ], x[j, ])
    rbf_matrix[i, j] <- rbf_kernel(x[i, ], x[j, ])
    sigmoid_matrix[i, j] <- sigmoid_kernel(x[i, ], x[j, ])
  }
}

# Visualize the kernel matrices
ggplot(data.frame(x = 1:nrow(linear_matrix), y = 1:nrow(linear_matrix), z = as.vector(linear_matrix)), aes(x, y, fill = z)) +
  geom_tile() +
  labs(title = "Linear Kernel")

ggplot(data.frame(x = 1:nrow(polynomial_matrix), y = 1:nrow(polynomial_matrix), z = as.vector(polynomial_matrix)), aes(x, y, fill = z)) +
  geom_tile() +
  labs(title = "Polynomial Kernel")

ggplot(data.frame(x = 1:nrow(rbf_matrix), y = 1:nrow(rbf_matrix), z = as.vector(rbf_matrix)), aes(x, y, fill = z)) +
  geom_tile() +
  labs(title = "RBF Kernel")

ggplot(data.frame(x = 1:nrow(sigmoid_matrix), y = 1:nrow(sigmoid_matrix), z = as.vector(sigmoid_matrix)), aes(x, y, fill = z)) +
  geom_tile() +
  labs(title = "Sigmoid Kernel")

# Train an SVM model using the linear kernel
svm_model <- ksvm(x, y, type = "C-svc", kernel = "vanilladot")

# Visualize the decision boundary
ggplot(data.frame(x = x[, 1], y = x[, 2], z = y), aes(x, y, color = factor(z))) +
  geom_point() +
  labs(title = "Linear Kernel SVM")
