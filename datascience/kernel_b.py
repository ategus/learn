import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Define the kernel functions
def linear_kernel(x, y):
    return np.dot(x, y)

def polynomial_kernel(x, y, degree=2):
    return np.power(np.dot(x, y) + 1, degree)

def rbf_kernel(x, y, sigma=1):
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))

def sigmoid_kernel(x, y, alpha=1, beta=1):
    return np.tanh(alpha * np.dot(x, y) + beta)

# Generate some sample data
np.random.seed(0)
X = np.random.rand(100, 2)
y = np.where(X[:, 0] + X[:, 1] > 1, 1, -1)

# Apply the kernel functions
linear_matrix = np.zeros((len(X), len(X)))
polynomial_matrix = np.zeros((len(X), len(X)))
rbf_matrix = np.zeros((len(X), len(X)))
sigmoid_matrix = np.zeros((len(X), len(X)))

for i in range(len(X)):
    for j in range(len(X)):
        linear_matrix[i, j] = linear_kernel(X[i], X[j])
        polynomial_matrix[i, j] = polynomial_kernel(X[i], X[j])
        rbf_matrix[i, j] = rbf_kernel(X[i], X[j])
        sigmoid_matrix[i, j] = sigmoid_kernel(X[i], X[j])

# Visualize the kernel matrices
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

axs[0, 0].imshow(linear_matrix, cmap='hot', interpolation='nearest')
axs[0, 0].set_title('Linear Kernel')

axs[0, 1].imshow(polynomial_matrix, cmap='hot', interpolation='nearest')
axs[0, 1].set_title('Polynomial Kernel')

axs[1, 0].imshow(rbf_matrix, cmap='hot', interpolation='nearest')
axs[1, 0].set_title('RBF Kernel')

axs[1, 1].imshow(sigmoid_matrix, cmap='hot', interpolation='nearest')
axs[1, 1].set_title('Sigmoid Kernel')

plt.show()

# Train an SVM model using the linear kernel
svm_model = svm.SVC(kernel='linear')
svm_model.fit(X, y)

# Visualize the decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title('Linear Kernel SVM')
plt.show()

# Train an SVM model using the RBF kernel
svm_model = svm.SVC(kernel='rbf')
svm_model.fit(X, y)

# Visualize the decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title('RBF Kernel SVM')
plt.show()
