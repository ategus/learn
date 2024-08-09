import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, degree=3):
    return (1 + np.dot(x1, x2)) ** degree

def rbf_kernel(x1, x2, gamma=1):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

def sigmoid_kernel(x1, x2, gamma=1, c=1):
    return np.tanh(gamma * np.dot(x1, x2) + c)

# Generate sample data
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Create subplots
fig = plt.figure(figsize=(20, 15))

# Linear Kernel
ax1 = fig.add_subplot(221, projection='3d')
Z1 = np.array([linear_kernel(np.array([x, y]), np.array([1, 1])) for x, y in zip(np.ravel(X), np.ravel(Y))])
Z1 = Z1.reshape(X.shape)
ax1.plot_surface(X, Y, Z1, cmap='viridis')
ax1.set_title('Linear Kernel')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('K(X, [1,1])')

# Polynomial Kernel
ax2 = fig.add_subplot(222, projection='3d')
Z2 = np.array([polynomial_kernel(np.array([x, y]), np.array([1, 1]), degree=3) for x, y in zip(np.ravel(X), np.ravel(Y))])
Z2 = Z2.reshape(X.shape)
ax2.plot_surface(X, Y, Z2, cmap='viridis')
ax2.set_title('Polynomial Kernel (degree=3)')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('K(X, [1,1])')

# RBF Kernel
ax3 = fig.add_subplot(223, projection='3d')
Z3 = np.array([rbf_kernel(np.array([x, y]), np.array([1, 1]), gamma=0.5) for x, y in zip(np.ravel(X), np.ravel(Y))])
Z3 = Z3.reshape(X.shape)
ax3.plot_surface(X, Y, Z3, cmap='viridis')
ax3.set_title('RBF Kernel (gamma=0.5)')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('K(X, [1,1])')

# Sigmoid Kernel
ax4 = fig.add_subplot(224, projection='3d')
Z4 = np.array([sigmoid_kernel(np.array([x, y]), np.array([1, 1]), gamma=0.5, c=1) for x, y in zip(np.ravel(X), np.ravel(Y))])
Z4 = Z4.reshape(X.shape)
ax4.plot_surface(X, Y, Z4, cmap='viridis')
ax4.set_title('Sigmoid Kernel (gamma=0.5, c=1)')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('K(
