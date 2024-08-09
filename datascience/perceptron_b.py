import numpy as np
import matplotlib.pyplot as plt

# Define the Perceptron class
class Perceptron:
    def __init__(self, learning_rate=0.01, num_iterations=100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _forward_pass(self, inputs):
        return self._sigmoid(np.dot(inputs, self.weights) + self.bias)

    def _error_calculation(self, predicted, actual):
        return actual - predicted

    def _weight_update(self, error, inputs):
        return self.weights + self.learning_rate * error * inputs

    def _bias_update(self, error):
        return self.bias + self.learning_rate * error

    def fit(self, inputs, targets):
        num_samples, num_features = inputs.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            for sample in range(num_samples):
                predicted = self._forward_pass(inputs[sample])
                error = self._error_calculation(predicted, targets[sample])
                self.weights = self._weight_update(error, inputs[sample])
                self.bias = self._bias_update(error)

    def predict(self, inputs):
        return np.where(self._forward_pass(inputs) >= 0.5, 1, 0)

# Generate some sample data
np.random.seed(0)
inputs = np.random.rand(100, 2)
targets = np.where(inputs[:, 0] + inputs[:, 1] > 1, 1, 0)

# Train the Perceptron model
model = Perceptron()
model.fit(inputs, targets)

# Make predictions
predictions = model.predict(inputs)

# Plot the results
plt.scatter(inputs[:, 0], inputs[:, 1], c=targets, label='Actual')
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions, marker='x', label='Predicted')
plt.legend()
plt.show()
