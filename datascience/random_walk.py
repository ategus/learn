import numpy as np
import matplotlib.pyplot as plt

# Define the number of steps
n_steps = 300

# Define the step size
step_size = 1.5

# Initialize the position
position = 0.0

# Create an array to store the positions
positions = np.zeros(n_steps)

# Simulate the random walk
for i in range(n_steps):
    # Generate a random step
    step = np.random.normal(0, step_size)
    
    # Update the position
    position += step
    
    # Store the position
    positions[i] = position

# Plot the positions
plt.plot(positions)
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Random Walk')
plt.show()
