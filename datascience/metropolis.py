import numpy as np
import matplotlib.pyplot as plt

def metropolis_hastings(p, q, x0, num_samples, burn_in):
    """
    Metropolis-Hastings algorithm

    Parameters:
    p (function): target distribution
    q (function): proposal distribution
    x0 (float): initial state
    num_samples (int): number of samples to generate
    burn_in (int): number of burn-in samples

    Returns:
    samples (numpy array): generated samples
    """
    samples = np.zeros(num_samples)
    x_t = x0

    for t in range(num_samples + burn_in):
        # Proposal
        x_prime = q(x_t)

        # Acceptance
        alpha = min(1, (p(x_prime) / p(x_t)) * (q(x_t) / q(x_prime)))

        # Accept or Reject
        u = np.random.uniform(0, 1)
        if u < alpha:
            x_t = x_prime

        # Store sample after burn-in
        if t >= burn_in:
            samples[t - burn_in] = x_t

    return samples

# Target distribution (e.g., Gaussian)
def p(x):
    return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

# Proposal distribution (e.g., uniform)
def q(x):
    return np.random.uniform(-1, 1)

# Initial state
x0 = 0

# Number of samples
num_samples = 1000

# Burn-in period
burn_in = 100

# Generate samples
samples = metropolis_hastings(p, q, x0, num_samples, burn_in)

# Plot histogram of samples
plt.hist(samples, bins=30, density=True)
plt.xlabel('x')
plt.ylabel('Probability density')
plt.title('Metropolis-Hastings samples')
plt.show()
