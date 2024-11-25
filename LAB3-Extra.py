import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate input values
x = np.linspace(0.1, 1, 20)
# Calculate desired output values
y = (1 + 0.6 * np.sin(2 * np.pi * x / 0.7) + 0.3 * np.sin(2 * np.pi * x)) / 2

# Define Gaussian RBF
def gaussian_rbf(x, c, r):
    return np.exp(-((x - c) ** 2) / (2 * r ** 2))

# Manually select initial parameters
c1, r1 = 0.2, 0.2
c2, r2 = 0.8, 0.2

# Compute initial RBF outputs
phi1 = gaussian_rbf(x, c1, r1)
phi2 = gaussian_rbf(x, c2, r2)

# Construct design matrix
Phi = np.vstack([phi1, phi2, np.ones_like(x)]).T

# Train the network using the perceptron training algorithm
weights = np.linalg.pinv(Phi).dot(y)
w1, w2, w0 = weights

# Evaluate the network
y_pred = Phi.dot(weights)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'bo-', label="Desired Output")
plt.plot(x, y_pred, 'r--', label="RBF Network Output")
plt.title("RBF Network Approximation")
plt.xlabel("Input (x)")
plt.ylabel("Output (y)")
plt.legend()
plt.grid(True)
plt.show()

# Additional Task: Alternative RBF Training Algorithm
# Use K-Means clustering to update centers and radii
kmeans = KMeans(n_clusters=2)
kmeans.fit(x.reshape(-1, 1))
c1, c2 = kmeans.cluster_centers_.flatten()
r1, r2 = np.std(x) / 2, np.std(x) / 2

# Recompute RBF outputs with updated parameters
phi1 = gaussian_rbf(x, c1, r1)
phi2 = gaussian_rbf(x, c2, r2)

# Reconstruct design matrix with updated RBFs
Phi = np.vstack([phi1, phi2, np.ones_like(x)]).T

# Retrain the network with updated RBFs
weights = np.linalg.pinv(Phi).dot(y)
w1, w2, w0 = weights

# Re-evaluate the network with updated RBFs
y_pred = Phi.dot(weights)

# Print updated weights
print("Weights:")
print(f"w1: {w1}, w2: {w2}, w0: {w0}")

# Plot the updated results
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'bo-', label="Desired Output")
plt.plot(x, y_pred, 'g--', label="Updated RBF Network Output")
plt.title("Updated RBF Network Approximation")
plt.xlabel("Input (x)")
plt.ylabel("Output (y)")
plt.legend()
plt.grid(True)
plt.show()
