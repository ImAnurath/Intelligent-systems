# IS LAB2-Extra

import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

x1 = np.linspace(0, 1, 20)
x2 = np.linspace(0, 1, 20)
x1, x2 = np.meshgrid(x1, x2)
X = np.c_[x1.ravel(), x2.ravel()] #tf is ravel :'')
y = (np.sin(np.pi * x1) * np.cos(np.pi * x2)).ravel()

mlp = MLPRegressor(hidden_layer_sizes=(10,), activation='tanh', solver='lbfgs', max_iter=1000, random_state=1)
mlp.fit(X, y)

y_pred = mlp.predict(X)
fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(x1, x2, y.reshape(x1.shape), cmap='viridis')
ax1.set_title('Actual Surface')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('y')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(x1, x2, y_pred.reshape(x1.shape), cmap='viridis')
ax2.set_title('Predicted Surface')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('y_pred')

plt.show()
