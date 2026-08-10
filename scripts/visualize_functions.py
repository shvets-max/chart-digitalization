import matplotlib.pyplot as plt
import numpy as np

from function import Linear, Logarithmic

# Example parameters
y0, y1 = 0, 100
v0, v1 = 100, 1000

# Linear scale
linear = Linear(y0, y1, v0, v1)
y_vals = np.linspace(y0, y1, 200)
v_linear = linear(y_vals)
y_inv_linear = linear.invert(v_linear)

# Logarithmic scale
logarithmic = Logarithmic(y0, y1, v0, v1)
v_log = logarithmic(y_vals)
y_inv_log = logarithmic.invert(v_log)

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Linear: y -> v
axs[0, 0].plot(y_vals, v_linear)
axs[0, 0].set_title("Linear: y → v")
axs[0, 0].set_xlabel("y")
axs[0, 0].set_ylabel("v")

# Linear: v -> y (invert)
axs[0, 1].plot(v_linear, y_inv_linear)
axs[0, 1].set_title("Linear: v → y (invert)")
axs[0, 1].set_xlabel("v")
axs[0, 1].set_ylabel("y")

# Logarithmic: y -> v
axs[1, 0].plot(y_vals, v_log)
axs[1, 0].set_title("Logarithmic: y → v")
axs[1, 0].set_xlabel("y")
axs[1, 0].set_ylabel("v")

# Logarithmic: v -> y (invert)
axs[1, 1].plot(v_log, y_inv_log)
axs[1, 1].set_title("Logarithmic: v → y (invert)")
axs[1, 1].set_xlabel("v")
axs[1, 1].set_ylabel("y")

plt.tight_layout()
plt.show()
