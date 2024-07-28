import numpy as np

# Create a 1D array
arr = np.arange(8)
print("Original array:", arr)

# Reshape to a 4x2 array
reshaped_arr = arr.reshape((4, 2))
print("Reshaped array (4x2):", reshaped_arr)

# Reshape to a 2x4 array
reshaped_again = reshaped_arr.reshape((2, 4))
print("Reshaped again (2x4):", reshaped_again)

# Using -1 to infer dimension
arr = np.arange(15)
inferred_shape = arr.reshape((5, -1))
print("Reshaped with inferred dimension (5x3):", inferred_shape)


#Example-02
import numpy as np

# Example 1: Reshape a 1D array to 3D array
arr = np.arange(24)
reshaped_3d = arr.reshape((2, 3, 4))
print("Reshaped to 3D (2x3x4):\n", reshaped_3d)

# Example 2: Reshape a 2D array to 1D array
arr = np.array([[1, 2, 3], [4, 5, 6]])
reshaped_1d = arr.reshape(-1)
print("Reshaped to 1D:\n", reshaped_1d)


