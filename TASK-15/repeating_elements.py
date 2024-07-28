# Create a 1D array
import numpy as np
arr = np.arange(3)
print("Original array:", arr)

# Repeat each element 3 times
repeated_arr = arr.repeat(3)
print("Repeated array:", repeated_arr)

# Create a 2D array
arr = np.random.randn(2, 2)
print("Original 2D array:\n", arr)

# Repeat along the first axis
repeated_axis0 = arr.repeat(2, axis=0)
print("Repeated along axis 0:\n", repeated_axis0)

# Repeat along the second axis
repeated_axis1 = arr.repeat(2, axis=1)
print("Repeated along axis 1:\n", repeated_axis1)

# Tile the array
tiled_arr = np.tile(arr, (2, 1))
print("Tiled array:\n", tiled_arr)



#Example-02
# Example 1: Using repeat on a 1D array
arr = np.array([1, 2, 3])
repeated_arr = arr.repeat(3)
print("Repeated array (each element 3 times):", repeated_arr)

# Example 2: Using repeat on a 2D array along axis 0
arr = np.array([[1, 2], [3, 4]])
repeated_axis0 = arr.repeat(2, axis=0)
print("Repeated along axis 0:\n", repeated_axis0)

# Example 3: Using repeat on a 2D array along axis 1
repeated_axis1 = arr.repeat(2, axis=1)
print("Repeated along axis 1:\n", repeated_axis1)

# Example 4: Using tile on a 1D array
tiled_arr = np.tile(arr, 2)
print("Tiled array (1D repeated 2 times):\n", tiled_arr)

# Example 5: Using tile on a 2D array
tiled_2d_arr = np.tile(arr, (2, 1))
print("Tiled 2D array (repeated 2 times along axis 0):\n", tiled_2d_arr)

