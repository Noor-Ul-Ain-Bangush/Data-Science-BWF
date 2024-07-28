# Create a 2D array
import numpy as np
arr = np.arange(15).reshape((5, 3))
print("Original 2D array:\n", arr)

# Ravel the array
raveled_arr = arr.ravel()
print("Raveled array:", raveled_arr)

# Flatten the array
flattened_arr = arr.flatten()
print("Flattened array:", flattened_arr)


#Exanple-02
# Example 1: Using ravel
arr = np.array([[1, 2, 3], [4, 5, 6]])
raveled_arr = arr.ravel()
print("Raveled array:", raveled_arr)

# Example 2: Using flatten
flattened_arr = arr.flatten()
print("Flattened array:", flattened_arr)

# Difference between ravel and flatten (flatten returns a copy)
arr[0, 0] = 99
print("Modified original array:\n", arr)
print("Raveled array after modification:", raveled_arr)
print("Flattened array after modification:", flattened_arr)

