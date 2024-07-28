# Create a 2D array
import numpy as np
arr = np.random.randn(5, 2)
print("Original array:\n", arr)

# Split the array into three parts along the first axis
first, second, third = np.split(arr, [1, 3])
print("First part:\n", first)
print("Second part:\n", second)
print("Third part:\n", third)



#Example-02
# Example 1: Splitting a 1D array
arr = np.arange(10)
split_arr = np.split(arr, [3, 6])
print("Splitting a 1D array at indices 3 and 6:", split_arr)

# Example 2: hsplit a 2D array
arr = np.arange(16).reshape((4, 4))
hsplit_arr = np.hsplit(arr, 2)
print("Horizontally split 2D array into 2 parts:\n", hsplit_arr[0], "\n", hsplit_arr[1])

# Example 3: vsplit a 2D array
vsplit_arr = np.vsplit(arr, 2)
print("Vertically split 2D array into 2 parts:\n", vsplit_arr[0], "\n", vsplit_arr[1])

