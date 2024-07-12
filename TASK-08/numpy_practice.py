#   <<<-------Numpy Library--------->>>

import numpy as np

#       <<.........EXP-1: Basic Array Creation & Operations...........>>

# Creating a 1D array
arr = np.array([1, 2, 3, 4, 5])
print("Array:", arr)

# Adding a scalar to the array
arr_plus_2 = arr + 2
print("Array + 2:", arr_plus_2)

# Multiplying the array by a scalar
arr_times_3 = arr * 3
print("Array * 3:", arr_times_3)

# Summing all elements in the array
sum_of_elements = np.sum(arr)
print("Sum of elements:", sum_of_elements)

# Finding the mean of the array
mean_of_elements = np.mean(arr)
print("Mean of elements:", mean_of_elements)


#       <<.........EXP-2: Element-wise Operations...........>>

# Creating two 1D arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([5, 4, 3, 2, 1])

# Element-wise addition
addition = arr1 + arr2
print("Addition:", addition)

# Element-wise multiplication
multiplication = arr1 * arr2
print("Multiplication:", multiplication)

# Element-wise comparison
comparison = arr1 > arr2
print("Comparison (arr1 > arr2):", comparison)


#       <<.........EXP-3: Reshaping & Slicing...........>>

# Creating a 1D array
arr = np.array([1, 2, 3, 4, 5, 6])

# Reshaping to a 2x3 array
reshaped_arr = arr.reshape(2, 3)
print("Reshaped Array (2x3):\n", reshaped_arr)

# Slicing the reshaped array
sliced_arr = reshaped_arr[0, 1:3]
print("Sliced Array (first row, columns 1 and 2):", sliced_arr)


#       <<<<<<...............................END...........................................>>>>>>