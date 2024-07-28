# Create two 2D arrays
import numpy as np
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8, 9], [10, 11, 12]])

# Concatenate along the first axis (rows)
concat_axis0 = np.concatenate([arr1, arr2], axis=0)
print("Concatenated along axis 0:\n", concat_axis0)

# Concatenate along the second axis (columns)
concat_axis1 = np.concatenate([arr1, arr2], axis=1)
print("Concatenated along axis 1:\n", concat_axis1)

# Using vstack and hstack
vstacked = np.vstack((arr1, arr2))
print("Vstacked arrays:\n", vstacked)

hstacked = np.hstack((arr1, arr2))
print("Hstacked arrays:\n", hstacked)


#Example-02
# Example 1: Concatenate 1D arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
concatenated = np.concatenate([arr1, arr2])
print("Concatenated 1D arrays:", concatenated)

# Example 2: Concatenate 3D arrays along different axes
arr1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
arr2 = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
concat_axis0 = np.concatenate([arr1, arr2], axis=0)
concat_axis1 = np.concatenate([arr1, arr2], axis=1)
concat_axis2 = np.concatenate([arr1, arr2], axis=2)
print("Concatenated along axis 0:\n", concat_axis0)
print("Concatenated along axis 1:\n", concat_axis1)
print("Concatenated along axis 2:\n", concat_axis2)


