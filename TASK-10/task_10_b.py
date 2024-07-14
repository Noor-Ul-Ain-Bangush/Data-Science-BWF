import pandas as pd
import numpy as np

# Creating DataFrame from the Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
iris_df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])

# Exploring DataFrame information
print("DataFrame Information:")
print(iris_df.info())

# Displaying first few rows of the DataFrame
print("\nFirst Few Rows of DataFrame:")
print(iris_df.head())

# Displaying last few rows of the DataFrame
print("\nLast Few Rows of DataFrame:")
print(iris_df.tail())

# Dealing with Missing Data
example1 = pd.Series([0, np.nan, '', None])
print("\nDetecting Null Values:")
print(example1.isnull())

# Dropping null values
example1 = example1.dropna()
print("\nDropping Null Values:")
print(example1)

# Creating a DataFrame with null values
example2 = pd.DataFrame([[1, np.nan, 7], [2, 5, 8], [np.nan, 6, 9]])
print("\nDataFrame with Null Values:")
print(example2)

# Dropping rows with any null values
print("\nDropping Rows with Null Values:")
print(example2.dropna())

# Filling null values with forward fill
print("\nFilling Null Values (Forward Fill):")
print(example2.fillna(method='ffill'))

# Removing Duplicate Data
example4 = pd.DataFrame({'letters': ['A','B'] * 2 + ['B'], 'numbers': [1, 2, 1, 3, 3]})
print("\nIdentifying Duplicates:")
print(example4.duplicated())

# Dropping duplicates
print("\nDropping Duplicates:")
print(example4.drop_duplicates())
