import pandas as pd
import numpy as np

# Handling Missing Data

# Creating a Series with missing values
string_data = pd.Series(['aardvark', 'artichoke', np.nan, 'avocado'])
# Checking for missing values
print(string_data.isnull())

# Adding a missing value represented by None
string_data[0] = None
# Checking for missing values again
print(string_data.isnull())

# DataFrame with missing values
data = pd.DataFrame([[1., 6.5, 3.], [1., np.nan, np.nan],
                     [np.nan, np.nan, np.nan], [np.nan, 6.5, 3.]])
# Dropping rows with missing values
cleaned = data.dropna()
print(cleaned)

# Dropping columns with missing values
cleaned = data.dropna(axis=1)
print(cleaned)

# Filling Missing Data

# Filling missing values with a specified value
df = pd.DataFrame(np.random.randn(7, 3))
df.iloc[:4, 1] = np.nan
df.iloc[:2, 2] = np.nan
filled = df.fillna(0)
print(filled)

# Forward filling missing values with a limit
filled = df.fillna(method='ffill', limit=2)
print(filled)

# Filling with the mean of the Series
data = pd.Series([1., np.nan, 3.5, np.nan, 7])
filled = data.fillna(data.mean())
print(filled)

# Removing Duplicates

# Creating a DataFrame with duplicate rows
data = pd.DataFrame({'k1': ['one', 'two'] * 3 + ['two'],
                     'k2': [1, 1, 2, 3, 3, 4, 4]})
# Detecting duplicates
print(data.duplicated())
# Removing duplicates
data_cleaned = data.drop_duplicates()
print(data_cleaned)

# Specifying subset of columns for detecting duplicates
data['v1'] = range(7)
print(data.drop_duplicates(['k1']))

# Keeping the last occurrence of duplicates
print(data.drop_duplicates(['k1', 'k2'], keep='last'))

# Transforming Data Using a Function or Mapping

# Creating a DataFrame with food data
data = pd.DataFrame({'food': ['bacon', 'pulled pork', 'bacon', 'Pastrami', 'corned beef',
                              'Bacon', 'pastrami', 'honey ham', 'nova lox'],
                     'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
meat_to_animal = {'bacon': 'pig', 'pulled pork': 'pig', 'pastrami': 'cow',
                  'corned beef': 'cow', 'honey ham': 'pig', 'nova lox': 'salmon'}
# Mapping to new values using a function
data['animal'] = data['food'].map(lambda x: meat_to_animal[x.lower()])
print(data)

# Replacing Values

# Creating a Series with sentinel values
data = pd.Series([1., -999., 2., -999., -1000., 3.])
# Replacing a single value
print(data.replace(-999, np.nan))
# Replacing multiple values
print(data.replace([-999, -1000], np.nan))
# Replacing with different values
print(data.replace([-999, -1000], [np.nan, 0]))
# Replacing using a dictionary
print(data.replace({-999: np.nan, -1000: 0}))

# Detecting and Filtering Outliers

# Creating a DataFrame with normally distributed data
data = pd.DataFrame(np.random.randn(1000, 4))
# Detecting outliers in a column
col = data[2]
print(col[np.abs(col) > 3])

# Capping values outside a range
data[np.abs(data) > 3] = np.sign(data) * 3
print(data.describe())

# Binning and Discretization

# Creating a DataFrame with random data
data = np.random.randn(1000)
# Discretizing data into quartiles
cats = pd.qcut(data, 4)
print(pd.value_counts(cats))

# Discretizing data with custom quantiles
cats = pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.])
print(pd.value_counts(cats))
