import pandas as pd
import numpy as np

# Handling Missing Data

# Creating a DataFrame with missing values
employee_data = pd.DataFrame({
    'EmployeeID': [101, 102, 103, 104, 105],
    'Name': ['Alice', 'Bob', 'Charlie', np.nan, 'Eve'],
    'Department': ['HR', 'Engineering', 'HR', 'Engineering', np.nan],
    'Salary': [70000, 80000, np.nan, 95000, 75000]
})

# Dropping rows with any missing values
print(employee_data.dropna())

# Dropping columns with any missing values
print(employee_data.dropna(axis=1))

# Filling Missing Data

# Creating a DataFrame with NaN values
sales_data = pd.DataFrame({
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
    'ProductA': [50, np.nan, 40, 45, np.nan],
    'ProductB': [30, 35, np.nan, np.nan, 25]
})

# Filling NaN values with a constant
print(sales_data.fillna(0))

# Filling NaN values with a method (backfill)
print(sales_data.fillna(method='bfill'))

# Filling NaN values in a specific column with the mean of that column
sales_data['ProductA'] = sales_data['ProductA'].fillna(sales_data['ProductA'].mean())
print(sales_data)

# Removing Duplicates

# Creating a DataFrame with duplicate rows
order_data = pd.DataFrame({
    'OrderID': [2001, 2002, 2001, 2003, 2003],
    'Customer': ['Alice', 'Bob', 'Alice', 'Charlie', 'Charlie'],
    'Amount': [250, 400, 250, 300, 300]
})

# Detecting duplicates based on specific columns
print(order_data.duplicated(['OrderID']))

# Removing duplicates based on specific columns
print(order_data.drop_duplicates(['OrderID']))

# Detecting and Filtering Outliers

# Creating a DataFrame with normally distributed data
transaction_data = pd.DataFrame({
    'TransactionID': range(1, 101),
    'Amount': np.random.randn(100) * 100 + 1000
})

# Detecting outliers in the 'Amount' column
amount_col = transaction_data['Amount']
print(amount_col[np.abs(amount_col - amount_col.mean()) > (3 * amount_col.std())])

# Capping outliers to a specified range
transaction_data_capped = transaction_data.copy()
transaction_data_capped['Amount'] = np.where(
    np.abs(transaction_data_capped['Amount'] - amount_col.mean()) > (3 * amount_col.std()),
    np.sign(transaction_data_capped['Amount'] - amount_col.mean()) * 3 * amount_col.std() + amount_col.mean(),
    transaction_data_capped['Amount']
)
print(transaction_data_capped.describe())

# Replacing Values

# Creating a Series with sentinel values
temperature_data = pd.Series([25.0, -999.0, 22.5, -999.0, -888.0, 30.0])

# Replacing a single value
print(temperature_data.replace(-999, np.nan))

# Replacing multiple values with different values
print(temperature_data.replace([-999, -888], [np.nan, 0]))

# Replacing using a dictionary
print(temperature_data.replace({-999: np.nan, -888: 0}))

# Binning and Discretization

# Creating a Series with continuous values representing ages
age_data = np.random.randint(18, 70, size=20)

# Discretizing data into bins
age_bins = [0, 20, 40, 60, 80]
age_categories = pd.cut(age_data, age_bins)
print(pd.value_counts(age_categories))

# Discretizing data into equal-sized bins
age_categories = pd.qcut(age_data, 4)
print(pd.value_counts(age_categories))

# Custom Binning
age_bins = [-np.inf, 25, 45, np.inf]
age_labels = ["Young", "Middle-aged", "Senior"]
age_categories = pd.cut(age_data, bins=age_bins, labels=age_labels)
print(pd.value_counts(age_categories))
