import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Generating a Date Range for Time Series
start_date = "2020-01-01"
end_date = "2020-03-31"
idx = pd.date_range(start_date, end_date)

# Creating a Series with Random Sales Data
np.random.seed(0)  # For reproducibility
items_sold = pd.Series(np.random.randint(25, 50, size=len(idx)), index=idx)
print("Items Sold Series:")
print(items_sold.head())

# Plotting the Time Series Data
items_sold.plot(title='Daily Ice-Cream Sales', ylabel='Items Sold')
plt.show()

# Adding Weekly Additional Sales Data
additional_items = pd.Series(10, index=pd.date_range(start_date, end_date, freq="W"))
total_items = items_sold.add(additional_items, fill_value=0)
print("Total Items Series with Additional Weekly Sales:")
print(total_items.head())

# Plotting the Total Sales Data
total_items.plot(title='Total Daily Ice-Cream Sales with Weekly Additions', ylabel='Total Items Sold')
plt.show()

# Resampling to Compute Monthly Averages
monthly = total_items.resample("M").mean()
print("Monthly Average Sales:")
print(monthly)

# Plotting Monthly Average Sales Data
monthly.plot(kind='bar', title='Monthly Average Ice-Cream Sales', ylabel='Average Items Sold')
plt.show()

# Creating a DataFrame
a = pd.Series(range(1, 10), name='A')
b = pd.Series(["I", "like", "to", "play", "games", "and", "will", "not", "change"], name='B')
df = pd.DataFrame({'A': a, 'B': b})
print("DataFrame Example:")
print(df)

# Adding New Columns to DataFrame
df['DivA'] = df['A'] - df['A'].mean()
df['LenB'] = df['B'].apply(len)
print("DataFrame with New Columns:")
print(df)

# Filtering Rows
filtered_df = df[df['A'] > 5]
print("Filtered DataFrame (A > 5):")
print(filtered_df)

# Grouping and Aggregating Data
grouped_df = df.groupby(by='LenB').aggregate({'DivA': 'count', 'A': 'mean'}).rename(columns={'DivA': 'Count', 'A': 'Mean'})
print("Grouped and Aggregated DataFrame:")
print(grouped_df)

# Loading Data from CSV
df_csv = pd.read_csv(r'E:\Data-Science-BWF-Noor Ul Ain (Task10)\minnesota_birds.csv')
print("DataFrame Loaded from CSV:")
print(df_csv.head())

# Example: Generating Normal Distribution Data and Plotting
mean, std_dev = 0, 0.1  # mean and standard deviation
s = np.random.normal(mean, std_dev, 1000)

# Plotting the histogram of the data
plt.hist(s, bins=30, density=True)

# Plotting the probability density function
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mean, std_dev)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mean = %.2f,  std_dev = %.2f" % (mean, std_dev)
plt.title(title)

plt.show()
