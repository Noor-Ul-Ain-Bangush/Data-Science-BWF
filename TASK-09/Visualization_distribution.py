# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the birds dataset
birds = pd.read_csv(r'E:/Data-Science-BWF-Noor Ul Ain (Task9)/pythonProject/minnesota_birds.csv')

# Display column names to understand the correct names
print(birds.columns)

# Sample first few rows to confirm data structure
print(birds.head())

# Check for missing values
print(birds.isna().sum())

# Drop rows with missing values in relevant columns
birds = birds.dropna(subset=['Body Mass Max (g)', 'Wingspan Max (cm)', 'Order'])

# Scatter plot: Max Wingspan per Order
plt.figure(figsize=(12, 8))
birds.plot(kind='scatter', x='Wingspan Max (cm)', y='Order', figsize=(12, 8))
plt.title('Max Wingspan per Order')
plt.ylabel('Order')
plt.xlabel('Max Wingspan (cm)')
plt.show()

# Histogram: Max Body Mass distribution
plt.figure(figsize=(12, 12))
birds['Body Mass Max (g)'].plot(kind='hist', bins=10)
plt.title('Distribution of Max Body Mass (bins=10)')
plt.xlabel('Max Body Mass (g)')
plt.ylabel('Frequency')
plt.show()

# Histogram with more bins
plt.figure(figsize=(12, 12))
birds['Body Mass Max (g)'].plot(kind='hist', bins=30)
plt.title('Distribution of Max Body Mass (bins=30)')
plt.xlabel('Max Body Mass (g)')
plt.ylabel('Frequency')
plt.show()

# Filter data: Max Body Mass under 60
filteredBirds = birds[(birds['Body Mass Max (g)'] > 1) & (birds['Body Mass Max (g)'] < 60)]

# Histogram for filtered data
plt.figure(figsize=(12, 12))
filteredBirds['Body Mass Max (g)'].plot(kind='hist', bins=40)
plt.title('Filtered Distribution of Max Body Mass (bins=40)')
plt.xlabel('Max Body Mass (g)')
plt.ylabel('Frequency')
plt.show()

# 2D Histogram: Max Body Mass vs. Max Wingspan
x = filteredBirds['Body Mass Max (g)']
y = filteredBirds['Wingspan Max (cm)']
plt.figure(figsize=(12, 12))
plt.hist2d(x, y, bins=30, cmap='Blues')
plt.colorbar()
plt.title('2D Histogram: Max Body Mass vs. Max Wingspan')
plt.xlabel('Max Body Mass (g)')
plt.ylabel('Max Wingspan (cm)')
plt.show()

# Histogram: Conservation Status vs. Min Wingspan
x1 = filteredBirds.loc[filteredBirds['Conservation status']=='EX', 'Wingspan Min (cm)']
x2 = filteredBirds.loc[filteredBirds['Conservation status']=='CR', 'Wingspan Min (cm)']
x3 = filteredBirds.loc[filteredBirds['Conservation status']=='EN', 'Wingspan Min (cm)']
x4 = filteredBirds.loc[filteredBirds['Conservation status']=='NT', 'Wingspan Min (cm)']
x5 = filteredBirds.loc[filteredBirds['Conservation status']=='VU', 'Wingspan Min (cm)']
x6 = filteredBirds.loc[filteredBirds['Conservation status']=='LC', 'Wingspan Min (cm)']

kwargs = dict(alpha=0.5, bins=20)

plt.figure(figsize=(12, 8))
plt.hist(x1, **kwargs, color='red', label='Extinct')
plt.hist(x2, **kwargs, color='orange', label='Critically Endangered')
plt.hist(x3, **kwargs, color='yellow', label='Endangered')
plt.hist(x4, **kwargs, color='green', label='Near Threatened')
plt.hist(x5, **kwargs, color='blue', label='Vulnerable')
plt.hist(x6, **kwargs, color='gray', label='Least Concern')

plt.title('Conservation Status vs. Min Wingspan')
plt.xlabel('Min Wingspan (cm)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Density plot: Min Wingspan
plt.figure(figsize=(12, 8))
sns.kdeplot(filteredBirds['Wingspan Min (cm)'], bw_adjust=0.5)
plt.title('Density Plot: Min Wingspan')
plt.xlabel('Min Wingspan (cm)')
plt.ylabel('Density')
plt.show()

# Density plot: Max Body Mass
plt.figure(figsize=(12, 8))
sns.kdeplot(filteredBirds['Body Mass Max (g)'], bw_adjust=0.5)
plt.title('Density Plot: Max Body Mass')
plt.xlabel('Max Body Mass (g)')
plt.ylabel('Density')
plt.show()

# Density plot: Max Body Mass per Order
plt.figure(figsize=(12, 8))
sns.kdeplot(data=filteredBirds, x='Body Mass Max (g)', hue='Order', fill=True, common_norm=False, palette='crest', alpha=0.5, linewidth=0)
plt.title('Density Plot: Max Body Mass per Order')
plt.xlabel('Max Body Mass (g)')
plt.ylabel('Density')
plt.show()

# Density plot: Min Length vs. Max Length by Conservation Status
plt.figure(figsize=(12, 8))
sns.kdeplot(data=filteredBirds, x='Length Min (cm)', y='Length Max (cm)', hue='Conservation status')
plt.title('Density Plot: Min Length vs. Max Length by Conservation Status')
plt.xlabel('Min Length (cm)')
plt.ylabel('Max Length (cm)')
plt.show()
