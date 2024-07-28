# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# 1. Linear Regression for House Prices Prediction
# Generate synthetic dataset
np.random.seed(42)
size = np.random.randint(500, 3500, 1000)  # size in square feet
rooms = np.random.randint(1, 10, 1000)     # number of rooms
location = np.random.choice(['A', 'B', 'C'], 1000)  # location categories
prices = size * 300 + rooms * 5000 + np.where(location == 'A', 10000, np.where(location == 'B', 20000, 30000)) + np.random.randint(-20000, 20000, 1000)

# Create DataFrame
df = pd.DataFrame({'size': size, 'rooms': rooms, 'location': location, 'price': prices})

# One-hot encode location
df = pd.get_dummies(df, columns=['location'], drop_first=True)

# Split the data
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = lin_reg.predict(X_train_scaled)
y_test_pred = lin_reg.predict(X_test_scaled)

# Evaluate the Linear Regression model
print("Linear Regression:")
print("Train RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
print("Train R^2:", r2_score(y_train, y_train_pred))
print("Test R^2:", r2_score(y_test, y_test_pred))

# Plot predictions
plt.scatter(y_test, y_test_pred)
plt.xlabel("True Prices")
plt.ylabel("Predicted Prices")
plt.title("Linear Regression: True vs Predicted Prices")
plt.show()

# 2. Ridge Regression for Predicting Car Fuel Efficiency
# Load the Auto MPG dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']
df = pd.read_csv(url, names=column_names, sep='\s+', na_values='?')

# Handle missing values
df = df.dropna()

# Drop 'car name' column
df = df.drop('car name', axis=1)

# Split the data
X = df.drop('mpg', axis=1)
y = df['mpg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Ridge Regression model
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = ridge_reg.predict(X_train_scaled)
y_test_pred = ridge_reg.predict(X_test_scaled)

# Evaluate the Ridge Regression model
print("\nRidge Regression:")
print("Train RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
print("Train R^2:", r2_score(y_train, y_train_pred))
print("Test R^2:", r2_score(y_test, y_test_pred))

# Plot predictions
plt.scatter(y_test, y_test_pred)
plt.xlabel("True MPG")
plt.ylabel("Predicted MPG")
plt.title("Ridge Regression: True vs Predicted MPG")
plt.show()

# 3. Lasso Regression for Predicting California Housing Prices
# Load the California housing dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Split the dataset into features and target variable
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Lasso Regression model
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = lasso_reg.predict(X_train_scaled)
y_test_pred = lasso_reg.predict(X_test_scaled)

# Evaluate the Lasso Regression model
print("\nLasso Regression:")
print("Train RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
print("Train R^2:", r2_score(y_train, y_train_pred))
print("Test R^2:", r2_score(y_test, y_test_pred))

# Plot predictions
plt.scatter(y_test, y_test_pred)
plt.xlabel("True Prices")
plt.ylabel("Predicted Prices")
plt.title("Lasso Regression: True vs Predicted Prices")
plt.show()

# 4. Polynomial Regression for Predicting House Prices
# Generate synthetic dataset
np.random.seed(42)
size = np.random.randint(500, 3500, 1000)  # size in square feet
rooms = np.random.randint(1, 10, 1000)     # number of rooms
prices = size * 300 + rooms * 5000 + np.random.randint(-20000, 20000, 1000)

# Create DataFrame
df = pd.DataFrame({'size': size, 'rooms': rooms, 'price': prices})

# Split the data
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polynomial features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Standardize features
scaler = StandardScaler()
X_train_poly_scaled = scaler.fit_transform(X_train_poly)
X_test_poly_scaled = scaler.transform(X_test_poly)

# Train the Polynomial Regression model
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly_scaled, y_train)

# Make predictions
y_train_pred = poly_reg.predict(X_train_poly_scaled)
y_test_pred = poly_reg.predict(X_test_poly_scaled)

# Evaluate the Polynomial Regression model
print("\nPolynomial Regression:")
print("Train RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
print("Train R^2:", r2_score(y_train, y_train_pred))
print("Test R^2:", r2_score(y_test, y_test_pred))

# Plot predictions
plt.scatter(y_test, y_test_pred)
plt.xlabel("True Prices")
plt.ylabel("Predicted Prices")
plt.title("Polynomial Regression: True vs Predicted Prices")
plt.show()

# 5. Elastic Net for Predicting Wine Quality
# Load the Wine Quality dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')

# Split the data
X = df.drop('quality', axis=1)
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Elastic Net model
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = elastic_net.predict(X_train_scaled)
y_test_pred = elastic_net.predict(X_test_scaled)

# Evaluate the Elastic Net model
print("\nElastic Net:")
print("Train RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
print("Train R^2:", r2_score(y_train, y_train_pred))
print("Test R^2:", r2_score(y_test, y_test_pred))

# Plot predictions
plt.scatter(y_test, y_test_pred)
plt.xlabel("True Quality")
plt.ylabel("Predicted Quality")
plt.title("Elastic Net: True vs Predicted Quality")
plt.show()
