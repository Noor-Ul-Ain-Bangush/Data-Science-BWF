import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# Example 1: KNN Classification with the Iris Dataset
print("Example 1: KNN Classification")

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the KNN classifier with K=3
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = knn_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Example 2: KNN Classification with Visualization
print("\nExample 2: KNN Classification with Visualization")

# Generate synthetic data
from sklearn.datasets import make_classification

# Create synthetic data with a valid configuration
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the KNN classifier with K=5
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# Plot decision boundaries
h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = knn_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', marker='o')
plt.title("KNN Classification Decision Boundaries (K=5)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Example 3: KNN Regression with Synthetic Data
print("\nExample 3: KNN Regression")

# Generate synthetic data for regression
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the KNN regressor with K=5
knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_train, y_train)

# Predict on the test set
y_pred = knn_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Plot regression results
plt.scatter(X_test, y_test, color='blue', label='Actual data')
plt.scatter(X_test, y_pred, color='red', label='Predicted data')
plt.title("KNN Regression Results (K=5)")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.show()

# Example 4: KNN Imputation
print("\nExample 4: KNN Imputation")

# Generate a dataset with missing values
X = np.array([[1, 2, np.nan], [3, np.nan, 2], [np.nan, 2, 3], [1, 1, 1], [2, 2, 2]])
print("Original Data with Missing Values:")
print(X)

# Initialize the KNN imputer
imputer = KNNImputer(n_neighbors=2)
X_imputed = imputer.fit_transform(X)

print("\nImputed Data:")
print(X_imputed)
