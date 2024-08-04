# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Example 1: K-Means on Synthetic Data
print("Example 1: K-Means Clustering on Synthetic Data")

# Generate synthetic data
X_synthetic, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Apply K-Means
kmeans_synthetic = KMeans(n_clusters=4, init='k-means++', random_state=42)
y_synthetic = kmeans_synthetic.fit_predict(X_synthetic)

# Plot the clusters
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(X_synthetic[:, 0], X_synthetic[:, 1], c=y_synthetic, s=50, cmap='viridis')
plt.scatter(kmeans_synthetic.cluster_centers_[:, 0], kmeans_synthetic.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
plt.title('Synthetic Data Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Example 2: K-Means on Iris Dataset
print("\nExample 2: K-Means Clustering on Iris Dataset")

# Load Iris dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# Apply K-Means
kmeans_iris = KMeans(n_clusters=3, init='k-means++', random_state=42)
y_iris_pred = kmeans_iris.fit_predict(X_iris)

# Plot the clusters
plt.subplot(1, 3, 2)
plt.scatter(X_iris[:, 0], X_iris[:, 1], c=y_iris_pred, s=50, cmap='viridis')
plt.scatter(kmeans_iris.cluster_centers_[:, 0], kmeans_iris.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
plt.title('Iris Data Clusters')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()

# Example 3: K-Means with Scaling on Mall Customers Dataset
print("\nExample 3: K-Means Clustering on Mall Customers Dataset with Scaling")

# Load and preprocess dataset
dataset = pd.read_csv(r'E:\Data-Science-BWF-Noor Ul Ain (Task16)\TASK-16\Mall_Customers.csv')
X_mall = dataset.iloc[:, [3, 4]].values

# Feature Scaling
scaler = StandardScaler()
X_mall_scaled = scaler.fit_transform(X_mall)

# Apply K-Means
kmeans_mall = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_mall_pred = kmeans_mall.fit_predict(X_mall_scaled)

# Plot the clusters
plt.subplot(1, 3, 3)
plt.scatter(X_mall_scaled[y_mall_pred == 0, 0], X_mall_scaled[y_mall_pred == 0, 1], s=50, c='blue', label='Cluster 1')
plt.scatter(X_mall_scaled[y_mall_pred == 1, 0], X_mall_scaled[y_mall_pred == 1, 1], s=50, c='green', label='Cluster 2')
plt.scatter(X_mall_scaled[y_mall_pred == 2, 0], X_mall_scaled[y_mall_pred == 2, 1], s=50, c='red', label='Cluster 3')
plt.scatter(X_mall_scaled[y_mall_pred == 3, 0], X_mall_scaled[y_mall_pred == 3, 1], s=50, c='cyan', label='Cluster 4')
plt.scatter(X_mall_scaled[y_mall_pred == 4, 0], X_mall_scaled[y_mall_pred == 4, 1], s=50, c='magenta', label='Cluster 5')
plt.scatter(kmeans_mall.cluster_centers_[:, 0], kmeans_mall.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Mall Customers Clusters')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.legend()

plt.tight_layout()
plt.show()
