# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import warnings

# Suppress warnings
warnings.simplefilter(action='ignore')

# Load data
iris = pd.read_csv(r"E:\Data-Science-BWF-Noor Ul Ain (Task18)/Iris.csv")
x = iris.iloc[:, [1, 2, 3, 4]].values

# Display the first few rows of the dataset
print(iris.head().style.background_gradient(cmap=sns.cubehelix_palette(as_cmap=True)))

# Dataset info
print(iris.info())

# Checking simple statistical parameters
print(iris.describe(include='all'))

# Setting training data
X = iris.iloc[:, 1:-1].values
y = iris.iloc[:, -1].values

# EDA
# Checking the number of rows and columns in the dataset
rows, col = x.shape
print('Row:', rows, '\nColumns:', col)

# Number of null values
print(iris.isnull().sum())

# Number of unique elements in each column
print(iris.nunique())

# Scatter plots using Plotly Express
fig = px.scatter(data_frame=iris, x='SepalLengthCm', y='SepalWidthCm', color='Species')
fig.update_layout(width=800, height=600, xaxis=dict(title='Sepal Length', color="#BF40BF"), yaxis=dict(title="Sepal Width", color="#BF40BF"))
fig.show()

fig = px.scatter(data_frame=iris, x='SepalWidthCm', y='PetalLengthCm', color='Species')
fig.update_layout(width=800, height=600, xaxis=dict(title='Sepal Width', color="#BF40BF"), yaxis=dict(title="Petal Length", color="#BF40BF"))
fig.show()

fig = px.scatter(data_frame=iris, x='PetalLengthCm', y='PetalWidthCm', color='Species')
fig.update_layout(width=800, height=600, xaxis=dict(title='Petal Length', color="#BF40BF"), yaxis=dict(title="Petal Width", color="#BF40BF"))
fig.show()

fig = px.scatter(data_frame=iris, x='PetalWidthCm', y='SepalLengthCm', color='Species')
fig.update_layout(width=800, height=600, xaxis=dict(title='Petal Width', color="#BF40BF"), yaxis=dict(title="Sepal Length", color="#BF40BF"))
fig.show()

# K-MEANS Clustering
kmeans_set = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}

scaler = StandardScaler()
scaled_features = scaler.fit_transform(x)

# Finding the optimal number of clusters using the Elbow method
List = []
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k, **kmeans_set)
    kmeans.fit(scaled_features)
    List.append(kmeans.inertia_)

plt.style.use("fivethirtyeight")
plt.plot(range(1, 20), List)
plt.xticks(range(1, 20))
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Silhouette coefficients for different values of k
silhouette_coefficients = []
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k, **kmeans_set)
    kmeans.fit(scaled_features)
    score = silhouette_score(scaled_features, kmeans.labels_)
    silhouette_coefficients.append(score)

plt.style.use("fivethirtyeight")
plt.plot(range(2, 20), silhouette_coefficients)
plt.xticks(range(2, 20))
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Coefficients')
plt.show()

# Applying KMeans clustering with optimal number of clusters
kmeans = KMeans(n_clusters=3, **kmeans_set)
y_kmeans = kmeans.fit_predict(x)

centroids = kmeans.cluster_centers_
print(centroids)

# Visualizing the clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=100, c='purple', label='Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=100, c='orange', label='Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50, label='Centroids')
plt.legend()
plt.show()

# 3D scatterplot using matplotlib
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], x[y_kmeans == 0, 2], s=100, c='purple', label='Cluster 1')
ax.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], x[y_kmeans == 1, 2], s=100, c='orange', label='Cluster 2')
ax.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], x[y_kmeans == 2, 2], s=100, c='green', label='Cluster 3')
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=100, c='red', label='Centroids')
plt.legend()
plt.show()

# Adding cluster labels to the dataset
iris["cluster_no"] = kmeans.labels_
print(iris.head(12))

# Display the last few rows of the dataset
print(iris.tail())

# Scatter plot using Plotly for visualizing clusters
fig = go.Figure()
fig.add_trace(go.Scatter(x=X[y_kmeans == 0, 0], y=X[y_kmeans == 0, 1], mode='markers', marker_color='#DB4CB2', name='Cluster 1'))
fig.add_trace(go.Scatter(x=X[y_kmeans == 1, 0], y=X[y_kmeans == 1, 1], mode='markers', marker_color='#c9e9f6', name='Cluster 2'))
fig.add_trace(go.Scatter(x=X[y_kmeans == 2, 0], y=X[y_kmeans == 2, 1], mode='markers', marker_color='#7D3AC1', name='Cluster 3'))
fig.add_trace(go.Scatter(x=centroids[:, 0], y=centroids[:, 1], mode='markers', marker_color='#CAC9CD', marker_symbol=4, marker_size=13, name='Centroids'))
fig.update_layout(template='plotly_dark', width=1000, height=500)
fig.show()

# Hierarchical clustering
dendogram = dendrogram(linkage(x, method="ward"))
plt.show()

# Dendogram for hierarchical clustering
plt.figure(figsize=(10, 10))
plt.title("Hierarchical Clustering Dendogram")
plt.xlabel("Observation Units")
plt.ylabel("Distances")
dendrogram(linkage(x, "complete"), leaf_font_size=10, truncate_mode="lastp", p=10, show_contracted=True)
plt.show()
