import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np

# Load the customer purchase history data
data = pd.read_csv('Mall_Customers.csv')

# Select the features to use for clustering (e.g. age, income, spending score)
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# Scale the features using StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])

# Determine the optimal number of clusters using the elbow method
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Choose the optimal number of clusters (e.g. 5)
n_clusters = 5

# Perform K-means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(data_scaled)

# Get the cluster labels for each customer
labels = kmeans.labels_

# Add the cluster labels to the original data
data['Cluster'] = labels

# Analyze the clusters
print("Cluster sizes:")
print(data['Cluster'].value_counts())

print("\nCluster means:")
print(data.groupby('Cluster')[features].mean())

print("\nCluster stds:")
print(data.groupby('Cluster')[features].std())

# Plot the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Annual Income (k$)', hue='Cluster', data=data)
plt.title('Clusters')
plt.show()

# Plot the clusters in 3D
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['Age'], data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['Cluster'])
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
plt.title('Clusters in 3D')
plt.show()

# Plot the distribution of each feature within each cluster
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster', y=feature, data=data)
    plt.title(f'Distribution of {feature} within each Cluster')
    plt.show()