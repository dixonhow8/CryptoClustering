# Module 11 Challenge (Crypto Clustering)

## Overview

In this challenge, I applied my knowledge of the K-means algorithm and Principal Component Analysis (PCA) to classify cryptocurrencies based on their price fluctuations over various timeframes. The goal is to analyze how price changes over different intervals can be used to group cryptocurrencies into clusters.

## Objectives

- **Apply PCA** to reduce the dimensionality of cryptocurrency price data.
- **Use the K-means algorithm** to classify cryptocurrencies into clusters based on their price fluctuations.
- **Examine price changes** over the following intervals:
  - 24 hours
  - 7 days
  - 30 days
  - 60 days
  - 200 days
  - 1 year

## Data Description

The dataset contains cryptocurrency price data across different timeframes. Each entry in the dataset includes the price fluctuations over the specified intervals.

## Steps

1. **Prepare the Data**
   - Load the cryptocurrency price data.
   - Preprocess the data, including scaling the features for PCA and K-means.

2. **Find the Best Value for k Using the Original Scaled DataFrame**
   - Determine the optimal number of clusters (`k`) for the K-means algorithm using the original scaled data.
   - Utilize techniques such as the Elbow Method or Silhouette Analysis.

3. **Cluster Cryptocurrencies with K-Means Using the Original Scaled Data**
   - Apply K-means clustering to the original scaled data using the optimal number of clusters.

4. **Optimize Clusters with Principal Component Analysis (PCA)**
   - Apply PCA to the scaled data to reduce its dimensionality while retaining important features.
   - Visualize the variance explained by each principal component.

5. **Find the Best Value for k Using the PCA Data**
   - Determine the optimal number of clusters (`k`) for the K-means algorithm using the PCA-reduced data.
   - Utilize technique such as the Elbow Method.

6. **Cluster Cryptocurrencies with K-Means Using the PCA Data**
   - Apply K-means clustering to the PCA-reduced data using the optimal number of clusters.

7. **Determine the Weights of Each Feature on Each Principal Component**
   - Analyze the principal components to understand the influence of each original feature on the principal components.

8. **Analysis**
   - Examine the clusters to understand the classification of cryptocurrencies.
   - Compare the clustering results from the original scaled data and the PCA-reduced data.

9. **Visualization**
   - Create scatter plots and other visualizations to represent the clusters and understand the relationships between cryptocurrencies.
   - Visualize the principal components and feature weights.

## Example Code

Here's a sample code snippet for applying PCA and K-means clustering:

```python
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import hvplot.pandas

# Load your data
data = pd.read_csv('path/to/your/data.csv')

# Data preprocessing
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Determine the best value for k
# ...

# Apply K-means to the original scaled data
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
clusters = kmeans.fit_predict(scaled_data)

# Apply PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Determine the best value for k using PCA data
# ...

# Fit K-means to PCA data
kmeans_pca = KMeans(n_clusters=optimal_k_pca, random_state=0)
clusters_pca = kmeans_pca.fit_predict(pca_data)

# Create DataFrames for visualization
df_pca_clusters = pd.DataFrame(pca_data, columns=['PCA1', 'PCA2'])
df_pca_clusters['cluster'] = clusters_pca

# Create a scatter plot
scatter_plot = df_pca_clusters.hvplot.scatter(
    x='PCA1',
    y='PCA2',
    c='cluster',
    cmap='Category10',
    title='Cryptocurrency Clusters (PCA Data)',
    xlabel='PCA1',
    ylabel='PCA2'
)

# Display the plot
scatter_plot

# Determine the weights of each feature on each principal component
weights = pd.DataFrame(pca.components_, columns=data.columns, index=['PC1', 'PC2'])
print(weights)
