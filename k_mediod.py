# -*- coding: utf-8 -*-
"""
K-Medoids Clustering Example
This script performs K-Medoids clustering on the Wholesale Customers dataset from UCI,
compares it with K-Means, and visualizes the clusters using PCA.
"""
# -------------------------------------------------------------understanding k-medoid Before diving into the
# code---------------------------------#

# it’s important to understand what K-Medoids is. K-Medoids is a clustering algorithm, very similar to K-Means,
# but with an important distinction: instead of using the mean of the points as the cluster center (like K-Means),
# K-Medoids chooses an actual data point as the center (called a medoid). A cluster is a group of similar items. For
# example, in a retail store, one cluster might be customers who spend heavily on Fresh products, and another cluster
# might be customers who mostly spend on Grocery and Detergents. A point represents an individual entity with
# features. In your dataset, a point is a customer, represented by six numeric features: Fresh, Milk, Grocery,
# Frozen, Detergents_Paper, and Delicassen. A medoid is like a “representative customer” of the cluster, the one who
# is closest to all other points in that cluster, minimizing total distance. K-Medoids works by repeatedly selecting
# medoids, assigning points to the nearest medoid, and updating medoids to reduce the sum of distances within the
# clusters. The goal is that every customer is assigned to a cluster whose medoid is closest to them, so we can group
# similar customers together.

# -------------------------------------------------Real world example to understand k-mediod------------------------
# Imagine you are a
# manager at a wholesale store and want to identify types of customers. You pick a few customers randomly as
# “prototypes” (medoids) and ask, “which of my customers are most similar to this prototype?” Then, you re-evaluate:
# “Is there a better customer that could represent this group more accurately?” You keep iterating until the groups#
# are stable.
# --------------------------------------------------------------------------------------------------------------------


# ---------------------------- ------------------------------IMPORT LIBRARIES ---------------------------- numpy →
# used for numerical operations and creating arrays of cluster labels. pandas → for reading CSV data and managing
# tabular datasets. matplotlib.pyplot → for plotting cluster visualizations. pyclustering.cluster.kmedoids.kmedoids →
# the actual K-Medoids implementation. sklearn.cluster.KMeans → used to compare K-Medoids with K-Means, showing which
# algorithm better clusters customers. sklearn.metrics.silhouette_score → evaluates how well the clusters are
# separated. sklearn.preprocessing.MinMaxScaler → normalizes features to the same scale (0–1) to avoid bias in
# distance calculations. sklearn.decomposition.PCA → reduces high-dimensional data (6 features) to 2D for visualization.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyclustering.cluster.kmedoids import kmedoids
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# ---------------------------- LOAD DATA -----------------------------------
# Load dataset directly from UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
df = pd.read_csv(url)

# Display first 5 rows to check the data
print("First 5 rows of the dataset:")
print(df.head())
# D:\checkdata\Scripts\python.exe D:\django\checkdata\check.py
# First 5 rows of the dataset:
#    Channel  Region  Fresh  Milk  Grocery  Frozen  Detergents_Paper  Delicassen
# 0        2       3  12669  9656     7561     214              2674        1338
# 1        2       3   7057  9810     9568    1762              3293        1776
# 2        2       3   6353  8808     7684    2405              3516        7844
# 3        1       3  13265  1196     4221    6404               507        1788
# 4        2       3  22615  5410     7198    3915              1777        5185


# -------------------------- DATA PREPARATION ------------------------------
# Select numeric columns for clustering
numeric_columns = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
df_numeric = df[numeric_columns]

# ---------------------------- NORMALIZATION -------------------------------
# Min-Max scaling: bring all numeric features to range [0,1]
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df_numeric)

print("\nNormalized data (first 5 rows):")
print(normalized_data[:5])

# ------------------------- K-MEDOIDS CLUSTERING ---------------------------
# Step 1: Initialize medoids (choose indices from dataset)
# Let's choose 3 medoids for k=3 clustering
initial_medoids = [0, 10, 20]

# Step 2: Create K-Medoids instance
kmedoid_instance = kmedoids(normalized_data, initial_medoids)

# Step 3: Run clustering
# backend working of .process()
# .process()?  Assign each customer to the nearest medoid (based on Euclidean distance).
# For each cluster, check if another customer in the cluster can become a better medoid by reducing total distance.
# Update medoids if better ones are found. Repeat until medoids stabilize (no change).
##Real-world example
#You are choosing three “representative customers” and asking, “which other customers are
# closest to me?” Then you check, “Is there a better representative?” until your clusters make the most sense.

# Iteratively update medoids and assign points
kmedoid_instance.process()

# Step 4: Get final clusters and medoids

# clusters → list of lists, each sublist contains indices of customers in that cluster. group of customers in each
# cluster
clusters = kmedoid_instance.get_clusters()  # List of clusters (indices of points)
#those representatives around which other customer form cluster
final_medoids = kmedoid_instance.get_medoids()  # Indices of medoids
print("\nFinal medoids indices:", final_medoids)

# ------------------------- SILHOUETTE SCORE -------------------------------
# Convert clusters into a flat label array for silhouette calculation

#labels for each customer row, define an array[], inside that arraw, clusters will be assigned to that customer
labels = np.zeros(len(normalized_data))
# Imagine clusters as a collection of groups of customers. K-Medoids, after it runs, doesn’t just label every
# customer directly. Instead, it tells you, “Here is Cluster 0, and it contains these customers, here is Cluster 1
# with these customers, and here is Cluster 2 with these customers.” Each cluster is actually a list of indices,
# each index pointing to a row in your dataset. So, if customer number 0, customer number 3, and customer number 5
# belong together, K-Medoids gives you [0, 3, 5] as the first cluster. That’s what clusters[0] would be. Similarly,
# other clusters are lists of indices too.
#
# Now, enumerate(clusters) is a Python function that does two things simultaneously. It gives you both the position
# of the cluster in the list (cluster_id), and the content of that cluster (cluster). The position here is crucial
# because it is exactly what we want to use as the “label” for that cluster. Cluster 0 will have label 0,
# cluster 1 will have label 1, and so on. This is how we convert a collection of index lists into a flat array of
# labels for every customer.
#
# Inside this loop, we have another loop: for index in cluster. Here, cluster is the list of customer indices,
# so we are iterating over each customer in that cluster. Suppose the cluster contains [0, 3, 5]. This inner loop
# will first take index = 0, then index = 3, and finally index = 5. For each customer index, we are telling the
# labels array: “This customer belongs to cluster number cluster_id.” So, labels[index] = cluster_id is assigning the
# cluster label to the correct position in the labels array. Before this, all values in labels were zero; after this,
# every position in the array reflects the cluster assignment for that particular customer.
#
# Think of it in real-world terms. Imagine you are a store manager with 440 customers, and after running K-Medoids,
# you have three groups: high spenders, moderate spenders, and low spenders. K-Medoids tells you “these customer IDs
# go into high spenders, these go into moderate, these go into low.” But to calculate metrics like silhouette score,
# you need a list where every customer knows its group. So, labels is like giving each customer a badge with their
# group number. Cluster 0 customers wear badge 0, Cluster 1 customers wear badge 1, and Cluster 2 customers wear
# badge 2.
#
# This loop does nothing more than distributing the cluster labels to the customers so that the next function,
# silhouette_score, can work. The silhouette function doesn’t care about lists of indices; it only understands,
# for customer 0, customer 1, etc., which group they belong to. So this loop is essentially a translator, converting
# K-Medoids’ output into a format that can be analyzed numerically. After this loop runs, you have an array of length
# equal to the number of customers, where each value is the cluster ID for that customer, exactly what silhouette
# score needs to calculate the cohesion and separation of clusters.
#
# In simpler terms, without this loop, the silhouette score would have no way of knowing which customers belong
# together. This loop is the bridge between the raw clustering output (lists of indices) and a usable format for
# evaluation (flat label array). It’s not doing any mathematics itself; it’s preparing the data structure so the next
# step, the silhouette calculation, can actually measure how well the clustering performed.

# cluster_id = index number of cluster in clusters list, cluster contains the acutal content inside the cluster
# eg:cluster0 =(1,4,5) cluster0 has customer 1,4,5 these are row number in customers
for cluster_id, cluster in enumerate(clusters):
    # here we go inside the cluster (2,4,9), from inside we are getting the index number of customers. customers here
    # are 2,4,9 these are row number and the index number will be for them will be 0, 1, 2
    for index in cluster:
        # label those index number with the cluster id means assign each customer insdie the cluster with the cluster
        # id which tells each customer insdie the cluster belongs to which cluster the labels tells us which customer
        # belongss to which group
        labels[index] = cluster_id

# K-Medoids silhouette score The first line is where the actual evaluation of your clusters happens. We already
# created the labels array in the previous step, which tells for every customer which cluster they belong to. The
# normalized_data contains the features for each customer — their spending on Fresh products, Milk, Grocery, Frozen,
# Detergents_Paper, and Delicassen, all scaled between 0 and 1. So, every customer is a point in a 6-dimensional
# space, and the labels array tells the function which cluster each point belongs to.
#
# The function silhouette_score() comes from scikit-learn and is specifically designed to measure how well-separated
# your clusters are, and how tightly packed the points inside each cluster are. Internally, it does something very
# logical: for every single customer (or point), it calculates two key things. The first is called a, which is the
# average distance of that point to all other points in its own cluster. Imagine a customer is in the “high spender”
# group: a tells us how close this customer is to the other high spenders. If this value is small, it means the
# customer is very similar to others in the same group — the cluster is cohesive.
#
# The second value is called b, which is the smallest average distance from this point to all points in any other
# cluster. This is effectively the distance to the “nearest competing group.” For example, if the point is a high
# spender, b measures how far this customer is from the moderate spenders or low spenders. A large b means the
# customer is far from other clusters, which is desirable, because it indicates clear separation between clusters.
#
# Once it has a and b for a point, the silhouette score for that point is computed as (b - a) / max(a,
# b). This formula has a very logical property: if an is much smaller than b, meaning the point is close to its own
# cluster and far from other clusters, the score is close to 1. If a and b are similar, meaning the point is equally
# close to its own cluster and others, the score is near 0, indicating the point is on the boundary. If an is larger
# than b, meaning the point is actually closer to another cluster than its own, the score is negative, signaling a
# misclassified point.
#
# The silhouette_score() function then calculates this value for every customer in the dataset and takes the average
# to produce a single number: kmedoid_silhouette. This single number is the summary metric of cluster quality. In our
# code, it gets stored in the variable kmedoid_silhouette. For example, if it prints 0.42, it tells us that,
# on average, customers are fairly well-clustered: they are closer to their own group than to other groups,
# but there is still some overlap. A value close to 1 would mean extremely tight clusters, and a value near 0 or
# negative would indicate poor clustering.
#
# Finally, the print() statement simply outputs this number to the console. It is not a graph or plot — it’s a
# quantitative measure that allows you to compare clustering algorithms. Later, you can compare the K-Medoids
# silhouette score with the K-Means silhouette score to see which algorithm grouped the customers more meaningfully.
#
# Real-world analogy: Imagine you are the store manager, and you grouped your customers into “high spenders,
# ” “moderate spenders,” and “low spenders.” The silhouette score is like a report card for your grouping. If the
# score is high, it means the customers in each group truly behave similarly and are far from other types. If the
# score is low, it’s like saying, “Some high spenders are behaving just like moderate spenders — maybe I need to
# rethink how I grouped them.”

kmedoid_silhouette = silhouette_score(normalized_data, labels)
print("K-Medoid Silhouette Score:", kmedoid_silhouette)

# ---------------------- COMPARISON WITH K-MEANS ---------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(normalized_data)
kmeans_silhouette = silhouette_score(normalized_data, kmeans_labels)
print("K-Means Silhouette Score:", kmeans_silhouette)

# ----------------------- PCA VISUALIZATION --------------------------------
# Reduce to 2 dimensions for plotting
pca = PCA(n_components=2)
pca_data = pca.fit_transform(normalized_data)

# Plot K-Medoids clusters
plt.figure(figsize=(10, 6))
for cluster_id, cluster in enumerate(clusters):
    plt.scatter(pca_data[cluster, 0], pca_data[cluster, 1], label=f'Cluster {cluster_id}')
# Plot medoids
plt.scatter(pca_data[final_medoids, 0], pca_data[final_medoids, 1],
            color='red', marker='X', s=200, label='Medoids')
plt.title('K-Medoids Clustering (k=3) - Wholesale Customers')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()

# Overlay K-Means clusters for comparison
plt.figure(figsize=(10, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=kmeans_labels, cmap='coolwarm', alpha=0.5)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            color='blue', marker='X', s=200, label='K-Means Centroids')
plt.title('K-Means Clustering (k=3) - Wholesale Customers')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()
