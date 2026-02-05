# -------------------------- HIERARCHICAL CLUSTERING ----------------------- We first import scipy.cluster.hierarchy
# because this library provides everything for hierarchical clustering, including computing distances between
# clusters (linkage) and drawing dendrograms (dendrogram). We also import fcluster, which allows us to cut the
# dendrogram at a chosen distance to produce final cluster labels.

import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
df = pd.read_csv(url)

# Select numeric columns for clustering
numeric_columns = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
df_numeric = df[numeric_columns]

# Normalize the data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df_numeric)

# Compute linkage matrices for different methods
linkage_avg = sch.linkage(normalized_data, method='average')
linkage_single = sch.linkage(normalized_data, method='single')
linkage_complete = sch.linkage(normalized_data, method='complete')

# Plot dendrograms
plt.figure(figsize=(10, 6))
sch.dendrogram(linkage_avg)
plt.title("Dendrogram - Average Linkage")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.show()

plt.figure(figsize=(10, 6))
sch.dendrogram(linkage_single)
plt.title("Dendrogram - Single Linkage")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.show()

plt.figure(figsize=(10, 6))
sch.dendrogram(linkage_complete)
plt.title("Dendrogram - Complete Linkage")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.show()

# Cut dendrogram to form clusters (choose threshold distance)
clusters_avg = fcluster(linkage_avg, t=0.7, criterion='distance')
clusters_single = fcluster(linkage_single, t=0.7, criterion='distance')
clusters_complete = fcluster(linkage_complete, t=0.7, criterion='distance')

# Add cluster labels to DataFrame
df['Cluster_Avg'] = clusters_avg
df['Cluster_Single'] = clusters_single
df['Cluster_Complete'] = clusters_complete
