#predictive ai vs classification ai vs clustering ai
#predictive ai
# 1️⃣ Prediction AI (What you were thinking of)
#
# This is when AI predicts something that will happen or estimates an unknown value.
#
# Examples:
# • Predict house prices
# • Predict stock market movement
# • Predict whether a customer will leave a company
# • Predict tomorrow’s weather
#
# Algorithms used:
# • Linear Regression
# • Logistic Regression
# • Neural Networks
# • LSTM (like your stock project)
#
# 👉 This type uses past data to estimate future or unknown results

# 2️⃣ Classification AI
#
# Here AI labels or categorizes data, not predicts the future.
#
# Examples:
# • Email spam vs not spam
# • Disease detection from medical reports
# • Image recognition (cat vs dog)
#
# Here AI is answering:
#
# "What category does this belong to?"


# 3️⃣ Clustering AI (Your K-Means comes here)
#
# This is where things become interesting.
#
# Clustering AI:
# • Does NOT predict future
# • Does NOT use labels
# • Instead, it discovers hidden patterns inside data
#
# Example:
# A mall has thousands of customers but doesn’t know customer types.
#
# K-Means helps answer:
# • Which customers behave similarly?
# • Which customers spend alike?
# • Which customers buy similar products?
#
# 👉 It reveals structure in data that humans may not notice easily

# 💡 Simple Way To Remember
#
# Prediction AI → "What will happen?"
#
# Classification AI → "What category is this?"
#
# Clustering AI → "What hidden groups exist?"

# ----------------clustering is hugely used in real companies for:
# • Customer segmentation
# • Fraud pattern discovery
# • Recommendation systems
# • Market analysis

# --------------------------- K-MEANS CLUSTERING ---------------------------
# is called AI because the system:
# • learns patterns from data
# • makes decisions without being explicitly programmed with rules
# • finds structure on its own
#
# You never tell it:
# “this customer is low spender”
# “this customer is high spender”
#
# It figures that out by itself using mathematics and data. That ability to learn patterns from data is why it qualifies as AI.
# # make clusters k, in the data points and update and ssigned data points to updating clusters

#data points= customers
#cluster 1 high spending customer
#cluster 2 low spending customer

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset directly from UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
df = pd.read_csv(url)

# Select numeric columns for clustering
numeric_columns = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
df_numeric = df[numeric_columns]

# Normalize data to 0-1 scale to avoid features with large numbers dominating
scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df_numeric)

# --------------------------- ELBOW METHOD ---------------------------
#DETERMING K
# “How many customer groups (clusters) should we create so that customers inside a group are similar, but different from other groups?”
# Determine optimal k by plotting inertia (sum of squared distances to centroids)

#Inertia is the value which tells how closly data point is tightly packed arround its centroid
# we need that much clusters k in which are data has low inertia and also some clusters
# increaing cluster decreses the inertia but we also dont want many clusters which defeats the purpose of clustering

inertia = []
# here we are trying different k values from 2 clusters to 6 clusters to check how many clusters reduces the inertia yet
# also serves to be meaningful
for k in range(2, 7):
    #intialization of k-means
    # n_clusters=k each iteration assign k value with [2->3->4->5->6]
    # k=2 first iteration -> k=3 second iteration

    #--- iteration number 1
    # “What happens if I divide customers into 2 groups?”
    # this line of code only do:
    #   Creates a K-Means object
    # • Stores the idea “I will create 2 clusters”
    # • Sets a seed so results are reproducible



    #-----------------the first time we use it for the purpsoe of
    #only to read inertia and determine how many k are good
    kmeans = KMeans(n_clusters=k, random_state=42)

    # here the model is train and fit
    kmeans.fit(df_normalized)
    inertia.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(2, 7), inertia, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Sum of Squared Distances)")
plt.show()

# --------------------------- K-MEANS WITH CHOSEN k ---------------------------


optimal_k = 3  # Assume 3 based on elbow curve
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(df_normalized)

# Add cluster labels to DataFrame
df['KMeans_Cluster'] = cluster_labels

# --------------------------- VISUALIZE CLUSTERS ---------------------------
# Use first two features for 2D scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(df_normalized[:, 0], df_normalized[:, 1],
            c=cluster_labels, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            color='red', marker='X', s=200, label='Centroids')
plt.title("K-Means Clustering (k=3)")
plt.xlabel(numeric_columns[0])
plt.ylabel(numeric_columns[1])
plt.legend()
plt.show()

print("K-Means clustering complete. Cluster counts:")
print(df['KMeans_Cluster'].value_counts())
