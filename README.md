# Supermarket Customer Clustering Analysis

This project analyzes supermarket customer purchasing behavior using multiple clustering algorithms to determine which technique produces the most meaningful customer segmentation for business decision-making.

The analysis is performed on the Wholesale Customers dataset from the UCI Machine Learning Repository and focuses on identifying natural groupings based on purchasing patterns rather than only spending magnitude.

---

## Dataset Description

The dataset contains annual spending information of customers across different product categories:

- Fresh
- Milk
- Grocery
- Frozen
- Detergents_Paper
- Delicassen

Each row represents a customer, and each column represents spending behavior in a specific product category.

---

## Objective

The goal of this project is to evaluate and compare different clustering algorithms to determine which method best captures real-world supermarket customer segments.

The algorithms evaluated are:

- K-Means Clustering
- K-Medoids Clustering
- Hierarchical Clustering

---

## Data Preprocessing

All numerical features are normalized using Min-Max Scaling to ensure that no single product category dominates distance calculations due to scale differences.

---

## Clustering Algorithms Used

### K-Means Clustering
K-Means partitions customers into a fixed number of clusters by minimizing within-cluster variance. While computationally efficient, it assumes spherical clusters and is sensitive to outliers.

### K-Medoids Clustering
K-Medoids improves upon K-Means by selecting actual data points as cluster centers (medoids), making it more robust to extreme spenders and outliers commonly present in retail data.

### Hierarchical Clustering
Hierarchical clustering builds a tree-like structure (dendrogram) based on customer similarity. It does not assume any predefined cluster shape and allows for more flexible, behavior-based segmentation.

---

## Evaluation Metric

Silhouette Score is used to evaluate clustering quality. It measures how similar a customer is to its own cluster compared to other clusters.

Higher silhouette values indicate better-defined and more meaningful clusters.

---

## Results and Comparison

- K-Means produced clusters primarily driven by spending magnitude and was influenced by extreme high-value customers.
- K-Medoids generated more stable clusters by reducing the impact of outliers and showed improved silhouette scores.
- Hierarchical Clustering produced the most interpretable and behavior-driven customer segments, clearly separating reseller-type buyers, horeca (hotel/restaurant) customers, and mixed buyers.

---

## Conclusion

Hierarchical Clustering performed best for supermarket customer segmentation because customer purchasing data is non-spherical, skewed, and contains outliers. Unlike centroid-based methods, hierarchical clustering groups customers based on true similarity in purchasing behavior, making the results more meaningful for business strategy and marketing decisions.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- PyClustering
- Matplotlib
- SciPy

---

## Author
Engineer Hamza Ahmed

Hamza Aftab Ahmed
