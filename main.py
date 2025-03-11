import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic customer data
data = {
    "CustomerID": range(1, 101),
    "Age": np.random.randint(18, 70, 100),
    "Income": np.random.randint(20000, 120000, 100),
    "SpendingScore": np.random.randint(1, 100, 100)
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("customer_data.csv", index=False)
df = pd.read_csv("customer_data.csv")

# Explore dataset
print(df.head())
print(df.info())

# Select relevant features (e.g., Annual Income and Spending Score)
features = ["Income", "SpendingScore"]
X = df[features]

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow curve
plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker="o", linestyle="--")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method for Optimal K")
plt.show()

# Apply K-Means with optimal K (assume K=5 from Elbow method)
kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Visualize Clusters
plt.figure(figsize=(10,6))
sns.scatterplot(x=df["Annual Income (k$)"], y=df["Spending Score (1-100)"], hue=df["Cluster"], palette="Set1", s=100)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.title("Customer Segments")
plt.show()
