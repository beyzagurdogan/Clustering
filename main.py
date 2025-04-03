import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def load_data(filename):
    """Load data from a text file assuming space-separated values."""
    data = np.loadtxt(filename)
    return data


def plot_clusters(data, labels, centroids, title):
    """Plot clustered data with centroids."""
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.title(title)
    plt.legend()
    plt.show()


def kmeans_clustering(data, n_clusters):
    """Apply K-means clustering."""
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels, kmeans.cluster_centers_


if __name__ == "__main__":
    files = ["s1.txt", "s2.txt", "s3.txt", "s4.txt"]
    num_clusters = 4  # Adjust as needed

    for file in files:
        data = load_data(file)
        labels, centroids = kmeans_clustering(data, num_clusters)
        plot_clusters(data, labels, centroids, title=f'K-Means Clustering for {file}')
