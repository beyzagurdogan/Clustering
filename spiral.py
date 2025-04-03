import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def load_spiral_data(filename):
    """Load spiral dataset with three columns (x, y, cluster)."""
    data = np.loadtxt(filename)
    return data[:, :2], data[:, 2]  # Separate coordinates and true labels


def plot_spiral_clusters(data, labels, centroids, title):
    """Plot clustered spiral data with centroids."""
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.title(title)
    plt.legend()
    plt.show()


def kmeans_spiral(data, n_clusters):
    """Apply K-means clustering on spiral dataset."""
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels, kmeans.cluster_centers_


if __name__ == "__main__":
    filename = "spiral (1).txt"
    data, true_labels = load_spiral_data(filename)

    num_clusters = 3  # Given assumption
    labels, centroids = kmeans_spiral(data, num_clusters)

    plot_spiral_clusters(data, labels, centroids, title='K-Means Clustering on Spiral Data')
