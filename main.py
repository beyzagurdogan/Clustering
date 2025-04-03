import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# 1. Veri Okuma Fonksiyonu (Hatalı satırları atlayan)
def read_data(file_path):
    if not os.path.exists(file_path):  # Eğer dosya yoksa hata vermesin
        print(f"Dosya bulunamadı: {file_path}")
        return None

    try:
        data = np.loadtxt(file_path, comments="#", delimiter=None)  # Sayısal veriyi oku
        return data
    except ValueError:
        with open(file_path, "r") as f:
            lines = f.readlines()
        clean_data = []
        for line in lines:
            try:
                values = list(map(float, line.split()))  # Satırı sayıya çevir
                clean_data.append(values)
            except ValueError:
                print(f"Hatalı satır atlandı: {line.strip()}")
        return np.array(clean_data)


# 2. Verileri Okuma
files = ["s1.txt", "s2.txt", "s3.txt", "s4.txt", "spiral.txt"]
datasets = {file: read_data(file) for file in files if os.path.exists(file)}


# 3. Veri Setlerini Görselleştirme
def plot_data(data, title):
    if data is None:  # Eğer veri yoksa, işlemi atla
        print(f"Veri bulunamadığı için {title} görselleştirilemiyor.")
        return

    plt.scatter(data[:, 0], data[:, 1], s=5, c='blue')
    plt.title(title)
    plt.show()


for file, data in datasets.items():
    plot_data(data, f"Veri Seti: {file}")


# 4. K-Means Kümeleme Fonksiyonu
def kmeans_clustering(data, n_clusters, title):
    if data is None:
        print(f"Veri eksik olduğu için {title} kümeleme yapılamıyor.")
        return

    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    labels = kmeans.fit_predict(data[:, :2])

    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=5)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=100,
                label='Centroids')
    plt.title(title)
    plt.legend()
    plt.show()

    return kmeans


# 5. Her Veri Seti İçin Kümeleme Sonuçlarını Görselleştirme
for file, data in datasets.items():
    if "spiral" in file:
        kmeans_clustering(data, 3, f"K-Means Kümeleme: {file}")
    else:
        kmeans_clustering(data, 4, f"K-Means Kümeleme: {file}")
