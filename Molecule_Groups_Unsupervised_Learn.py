# Author: Alex Seager
# Last Version: 7/7/25
#
# Exploratory file to group similar molecules
# Description: Attempting to learn to classify important molecules by group without labels
# 

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import hdbscan

# Import and clean GNPS data
def LoadRealMatrix(csv_path, normalize_spectra=True):
    df_full = pd.read_csv(csv_path, index_col=0)
    df = df_full.loc[50:787]  # ensure numeric slicing works correctly

    A = df.values  # shape: (numWavelengths, numMolecules)

    if normalize_spectra:
        norms = np.linalg.norm(A, axis=0)
        norms[norms == 0] = 1  # avoid division by zero
        A = A / norms

    return A, df

# AGGROMATIVE CLUSTER
# Plot dendrogram of aggromerative clusters using Ward’s method
def plot_dendrogram(data, labels, truncate_level=None):
    # Compute pairwise distances and linkage matrix
    linkage_matrix = linkage(data, method='ward')  # Ward minimizes variance

    plt.figure(figsize=(12, 6))
    dendrogram(
        linkage_matrix,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=10,
        truncate_mode='level' if truncate_level else None,
        p=truncate_level if truncate_level else None,
        color_threshold=None  # You can set this to color branches by height
    )
    plt.title("Hierarchical Clustering Dendrogram (Ward Linkage)")
    plt.xlabel("Molecule")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()


#DBSCAN
def run_dbscan(A, df, eps=0.5, min_samples=5):
    # A is already normalized — but we standardize features across m/z for DBSCAN
    A_scaled = StandardScaler().fit_transform(A.T)  # transpose to shape (n_samples, n_features)

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(A_scaled)
    labels = db.labels_

    # Print cluster summary
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"Estimated clusters: {n_clusters}")
    print(f"Noise points: {n_noise}")

    # Assign labels to molecule names
    mol_names = df.columns.tolist()
    cluster_df = pd.DataFrame({'Molecule': mol_names, 'Cluster': labels})
    print(cluster_df.sort_values('Cluster'))

    return cluster_df

# running DBSCAN  
def run_and_plot_dbscan(A, df, eps=1.5, min_samples=4):
    # Transpose to shape (n_samples, n_features)
    A_T = A.T

    # Standardize features (important for DBSCAN)
    A_scaled = StandardScaler().fit_transform(A_T)

    # Run DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(A_scaled)
    labels = db.labels_

    # PCA to 2D
    pca = PCA(n_components=2)
    A_pca = pca.fit_transform(A_scaled)

    # Molecule names
    mol_names = df.columns.tolist()

    # Plot
    unique_labels = set(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    plt.figure(figsize=(12, 8))
    for label, color in zip(unique_labels, colors):
        idx = labels == label
        label_str = f"Cluster {label}" if label != -1 else "Noise"
        plt.scatter(A_pca[idx, 0], A_pca[idx, 1], c=[color], label=label_str, s=60, alpha=0.7, edgecolors='k')

        # Annotate each point with molecule name
        for x, y, name in zip(A_pca[idx, 0], A_pca[idx, 1], np.array(mol_names)[idx]):
            plt.text(x, y, name, fontsize=8, alpha=0.7)

    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("DBSCAN Clusters on Mass Spec Matrix (PCA Projected)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Save cluster assignments
    cluster_df = pd.DataFrame({'Molecule': mol_names, 'Cluster': labels})
    return cluster_df

def run_and_plot_hdbscan(A, df, min_cluster_size=5, min_samples=None):
    A_scaled = StandardScaler().fit_transform(A.T)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                prediction_data=True)
    labels = clusterer.fit_predict(A_scaled)

    # PCA for 2D plotting
    pca = PCA(n_components=2)
    A_pca = pca.fit_transform(A_scaled)

    # Plot clusters
    unique_labels = set(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    plt.figure(figsize=(10, 6))
    for label, color in zip(unique_labels, colors):
        idx = labels == label
        label_str = f"Cluster {label}" if label != -1 else "Noise"
        plt.scatter(A_pca[idx, 0], A_pca[idx, 1], c=[color], label=label_str, s=60, alpha=0.7, edgecolors='k')

    # Label each point with molecule name
    mol_names = df.columns.tolist()
    for i, name in enumerate(mol_names):
        plt.text(A_pca[i, 0] + 0.01, A_pca[i, 1] + 0.01, name, fontsize=6)

    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("HDBSCAN Clusters on Mass Spec Matrix (PCA Projected)")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    # Return cluster assignments
    return pd.DataFrame({'Molecule': mol_names, 'Cluster': labels})



# Main
def main():
    file = "mass_spectra_individual.csv"
    A, df = LoadRealMatrix(file)
    
    # cluster_df = run_and_plot_hdbscan(A, df, min_cluster_size=2)
  
  
    # Transpose for clustering: shape (molecules, features)
    A_T = A.T
    molecule_names = df.columns.tolist()

    # # # Plot full or truncated dendrogram
    plot_dendrogram(A_T, labels=molecule_names, truncate_level=20)  # change truncate_level=None for full tree

if __name__ == "__main__":
    main()
