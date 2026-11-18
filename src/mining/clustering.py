"""
Unsupervised Clustering Module: K-Means, Hierarchical & DBSCAN
Discovers natural nutritional groupings and detects outlier health-washers.
Supports DMDW Lab Module 11 (K-Means, Agglomerative Clustering, DBSCAN, Silhouette Score).
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

FEATURE_COLS = [
    "energy_kcal", "carbohydrates_g", "sugars_g", "added_sugars_g",
    "fat_g", "saturated_fat_g", "protein_g", "fiber_g", "sodium_mg"
]

def run_kmeans_clustering(df: pd.DataFrame, n_clusters: int = 4) -> dict:
    """
    Performs K-Means clustering, computes Elbow curve (k=2..7) and cluster profiles.
    """
    X = df[FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Evaluate optimal k across 2 to 7
    k_range = list(range(2, 8))
    inertias = []
    silhouettes = []
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(X_scaled, km.labels_)))
        
    # Fit selected k
    optimal_km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = optimal_km.fit_predict(X_scaled)
    
    df_clustered = df.copy()
    df_clustered["cluster"] = cluster_labels
    
    # Cluster profiling: mean nutritional values
    cluster_profiles = df_clustered.groupby("cluster")[FEATURE_COLS].mean().round(1)
    cluster_profiles["product_count"] = df_clustered.groupby("cluster").size()
    cluster_profiles["dominant_verdict"] = df_clustered.groupby("cluster")["verdict"].agg(lambda s: s.mode()[0])
    
    # Semantic name assignment
    cluster_names = {}
    for c_id, row in cluster_profiles.iterrows():
        if row["sugars_g"] > 20:
            cluster_names[c_id] = "High Sugar / Confectionery Cluster"
        elif row["fat_g"] > 25:
            cluster_names[c_id] = "High Fat / Deep Fried Namkeen Cluster"
        elif row["protein_g"] > 12 or row["fiber_g"] > 6:
            cluster_names[c_id] = "Nutrient Dense / High Protein Cluster"
        else:
            cluster_names[c_id] = "Balanced / Moderate Impact Cluster"
            
    df_clustered["cluster_name"] = df_clustered["cluster"].map(cluster_names)
    
    return {
        "k_range": k_range,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "optimal_k": n_clusters,
        "silhouette_score": round(float(silhouette_score(X_scaled, cluster_labels)), 3),
        "cluster_profiles": cluster_profiles,
        "clustered_df": df_clustered
    }

def run_hierarchical_clustering(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    Performs Agglomerative Hierarchical Clustering with Ward linkage.
    """
    X = df[FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    agg = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")
    labels = agg.fit_predict(X_scaled)
    
    df_res = df.copy()
    df_res["hierarchical_cluster"] = labels
    return df_res

def run_dbscan_anomaly_detection(df: pd.DataFrame, eps: float = 1.8, min_samples: int = 3) -> dict:
    """
    DBSCAN density-based clustering to identify outliers (label -1).
    Outliers often correspond to extreme health-washers or extreme nutritional outliers.
    """
    X = df[FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    
    df_res = df.copy()
    df_res["dbscan_cluster"] = labels
    df_res["is_outlier"] = (labels == -1).astype(int)
    
    outliers = df_res[df_res["is_outlier"] == 1]
    
    return {
        "num_clusters": len(set(labels)) - (1 if -1 in labels else 0),
        "num_outliers": len(outliers),
        "outliers_df": outliers[["product_name", "brand", "category", "sugars_g", "fat_g", "sodium_mg", "verdict"]],
        "full_df": df_res
    }
