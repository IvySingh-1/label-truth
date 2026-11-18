"""
Model Evaluation & Benchmarking Runner
Executes 10-Fold CV Classification and Clustering Evaluation.
Supports DMDW Lab Module 12 (Model Evaluation, ROC-AUC, Silhouette Scores).
"""

import os
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.mining.classification import evaluate_classification_models
from src.mining.clustering import run_kmeans_clustering, run_dbscan_anomaly_detection

def run_full_ml_evaluation():
    csv_path = os.path.join(PROJECT_ROOT, "data/processed/openfoodfacts_india_processed.csv")
    df = pd.read_csv(csv_path)
    
    print("====================================================================")
    print("DMDW LAB MODULE 9-11: MACHINE LEARNING & EVALUATION SUITE")
    print("====================================================================\n")
    
    # 1. Classification Benchmarking
    print("--- 1. Multi-Classifier Comparison (10-Fold Stratified Cross-Validation) ---")
    cls_res = evaluate_classification_models(df, target_col="verdict")
    print(cls_res["leaderboard"].to_string(index=False))
    
    # 2. K-Means Clustering & Silhouette Scores
    print("\n--- 2. K-Means Clustering & Cluster Quality Metrics ---")
    km_res = run_kmeans_clustering(df, n_clusters=4)
    print(f"Optimal k: {km_res['optimal_k']} | Silhouette Score: {km_res['silhouette_score']}")
    print("\nElbow Inertia and Silhouette across k=2 to 7:")
    for k, inert, sil in zip(km_res["k_range"], km_res["inertias"], km_res["silhouettes"]):
        print(f"  k={k} -> Inertia: {inert:7.1f} | Silhouette: {sil:6.3f}")
        
    print("\nCluster Profiles (Nutrient Means per Cluster):")
    profiles = km_res["cluster_profiles"][["sugars_g", "fat_g", "protein_g", "fiber_g", "product_count", "dominant_verdict"]]
    print(profiles.to_string())
    
    # 3. DBSCAN Anomaly Detection
    print("\n--- 3. DBSCAN Density-Based Outlier Detection ---")
    db_res = run_dbscan_anomaly_detection(df, eps=1.8, min_samples=3)
    print(f"Clusters Formed: {db_res['num_clusters']} | Outliers / Extreme Health-Washers Identified: {db_res['num_outliers']}")
    if db_res["num_outliers"] > 0:
        print("\nSample Outliers:")
        print(db_res["outliers_df"].head(5).to_string(index=False))

if __name__ == "__main__":
    run_full_ml_evaluation()
