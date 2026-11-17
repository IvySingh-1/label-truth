"""
Preprocessing Pipeline Verification Runner
Tests feature scaling, binning, PCA reduction, and TF-IDF text mining.
"""

import os
import sys
import pandas as pd

# Add project root and module path to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.feature_scaling import apply_feature_scaling
from src.preprocessing.binning import traffic_light_binning, equal_width_binning, equal_frequency_binning
from src.preprocessing.pca_reduction import perform_pca_reduction
from src.preprocessing.text_mining import extract_tfidf_features, analyze_claim_keywords_vs_verdict

def run_all_preprocessing():
    csv_path = os.path.join(PROJECT_ROOT, "data/processed/openfoodfacts_india_processed.csv")
    df = pd.read_csv(csv_path)
    print(f"Loaded processed dataset with {len(df)} records.\n")
    
    # 1. Feature Scaling
    print("--- 1. Feature Scaling (MinMax & Standard) ---")
    minmax_df, _ = apply_feature_scaling(df, method="minmax")
    std_df, _ = apply_feature_scaling(df, method="standard")
    print("MinMax Scaled Sample:")
    print(minmax_df[["product_name", "sugars_g_scaled", "fat_g_scaled", "sodium_mg_scaled"]].head(3))
    
    # 2. Binning
    print("\n--- 2. Nutritional Discretization & Traffic Lights ---")
    binned_df = traffic_light_binning(df)
    print("Sugar Traffic Light Distribution:")
    print(binned_df["sugar_traffic_light"].value_counts())
    print("\nSodium Traffic Light Distribution:")
    print(binned_df["sodium_traffic_light"].value_counts())
    
    # 3. PCA Reduction
    print("\n--- 3. Principal Component Analysis (PCA) ---")
    pca_res = perform_pca_reduction(df, n_components=3)
    exp_var = pca_res["explained_variance_ratio"]
    cum_var = pca_res["cumulative_variance"]
    for i, (ev, cv) in enumerate(zip(exp_var, cum_var), 1):
        print(f"PC{i}: Explained Variance = {ev*100:.2f}%, Cumulative = {cv*100:.2f}%")
    print("\nPC1 Loadings (Top driving nutrients):")
    print(pca_res["loadings"]["PC1"].sort_values(ascending=False))
    
    # 4. TF-IDF Text Mining
    print("\n--- 4. TF-IDF Text Mining on Ingredient Lists ---")
    tfidf_res = extract_tfidf_features(df, text_column="ingredients_text", max_features=10)
    print("Top 10 Ingredient Terms by Mean TF-IDF:")
    print(tfidf_res["mean_scores"])
    
    print("\nClaim Keywords Correlation with Health-Washing Flag (is_misleading):")
    claim_corr = analyze_claim_keywords_vs_verdict(df)
    print(claim_corr.head(5))

if __name__ == "__main__":
    run_all_preprocessing()
