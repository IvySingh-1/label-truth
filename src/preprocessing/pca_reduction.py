"""
Principal Component Analysis (PCA) Dimensionality Reduction
Supports DMDW Lab Module 7 (Data Reduction & Feature Extraction).
Reduces 9 continuous nutritional attributes into orthogonal principal components.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

NUMERIC_NUTRIENTS = [
    "energy_kcal", "carbohydrates_g", "sugars_g", "added_sugars_g",
    "fat_g", "saturated_fat_g", "protein_g", "fiber_g", "sodium_mg"
]

def perform_pca_reduction(df: pd.DataFrame, n_components: int = 3):
    """
    Standardizes features and computes PCA.
    
    Returns:
        transformed_df: DataFrame with PC1, PC2, ... and metadata
        explained_variance_ratio: Array of explained variance per PC
        cumulative_variance: Cumulative variance explained
        loadings: DataFrame of eigenvectors/loadings per nutrient
    """
    X = df[NUMERIC_NUTRIENTS].values
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_std)
    
    columns = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(X_pca, columns=columns, index=df.index)
    
    meta_cols = ["product_id", "product_name", "brand", "category", "nutriscore_grade", "verdict"]
    result_df = pd.concat([df[[c for c in meta_cols if c in df.columns]], pca_df], axis=1)
    
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=columns,
        index=NUMERIC_NUTRIENTS
    )
    
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    
    return {
        "transformed_df": result_df,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": cum_var,
        "loadings": loadings,
        "pca_model": pca
    }
