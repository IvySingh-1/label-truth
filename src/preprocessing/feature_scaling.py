"""
Feature Scaling Module
Implements StandardScaler, MinMaxScaler, and RobustScaler for nutritional attributes.
Supports DMDW Lab Module 6 (Preprocessing & Normalization).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

NUMERIC_NUTRIENTS = [
    "energy_kcal", "carbohydrates_g", "sugars_g", "added_sugars_g",
    "fat_g", "saturated_fat_g", "protein_g", "fiber_g", "sodium_mg"
]

def apply_feature_scaling(df: pd.DataFrame, method: str = "minmax"):
    """
    Applies feature scaling on nutritional columns.
    
    Parameters:
        df (pd.DataFrame): Dataframe with nutritional values.
        method (str): 'standard' for Z-score normalization,
                      'minmax' for [0, 1] scaling,
                      'robust' for median/IQR scaling.
    
    Returns:
        pd.DataFrame: Scaled dataframe with preserved IDs and metadata.
    """
    df_copy = df.copy()
    features = df_copy[NUMERIC_NUTRIENTS].values
    
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler(feature_range=(0, 1))
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
        
    scaled_values = scaler.fit_transform(features)
    scaled_df = pd.DataFrame(scaled_values, columns=[f"{col}_scaled" for col in NUMERIC_NUTRIENTS], index=df.index)
    
    result = pd.concat([df_copy[["product_id", "product_name", "brand", "category", "verdict"]], scaled_df], axis=1)
    return result, scaler
