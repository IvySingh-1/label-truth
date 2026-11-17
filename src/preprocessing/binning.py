"""
Nutritional Discretization & Binning Module
Supports Equal-Width, Equal-Frequency (Quantile), and FSSAI Regulatory Threshold Binning.
Supports DMDW Lab Module 6 (Discretization & Concept Hierarchies).
"""

import pandas as pd
import numpy as np

def equal_width_binning(series: pd.Series, bins: int = 3, labels=None):
    """Discretizes continuous series into equal-width intervals."""
    if labels is None:
        labels = [f"Bin_{i+1}" for i in range(bins)]
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True)

def equal_frequency_binning(series: pd.Series, q: int = 3, labels=None):
    """Discretizes continuous series into equal-frequency (quantile) intervals."""
    if labels is None:
        labels = [f"Quantile_{i+1}" for i in range(q)]
    return pd.qcut(series, q=q, labels=labels, duplicates='drop')

def traffic_light_binning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies FSSAI / Traffic-Light regulatory thresholds to classify
    Sugar, Saturated Fat, and Sodium into 'LOW', 'MEDIUM', 'HIGH'.
    
    Thresholds (Solid foods per 100g):
    - Sugar: Low <= 5g, Medium 5-15g, High > 15g
    - Saturated Fat: Low <= 1.5g, Medium 1.5-5g, High > 5g
    - Sodium: Low <= 120mg, Medium 120-600mg, High > 600mg
    """
    df_binned = df.copy()
    
    # Sugar Traffic Light
    sugar_bins = [-np.inf, 5.0, 15.0, np.inf]
    df_binned["sugar_traffic_light"] = pd.cut(df_binned["sugars_g"], bins=sugar_bins, labels=["LOW", "MEDIUM", "HIGH"])
    
    # Saturated Fat Traffic Light
    sat_fat_bins = [-np.inf, 1.5, 5.0, np.inf]
    df_binned["sat_fat_traffic_light"] = pd.cut(df_binned["saturated_fat_g"], bins=sat_fat_bins, labels=["LOW", "MEDIUM", "HIGH"])
    
    # Sodium Traffic Light
    sodium_bins = [-np.inf, 120.0, 600.0, np.inf]
    df_binned["sodium_traffic_light"] = pd.cut(df_binned["sodium_mg"], bins=sodium_bins, labels=["LOW", "MEDIUM", "HIGH"])
    
    return df_binned
