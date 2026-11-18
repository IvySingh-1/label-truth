"""
Health-Washing Index (HWI) Calculator Module
Computes product-level, category-level, and brand-level Health-Washing Index metrics.
Formula: HWI (%) = (Misleading Products with Claims / Total Products with Claims) * 100
Supports DMDW Lab Module 12-14 (Evaluation & Quantitative Metrics).
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple

def compute_market_hwi(df: pd.DataFrame) -> Dict:
    """
    Computes overall market Health-Washing Index and breakdown.
    """
    total_products = len(df)
    with_claims = df[df["has_health_claim"] == 1]
    total_with_claims = len(with_claims)
    
    misleading_products = len(with_claims[with_claims["is_misleading"] == 1])
    honest_products = len(with_claims[with_claims["is_misleading"] == 0])
    
    hwi_pct = (misleading_products / total_with_claims * 100.0) if total_with_claims > 0 else 0.0
    
    return {
        "total_products": total_products,
        "total_with_claims": total_with_claims,
        "misleading_products": misleading_products,
        "honest_products": honest_products,
        "hwi_percentage": round(hwi_pct, 2)
    }

def compute_brand_hwi_leaderboard(df: pd.DataFrame, min_products: int = 2) -> pd.DataFrame:
    """
    Computes HWI leaderboard ranking brands by health-washing severity.
    """
    records = []
    for brand, group in df.groupby("brand"):
        total = len(group)
        with_claims = len(group[group["has_health_claim"] == 1])
        misleading = len(group[(group["has_health_claim"] == 1) & (group["is_misleading"] == 1)])
        honest = len(group[group["verdict"] == "HONEST"])
        
        hwi = (misleading / with_claims * 100.0) if with_claims > 0 else 0.0
        avg_sugar_tsp = group["sugar_teaspoons"].mean()
        avg_hidden_sugars = group["hidden_sugar_count"].mean()
        
        records.append({
            "brand": brand,
            "total_products": total,
            "products_with_claims": with_claims,
            "misleading_products": misleading,
            "honest_products": honest,
            "hwi_pct": round(hwi, 1),
            "avg_sugar_teaspoons": round(avg_sugar_tsp, 1),
            "avg_hidden_sugar_count": round(avg_hidden_sugars, 1)
        })
        
    res_df = pd.DataFrame(records)
    if min_products > 1:
        res_df = res_df[res_df["total_products"] >= min_products]
    return res_df.sort_values(by=["hwi_pct", "misleading_products", "avg_sugar_teaspoons"], ascending=[False, False, False])

def compute_category_hwi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes HWI and nutritional averages per food category.
    """
    records = []
    for cat, group in df.groupby("category"):
        total = len(group)
        with_claims = len(group[group["has_health_claim"] == 1])
        misleading = len(group[(group["has_health_claim"] == 1) & (group["is_misleading"] == 1)])
        
        hwi = (misleading / with_claims * 100.0) if with_claims > 0 else 0.0
        avg_sugar_tsp = group["sugar_teaspoons"].mean()
        avg_sat_fat = group["saturated_fat_g"].mean()
        avg_sodium = group["sodium_mg"].mean()
        
        records.append({
            "category": cat,
            "total_products": total,
            "products_with_claims": with_claims,
            "misleading_count": misleading,
            "hwi_pct": round(hwi, 1),
            "avg_sugar_tsp": round(avg_sugar_tsp, 1),
            "avg_sat_fat_g": round(avg_sat_fat, 1),
            "avg_sodium_mg": round(avg_sodium, 0)
        })
        
    return pd.DataFrame(records).sort_values(by="hwi_pct", ascending=False)
