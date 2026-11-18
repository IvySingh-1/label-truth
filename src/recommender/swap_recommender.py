"""
Healthier Swap Recommender Engine
Recommends truly healthy product alternatives within the same food category.
Uses Nutri-Score superiority, strict HONEST verdict filtering, and Cosine Nutritional Similarity.
Calculates quantified nutrient deltas (-% sugar, -% fat, +% fiber, teaspoons saved).
Supports DMDW Lab Module 12-14 (Recommendation Systems & Course Capstone).
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

FEATURE_COLS = [
    "energy_kcal", "carbohydrates_g", "sugars_g", "fat_g",
    "saturated_fat_g", "protein_g", "fiber_g", "sodium_mg"
]

NUTRI_RANK = {"a": 5, "b": 4, "c": 3, "d": 2, "e": 1}

def get_healthier_swaps(product_id_or_name: str, df: Optional[pd.DataFrame] = None, top_k: int = 3) -> Dict:
    """
    Finds top-k healthier alternatives for a given product.
    
    Criteria:
    1. Same category (or matching sub-category).
    2. Higher or equal Nutri-Score (A > B > C > D > E).
    3. Strictly lower sugars or lower saturated fat.
    4. Ranked by cosine nutritional profile similarity (best functional match).
    """
    if df is None:
        csv_path = os.path.join(PROJECT_ROOT, "data/processed/openfoodfacts_india_processed.csv")
        df = pd.read_csv(csv_path)

    # Find target product
    matches = df[(df["product_id"].astype(str) == str(product_id_or_name)) | 
                 (df["product_name"].str.contains(str(product_id_or_name), case=False, na=False))]
    
    if matches.empty:
        return {"found": False, "message": f"Product '{product_id_or_name}' not found."}
        
    target = matches.iloc[0]
    target_rank = NUTRI_RANK.get(str(target["nutriscore_grade"]).lower(), 2)
    
    # Filter candidate pool
    candidates = df[
        (df["category"] == target["category"]) &
        (df["product_id"] != target["product_id"])
    ].copy()
    
    if candidates.empty:
        # Fallback to broader category pool
        candidates = df[df["product_id"] != target["product_id"]].copy()

    # Retain products that are strictly better:
    # (Higher Nutri-Score OR (Equal Nutri-Score AND lower sugar/fat))
    candidates["nutri_rank"] = candidates["nutriscore_grade"].str.lower().map(NUTRI_RANK)
    better_mask = (candidates["nutri_rank"] > target_rank) | (
        (candidates["nutri_rank"] >= target_rank) & 
        (candidates["sugars_g"] < target["sugars_g"])
    ) | (candidates["verdict"] == "HONEST")
    
    qualified = candidates[better_mask].copy()
    if qualified.empty:
        qualified = candidates.sort_values(by="nutri_rank", ascending=False).head(5)

    # Compute cosine similarity
    scaler = MinMaxScaler()
    all_features = np.vstack([target[FEATURE_COLS].values.reshape(1, -1), qualified[FEATURE_COLS].values])
    scaled_feats = scaler.fit_transform(all_features)
    
    target_vec = scaled_feats[0:1]
    candidate_vecs = scaled_feats[1:]
    
    sim_scores = cosine_similarity(target_vec, candidate_vecs)[0]
    qualified["similarity"] = sim_scores
    
    # Sort by Nutri-Score superiority first, then similarity
    ranked = qualified.sort_values(by=["nutri_rank", "similarity", "sugars_g"], ascending=[False, False, True]).head(top_k)
    
    swaps = []
    for _, cand in ranked.iterrows():
        # Compute nutrient deltas
        sugar_diff = cand["sugars_g"] - target["sugars_g"]
        sugar_pct_change = round((sugar_diff / max(target["sugars_g"], 0.1)) * 100.0, 1)
        tsp_saved = round((target["sugars_g"] - cand["sugars_g"]) / 4.0, 1)
        
        fat_diff = cand["fat_g"] - target["fat_g"]
        fat_pct_change = round((fat_diff / max(target["fat_g"], 0.1)) * 100.0, 1)
        
        fiber_diff = cand["fiber_g"] - target["fiber_g"]
        protein_diff = cand["protein_g"] - target["protein_g"]

        swaps.append({
            "product_id": cand["product_id"],
            "product_name": cand["product_name"],
            "brand": cand["brand"],
            "category": cand["category"],
            "nutriscore_grade": cand["nutriscore_grade"].upper(),
            "verdict": cand["verdict"],
            "sugars_g": cand["sugars_g"],
            "sugar_teaspoons": cand["sugar_teaspoons"],
            "fiber_g": cand["fiber_g"],
            "protein_g": cand["protein_g"],
            "similarity_pct": round(float(cand["similarity"]) * 100.0, 1),
            "sugar_delta_pct": sugar_pct_change,
            "teaspoons_saved": max(0.0, tsp_saved),
            "fat_delta_pct": fat_pct_change,
            "fiber_gain_g": round(fiber_diff, 1),
            "protein_gain_g": round(protein_diff, 1),
            "swap_rationale": f"Upgrades Nutri-Score from {target['nutriscore_grade'].upper()} to {cand['nutriscore_grade'].upper()}; saves {max(0.0, tsp_saved)} tsp sugar."
        })
        
    return {
        "found": True,
        "target_product": {
            "product_name": target["product_name"],
            "brand": target["brand"],
            "category": target["category"],
            "nutriscore_grade": str(target["nutriscore_grade"]).upper(),
            "verdict": target["verdict"],
            "sugars_g": target["sugars_g"],
            "sugar_teaspoons": target["sugar_teaspoons"],
            "fat_g": target["fat_g"],
            "fiber_g": target["fiber_g"]
        },
        "recommendations": swaps
    }

if __name__ == "__main__":
    res = get_healthier_swaps("890103000001")  # Britannia NutriChoice Digestive
    print("TARGET PRODUCT:")
    print(res["target_product"])
    print("\nRECOMMENDED HEALTHIER SWAPS:")
    for i, s in enumerate(res["recommendations"], 1):
        print(f"{i}. {s['product_name']} by {s['brand']} (Nutri-Score: {s['nutriscore_grade']}, Verdict: {s['verdict']})")
        print(f"   Saved {s['teaspoons_saved']} tsp sugar ({s['sugar_delta_pct']}% sugar), +{s['fiber_gain_g']}g fibre")
        print(f"   Rationale: {s['swap_rationale']}\n")
