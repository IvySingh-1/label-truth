"""
Rules Engine Demonstration & Verification Runner
Tests Hidden Sugar Detector, Teaspoon Calculator, Claim Verifier, and HWI Engine.
"""

import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rules_engine.hidden_sugars import analyze_hidden_sugars
from src.rules_engine.teaspoon_meter import calculate_sugar_teaspoons
from src.rules_engine.claim_verifier import audit_all_product_claims
from src.rules_engine.hwi_calculator import compute_market_hwi, compute_brand_hwi_leaderboard, compute_category_hwi

def run_rules_demo():
    print("====================================================================")
    print("LABELTRUTH RULES ENGINE & AUDIT SUITE")
    print("====================================================================\n")

    # 1. Test Hidden Sugar Detector on realistic label
    sample_ingredients = "Refined Wheat Flour, Sugar, Invert Sugar Syrup, Maltodextrin, Liquid Glucose, Palm Oil, Cocoa Solids, Salt"
    hs_res = analyze_hidden_sugars(sample_ingredients)
    print("--- 1. Hidden Sugar Detection Test ---")
    print(f"Ingredients: {sample_ingredients}")
    print(f"Total Disguised Sugars Found: {hs_res['total_hidden_count']}")
    print(f"Detected Terms: {hs_res['detected_sugars']}")
    print(f"Biochemical Groups: {list(hs_res['groups_detected'].keys())}")
    print(f"Sugar Disguise Score: {hs_res['disguise_score']}/100 ({hs_res['risk_level']} RISK)\n")

    # 2. Test Teaspoon Calculator
    print("--- 2. Sugar in Teaspoons & WHO Guideline Auditor ---")
    tsp_res = calculate_sugar_teaspoons(sugars_100g=24.5, serving_size_g=35.0)
    print(f"Product Sugar: 24.5g / 100g | Serving Size: 35g")
    print(f"Teaspoons per 100g: {tsp_res['teaspoons_per_100g']} tsp")
    print(f"Teaspoons per Single Serving: {tsp_res['teaspoons_per_serving']} tsp ({tsp_res['visual_spoon_display']})")
    print(f"WHO Daily Limit Used by 1 Serving: {tsp_res['who_adult_daily_pct']}% of adult max ({tsp_res['alert_level']})")
    print(f"Consumer Alert: {tsp_res['alert_desc']}\n")

    # 3. Test Claim Verifier on sample product
    print("--- 3. Claim Verifier Audit on 'Digestive High Fibre' Biscuit ---")
    product_profile = {
        "product_name": "Sample Digestive Cookie",
        "category": "Snacks & Bakery",
        "sugars_g": 18.0,
        "added_sugars_g": 16.5,
        "fat_g": 21.0,
        "saturated_fat_g": 9.5,
        "fiber_g": 5.8,
        "protein_g": 7.5,
        "sodium_mg": 460.0,
        "ingredients_text": "Refined Wheat Flour (Maida), Sugar, Palm Oil, Invert Sugar, Maltodextrin, Salt"
    }
    claims_str = "High Fibre, 100% Whole Wheat, Digestive, 0% Trans Fat"
    audit_res = audit_all_product_claims(claims_str, product_profile)
    print(f"Claims on Pack: {claims_str}")
    print(f"Overall Audit Verdict: {audit_res['overall_status']}")
    print(f"Misleading Claims Count: {audit_res['misleading_claims_count']}/{audit_res['total_claims']}")
    for audit in audit_res["claims_audited"]:
        print(f"  • Claim: '{audit['claim']}' -> [{audit['status']}] - {audit['reason']}")

    # 4. Market HWI and Leaderboard
    csv_path = os.path.join(PROJECT_ROOT, "data/processed/openfoodfacts_india_processed.csv")
    df = pd.read_csv(csv_path)
    market_hwi = compute_market_hwi(df)
    print(f"\n--- 4. Market Health-Washing Index (HWI) ---")
    print(f"Total Products Analyzed: {market_hwi['total_products']}")
    print(f"Products with Health Claims: {market_hwi['total_with_claims']}")
    print(f"Misleading Products: {market_hwi['misleading_products']}")
    print(f"Market HWI (%): {market_hwi['hwi_percentage']}%\n")

    print("Top 5 Brand Health-Washing Leaderboard:")
    brand_lb = compute_brand_hwi_leaderboard(df, min_products=4)
    print(brand_lb.head(5).to_string(index=False))

if __name__ == "__main__":
    run_rules_demo()
