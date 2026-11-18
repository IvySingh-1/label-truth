"""
Association Rule Mining Module (Apriori & FP-Growth)
Discovers frequent itemsets and association rules connecting front packaging claims
to nutritional red flags (e.g. {Claim_Multigrain} => {Flag_High_Sugar, Flag_High_Sat_Fat}).
Supports DMDW Lab Module 9 (Apriori, FP-Growth, Support, Confidence, Lift).
"""

import time
import os
import sys
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def create_food_transactions(df: pd.DataFrame) -> list:
    """
    Transforms tabular food dataset into basket-style transactions of Claims and Nutritional Red Flags.
    """
    transactions = []
    
    for _, row in df.iterrows():
        basket = []
        claims_str = str(row["front_claims"]).lower()
        ing_str = str(row["ingredients_text"]).lower()
        
        # Claims tags
        if "fibre" in claims_str or "fiber" in claims_str:
            basket.append("Claim_High_Fibre")
        if "no added sugar" in claims_str or "sugar free" in claims_str:
            basket.append("Claim_No_Added_Sugar")
        if "multigrain" in claims_str or "5 grain" in claims_str or "whole wheat" in claims_str or "atta" in claims_str:
            basket.append("Claim_Multigrain")
        if "baked" in claims_str:
            basket.append("Claim_Baked_Not_Fried")
        if "protein" in claims_str:
            basket.append("Claim_High_Protein")
        if "100% natural" in claims_str or "clean" in claims_str:
            basket.append("Claim_Clean_Label")
        if "diet" in claims_str or "fit" in claims_str or "slim" in claims_str:
            basket.append("Claim_Diet_Fit")
            
        # Nutritional Red Flags
        if row["sugars_g"] > 15.0:
            basket.append("Flag_High_Sugar")
        elif row["sugars_g"] <= 5.0:
            basket.append("Flag_Low_Sugar")
            
        if row["saturated_fat_g"] > 5.0:
            basket.append("Flag_High_Sat_Fat")
        elif row["saturated_fat_g"] <= 1.5:
            basket.append("Flag_Low_Sat_Fat")
            
        if row["sodium_mg"] > 600.0:
            basket.append("Flag_Excess_Sodium")
            
        if row["hidden_sugar_count"] >= 2:
            basket.append("Flag_Disguised_Sugars")
            
        if "maida" in ing_str or "refined wheat flour" in ing_str:
            basket.append("Flag_Maida_Dominant")
            
        if row["nutriscore_grade"] in ["d", "e"]:
            basket.append("NutriScore_D_or_E")
        elif row["nutriscore_grade"] in ["a", "b"]:
            basket.append("NutriScore_A_or_B")
            
        if row["verdict"] == "MISLEADING":
            basket.append("Verdict_MISLEADING")
        elif row["verdict"] == "HONEST":
            basket.append("Verdict_HONEST")
            
        transactions.append(basket)
        
    return transactions

def run_association_mining(min_support: float = 0.10, min_confidence: float = 0.50):
    csv_path = os.path.join(PROJECT_ROOT, "data/processed/openfoodfacts_india_processed.csv")
    df = pd.read_csv(csv_path)
    
    transactions = create_food_transactions(df)
    
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    encoded_df = pd.DataFrame(te_ary, columns=te.columns_)
    
    # 1. Benchmark Apriori vs FP-Growth
    t0 = time.time()
    apriori_frequent = apriori(encoded_df, min_support=min_support, use_colnames=True)
    t_apriori = time.time() - t0
    
    t0 = time.time()
    fpgrowth_frequent = fpgrowth(encoded_df, min_support=min_support, use_colnames=True)
    t_fpgrowth = time.time() - t0
    
    # 2. Generate Association Rules
    rules = association_rules(fpgrowth_frequent, metric="confidence", min_threshold=min_confidence)
    
    # Filter for Health-Washing Rules (Antecedent has Claim, Consequent has Flag or Misleading)
    hw_mask = rules.apply(
        lambda r: any("Claim_" in x for x in r["antecedents"]) and 
                  any("Flag_" in x or "Verdict_MISLEADING" in x for x in r["consequents"]),
        axis=1
    )
    hw_rules = rules[hw_mask].copy()
    hw_rules = hw_rules.sort_values(by=["lift", "confidence"], ascending=[False, False])
    
    # Format frozensets to readable strings
    hw_rules["antecedents_str"] = hw_rules["antecedents"].apply(lambda s: ", ".join(list(s)))
    hw_rules["consequents_str"] = hw_rules["consequents"].apply(lambda s: ", ".join(list(s)))
    
    # Round numerical metrics
    metrics = ["support", "confidence", "lift", "leverage", "conviction"]
    for m in metrics:
        hw_rules[m] = hw_rules[m].round(3)
        
    # Save rules
    out_csv = os.path.join(PROJECT_ROOT, "data/processed/health_washing_association_rules.csv")
    hw_rules.to_csv(out_csv, index=False)
    
    return {
        "apriori_time_ms": round(t_apriori * 1000, 2),
        "fpgrowth_time_ms": round(t_fpgrowth * 1000, 2),
        "total_frequent_itemsets": len(fpgrowth_frequent),
        "total_rules": len(rules),
        "health_washing_rules_count": len(hw_rules),
        "top_rules": hw_rules[[
            "antecedents_str", "consequents_str", "support", "confidence", "lift", "conviction"
        ]].head(10)
    }

if __name__ == "__main__":
    res = run_association_mining()
    print("====================================================================")
    print("ASSOCIATION RULE MINING: APRIORI vs FP-GROWTH")
    print("====================================================================")
    print(f"Apriori Runtime:   {res['apriori_time_ms']} ms")
    print(f"FP-Growth Runtime: {res['fpgrowth_time_ms']} ms")
    print(f"Total Itemsets:    {res['total_frequent_itemsets']}")
    print(f"Health-Washing Rules Mined: {res['health_washing_rules_count']}\n")
    print("--- Top 10 Mined Health-Washing Association Rules ---")
    print(res["top_rules"].to_string(index=False))
