"""
ETL Pipeline for LabelTruth Data Warehouse
Extracts from Open Food Facts India raw CSV, transforms, validates,
computes health-washing metrics, and loads into SQLite / MySQL Star-Snowflake Warehouse.
"""

import os
import re
import sqlite3
import pandas as pd
import numpy as np

# Disguised / hidden sugar keywords dictionary
HIDDEN_SUGAR_SYNONYMS = [
    "maltodextrin", "invert sugar", "invert syrup", "liquid glucose", "glucose syrup",
    "high fructose corn syrup", "corn syrup", "dextrose", "barley malt extract",
    "malt extract", "cane sugar", "raw sugar", "brown sugar", "sucrose",
    "fructose", "maltose", "honey", "date syrup", "agave", "fruit juice concentrate",
    "concentrated juice", "caramel", "ethyl maltol", "treacle", "molasses",
    "dextran", "galactose", "corn syrup solids", "rice syrup"
]

def detect_hidden_sugars(ingredients_text: str):
    """Counts number of distinct hidden sugar aliases found in ingredient text."""
    if not isinstance(ingredients_text, str):
        return 0, []
    text_lower = ingredients_text.lower()
    found = [syn for syn in HIDDEN_SUGAR_SYNONYMS if syn in text_lower]
    return len(found), found

def determine_verdict(row):
    """
    Classifies product verdict:
    - MISLEADING: Claims made, but product exceeds FSSAI/WHO thresholds for sugar, sat fat, sodium or lacks fibre
    - UNHEALTHY: Poor nutritional characteristics (Nutri-Score D/E or high sugar/fat) without claim
    - HONEST: Good nutritional characteristics or claims match actual nutrition profile
    """
    claims = str(row["front_claims"]).strip()
    has_claim = claims != "" and claims.lower() != "none"
    
    is_liquid = row["category"] == "Beverages"
    high_sugar = row["sugars_g"] > (7.5 if is_liquid else 15.0)
    high_sat_fat = row["saturated_fat_g"] > 5.0
    high_sodium = row["sodium_mg"] > 600.0
    
    fibre_claim = "fibre" in claims.lower() or "fiber" in claims.lower() or "multigrain" in claims.lower() or "whole wheat" in claims.lower()
    low_fibre = row["fiber_g"] < 4.0
    
    sugar_claim = "no added sugar" in claims.lower() or "sugar free" in claims.lower() or "low sugar" in claims.lower()
    violates_sugar = sugar_claim and (row["added_sugars_g"] > 1.0 or row["sugars_g"] > 10.0)
    
    baked_claim = "baked" in claims.lower()
    violates_fat = baked_claim and (row["fat_g"] > 15.0 or high_sat_fat)

    failing_criteria = False
    if has_claim:
        if sugar_claim and violates_sugar:
            failing_criteria = True
        elif fibre_claim and (low_fibre or high_sugar or high_sat_fat):
            failing_criteria = True
        elif baked_claim and violates_fat:
            failing_criteria = True
        elif high_sugar or high_sat_fat or high_sodium or row["nutriscore_grade"] in ["d", "e"]:
            failing_criteria = True

    if has_claim and failing_criteria:
        return "MISLEADING", 1, 1
    elif not has_claim and (high_sugar or high_sat_fat or high_sodium or row["nutriscore_grade"] in ["d", "e"]):
        return "UNHEALTHY", 0, 0
    else:
        return "HONEST", int(has_claim), 0

def run_etl():
    raw_path = "data/raw/openfoodfacts_india_raw.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}")

    print("Step 1: Extracting raw Open Food Facts India data...")
    df = pd.read_csv(raw_path)
    print(f"Loaded {len(df)} records.")

    print("Step 2: Transforming and applying health-washing detection...")
    # Missing value imputation if any
    numeric_cols = ["energy_kcal", "carbohydrates_g", "sugars_g", "added_sugars_g", "fat_g", "saturated_fat_g", "protein_g", "fiber_g", "sodium_mg"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

    # Sugar in teaspoons
    df["sugar_teaspoons"] = (df["sugars_g"] / 4.0).round(2)

    # Hidden sugar detection
    hs_counts = []
    hs_lists = []
    for ing in df["ingredients_text"]:
        cnt, found = detect_hidden_sugars(ing)
        hs_counts.append(cnt)
        hs_lists.append("; ".join(found))
    df["hidden_sugar_count"] = hs_counts
    df["hidden_sugars_found"] = hs_lists

    # Verdict and health claim flags
    verdicts = []
    has_claims = []
    is_misleadings = []
    for _, row in df.iterrows():
        v, hc, im = determine_verdict(row)
        verdicts.append(v)
        has_claims.append(hc)
        is_misleadings.append(im)

    df["verdict"] = verdicts
    df["has_health_claim"] = has_claims
    df["is_misleading"] = is_misleadings

    # Save processed CSV
    processed_path = "data/processed/openfoodfacts_india_processed.csv"
    df.to_csv(processed_path, index=False)
    print(f"Processed dataset saved to {processed_path}")

    print("Step 3: Building SQLite Data Warehouse (Star / Snowflake Schema)...")
    db_path = "database/labeltruth_dw.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Execute DDL schema
    with open("database/schema.sql", "r") as f:
        cursor.executescript(f.read())

    # Populate Dim_Brand
    brands = df["brand"].unique()
    brand_map = {}
    for idx, b in enumerate(sorted(brands), start=1):
        parent = "PepsiCo India" if b in ["Lay's", "Kurkure", "Tropicana", "Quaker"] else \
                 "ITC Limited" if "ITC" in b or b in ["Sunfeast", "B Natural"] else \
                 "Nestle S.A." if "Nestle" in b else \
                 "Britannia Industries" if "Britannia" in b else \
                 "Parle Products" if "Parle" in b else \
                 "Mondelez International" if b in ["Mondelez", "Cadbury"] else \
                 "Marico Ltd" if b in ["Marico", "Saffola"] else \
                 "Gujarat Co-operative Milk (GCMMF)" if "Amul" in b else \
                 f"{b} Group"
        cursor.execute("""
            INSERT INTO Dim_Brand (brand_id, brand_name, parent_company, country_of_origin)
            VALUES (?, ?, ?, 'India')
        """, (idx, b, parent))
        brand_map[b] = idx

    # Populate Dim_Category (Snowflake Hierarchy)
    category_df = df[["product_type", "sub_category", "category"]].drop_duplicates()
    cat_map = {}
    for idx, row in enumerate(category_df.itertuples(index=False), start=1):
        ptype, subcat, cat = row
        cursor.execute("""
            INSERT INTO Dim_Category (category_id, product_type, sub_category, category_name)
            VALUES (?, ?, ?, ?)
        """, (idx, ptype, subcat, cat))
        cat_map[(ptype, subcat, cat)] = idx

    # Populate Dim_NutriScore
    nutriscore_meta = [
        (1, "A", -15, -1, "#038141", "Very Good Nutritional Quality"),
        (2, "B", 0, 2, "#85BB2F", "Good Nutritional Quality"),
        (3, "C", 3, 10, "#FECB02", "Moderate Nutritional Quality"),
        (4, "D", 11, 18, "#EE8100", "Poor Nutritional Quality"),
        (5, "E", 19, 40, "#E63E11", "Very Unfavourable Nutritional Quality")
    ]
    for n in nutriscore_meta:
        cursor.execute("""
            INSERT INTO Dim_NutriScore (nutriscore_id, grade, score_min, score_max, color_hex, health_grade_desc)
            VALUES (?, ?, ?, ?, ?, ?)
        """, n)
    grade_to_id = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}

    # Populate Dim_Time
    cursor.execute("""
        INSERT INTO Dim_Time (time_id, year, quarter, month, month_name, academic_semester)
        VALUES (1, 2026, 'Q4', 11, 'November', 'Odd Semester 2026')
    """)

    # Populate Dim_Claims and collect claims for bridge table
    all_claim_tags = set()
    for c_str in df["front_claims"]:
        for c in str(c_str).split(","):
            c_clean = c.strip()
            if c_clean and c_clean.lower() != "none":
                all_claim_tags.add(c_clean)
    all_claim_tags.add("None")

    claim_map = {}
    for idx, c in enumerate(sorted(all_claim_tags), start=1):
        ctype = "Nutrient Claim" if any(w in c.lower() for w in ["fibre", "protein", "fat", "sugar", "calcium", "vitamin", "trans fat"]) else \
                "Processing Claim" if any(w in c.lower() for w in ["baked", "roasted", "cold pressed"]) else \
                "Clean Label" if any(w in c.lower() for w in ["100%", "clean", "natural", "no preservatives"]) else \
                "General Health Claim"
        cursor.execute("INSERT INTO Dim_Claims (claim_id, claim_tag, claim_type) VALUES (?, ?, ?)", (idx, c, ctype))
        claim_map[c] = idx

    # Populate Fact_Product_Nutrition and Bridge_Product_Claims
    for _, row in df.iterrows():
        b_id = brand_map[row["brand"]]
        c_id = cat_map[(row["product_type"], row["sub_category"], row["category"])]
        n_id = grade_to_id[str(row["nutriscore_grade"]).lower()]

        cursor.execute("""
            INSERT INTO Fact_Product_Nutrition (
                product_id, product_name, brand_id, category_id, nutriscore_id, time_id,
                energy_kcal, carbohydrates_g, sugars_g, added_sugars_g, sugar_teaspoons,
                fat_g, saturated_fat_g, trans_fat_g, protein_g, fiber_g, sodium_mg, salt_g,
                serving_size_g, nova_group, trust_score, hidden_sugar_count,
                has_health_claim, is_misleading, verdict
            ) VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(row["product_id"]),
            str(row["product_name"]),
            b_id,
            c_id,
            n_id,
            float(row["energy_kcal"]),
            float(row["carbohydrates_g"]),
            float(row["sugars_g"]),
            float(row["added_sugars_g"]),
            float(row["sugar_teaspoons"]),
            float(row["fat_g"]),
            float(row["saturated_fat_g"]),
            float(row["trans_fat_g"]),
            float(row["protein_g"]),
            float(row["fiber_g"]),
            float(row["sodium_mg"]),
            float(row["salt_g"]),
            float(row["serving_size_g"]),
            int(row["nova_group"]),
            float(row["trust_score"]),
            int(row["hidden_sugar_count"]),
            int(row["has_health_claim"]),
            int(row["is_misleading"]),
            str(row["verdict"])
        ))

        # Bridge table mapping
        claims_in_row = [c.strip() for c in str(row["front_claims"]).split(",") if c.strip() and c.strip().lower() != "none"]
        if not claims_in_row:
            claims_in_row = ["None"]
        for c_tag in claims_in_row:
            if c_tag in claim_map:
                cursor.execute("""
                    INSERT OR IGNORE INTO Bridge_Product_Claims (product_id, claim_id)
                    VALUES (?, ?)
                """, (str(row["product_id"]), claim_map[c_tag]))

    conn.commit()

    # Export Snacks & Beverages Data Mart to CSV for Power BI / OLAP
    mart_df = pd.read_sql_query("SELECT * FROM View_Snacks_Beverages_Mart", conn)
    mart_path = "data/data_marts/mart_snacks_beverages.csv"
    mart_df.to_csv(mart_path, index=False)
    print(f"Data Mart exported: {len(mart_df)} rows to {mart_path}")

    # Summary statistics
    fact_count = cursor.execute("SELECT COUNT(*) FROM Fact_Product_Nutrition").fetchone()[0]
    misleading_count = cursor.execute("SELECT COUNT(*) FROM Fact_Product_Nutrition WHERE verdict = 'MISLEADING'").fetchone()[0]
    honest_count = cursor.execute("SELECT COUNT(*) FROM Fact_Product_Nutrition WHERE verdict = 'HONEST'").fetchone()[0]
    unhealthy_count = cursor.execute("SELECT COUNT(*) FROM Fact_Product_Nutrition WHERE verdict = 'UNHEALTHY'").fetchone()[0]
    claims_count = cursor.execute("SELECT COUNT(*) FROM Fact_Product_Nutrition WHERE has_health_claim = 1").fetchone()[0]

    hwi = (misleading_count / claims_count * 100.0) if claims_count > 0 else 0.0

    print(f"\n--- Data Warehouse Summary ---")
    print(f"Total Products Loaded: {fact_count}")
    print(f"Products with Claims:  {claims_count}")
    print(f"MISLEADING Products:   {misleading_count}")
    print(f"HONEST Products:       {honest_count}")
    print(f"UNHEALTHY Products:    {unhealthy_count}")
    print(f"Health-Washing Index:  {hwi:.1f}%\n")

    conn.close()

if __name__ == "__main__":
    run_etl()
