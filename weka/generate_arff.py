"""
WEKA ARFF Dataset Generator
Exports Open Food Facts India data into standard Weka ARFF (Attribute-Relation File Format)
Supporting DMDW Lab 1-2 (WEKA Explorer, J48 Classifier & 10-Fold Cross-Validation).
"""

import os
import pandas as pd

def create_arff_file(df: pd.DataFrame, output_path: str, relation_name: str, attributes: list, class_col: str):
    """Generates an authentic ARFF file compatible with WEKA 3.8 / 3.9."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(f"% ====================================================================\n")
        f.write(f"% WEKA ARFF DATASET: {relation_name}\n")
        f.write(f"% Course: Data Mining and Data Warehousing Lab (26B16CS313)\n")
        f.write(f"% Project: LabelTruth - Detecting Health-Washing in Indian Packaged Food\n")
        f.write(f"% ====================================================================\n\n")
        f.write(f"@relation {relation_name}\n\n")
        
        # Write attributes
        for col_name, col_type in attributes:
            if isinstance(col_type, list):
                classes = "{" + ",".join([str(c) for c in col_type]) + "}"
                f.write(f"@attribute {col_name} {classes}\n")
            else:
                f.write(f"@attribute {col_name} {col_type}\n")
                
        f.write("\n@data\n")
        
        # Write rows
        cols = [a[0] for a in attributes]
        for _, row in df[cols].iterrows():
            line_vals = []
            for col, attr in attributes:
                val = row[col]
                if pd.isna(val):
                    line_vals.append("?")
                elif isinstance(attr, list) or attr == "string":
                    # Escape or quote if string
                    s = str(val).replace("'", "").replace('"', '').strip()
                    line_vals.append(f"'{s}'" if " " in s else s)
                else:
                    line_vals.append(str(round(float(val), 2)))
            f.write(",".join(line_vals) + "\n")
            
    print(f"Generated ARFF: {output_path} ({len(df)} instances)")

def generate_all_arff():
    csv_path = "data/processed/openfoodfacts_india_processed.csv"
    df = pd.read_csv(csv_path)
    
    # 1. Full Dataset ARFF
    attrs_full = [
        ("category", sorted(df["category"].unique().tolist())),
        ("energy_kcal", "numeric"),
        ("sugars_g", "numeric"),
        ("sugar_teaspoons", "numeric"),
        ("fat_g", "numeric"),
        ("saturated_fat_g", "numeric"),
        ("protein_g", "numeric"),
        ("fiber_g", "numeric"),
        ("sodium_mg", "numeric"),
        ("nova_group", "numeric"),
        ("hidden_sugar_count", "numeric"),
        ("has_health_claim", ["0", "1"]),
        ("verdict", ["HONEST", "MISLEADING", "UNHEALTHY"])
    ]
    df["has_health_claim_str"] = df["has_health_claim"].astype(str)
    attrs_full[11] = ("has_health_claim_str", ["0", "1"])
    create_arff_file(df, "weka/datasets/openfoodfacts_india.arff", "LabelTruth_OpenFoodFacts_India", attrs_full, "verdict")
    
    # 2. Claims Verification ARFF for J48 Decision Tree
    attrs_j48 = [
        ("energy_kcal", "numeric"),
        ("sugars_g", "numeric"),
        ("fat_g", "numeric"),
        ("saturated_fat_g", "numeric"),
        ("protein_g", "numeric"),
        ("fiber_g", "numeric"),
        ("sodium_mg", "numeric"),
        ("hidden_sugar_count", "numeric"),
        ("verdict", ["HONEST", "MISLEADING", "UNHEALTHY"])
    ]
    create_arff_file(df, "weka/datasets/claims_verification.arff", "LabelTruth_J48_Claims_Verification", attrs_j48, "verdict")
    
    # 3. NutriScore Classification ARFF
    df["nutriscore_upper"] = df["nutriscore_grade"].str.upper()
    attrs_nutri = [
        ("category", sorted(df["category"].unique().tolist())),
        ("energy_kcal", "numeric"),
        ("sugars_g", "numeric"),
        ("fat_g", "numeric"),
        ("saturated_fat_g", "numeric"),
        ("protein_g", "numeric"),
        ("fiber_g", "numeric"),
        ("sodium_mg", "numeric"),
        ("nutriscore_upper", ["A", "B", "C", "D", "E"])
    ]
    create_arff_file(df, "weka/datasets/nutriscore_classification.arff", "LabelTruth_NutriScore_Classification", attrs_nutri, "nutriscore_upper")

if __name__ == "__main__":
    generate_all_arff()
