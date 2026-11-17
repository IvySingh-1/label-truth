"""
LabelTruth OLAP Operations Execution Script
Executes Roll-Up, Drill-Down, Slice, Dice, and Pivot queries against SQLite DW.
"""

import sqlite3
import pandas as pd

def execute_olap():
    db_path = "database/labeltruth_dw.db"
    conn = sqlite3.connect(db_path)

    queries = {
        "1. ROLL-UP (Category Hierarchy)": """
            SELECT 
                c.category_name,
                c.sub_category,
                COUNT(f.product_id) AS total_items,
                SUM(f.has_health_claim) AS with_claims,
                SUM(f.is_misleading) AS misleading,
                ROUND(AVG(f.sugars_g), 1) AS avg_sugars,
                ROUND(AVG(f.sugar_teaspoons), 1) AS avg_sugar_tsp,
                ROUND((CAST(SUM(f.is_misleading) AS REAL) / NULLIF(SUM(f.has_health_claim), 0)) * 100, 1) AS hwi_pct
            FROM Fact_Product_Nutrition f
            JOIN Dim_Category c ON f.category_id = c.category_id
            GROUP BY c.category_name, c.sub_category
            ORDER BY c.category_name
        """,
        "2. DRILL-DOWN (Snacks & Bakery Brands)": """
            SELECT 
                b.brand_name,
                COUNT(f.product_id) AS total_products,
                SUM(f.is_misleading) AS misleading,
                ROUND(AVG(f.sugars_g), 1) AS avg_sugar,
                ROUND(AVG(f.fat_g), 1) AS avg_fat,
                ROUND(AVG(f.hidden_sugar_count), 1) AS avg_hidden_sugar_aliases,
                ROUND((CAST(SUM(f.is_misleading) AS REAL) / NULLIF(SUM(f.has_health_claim), 0)) * 100, 1) AS hwi_pct
            FROM Fact_Product_Nutrition f
            JOIN Dim_Category c ON f.category_id = c.category_id
            JOIN Dim_Brand b ON f.brand_id = b.brand_id
            WHERE c.category_name = 'Snacks & Bakery'
            GROUP BY b.brand_name
            ORDER BY misleading DESC, avg_sugar DESC
            LIMIT 8
        """,
        "3. SLICE (Category = 'Breakfast Cereals')": """
            SELECT 
                f.product_name,
                b.brand_name,
                n.grade AS nutriscore,
                f.sugars_g,
                f.sugar_teaspoons,
                f.fiber_g,
                f.protein_g,
                f.verdict
            FROM Fact_Product_Nutrition f
            JOIN Dim_Category c ON f.category_id = c.category_id
            JOIN Dim_Brand b ON f.brand_id = b.brand_id
            JOIN Dim_NutriScore n ON f.nutriscore_id = n.nutriscore_id
            WHERE c.category_name = 'Breakfast Cereals'
            ORDER BY f.sugars_g DESC
            LIMIT 8
        """,
        "4. DICE (Snacks/Beverages + Nutri-Score D/E + With Claims)": """
            SELECT 
                f.product_name,
                b.brand_name,
                c.category_name,
                n.grade AS nutriscore,
                f.sugars_g,
                f.fat_g,
                f.hidden_sugar_count,
                f.verdict
            FROM Fact_Product_Nutrition f
            JOIN Dim_Category c ON f.category_id = c.category_id
            JOIN Dim_Brand b ON f.brand_id = b.brand_id
            JOIN Dim_NutriScore n ON f.nutriscore_id = n.nutriscore_id
            WHERE c.category_name IN ('Snacks & Bakery', 'Beverages')
              AND n.grade IN ('D', 'E')
              AND f.has_health_claim = 1
            ORDER BY f.sugars_g DESC
            LIMIT 8
        """,
        "5. PIVOT (Cross-Tab: Category vs Nutri-Score Grade)": """
            SELECT 
                c.category_name,
                SUM(CASE WHEN n.grade = 'A' THEN 1 ELSE 0 END) AS Grade_A,
                SUM(CASE WHEN n.grade = 'B' THEN 1 ELSE 0 END) AS Grade_B,
                SUM(CASE WHEN n.grade = 'C' THEN 1 ELSE 0 END) AS Grade_C,
                SUM(CASE WHEN n.grade = 'D' THEN 1 ELSE 0 END) AS Grade_D,
                SUM(CASE WHEN n.grade = 'E' THEN 1 ELSE 0 END) AS Grade_E,
                ROUND(AVG(f.sugar_teaspoons), 1) AS Avg_Sugar_Tsp
            FROM Fact_Product_Nutrition f
            JOIN Dim_Category c ON f.category_id = c.category_id
            JOIN Dim_NutriScore n ON f.nutriscore_id = n.nutriscore_id
            GROUP BY c.category_name
            ORDER BY Grade_E DESC
        """
    }

    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 1000)

    for op_name, sql in queries.items():
        print(f"\n=======================================================")
        print(f"OLAP OPERATION: {op_name}")
        print(f"=======================================================")
        res = pd.read_sql_query(sql, conn)
        print(res.to_string(index=False))

    conn.close()

if __name__ == "__main__":
    execute_olap()
