-- ====================================================================
-- LABELTRUTH OLAP OPERATIONS (Roll-Up, Drill-Down, Slice, Dice, Pivot)
-- Course: Data Mining and Data Warehousing Lab (26B16CS313)
-- Target: SQLite & MySQL Compliant Analytical Queries
-- ====================================================================

-- --------------------------------------------------------------------
-- 1. ROLL-UP OPERATION
-- Aggregation across the Category Hierarchy (Product Type -> Sub-Category -> Category)
-- --------------------------------------------------------------------
SELECT 
    c.category_name,
    c.sub_category,
    COUNT(f.product_id) AS total_products,
    SUM(f.has_health_claim) AS products_with_claims,
    SUM(f.is_misleading) AS misleading_count,
    ROUND(AVG(f.sugars_g), 1) AS avg_sugar_g,
    ROUND(AVG(f.sugar_teaspoons), 1) AS avg_sugar_tsp,
    ROUND(AVG(f.saturated_fat_g), 1) AS avg_sat_fat_g,
    ROUND(AVG(f.sodium_mg), 0) AS avg_sodium_mg,
    ROUND((CAST(SUM(f.is_misleading) AS REAL) / NULLIF(SUM(f.has_health_claim), 0)) * 100, 1) AS category_hwi_pct
FROM Fact_Product_Nutrition f
JOIN Dim_Category c ON f.category_id = c.category_id
GROUP BY c.category_name, c.sub_category
ORDER BY c.category_name, misleading_count DESC;

-- --------------------------------------------------------------------
-- 2. DRILL-DOWN OPERATION
-- Expanding from Category level down to individual Brands within "Snacks & Bakery"
-- --------------------------------------------------------------------
SELECT 
    c.category_name,
    b.brand_name,
    b.parent_company,
    COUNT(f.product_id) AS total_products,
    SUM(f.is_misleading) AS misleading_products,
    ROUND(AVG(f.sugars_g), 1) AS avg_sugars_g,
    ROUND(AVG(f.fat_g), 1) AS avg_fat_g,
    ROUND(AVG(f.hidden_sugar_count), 1) AS avg_hidden_sugar_aliases,
    ROUND((CAST(SUM(f.is_misleading) AS REAL) / NULLIF(SUM(f.has_health_claim), 0)) * 100, 1) AS brand_hwi_pct
FROM Fact_Product_Nutrition f
JOIN Dim_Category c ON f.category_id = c.category_id
JOIN Dim_Brand b ON f.brand_id = b.brand_id
WHERE c.category_name = 'Snacks & Bakery'
GROUP BY c.category_name, b.brand_name
ORDER BY brand_hwi_pct DESC, misleading_products DESC;

-- --------------------------------------------------------------------
-- 3. SLICE OPERATION
-- Analyzing a single dimensional slice: Category = 'Breakfast Cereals'
-- --------------------------------------------------------------------
SELECT 
    f.product_name,
    b.brand_name,
    c.product_type,
    n.grade AS nutriscore,
    f.sugars_g,
    f.sugar_teaspoons,
    f.fiber_g,
    f.protein_g,
    f.hidden_sugar_count,
    f.verdict
FROM Fact_Product_Nutrition f
JOIN Dim_Category c ON f.category_id = c.category_id
JOIN Dim_Brand b ON f.brand_id = b.brand_id
JOIN Dim_NutriScore n ON f.nutriscore_id = n.nutriscore_id
WHERE c.category_name = 'Breakfast Cereals'
ORDER BY f.sugars_g DESC;

-- --------------------------------------------------------------------
-- 4. DICE OPERATION
-- Multi-dimensional sub-cube:
-- Categories IN ('Snacks & Bakery', 'Beverages')
-- Nutri-Score IN ('D', 'E')
-- Has Health Claim = 1
-- --------------------------------------------------------------------
SELECT 
    f.product_id,
    f.product_name,
    b.brand_name,
    c.category_name,
    n.grade AS nutriscore,
    f.sugars_g,
    f.sugar_teaspoons,
    f.fat_g,
    f.saturated_fat_g,
    f.hidden_sugar_count,
    f.verdict
FROM Fact_Product_Nutrition f
JOIN Dim_Category c ON f.category_id = c.category_id
JOIN Dim_Brand b ON f.brand_id = b.brand_id
JOIN Dim_NutriScore n ON f.nutriscore_id = n.nutriscore_id
WHERE c.category_name IN ('Snacks & Bakery', 'Beverages')
  AND n.grade IN ('D', 'E')
  AND f.has_health_claim = 1
ORDER BY f.sugars_g DESC;

-- --------------------------------------------------------------------
-- 5. PIVOT OPERATION (Cross-Tabulation)
-- Matrix of Category vs. Nutri-Score Grade showing product counts
-- --------------------------------------------------------------------
SELECT 
    c.category_name,
    SUM(CASE WHEN n.grade = 'A' THEN 1 ELSE 0 END) AS Grade_A,
    SUM(CASE WHEN n.grade = 'B' THEN 1 ELSE 0 END) AS Grade_B,
    SUM(CASE WHEN n.grade = 'C' THEN 1 ELSE 0 END) AS Grade_C,
    SUM(CASE WHEN n.grade = 'D' THEN 1 ELSE 0 END) AS Grade_D,
    SUM(CASE WHEN n.grade = 'E' THEN 1 ELSE 0 END) AS Grade_E,
    COUNT(f.product_id) AS Total_Products,
    ROUND(AVG(f.sugar_teaspoons), 1) AS Avg_Sugar_Teaspoons
FROM Fact_Product_Nutrition f
JOIN Dim_Category c ON f.category_id = c.category_id
JOIN Dim_NutriScore n ON f.nutriscore_id = n.nutriscore_id
GROUP BY c.category_name
ORDER BY Grade_E DESC;
