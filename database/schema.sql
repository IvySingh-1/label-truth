-- ====================================================================
-- LABELTRUTH DATA WAREHOUSE SCHEMA
-- Course: Data Mining and Data Warehousing Lab (26B16CS313)
-- Architecture: Hybrid Star / Snowflake Schema for OLAP Processing
-- Compatible with: SQLite 3 and MySQL 8.x
-- ====================================================================

-- 1. Brand Dimension
CREATE TABLE IF NOT EXISTS Dim_Brand (
    brand_id INTEGER PRIMARY KEY,
    brand_name VARCHAR(100) NOT NULL UNIQUE,
    parent_company VARCHAR(100),
    country_of_origin VARCHAR(50) DEFAULT 'India',
    is_fssai_registered INTEGER DEFAULT 1
);

-- 2. Category Dimension (Snowflake Hierarchy: ProductType -> SubCategory -> Category)
CREATE TABLE IF NOT EXISTS Dim_Category (
    category_id INTEGER PRIMARY KEY,
    product_type VARCHAR(100) NOT NULL,
    sub_category VARCHAR(100) NOT NULL,
    category_name VARCHAR(100) NOT NULL,
    storage_type VARCHAR(50) DEFAULT 'Ambient Shelf'
);

-- 3. Claims Dimension
CREATE TABLE IF NOT EXISTS Dim_Claims (
    claim_id INTEGER PRIMARY KEY,
    claim_tag VARCHAR(100) NOT NULL UNIQUE,
    claim_type VARCHAR(50) NOT NULL,
    fssai_regulated INTEGER DEFAULT 1
);

-- 4. Nutri-Score Dimension
CREATE TABLE IF NOT EXISTS Dim_NutriScore (
    nutriscore_id INTEGER PRIMARY KEY,
    grade VARCHAR(2) NOT NULL UNIQUE,
    score_min INTEGER,
    score_max INTEGER,
    color_hex VARCHAR(10),
    health_grade_desc VARCHAR(50)
);

-- 5. Time Dimension
CREATE TABLE IF NOT EXISTS Dim_Time (
    time_id INTEGER PRIMARY KEY,
    year INTEGER NOT NULL,
    quarter VARCHAR(5) NOT NULL,
    month INTEGER NOT NULL,
    month_name VARCHAR(20) NOT NULL,
    academic_semester VARCHAR(20) DEFAULT 'Odd Semester 2026'
);

-- 6. Fact Table: Nutritional & Health-Washing Metrics
CREATE TABLE IF NOT EXISTS Fact_Product_Nutrition (
    fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id VARCHAR(50) NOT NULL UNIQUE,
    product_name VARCHAR(200) NOT NULL,
    brand_id INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    nutriscore_id INTEGER NOT NULL,
    time_id INTEGER NOT NULL DEFAULT 1,
    
    -- Nutritional measures (per 100g)
    energy_kcal REAL NOT NULL,
    carbohydrates_g REAL NOT NULL,
    sugars_g REAL NOT NULL,
    added_sugars_g REAL NOT NULL,
    sugar_teaspoons REAL NOT NULL,
    fat_g REAL NOT NULL,
    saturated_fat_g REAL NOT NULL,
    trans_fat_g REAL DEFAULT 0.0,
    protein_g REAL NOT NULL,
    fiber_g REAL NOT NULL,
    sodium_mg REAL NOT NULL,
    salt_g REAL NOT NULL,
    serving_size_g REAL DEFAULT 30.0,
    
    -- Processing and Trust measures
    nova_group INTEGER NOT NULL,
    trust_score REAL NOT NULL,
    hidden_sugar_count INTEGER DEFAULT 0,
    
    -- Health-washing flags and classification
    has_health_claim INTEGER DEFAULT 0,
    is_misleading INTEGER DEFAULT 0,
    verdict VARCHAR(20) NOT NULL, -- 'HONEST', 'MISLEADING', 'UNHEALTHY'
    
    FOREIGN KEY (brand_id) REFERENCES Dim_Brand(brand_id),
    FOREIGN KEY (category_id) REFERENCES Dim_Category(category_id),
    FOREIGN KEY (nutriscore_id) REFERENCES Dim_NutriScore(nutriscore_id),
    FOREIGN KEY (time_id) REFERENCES Dim_Time(time_id)
);

-- 7. Bridge Table for Many-to-Many Product-to-Claims Mapping
CREATE TABLE IF NOT EXISTS Bridge_Product_Claims (
    product_id VARCHAR(50) NOT NULL,
    claim_id INTEGER NOT NULL,
    PRIMARY KEY (product_id, claim_id),
    FOREIGN KEY (product_id) REFERENCES Fact_Product_Nutrition(product_id),
    FOREIGN KEY (claim_id) REFERENCES Dim_Claims(claim_id)
);

-- 8. Dedicated Data Mart: Snacks & Beverages Analysis
CREATE VIEW IF NOT EXISTS View_Snacks_Beverages_Mart AS
SELECT 
    f.product_id,
    f.product_name,
    b.brand_name,
    c.category_name,
    c.sub_category,
    c.product_type,
    n.grade AS nutriscore_grade,
    f.energy_kcal,
    f.sugars_g,
    f.sugar_teaspoons,
    f.fat_g,
    f.saturated_fat_g,
    f.fiber_g,
    f.protein_g,
    f.sodium_mg,
    f.hidden_sugar_count,
    f.has_health_claim,
    f.is_misleading,
    f.verdict,
    f.trust_score
FROM Fact_Product_Nutrition f
JOIN Dim_Brand b ON f.brand_id = b.brand_id
JOIN Dim_Category c ON f.category_id = c.category_id
JOIN Dim_NutriScore n ON f.nutriscore_id = n.nutriscore_id
WHERE c.category_name IN ('Snacks & Bakery', 'Beverages');
