"""
Packaging OCR Label Scanner & Information Extraction Module
Extracts nutritional facts, ingredients, and claims from food packaging images.
Uses regular expressions and domain heuristics to parse OCR text,
then pipes extracted nutrients into the Claim Verifier and Hidden Sugar Detector.
Supports DMDW Lab Module 8 (OCR, Image Processing & Text Extraction).
"""

import os
import re
import sys
from typing import Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rules_engine.hidden_sugars import analyze_hidden_sugars
from src.rules_engine.claim_verifier import audit_all_product_claims
from src.rules_engine.teaspoon_meter import calculate_sugar_teaspoons

# Regex patterns for nutritional label parsing
REGEX_PATTERNS = {
    "energy_kcal": [
        r"energy\s*(?:kcal)?\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)",
        r"calories\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)"
    ],
    "carbohydrates_g": [
        r"carbohydrate[s]?\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*g?",
        r"carbs\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*g?"
    ],
    "sugars_g": [
        r"(?:total\s+)?sugar[s]?\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*g?"
    ],
    "added_sugars_g": [
        r"added\s+sugar[s]?\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*g?"
    ],
    "fat_g": [
        r"(?:total\s+)?fat\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*g?"
    ],
    "saturated_fat_g": [
        r"saturated\s+fat\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*g?",
        r"sat(?:\.|\s+)fat\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*g?"
    ],
    "trans_fat_g": [
        r"trans\s+fat\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*g?"
    ],
    "protein_g": [
        r"protein[s]?\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*g?"
    ],
    "fiber_g": [
        r"(?:dietary\s+)?fib(?:re|er)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*g?"
    ],
    "sodium_mg": [
        r"sodium\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*mg",
        r"salt\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*g"
    ]
}

SAMPLE_PRESETS = {
    "Britannia NutriChoice Digestive Biscuit": """
PRODUCT: Britannia NutriChoice Digestive Biscuits
FRONT CLAIMS: High Fibre, Digestive, 100% Whole Wheat Flour, 0% Trans Fat
NUTRITIONAL FACTS (Per 100g):
Energy: 482 kcal
Carbohydrates: 68.0g
Total Sugars: 14.5g
Added Sugars: 13.0g
Total Fat: 20.0g
Saturated Fat: 9.5g
Trans Fat: 0.05g
Protein: 7.5g
Dietary Fibre: 6.2g
Sodium: 460 mg
INGREDIENTS: Refined Wheat Flour (Maida) 40%, Whole Wheat Flour (Atta) 25%, Refined Palm Oil, Sugar, Wheat Bran, Invert Sugar Syrup, Liquid Glucose, Maltodextrin, Raising Agents, Salt, Emulsifiers.
""",
    "Kellogg's Chocos Crunchy Bites": """
PRODUCT: Kellogg's Chocos Crunchy Bites
FRONT CLAIMS: Roti Jaisi Shakti, Made with Whole Grain, High Calcium & Protein
NUTRITIONAL FACTS (Per 100g):
Energy: 388 kcal
Carbohydrates: 83.5g
Total Sugars: 31.0g
Added Sugars: 29.5g
Total Fat: 2.8g
Saturated Fat: 1.2g
Protein: 7.2g
Dietary Fibre: 4.2g
Sodium: 380 mg
INGREDIENTS: Whole Grain Wheat Flour (Atta) 29%, Refined Wheat Flour (Maida) 29%, Sugar, Cocoa Solids, High Fructose Corn Syrup, Palmolein Oil, Malt Extract, Salt.
""",
    "Dabur Real Fruit Power Mixed Fruit Juice": """
PRODUCT: Real Fruit Power Mixed Fruit
FRONT CLAIMS: Real Goodness of 9 Fruits, No Added Preservatives, Rich in Vitamin C
NUTRITIONAL FACTS (Per 100g):
Energy: 56 kcal
Carbohydrates: 14.0g
Total Sugars: 13.5g
Added Sugars: 9.5g
Total Fat: 0.0g
Saturated Fat: 0.0g
Protein: 0.1g
Dietary Fibre: 0.2g
Sodium: 22 mg
INGREDIENTS: Water, Mixed Fruit Concentrate 18%, Sugar, Liquid Glucose, Acidity Regulator (330), Antioxidant (300).
""",
    "Too Yumm! Karare Munchy Masala": """
PRODUCT: Too Yumm! Karare Munchy Masala
FRONT CLAIMS: Baked Not Fried, 40% Less Fat, Multigrain Crunch, Guilt Free Snacking
NUTRITIONAL FACTS (Per 100g):
Energy: 465 kcal
Carbohydrates: 68.0g
Total Sugars: 4.5g
Added Sugars: 3.8g
Total Fat: 18.5g
Saturated Fat: 8.5g
Protein: 6.8g
Dietary Fibre: 3.5g
Sodium: 960 mg
INGREDIENTS: Rice Flour, Corn Grits, Palmolein Oil, Seasoning, Sugar, Maltodextrin, Salt, Spices, Flavour Enhancers (627, 631).
""",
    "The Whole Truth Clean Protein Bar": """
PRODUCT: The Whole Truth Double Cocoa Protein Bar
FRONT CLAIMS: No Added Sugar, 100% Clean Label, Whey Isolate, 6 Real Ingredients Only
NUTRITIONAL FACTS (Per 100g):
Energy: 395 kcal
Carbohydrates: 42.0g
Total Sugars: 26.0g
Added Sugars: 0.0g
Total Fat: 14.0g
Saturated Fat: 4.5g
Protein: 25.0g
Dietary Fibre: 7.5g
Sodium: 85 mg
INGREDIENTS: Dates (40%), Whey Protein Isolate (28%), Cashews (16%), Cocoa Butter, Cocoa Powder, Almonds.
"""
}

def parse_label_text(text: str) -> Dict[str, Any]:
    """
    Parses OCR plain text to extract nutritional table values, claims, and ingredients.
    """
    text_lower = text.lower()
    parsed_nutrients = {}
    
    # 1. Extract numerical nutrients
    for nutrient, patterns in REGEX_PATTERNS.items():
        val = 0.0
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    val = float(match.group(1))
                    break
                except ValueError:
                    pass
        parsed_nutrients[nutrient] = val

    # If sodium was captured as salt (salt = sodium * 2.5 / 1000)
    if "salt" in text_lower and parsed_nutrients.get("sodium_mg", 0.0) < 5.0:
        salt_match = re.search(r"salt\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*g", text_lower)
        if salt_match:
            try:
                salt_val = float(salt_match.group(1))
                parsed_nutrients["sodium_mg"] = round(salt_val * 400.0, 1)
            except ValueError:
                pass

    # 2. Extract Claims
    claims_match = re.search(r"front\s+claims?\s*[:\-]?\s*(.+)", text, re.IGNORECASE)
    if claims_match:
        front_claims = claims_match.group(1).split("\n")[0].strip()
    else:
        # Fallback keyword hunting for claims
        possible_claims = []
        for kw in ["high fibre", "no added sugar", "100% whole wheat", "multigrain", "baked not fried", "zero trans fat", "rich in protein"]:
            if kw in text_lower:
                possible_claims.append(kw.title())
        front_claims = ", ".join(possible_claims) if possible_claims else "None"

    # 3. Extract Ingredients
    ing_match = re.search(r"ingredients?\s*[:\-]?\s*(.+)", text, re.IGNORECASE)
    ingredients_text = ing_match.group(1).strip() if ing_match else text

    # 4. Product Name
    prod_match = re.search(r"product\s*(?:name)?\s*[:\-]?\s*(.+)", text, re.IGNORECASE)
    product_name = prod_match.group(1).split("\n")[0].strip() if prod_match else "Scanned Food Product"

    # 5. Run Rules Engine Audits
    hidden_sugars_audit = analyze_hidden_sugars(ingredients_text)
    claim_audit = audit_all_product_claims(front_claims, {
        **parsed_nutrients,
        "ingredients_text": ingredients_text,
        "category": "Packaged Food"
    })
    teaspoon_audit = calculate_sugar_teaspoons(
        sugars_100g=parsed_nutrients["sugars_g"],
        serving_size_g=30.0
    )

    return {
        "product_name": product_name,
        "front_claims": front_claims,
        "ingredients_text": ingredients_text,
        "nutrients_extracted": parsed_nutrients,
        "hidden_sugars_audit": hidden_sugars_audit,
        "claim_audit": claim_audit,
        "teaspoon_audit": teaspoon_audit
    }

def create_sample_label_images():
    """Generates synthetic food label test images in data/ocr/test_labels/."""
    label_dir = os.path.join(PROJECT_ROOT, "src/ocr/test_labels")
    os.makedirs(label_dir, exist_ok=True)
    
    for name, content in SAMPLE_PRESETS.items():
        filename = name.lower().replace(" ", "_").replace("'", "").replace("!", "") + ".png"
        img_path = os.path.join(label_dir, filename)
        
        # Create image card with PIL
        img = Image.new("RGB", (700, 480), color="#FBFBFC")
        draw = ImageDraw.Draw(img)
        
        # Header banner
        draw.rectangle([0, 0, 700, 60], fill="#1E293B")
        draw.text((20, 18), f"LABELTRUTH OCR TEST SAMPLE: {name[:35]}", fill="#FFFFFF")
        
        # Body text
        y = 75
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("PRODUCT:") or line.startswith("FRONT CLAIMS:") or line.startswith("NUTRITIONAL") or line.startswith("INGREDIENTS:"):
                draw.text((25, y), line, fill="#0F172A")
            else:
                draw.text((40, y), line, fill="#334155")
            y += 26
            if y > 450:
                break
                
        img.save(img_path)
    print(f"Created {len(SAMPLE_PRESETS)} sample OCR label test images in {label_dir}")

if __name__ == "__main__":
    create_sample_label_images()
    # Test scanner on first preset
    test_text = SAMPLE_PRESETS["Britannia NutriChoice Digestive Biscuit"]
    res = parse_label_text(test_text)
    print("\n--- OCR SCANNER TEST RESULT ---")
    print(f"Product: {res['product_name']}")
    print(f"Claims Detected: {res['front_claims']}")
    print(f"Extracted Nutrients: {res['nutrients_extracted']}")
    print(f"Claim Audit Overall Status: {res['claim_audit']['overall_status']}")
    print(f"Hidden Sugars Detected: {res['hidden_sugars_audit']['total_hidden_count']} ({res['hidden_sugars_audit']['detected_sugars']})")
    print(f"Teaspoons of Sugar (100g): {res['teaspoon_audit']['teaspoons_per_100g']} tsp")
