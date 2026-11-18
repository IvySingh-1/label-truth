"""
Open Food Facts Live API Client
Connects to official Open Food Facts v2 JSON API and search endpoints.
Documentation:
  - Product endpoint: https://world.openfoodfacts.org/api/v2/product/{barcode}.json
  - Search endpoint: https://in.openfoodfacts.org/cgi/search.pl?action=process&json=1
"""

import os
import json
import requests
from typing import Optional, Dict, Any

class OpenFoodFactsClient:
    """Official Open Food Facts v2 API wrapper for Indian packaged food data."""
    
    BASE_URL_WORLD = "https://world.openfoodfacts.org/api/v2/product"
    BASE_URL_INDIA = "https://in.openfoodfacts.org/api/v2/product"
    SEARCH_URL = "https://in.openfoodfacts.org/cgi/search.pl"
    USER_AGENT = "LabelTruth-DMDW-Project/1.0 (contact: student@jiit.ac.in)"

    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self.headers = {
            "User-Agent": self.USER_AGENT,
            "Accept": "application/json"
        }

    def fetch_product_by_barcode(self, barcode: str) -> Optional[Dict[str, Any]]:
        """
        Fetches live product details from Open Food Facts JSON API.
        Endpoint: https://world.openfoodfacts.org/api/v2/product/{barcode}.json
        """
        barcode = str(barcode).strip()
        url = f"{self.BASE_URL_WORLD}/{barcode}.json"
        
        try:
            resp = requests.get(url, headers=self.headers, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == 1 and "product" in data:
                    return self._parse_off_product(data["product"])
        except Exception as e:
            # Fallback or offline graceful handling
            pass
            
        return None

    def search_products(self, query: str, page_size: int = 10) -> list:
        """
        Searches Indian products on Open Food Facts by keyword or brand.
        """
        params = {
            "action": "process",
            "search_terms": query,
            "tagtype_0": "countries",
            "tag_contains_0": "contains",
            "tag_0": "india",
            "page_size": page_size,
            "json": 1
        }
        try:
            resp = requests.get(self.SEARCH_URL, params=params, headers=self.headers, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                products = data.get("products", [])
                return [self._parse_off_product(p) for p in products if p.get("product_name")]
        except Exception:
            pass
        return []

    def _parse_off_product(self, raw: dict) -> dict:
        """Standardizes Open Food Facts raw JSON dictionary into LabelTruth schema."""
        nutr = raw.get("nutriments", {})
        
        energy = nutr.get("energy-kcal_100g") or nutr.get("energy-kcal") or nutr.get("energy_100g", 0) / 4.184
        sugars = nutr.get("sugars_100g", 0.0)
        added_sugars = nutr.get("added-sugars_100g", max(0.0, sugars * 0.7 if sugars > 10 else 0.0))
        fat = nutr.get("fat_100g", 0.0)
        sat_fat = nutr.get("saturated-fat_100g", 0.0)
        trans_fat = nutr.get("trans-fat_100g", 0.0)
        carbs = nutr.get("carbohydrates_100g", 0.0)
        protein = nutr.get("proteins_100g", 0.0)
        fiber = nutr.get("fiber_100g", 0.0)
        sodium = nutr.get("sodium_100g", 0.0) * 1000.0 if nutr.get("sodium_100g") is not None else (nutr.get("salt_100g", 0.0) * 400.0)
        salt = nutr.get("salt_100g", round(sodium * 2.5 / 1000.0, 2))
        
        # Claims extraction from labels_tags
        labels = raw.get("labels_tags", [])
        clean_claims = [l.replace("en:", "").replace("-", " ").title() for l in labels]
        front_claims = ", ".join(clean_claims) if clean_claims else "None"

        # Categories
        cat_tags = raw.get("categories_tags", [])
        primary_cat = cat_tags[0].replace("en:", "").replace("-", " ").title() if cat_tags else "Packaged Food"

        return {
            "product_id": raw.get("code") or raw.get("_id", "unknown"),
            "product_name": raw.get("product_name") or raw.get("product_name_en", "Unknown Product"),
            "brand": raw.get("brands") or raw.get("brand_owner", "Generic Brand"),
            "category": primary_cat,
            "sub_category": cat_tags[1].replace("en:", "").title() if len(cat_tags) > 1 else primary_cat,
            "product_type": primary_cat,
            "front_claims": front_claims,
            "ingredients_text": raw.get("ingredients_text") or raw.get("ingredients_text_en", "Ingredients not specified on packaging."),
            "energy_kcal": round(float(energy or 0.0), 1),
            "carbohydrates_g": round(float(carbs or 0.0), 1),
            "sugars_g": round(float(sugars or 0.0), 1),
            "added_sugars_g": round(float(added_sugars or 0.0), 1),
            "sugar_teaspoons": round(float(sugars or 0.0) / 4.0, 2),
            "fat_g": round(float(fat or 0.0), 1),
            "saturated_fat_g": round(float(sat_fat or 0.0), 1),
            "trans_fat_g": round(float(trans_fat or 0.0), 2),
            "protein_g": round(float(protein or 0.0), 1),
            "fiber_g": round(float(fiber or 0.0), 1),
            "sodium_mg": round(float(sodium or 0.0), 1),
            "salt_g": round(float(salt or 0.0), 2),
            "serving_size_g": 30.0,
            "nutriscore_grade": str(raw.get("nutriscore_grade", "c")).lower(),
            "nutriscore_score": int(raw.get("nutriscore_score", 10)),
            "nova_group": int(raw.get("nova_group", 4)),
            "trust_score": 0.95,
            "image_url": raw.get("image_url") or raw.get("image_front_url", ""),
            "countries": raw.get("countries", "India")
        }
