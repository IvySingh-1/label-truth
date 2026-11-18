"""
FSSAI & CODEX Food Claim Verification Engine
Audits front-of-package marketing claims against nutritional thresholds and ingredients.
Generates itemized claim compliance verdicts: PASS, FAIL, or MISLEADING.
"""

from typing import Dict, List

def audit_single_claim(claim_text: str, product_data: dict) -> dict:
    """
    Audits an individual claim string against actual product nutrition and ingredients.
    """
    c_lower = claim_text.strip().lower()
    cat = str(product_data.get("category", "")).lower()
    is_liquid = "beverage" in cat or "drink" in cat or "milk" in cat or "juice" in cat
    
    sugars = float(product_data.get("sugars_g", 0.0))
    added_sugars = float(product_data.get("added_sugars_g", 0.0))
    fat = float(product_data.get("fat_g", 0.0))
    sat_fat = float(product_data.get("saturated_fat_g", 0.0))
    trans_fat = float(product_data.get("trans_fat_g", 0.0))
    fiber = float(product_data.get("fiber_g", 0.0))
    protein = float(product_data.get("protein_g", 0.0))
    sodium = float(product_data.get("sodium_mg", 0.0))
    energy = float(product_data.get("energy_kcal", 0.0))
    ingredients = str(product_data.get("ingredients_text", "")).lower()

    # Rule 1: No Added Sugar / Sugar Free
    if "no added sugar" in c_lower or "zero added sugar" in c_lower or "without added sugar" in c_lower:
        if added_sugars > 1.0:
            return {
                "claim": claim_text,
                "status": "MISLEADING",
                "reason": f"Product claims 'No Added Sugar' but contains {added_sugars}g added sugar per 100g.",
                "severity": "CRITICAL"
            }
        # Check for disguised syrups in ingredients
        syrup_matches = [s for s in ["maltodextrin", "liquid glucose", "invert sugar", "corn syrup", "fruit juice concentrate"] if s in ingredients]
        if syrup_matches:
            return {
                "claim": claim_text,
                "status": "MISLEADING",
                "reason": f"Technically claims no added cane sugar, but contains hidden concentrated sugars: {', '.join(syrup_matches)}.",
                "severity": "HIGH"
            }
        return {"claim": claim_text, "status": "PASS", "reason": "Verified: Zero refined added sugars and no disguised syrups.", "severity": "NONE"}

    # Rule 2: High Fibre / Rich in Fibre / Fibre
    if "fibre" in c_lower or "fiber" in c_lower:
        if fiber < 3.0:
            return {
                "claim": claim_text,
                "status": "MISLEADING",
                "reason": f"FSSAI requires at least 3.0g fiber/100g for source of fibre. Product has only {fiber}g.",
                "severity": "HIGH"
            }
        if fiber >= 6.0 and (sugars > 15.0 or sat_fat > 5.0):
            return {
                "claim": claim_text,
                "status": "MISLEADING",
                "reason": f"Contains {fiber}g fiber, but is health-washed with excessive sugar ({sugars}g) or saturated fat ({sat_fat}g).",
                "severity": "MODERATE"
            }
        return {"claim": claim_text, "status": "PASS", "reason": f"Meets fibre threshold ({fiber}g per 100g).", "severity": "NONE"}

    # Rule 3: Multigrain / Whole Wheat / Atta
    if "multigrain" in c_lower or "whole wheat" in c_lower or "atta" in c_lower or "oats" in c_lower:
        if "maida" in ingredients or "refined wheat flour" in ingredients:
            return {
                "claim": claim_text,
                "status": "MISLEADING",
                "reason": "Promotes whole grain/multigrain benefits, yet refined wheat flour (Maida) is a dominant ingredient.",
                "severity": "HIGH"
            }
        if fiber < 4.0:
            return {
                "claim": claim_text,
                "status": "MISLEADING",
                "reason": f"Multigrain claim is undermined by low actual dietary fibre ({fiber}g per 100g).",
                "severity": "MODERATE"
            }
        return {"claim": claim_text, "status": "PASS", "reason": "Verified whole grains without maida predominance.", "severity": "NONE"}

    # Rule 4: Baked Not Fried
    if "baked" in c_lower:
        if fat > 18.0 or sat_fat > 5.0:
            return {
                "claim": claim_text,
                "status": "MISLEADING",
                "reason": f"Claims 'Baked Not Fried' to imply low fat, but total fat is high ({fat}g/100g, {sat_fat}g sat fat) via palm oil spray.",
                "severity": "HIGH"
            }
        return {"claim": claim_text, "status": "PASS", "reason": f"Moderate fat content ({fat}g/100g) aligns with baked claim.", "severity": "NONE"}

    # Rule 5: Low Fat / 0% Fat / Fat Free
    if "low fat" in c_lower or "fat free" in c_lower:
        limit = 1.5 if is_liquid else 3.0
        if fat > limit:
            return {
                "claim": claim_text,
                "status": "MISLEADING",
                "reason": f"FSSAI threshold for low fat is {limit}g/100g. Product contains {fat}g.",
                "severity": "HIGH"
            }
        return {"claim": claim_text, "status": "PASS", "reason": f"Meets statutory low fat ceiling ({fat}g/100g).", "severity": "NONE"}

    # Rule 6: High Protein / Protein
    if "protein" in c_lower:
        threshold = 6.0 if is_liquid else 10.0
        if protein < threshold:
            return {
                "claim": claim_text,
                "status": "MISLEADING",
                "reason": f"FSSAI high protein claims require at least {threshold}g protein per 100g. Product has {protein}g.",
                "severity": "MODERATE"
            }
        return {"claim": claim_text, "status": "PASS", "reason": f"High protein verified: {protein}g per 100g.", "severity": "NONE"}

    # Rule 7: 100% Natural / Clean Label
    if "100% natural" in c_lower or "clean" in c_lower or "no preservatives" in c_lower:
        additives = [a for a in ["ins", "emulsifier", "artificial", "synthetic", "preservative", "maltodextrin", "liquid glucose"] if a in ingredients]
        if additives:
            return {
                "claim": claim_text,
                "status": "MISLEADING",
                "reason": f"Advertised as 100% natural/clean, but contains ultra-processed additives/stabilizers: {', '.join(additives[:3])}.",
                "severity": "HIGH"
            }
        return {"claim": claim_text, "status": "PASS", "reason": "No synthetic additives or disguised extracts identified.", "severity": "NONE"}

    # Rule 8: Diet / Light / Slim / Fit
    if "diet" in c_lower or "light" in c_lower or "slim" in c_lower or "fit" in c_lower:
        cal_limit = 20.0 if is_liquid else 40.0
        if energy > 350.0 or sugars > 15.0:
            return {
                "claim": claim_text,
                "status": "MISLEADING",
                "reason": f"Marketed as diet/fitness food, but is energy-dense ({energy} kcal) with substantial sugar ({sugars}g).",
                "severity": "CRITICAL"
            }
        return {"claim": claim_text, "status": "PASS", "reason": "Nutritional density aligns with light profile.", "severity": "NONE"}

    # Default general health claim verification
    if sugars > (7.5 if is_liquid else 15.0) or sat_fat > 5.0 or sodium > 600.0:
        return {
            "claim": claim_text,
            "status": "MISLEADING",
            "reason": f"General wellness claim is contradicted by high red flags: Sugar {sugars}g, Sat Fat {sat_fat}g, Sodium {sodium}mg.",
            "severity": "MODERATE"
        }

    return {"claim": claim_text, "status": "PASS", "reason": "Nutrition characteristics conform to reasonable expectations.", "severity": "NONE"}

def audit_all_product_claims(claims_str: str, product_data: dict) -> dict:
    """
    Splits multi-claim strings and evaluates each claim individually,
    returning aggregate verdict and detailed breakdown.
    """
    if not claims_str or str(claims_str).strip() == "" or str(claims_str).strip().lower() == "none":
        return {
            "has_claims": False,
            "overall_status": "NO_CLAIMS",
            "claims_audited": [],
            "misleading_claims_count": 0,
            "passed_claims_count": 0
        }
        
    claims_list = [c.strip() for c in str(claims_str).split(",") if c.strip() and c.strip().lower() != "none"]
    audits = [audit_single_claim(c, product_data) for c in claims_list]
    
    misleading_count = sum(1 for a in audits if a["status"] == "MISLEADING")
    passed_count = sum(1 for a in audits if a["status"] == "PASS")
    
    overall_status = "MISLEADING" if misleading_count > 0 else "HONEST"
    
    return {
        "has_claims": True,
        "overall_status": overall_status,
        "claims_audited": audits,
        "misleading_claims_count": misleading_count,
        "passed_claims_count": passed_count,
        "total_claims": len(audits)
    }
