"""
Teaspoon Sugar Calculator & WHO Guideline Auditor
Converts grams of sugar into household teaspoons (1 tsp = ~4.0g)
and evaluates against the World Health Organization (WHO) daily free-sugar limits.
"""

from typing import Dict

WHO_ADULT_DAILY_MAX_G = 25.0   # ~6 teaspoons
WHO_CHILD_DAILY_MAX_G = 19.0   # ~4.75 teaspoons
GRAMS_PER_TEASPOON = 4.0

def calculate_sugar_teaspoons(sugars_100g: float, serving_size_g: float = 30.0, added_sugars_100g: float = None) -> Dict:
    """
    Computes sugar in teaspoons and checks against WHO thresholds.
    
    Parameters:
        sugars_100g (float): Total sugar in grams per 100g.
        serving_size_g (float): Consumer portion size in grams.
        added_sugars_100g (float): Added refined sugars per 100g (optional).
    
    Returns:
        Dict with teaspoon counts, WHO percent utilization, and visual alert state.
    """
    sugars_100g = max(0.0, float(sugars_100g or 0.0))
    serving_size_g = max(1.0, float(serving_size_g or 30.0))
    
    sugar_per_serving_g = (sugars_100g * serving_size_g) / 100.0
    teaspoons_100g = round(sugars_100g / GRAMS_PER_TEASPOON, 2)
    teaspoons_serving = round(sugar_per_serving_g / GRAMS_PER_TEASPOON, 2)
    
    # Calculate WHO daily limit consumption for this single serving
    who_adult_pct = round((sugar_per_serving_g / WHO_ADULT_DAILY_MAX_G) * 100.0, 1)
    who_child_pct = round((sugar_per_serving_g / WHO_CHILD_DAILY_MAX_G) * 100.0, 1)
    
    # Alert rating
    if who_adult_pct < 20.0:
        alert_level = "GREEN"
        alert_desc = "Low Sugar Impact: Under 20% of WHO daily added sugar ceiling."
    elif who_adult_pct <= 50.0:
        alert_level = "AMBER"
        alert_desc = "Moderate Sugar Warning: Consumes 20% to 50% of WHO daily ceiling in one serving."
    else:
        alert_level = "RED"
        alert_desc = f"Critical Sugar Hazard: Single serving exhausts {who_adult_pct}% of daily WHO limit!"
        
    visual_icons = "🥄 " * min(15, max(1, int(round(teaspoons_serving))))
    
    return {
        "sugars_per_100g": sugars_100g,
        "sugar_per_serving_g": round(sugar_per_serving_g, 2),
        "teaspoons_per_100g": teaspoons_100g,
        "teaspoons_per_serving": teaspoons_serving,
        "serving_size_g": serving_size_g,
        "who_adult_daily_pct": who_adult_pct,
        "who_child_daily_pct": who_child_pct,
        "alert_level": alert_level,
        "alert_desc": alert_desc,
        "visual_spoon_display": visual_icons.strip()
    }
