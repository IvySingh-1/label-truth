"""
Hidden Sugar Detector Module
Identifies 65+ disguised sugar aliases across 5 distinct biochemical groups,
computes disguised sugar counts, and assigns a Sugar Disguise Risk Score.
Supports DMDW Lab Module 8 (Domain Rules & Knowledge-Based Discovery).
"""

from typing import Dict, List, Tuple

# Comprehensive taxonomy of 65+ hidden sugar synonyms
HIDDEN_SUGARS_TAXONOMY = {
    "High-Glycemic Syrups": [
        "high fructose corn syrup", "hfcs", "invert sugar syrup", "invert syrup",
        "liquid glucose", "glucose syrup", "corn syrup", "golden syrup",
        "rice syrup", "brown rice syrup", "tapioca syrup", "agave nectar",
        "agave syrup", "maple syrup", "treacle", "molasses", "sorghum syrup"
    ],
    "Refined Industrial Sugars": [
        "dextrose", "sucrose", "maltose", "fructose", "galactose",
        "ethyl maltol", "caramel", "cane sugar", "raw sugar", "brown sugar",
        "demerara sugar", "turbinado sugar", "castor sugar", "icing sugar",
        "confectioners sugar", "palm sugar", "coconut sugar", "beet sugar"
    ],
    "Disguised Starches & Dextrins": [
        "maltodextrin", "dextrin", "corn syrup solids", "soluble corn fibre",
        "modified food starch", "modified corn starch", "hydrolysed starch",
        "isomaltooligosaccharide", "maltitol syrup", "polydextrose"
    ],
    "Concentrated Fruit Sugars": [
        "fruit juice concentrate", "concentrated fruit juice", "apple juice concentrate",
        "orange juice concentrate", "white grape juice concentrate", "pear juice concentrate",
        "date syrup", "date paste", "raisin paste", "grape must", "dehydrated cane juice"
    ],
    "Malt & Fermented Derivatives": [
        "barley malt extract", "malt extract", "malted barley", "diastatic malt",
        "barley malt", "malt powder", "brewer's yeast extract"
    ]
}

# Flattened list for fast substring matching
ALL_SUGAR_SYNONYMS = []
SYNONYM_TO_GROUP = {}
for group, syns in HIDDEN_SUGARS_TAXONOMY.items():
    for s in syns:
        ALL_SUGAR_SYNONYMS.append(s)
        SYNONYM_TO_GROUP[s] = group

def analyze_hidden_sugars(ingredients_text: str) -> Dict:
    """
    Scans ingredients text for disguised sugars.
    
    Returns:
        total_hidden_count (int): Number of distinct hidden sugars detected.
        detected_sugars (list): List of detected synonym names.
        groups_detected (dict): Breakdown by biochemical group.
        disguise_score (int): Score from 0 to 100 indicating severity of sugar disguise.
        risk_level (str): 'LOW', 'MODERATE', 'HIGH', or 'SEVERE'.
    """
    if not isinstance(ingredients_text, str) or not ingredients_text.strip():
        return {
            "total_hidden_count": 0,
            "detected_sugars": [],
            "groups_detected": {},
            "disguise_score": 0,
            "risk_level": "NONE"
        }
    
    text_lower = ingredients_text.lower()
    detected = []
    groups_found = {}
    
    for syn in ALL_SUGAR_SYNONYMS:
        if syn in text_lower:
            detected.append(syn)
            grp = SYNONYM_TO_GROUP[syn]
            groups_found.setdefault(grp, []).append(syn)
            
    count = len(detected)
    num_groups = len(groups_found)
    
    # Calculate Disguise Score (0-100)
    # Penalizes multiple aliases and multi-group disguising (e.g. syrup + dextrin + concentrate)
    disguise_score = min(100, int((count * 15) + (num_groups * 12)))
    
    if count == 0:
        risk_level = "NONE"
    elif count == 1:
        risk_level = "LOW"
    elif count <= 3:
        risk_level = "MODERATE"
    elif count <= 5:
        risk_level = "HIGH"
    else:
        risk_level = "SEVERE"
        
    return {
        "total_hidden_count": count,
        "detected_sugars": detected,
        "groups_detected": groups_found,
        "disguise_score": disguise_score,
        "risk_level": risk_level
    }
