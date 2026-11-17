"""
Text Mining & TF-IDF Vectorization Module
Extracts keyword importance from packaging claims and ingredient lists.
Supports DMDW Lab Module 8 (Text Preprocessing & Mining).
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_tfidf_features(df: pd.DataFrame, text_column: str = "ingredients_text", max_features: int = 25):
    """
    Computes TF-IDF matrix for a text column (ingredients or claims).
    
    Returns:
        tfidf_df: DataFrame with top TF-IDF feature weights
        top_words: Series of highest mean TF-IDF terms
        vectorizer: Fitted TfidfVectorizer instance
    """
    clean_texts = df[text_column].fillna("").astype(str)
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b"
    )
    
    tfidf_matrix = vectorizer.fit_transform(clean_texts)
    feature_names = vectorizer.get_feature_names_out()
    
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=feature_names,
        index=df.index
    )
    
    # Calculate global mean TF-IDF score
    mean_scores = tfidf_df.mean(axis=0).sort_values(ascending=False)
    
    return {
        "tfidf_df": tfidf_df,
        "mean_scores": mean_scores,
        "feature_names": feature_names,
        "vectorizer": vectorizer
    }

def analyze_claim_keywords_vs_verdict(df: pd.DataFrame):
    """Correlates claim keywords with misleading vs honest verdicts."""
    tfidf_res = extract_tfidf_features(df, text_column="front_claims", max_features=20)
    claims_tfidf = tfidf_res["tfidf_df"]
    claims_tfidf["is_misleading"] = df["is_misleading"].values
    
    correlations = claims_tfidf.corr()["is_misleading"].drop("is_misleading").sort_values(ascending=False)
    return correlations
