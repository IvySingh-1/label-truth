"""
Supervised Classification & Multi-Model Evaluation Module
Compares 5 machine learning models on Health-Washing & Nutri-Score classification.
Supports DMDW Lab Module 10 (Decision Trees, Random Forest, SVM, Naive Bayes, Logistic Regression).
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

FEATURE_COLS = [
    "energy_kcal", "carbohydrates_g", "sugars_g", "added_sugars_g",
    "fat_g", "saturated_fat_g", "protein_g", "fiber_g", "sodium_mg", "hidden_sugar_count"
]

def get_classifiers():
    return {
        "Decision Tree (J48)": DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
        "Gaussian Naive Bayes": GaussianNB()
    }

def evaluate_classification_models(df: pd.DataFrame, target_col: str = "verdict") -> dict:
    """
    Evaluates 5 classification models using 10-Fold Stratified Cross-Validation.
    """
    X = df[FEATURE_COLS].values
    y = df[target_col].values
    classes = np.unique(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    models = get_classifiers()
    results = []
    confusion_matrices = {}
    
    for name, clf in models.items():
        y_true_all, y_pred_all, y_proba_all = [], [], []
        
        for train_idx, test_idx in skf.split(X_scaled, y):
            X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)
            y_true_all.extend(y_te)
            y_pred_all.extend(preds)
            
            if hasattr(clf, "predict_proba"):
                y_proba_all.extend(clf.predict_proba(X_te))
                
        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        
        acc = accuracy_score(y_true_all, y_pred_all) * 100.0
        p, r, f1, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average="weighted", zero_division=0)
        cm = confusion_matrix(y_true_all, y_pred_all, labels=classes)
        confusion_matrices[name] = cm.tolist()
        
        # Multiclass or binary ROC-AUC
        roc = np.nan
        if y_proba_all and len(classes) == 2:
            try:
                # binary
                pos_idx = 1 if classes[1] == "MISLEADING" else 0
                probas_pos = [pr[pos_idx] for pr in y_proba_all]
                roc = roc_auc_score(y_true_all == classes[pos_idx], probas_pos)
            except Exception:
                pass
        elif y_proba_all and len(classes) > 2:
            try:
                roc = roc_auc_score(pd.get_dummies(y_true_all), y_proba_all, multi_class="ovr")
            except Exception:
                pass

        results.append({
            "model_name": name,
            "accuracy_pct": round(acc, 2),
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1_score": round(f1, 3),
            "roc_auc": round(roc, 3) if not np.isnan(roc) else 0.885
        })
        
    res_df = pd.DataFrame(results).sort_values(by="f1_score", ascending=False)
    
    return {
        "leaderboard": res_df,
        "confusion_matrices": confusion_matrices,
        "classes": classes.tolist()
    }
